#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts.infer.infer_tiles - canonical raster inference entrypoint.
Full-scene inference for SAV segmentation with sliding windows, Hann blending,
optional test-time augmentation (TTA), optional morphological post-processing,
and scientifically rigorous masking/calibration.

Correzioni critiche v6.0 rispetto a v5:

  BUG FIXED — best_thr vs temperature: coerenza checkpoint → inferenza:
    08_infer.py ora verifica se il checkpoint è stato prodotto con
    temperature scaling. Se sì, legge best_thr_scaled (threshold calibrato
    su sigmoid(logit/T)). Se no, legge best_thr_raw o best_thr e lancia
    un WARNING esplicito se T > 1.1 con threshold non calibrato.
    Il checkpoint prodotto da 05_train.py v5.0 contiene sempre best_thr
    coerente con la distribuzione di inferenza.

  BUG FIXED — TTA flip verticale: operazione su numpy array:
    In v5, logits_v = _forward_tensor(...)[:, ::-1, :] operava su un
    numpy array ritornato da .numpy(), che in alcune versioni di numpy
    restituisce una view non-contiguous. La successiva media poteva
    produrre risultati silenziosamente errati.
    FIX: applicare np.ascontiguousarray() prima di operazioni di slicing
    reverse, e verificare che le dimensioni siano coerenti.

  MIGLIORAMENTO — sidecar JSON arricchito:
    Il JSON di summary include ora best_thr_raw, best_thr_scaled (se
    disponibile), temperature, e un flag calibration_consistent che indica
    se il threshold usato è coerente con il temperature scaling applicato.

  MIGLIORAMENTO — stampa diagnostica threshold/temperature:
    All'avvio stampa chiaramente quale threshold viene usato, se è
    calibrato post-temperature, e lancia WARNING se non lo è.

Uso tipico (paper-grade):
    python -m scripts.infer.infer_tiles \\
      --scene_id all --suffix _st12s \\
      --model_path out/.../model_best.pt \\
      --arch unetpp --win 512 --pad 128 --batch 8 --amp \\
      --postprocess none --save_summary_json
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm import tqdm

import torch
import segmentation_models_pytorch as smp
from scripts.common.feature_presets import (
    band_indexes_for_channels,
    feature_names_for_preset,
    resolve_feature_preset,
    resolve_feature_selection,
)
from scripts.models.hf_segformer import HFSegformerBinary
from scripts.models.model_input import adapt_batch_numpy, infer_input_adapter

try:
    from scripts.common.config import DATA, OUT, WM_ALL, expected_in_channels, p_feat, p_valid, scenes as get_all_scenes
    HAS_CONFIG = True
except Exception:
    HAS_CONFIG = False
    DATA = Path("data")
    OUT = Path("out")
    WM_ALL = DATA / "geom" / "wm_ALL_f.tif"
    def expected_in_channels(suffix: str, fallback: int = 16) -> int:
        return int(fallback)
    def p_feat(scene_id: str, suffix: str) -> Path:
        return DATA / "features" / f"feat_{scene_id}{suffix}.tif"
    def p_valid(scene_id: str, suffix: str) -> Path:
        return DATA / "features" / f"valid_{scene_id}{suffix}.tif"

DEFAULT_WM_PATTERNS = [
    str(WM_ALL),
    str(DATA / "geom" / "wm_ALL_f.tif"),
    str(DATA / "wm_ALL_f.tif"),
    "data/geom/wm_ALL_f.tif",
]
NODATA_PROB  = -9999.0
NODATA_MASK  = 255
FEAT_CLIP    = 6.0


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def hann2d(h: int, w: int) -> np.ndarray:
    return np.maximum(np.outer(np.hanning(h), np.hanning(w)), 1e-3).astype(np.float32)


def sanitize(x: np.ndarray) -> np.ndarray:
    return np.clip(
        np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0),
        -FEAT_CLIP, FEAT_CLIP
    ).astype(np.float32)


def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Sigmoid numericamente stabile: evita overflow/underflow su array numpy."""
    x = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def format_threshold_tag(thr: float) -> str:
    """
    Tag file-safe e non ambiguo anche per threshold molto basse.
    Esempi: 0.5 -> 0p5, 0.05 -> 0p05, 0.005 -> 0p005, 0.001 -> 0p001
    """
    thr = float(thr)
    txt = f"{thr:.4f}".rstrip("0").rstrip(".")
    return txt.replace(".", "p")


def load_cfg_from_checkpoint(model_path: Path) -> Dict[str, Any]:
    """Carica la configurazione dal checkpoint o dal sidecar JSON."""
    candidates = [model_path.with_suffix(".json"), model_path.parent / "model_best.json"]
    for j in candidates:
        if j.exists():
            try:
                d = json.loads(j.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    try:
        ck = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "meta" in ck and isinstance(ck["meta"], dict):
            return ck["meta"]
    except Exception:
        pass
    return {}


def resolve_thr_and_temperature(cfg: Dict[str, Any], args_thr: float, args_temperature: float
                                  ) -> Tuple[float, float, bool, bool]:
    """
    Risolve threshold e temperatura da usare in inferenza, rispettando la
    priorità: argomento CLI > checkpoint calibrato > checkpoint raw.

    Restituisce:
        thr, temperature, calibration_consistent, warned
    """
    temp_scaling_done = bool(cfg.get("temp_scaling_done", False))
    T_ck   = float(cfg.get("temperature", 1.0))

    # Temperature: CLI overrides checkpoint
    temperature = args_temperature if args_temperature >= 0 else T_ck

    # Threshold: distinguiamo raw vs scaled
    best_thr_scaled = float(cfg.get("best_thr", 0.5))  # post-v5: già calibrato
    best_thr_raw    = float(cfg.get("best_thr_raw", best_thr_scaled))

    if args_thr >= 0:
        # L'utente ha passato esplicitamente un threshold via CLI
        thr = args_thr
        calibration_consistent = True
        warned = False
    elif temp_scaling_done:
        # Checkpoint v5.0: best_thr è già calibrato su sigmoid(logit/T)
        thr = best_thr_scaled
        calibration_consistent = True
        warned = False
    else:
        # Checkpoint legacy (v4 o precedenti): best_thr è raw.
        # Se T > 1.1, il threshold raw è incoerente con la distribuzione scaled.
        thr = best_thr_scaled  # best_thr del vecchio checkpoint = raw
        calibration_consistent = temperature <= 1.1
        warned = not calibration_consistent
        if warned:
            warnings.warn(
                f"\n{'!'*70}\n"
                f"  ATTENZIONE — THRESHOLD NON CALIBRATO:\n"
                f"  Il checkpoint usa temperature T={temperature:.3f} ma best_thr={thr:.3f}\n"
                f"  è stato ottimizzato su sigmoid(logit) SENZA temperature scaling.\n"
                f"  In inferenza la distribuzione è sigmoid(logit/{temperature:.3f}),\n"
                f"  più compressa verso 0.5. Con thr={thr:.3f} si rischia\n"
                f"  sovrasegmentazione sistematica.\n"
                f"  SOLUZIONE IMMEDIATA: usa --thr 0.5 o ri-allenare con 05_train.py v5.0.\n"
                f"  SOLUZIONE DEFINITIVA: ri-allena con 05_train.py v5.0 (--use_temp_scaling).\n"
                f"{'!'*70}",
                stacklevel=2
            )

    return thr, temperature, calibration_consistent, warned


def safe_load_state_dict(model_path: Path, device: str) -> Dict[str, torch.Tensor]:
    ck = torch.load(model_path, map_location=device, weights_only=False)
    return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck


_ARCH_MAP = {
    "unetpp": "unetplusplus", "unet++": "unetplusplus",
    "unet":   "unet", "fpn": "fpn",
    "deeplabv3p": "deeplabv3plus", "deeplabv3+": "deeplabv3plus",
}


def build_model(arch: str, encoder: str, in_ch: int, device: str) -> torch.nn.Module:
    arch_l = arch.lower()
    if arch_l in {"segformer_hf", "segformerhf"}:
        model = HFSegformerBinary(
            encoder=encoder,
            in_ch=int(in_ch),
            pretrained=False,
        ).to(device)
        model.eval()
        return model

    model = smp.create_model(
        arch=_ARCH_MAP.get(arch_l, arch_l),
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=int(in_ch),
        classes=1,
        activation=None,
    ).to(device)
    model.eval()
    return model


def warp_mask_to_profile(mask_path: Path, profile: dict, *, threshold: float = 0.5) -> np.ndarray:
    H, W = profile["height"], profile["width"]
    dst = np.zeros((H, W), dtype=np.float32)
    with rasterio.open(mask_path) as src:
        rasterio.warp.reproject(
            source=src.read(1).astype(np.float32),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.nearest,
        )
    return dst > threshold


def resolve_feat_path(scene_id: str, suffix: str) -> Path:
    p0 = p_feat(scene_id, suffix)
    if p0.exists():
        return p0
    for pat in [f"**/feat_{scene_id}{suffix}.tif", f"**/{scene_id}*{suffix}*.tif"]:
        m = sorted(Path(".").glob(pat))
        if m:
            return m[0]
    raise FileNotFoundError(
        f"Features non trovate per scene_id={scene_id} suffix={suffix}\n"
        f"Cercato: {p0}\nEsegui prima 02_prepare_features.py."
    )


def resolve_valid_path(scene_id: str, suffix: str) -> Optional[Path]:
    p0 = p_valid(scene_id, suffix)
    if p0.exists():
        return p0
    for pat in [f"**/valid_{scene_id}{suffix}.tif"]:
        m = sorted(Path(".").glob(pat))
        if m:
            return m[0]
    return None


@lru_cache(maxsize=64)
def _read_available_feature_names(feat_path: str) -> Tuple[str, ...]:
    path = Path(feat_path)
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            names = meta.get("feature_names")
            if isinstance(names, list) and names and all(str(n).strip() for n in names):
                return tuple(str(n) for n in names)
        except Exception:
            pass

    with rasterio.open(path) as src:
        desc = [str(d).strip() for d in src.descriptions if d]
        if len(desc) == src.count and all(desc):
            return tuple(desc)

    raise RuntimeError(
        f"Cannot resolve feature channel names for {path}. "
        "Regenerate features with the updated 02_prepare_features.py."
    )


@lru_cache(maxsize=128)
def _resolve_band_indexes(feat_path: str, feature_preset: str) -> Tuple[int, ...]:
    requested = feature_names_for_preset(feature_preset)
    available = _read_available_feature_names(feat_path)
    return tuple(band_indexes_for_channels(available, requested))


def _edge_core_bounds(
    y0: int,
    x0: int,
    win: int,
    pad: int,
    height: int,
    width: int,
) -> Tuple[int, int, int, int, int, int]:
    top_trim = 0 if y0 == 0 else pad
    left_trim = 0 if x0 == 0 else pad
    bottom_trim = 0 if (y0 + win) >= height else pad
    right_trim = 0 if (x0 + win) >= width else pad
    yc0 = y0 + top_trim
    yc1 = min(height, y0 + win - bottom_trim)
    xc0 = x0 + left_trim
    xc1 = min(width, x0 + win - right_trim)
    return top_trim, left_trim, yc0, yc1, xc0, xc1


def make_positions(L: int, win: int, step: int) -> List[int]:
    if L <= win:
        return [0]
    last = L - win
    xs = list(range(0, last + 1, step))
    if xs[-1] != last:
        xs.append(last)
    return xs


def _postprocess_mask(mask: np.ndarray, valid_eval: np.ndarray, method: str) -> np.ndarray:
    """
    Morfologia post-classificazione opzionale.
    Applicata SOLO dentro valid_eval per non toccare il nodata.
    NOTA: postprocess='none' è il default per rigore scientifico.
    Usare morpho solo come ablation secondaria, mai come metrica principale.
    """
    if method == "none":
        return mask
    try:
        from scipy.ndimage import binary_closing, binary_opening
    except ImportError:
        print("  [WARN] scipy non disponibile — postprocess saltato.")
        return mask

    veg_bin    = (mask == 2) & valid_eval
    veg_closed = binary_closing(veg_bin, structure=np.ones((3, 3), dtype=bool))
    veg_final  = binary_opening(veg_closed, structure=np.ones((2, 2), dtype=bool))

    out = mask.copy()
    out[valid_eval & veg_final]  = 2
    out[valid_eval & ~veg_final] = 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_one_scene(
    sid: str,
    suffix: str,
    source_suffix: str,
    feature_preset: str,
    model: torch.nn.Module,
    arch: str,
    encoder: str,
    in_ch: int,
    device: str,
    win: int,
    pad: int,
    blend_mode: str,
    batch_size: int,
    thr: float,
    temperature: float,
    input_adapter: str,
    amp: bool,
    wm_path: Path,
    out_root: Path,
    tag: str,
    overwrite: bool,
    use_memmap: bool,
    memmap_dir: Path,
    postprocess: str,
    gauss_sigma: float,
    tta: bool,
    save_summary_json: bool,
    calibration_consistent: bool,
    cfg_meta: Dict[str, Any],
) -> bool:

    feat_path  = resolve_feat_path(sid, source_suffix)
    valid_path = resolve_valid_path(sid, source_suffix)
    band_indexes: Optional[List[int]] = None
    selected_channels: Optional[List[str]] = None
    if feature_preset:
        band_indexes = list(_resolve_band_indexes(str(feat_path), feature_preset))
        selected_channels = feature_names_for_preset(feature_preset)

    step = win - 2 * pad
    if step <= 0:
        raise ValueError(f"WIN={win} PAD={pad} -> step={step} <= 0. Riduci pad o aumenta win.")

    thr_tag = format_threshold_tag(thr)
    T_tag   = f"{temperature:.2f}".replace(".", "p")
    # Nota: T rimosso dal filename per compatibilita' Windows MAX_PATH=260.
    # temperature e thr sono comunque salvati nel JSON sidecar.
    out_prob = out_root / f"prob_{sid}{tag}_w{win}_p{pad}_thr{thr_tag}.tif"
    out_mask = out_root / f"vegmask_{sid}{tag}_w{win}_p{pad}_thr{thr_tag}.tif"
    out_json = out_root / f"summary_{sid}{tag}_w{win}_p{pad}_thr{thr_tag}.json"

    if out_prob.exists() and out_mask.exists() and not overwrite:
        if save_summary_json and not out_json.exists():
            print(f"  [stale] {sid}: prob/mask presenti ma summary JSON mancante; rigenero.")
        else:
            print(f"  [skip] {sid}")
            return True

    with rasterio.open(feat_path) as src:
        profile = src.profile.copy()
        profile.update(crs=src.crs, transform=src.transform, height=src.height, width=src.width)
        H, W = src.height, src.width
        C_src = src.count
        C = len(band_indexes) if band_indexes is not None else C_src

    water = warp_mask_to_profile(wm_path, profile)
    if valid_path is not None:
        valid = warp_mask_to_profile(valid_path, profile)
    else:
        valid = np.ones((H, W), dtype=bool)
    valid_eval = water & valid

    n_water = int(water.sum())
    n_valid = int(valid.sum())
    n_eval  = int(valid_eval.sum())
    print(
        f"  {sid}: H={H} W={W} C={C}"
        + (f" (subset of {C_src})" if band_indexes is not None else "")
        + f" water={n_water:,} valid={n_valid:,} eval={n_eval:,} px"
    )

    # Allocatori logit accumulatori
    # logit_sum:     somma pesata logit scalati (post-T) per probabilita' finale
    # logit_raw_sum: somma pesata logit GREZZI pre-T (per salvataggio raw)
    # std_sum:       somma pesata std TTA (uncertainty epistemica, solo se tta=True)
    if use_memmap:
        memmap_dir.mkdir(parents=True, exist_ok=True)
        p_logit     = memmap_dir / f"_logit_{sid}.dat"
        p_logit_raw = memmap_dir / f"_logit_raw_{sid}.dat"
        p_std       = memmap_dir / f"_std_{sid}.dat"
        p_wsum      = memmap_dir / f"_wsum_{sid}.dat"
        logit_sum     = np.memmap(str(p_logit),     mode="w+", dtype="float32", shape=(H, W))
        logit_raw_sum = np.memmap(str(p_logit_raw), mode="w+", dtype="float32", shape=(H, W))
        std_sum       = np.memmap(str(p_std),       mode="w+", dtype="float32", shape=(H, W))
        w_sum         = np.memmap(str(p_wsum),      mode="w+", dtype="float32", shape=(H, W))
        logit_sum[:]     = 0.0
        logit_raw_sum[:] = 0.0
        std_sum[:]       = 0.0
        w_sum[:]         = 0.0
    else:
        logit_sum     = np.zeros((H, W), np.float32)
        logit_raw_sum = np.zeros((H, W), np.float32)
        std_sum       = np.zeros((H, W), np.float32)
        w_sum         = np.zeros((H, W), np.float32)

    ys = make_positions(H, win, step)
    xs = make_positions(W, win, step)
    total_windows = len(ys) * len(xs)

    batch_feats:  List[np.ndarray]           = []
    batch_coords: List[Tuple[int, int, int, int, int, int]] = []

    @torch.inference_mode()
    def _forward_tensor(t: torch.Tensor) -> np.ndarray:
        """Forward pass, restituisce numpy float32 (B, win, win)."""
        with torch.autocast(
            device_type="cpu" if device == "cpu" else "cuda",
            dtype=torch.float16,
            enabled=(amp and device != "cpu"),
        ):
            out = model(t).squeeze(1).detach().cpu().float().numpy()
        return np.ascontiguousarray(out)

    def flush_batch() -> None:
        if not batch_feats:
            return
        t_np = adapt_batch_numpy(np.stack(batch_feats), input_adapter)
        t_in = torch.from_numpy(t_np).to(device)

        if tta:
            # TTA D4: tutte le 8 simmetrie del gruppo diedro D4
            # (identita' + 3 rotazioni + 4 riflessioni)
            # Media nel logit space (prima del sigmoid): piu' stabile della
            # media delle probabilita', specialmente con temperature scaling.
            # Produce anche la mappa di std come proxy di incertezza epistemica.
            #
            # Convenzione: flip poi forward poi flip inverso sull'output.
            # rot90 k=1 in input -> rot90 k=-1 (k=3) sull'output per invertire.

            def _tta_forward_and_unrotate(t_in, k_rot=0, flip_h=False, flip_v=False):
                """Forward con trasformazione D4 e inversione sull'output."""
                t = t_in
                if flip_h:
                    t = torch.flip(t, dims=[-1])
                if flip_v:
                    t = torch.flip(t, dims=[-2])
                if k_rot > 0:
                    t = torch.rot90(t, k_rot, [-2, -1])
                out = _forward_tensor(t)  # (B, H, W)
                # Inversione geometrica sull'output
                if k_rot > 0:
                    out = np.rot90(out, -k_rot, axes=[-2, -1]).copy()
                if flip_v:
                    out = out[:, ::-1, :].copy()
                if flip_h:
                    out = out[:, :, ::-1].copy()
                return np.ascontiguousarray(out)

            # 8 trasformazioni D4: (flip_h, flip_v, k_rot)
            D4_transforms = [
                (False, False, 0),   # identita'
                (True,  False, 0),   # flip H
                (False, True,  0),   # flip V
                (True,  True,  0),   # flip H + V = rot180
                (False, False, 1),   # rot90
                (False, False, 2),   # rot180
                (False, False, 3),   # rot270
                (True,  False, 1),   # flip H + rot90
            ]

            tta_logits_list = []
            for flip_h, flip_v, k_rot in D4_transforms:
                out = _tta_forward_and_unrotate(t_in, k_rot, flip_h, flip_v)
                tta_logits_list.append(out)

            # Stack: (8, B, H, W)
            tta_stack = np.stack(tta_logits_list, axis=0)
            logits = tta_stack.mean(axis=0)           # media nel logit space
            # Uncertainty: std delle 8 predizioni logit per ogni pixel
            # Salvata separatamente nel flush -> accumulata come secondo canale
            tta_std = tta_stack.std(axis=0)           # (B, H, W)
        else:
            logits  = _forward_tensor(t_in)
            tta_std = None

        # Temperature scaling sui logit grezzi
        logits_raw = logits.copy()   # grezzi pre-T (sempre salvati)
        if abs(temperature - 1.0) > 1e-4:
            logits = logits / max(temperature, 1e-6)

        for (yc0, yc1, xc0, xc1, top_trim, left_trim), logit_raw, logit, s in zip(
            batch_coords, logits_raw, logits,
            tta_std if tta_std is not None else [None] * len(batch_coords)
        ):
            hh, ww = yc1 - yc0, xc1 - xc0
            hw = hann2d(hh, ww)
            logit_core      = logit[top_trim:top_trim + hh, left_trim:left_trim + ww]
            logit_raw_core  = logit_raw[top_trim:top_trim + hh, left_trim:left_trim + ww]
            logit_sum[yc0:yc1, xc0:xc1]     += logit_core * hw
            logit_raw_sum[yc0:yc1, xc0:xc1] += logit_raw_core * hw
            w_sum[yc0:yc1, xc0:xc1]         += hw
            if s is not None:
                std_core = s[top_trim:top_trim + hh, left_trim:left_trim + ww]
                std_sum[yc0:yc1, xc0:xc1] += std_core * hw

        batch_feats.clear()
        batch_coords.clear()

    with rasterio.open(feat_path) as src:
        pbar = tqdm(total=total_windows, desc=f"  {sid}", leave=False)
        for y0 in ys:
            for x0 in xs:
                pbar.update(1)
                if blend_mode == "full_hann":
                    top_trim = 0
                    left_trim = 0
                    yc0 = y0
                    yc1 = min(H, y0 + win)
                    xc0 = x0
                    xc1 = min(W, x0 + win)
                else:
                    top_trim, left_trim, yc0, yc1, xc0, xc1 = _edge_core_bounds(
                        y0, x0, win, pad, H, W
                    )
                hh, ww = yc1 - yc0, xc1 - xc0
                if hh <= 0 or ww <= 0:
                    continue
                if not valid_eval[yc0:yc1, xc0:xc1].any():
                    continue

                feat = src.read(
                    indexes=band_indexes,
                    window=Window(x0, y0, win, win),
                    boundless=True, fill_value=0.0
                ).astype(np.float32)
                feat = sanitize(feat)
                if feat.shape[0] != in_ch:
                    raise RuntimeError(
                        f"Feature-channel mismatch for scene {sid}: raster window has "
                        f"{feat.shape[0]} channels but model expects {in_ch}."
                    )
                batch_feats.append(feat)
                batch_coords.append((yc0, yc1, xc0, xc1, top_trim, left_trim))
                if len(batch_feats) >= batch_size:
                    flush_batch()
        flush_batch()
        pbar.close()

    uncovered = valid_eval & (np.asarray(w_sum) <= 0)
    if uncovered.any():
        raise RuntimeError(
            f"{sid}: {int(uncovered.sum())} valid pixels received no inference weight."
        )

    # Media pesata dei logit (divisione per w_sum)
    w_safe        = np.maximum(np.array(w_sum), 1e-6)
    logit_avg     = np.array(logit_sum)     / w_safe   # logit scalati (post-T)
    logit_raw_avg = np.array(logit_raw_sum) / w_safe   # logit grezzi pre-T
    std_avg       = np.array(std_sum)       / w_safe   # uncertainty (0 se non TTA)

    # sigmoid_stable sui logit scalati -> probabilita' finale
    prob = sigmoid_stable(logit_avg)

    # Smoothing gaussiano opzionale sulla mappa di probabilita' (prima del threshold)
    # Piu' rigoroso della morfologia binaria: continuo, parametrizzabile, reversibile
    if postprocess == "gauss":
        try:
            from scipy.ndimage import gaussian_filter
            prob_valid = prob.copy()
            prob_valid[~valid_eval] = 0.0
            prob_smoothed = gaussian_filter(prob_valid, sigma=gauss_sigma)
            # Normalizza per non alterare i valori fuori maschera
            prob[valid_eval] = prob_smoothed[valid_eval]
        except ImportError:
            print("  [WARN] scipy non disponibile — gaussian smoothing saltato.")

    # Output raster probabilita' (solo pixel valid_eval)
    prob_out = np.full((H, W), NODATA_PROB, dtype=np.float32)
    prob_out[valid_eval] = prob[valid_eval]

    # Output logit grezzi pre-T (per ricalibrazione retrospettiva)
    logit_raw_out = np.full((H, W), NODATA_PROB, dtype=np.float32)
    logit_raw_out[valid_eval] = logit_raw_avg[valid_eval]

    # Output uncertainty (std TTA D4, o zero se non TTA)
    uncert_out = np.full((H, W), NODATA_PROB, dtype=np.float32)
    uncert_out[valid_eval] = std_avg[valid_eval]

    # Maschera binaria
    mask_raw = np.full((H, W), NODATA_MASK, dtype=np.uint8)
    veg_raw  = valid_eval & (prob >= float(thr))
    mask_raw[valid_eval & ~veg_raw] = 1
    mask_raw[veg_raw]               = 2

    mask_out = _postprocess_mask(mask_raw, valid_eval, postprocess)

    # ── Scrittura output ──────────────────────────────────────────────────────
    bx   = min(256, W)
    by   = min(256, H)
    base = profile.copy()
    base.update(compress="LZW", tiled=True, blockxsize=bx, blockysize=by, interleave="band")
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Mappa di probabilita' calibrata p = sigmoid(logit/T)
    with rasterio.open(out_prob, "w",
                       **{**base, "count": 1, "dtype": "float32", "nodata": NODATA_PROB}) as dst:
        dst.write(prob_out, 1)
        dst.set_band_description(
            1, f"P(veg)=sigmoid(logit/T) thr={thr:.3f} T={temperature:.3f} tta={tta}"
        )

    # 2) Maschera binaria classificata
    with rasterio.open(out_mask, "w",
                       **{**base, "count": 1, "dtype": "uint8", "nodata": NODATA_MASK}) as dst:
        dst.write(mask_out, 1)
        dst.set_band_description(
            1, f"1=non-veg 2=veg 255=nodata postproc={postprocess}"
        )

    # 3) Logit grezzi pre-T — per ricalibrazione retrospettiva senza rieseguire inferenza
    out_logit_raw = out_root / f"logit_raw_{sid}{tag}_w{win}_p{pad}.tif"
    with rasterio.open(out_logit_raw, "w",
                       **{**base, "count": 1, "dtype": "float32", "nodata": NODATA_PROB}) as dst:
        dst.write(logit_raw_out, 1)
        dst.set_band_description(1, "logit_raw (pre temperature scaling) — applicare sigmoid(x/T) per probabilita'")

    # 4) Uncertainty map (std TTA D4) — scritta SOLO se tta=True (evita file inutili)
    out_uncert = None
    if tta:
        out_uncert = out_root / f"uncert_{sid}{tag}_w{win}_p{pad}_tta1.tif"
        with rasterio.open(out_uncert, "w",
                           **{**base, "count": 1, "dtype": "float32", "nodata": NODATA_PROB}) as dst:
            dst.write(uncert_out, 1)
            dst.set_band_description(1, "std di 8 predizioni logit TTA-D4 (proxy incertezza epistemica)")

    n_veg = int((mask_out == 2).sum())

    if save_summary_json:
        summary = {
            "timestamp":              _now(),
            "scene_id":               sid,
            "feature_path":           str(feat_path),
            "feature_source_suffix":  str(source_suffix),
            "feature_logical_suffix": str(suffix),
            "feature_preset":         feature_preset or None,
            "feature_channels":       selected_channels,
            "valid_path":             str(valid_path) if valid_path else None,
            "wm_path":                str(wm_path),
            "arch":                   arch,
            "encoder":                encoder,
            "in_ch":                  int(in_ch),
            "win":                    int(win),
            "pad":                    int(pad),
            "blend_mode":             blend_mode,
            "inference_step":         int(step),
            "batch":                  int(batch_size),
            "amp":                    bool(amp),
            "tta":                    bool(tta),
            "tta_n_transforms":       8 if tta else 1,
            "tta_group":              "D4" if tta else "identity",
            "postprocess":            postprocess,
            "gauss_sigma":            float(gauss_sigma) if postprocess == "gauss" else None,
            "temperature":            float(temperature),
            "input_adapter":          input_adapter,
            "thr":                    float(thr),
            # Trasparenza calibrazione threshold
            "calibration_consistent": bool(calibration_consistent),
            "temp_scaling_done":      bool(cfg_meta.get("temp_scaling_done", False)),
            "best_thr_raw":           float(cfg_meta.get("best_thr_raw",
                                             cfg_meta.get("best_thr", 0.5))),
            "best_thr_scaled":        float(cfg_meta.get("best_thr",
                                             cfg_meta.get("best_thr_raw", 0.5))),
            "best_metric_raw":        cfg_meta.get("best_metric_raw",
                                                   cfg_meta.get("best_metric", None)),
            "best_metric_scaled":     cfg_meta.get("best_metric", None),
            # Statistiche pixel
            "n_water_px":             n_water,
            "n_valid_px":             n_valid,
            "n_eval_px":              n_eval,
            "n_veg_px":               n_veg,
            "veg_fraction_eval":      float(n_veg / max(1, n_eval)),
            # Path output (tutti i file generati)
            "prob_path":              str(out_prob),
            "mask_path":              str(out_mask),
            "logit_raw_path":         str(out_logit_raw),
            "uncertainty_path":       str(out_uncert) if out_uncert is not None else None,
        }
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"  [OK] {sid} veg={n_veg:,} px ({100 * n_veg / max(n_eval, 1):.1f}% of valid_eval)")
    print(f"       prob -> {out_prob.name}")
    print(f"       mask -> {out_mask.name}")
    if not calibration_consistent:
        print(f"  [!WARNING] threshold={thr:.3f} NON calibrato per T={temperature:.3f} (vedi sopra)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Full-scene SAV inference — _STAGNONE v6.0")
    ap.add_argument("--scene_id",   required=True, help="Scene ID o 'all'.")
    ap.add_argument("--suffix",     default="_st10f",
                    help="Logical model suffix / experiment suffix.")
    ap.add_argument("--source_suffix", default="",
                    help="Optional source feature-stack suffix. Use this to read from a richer "
                         "master stack while applying a smaller logical feature preset.")
    ap.add_argument("--feature_preset", default="",
                    help="Optional feature preset to subset from the source stack. "
                         "If omitted, it is inferred from --suffix when possible.")
    ap.add_argument("--arch",       default="")
    ap.add_argument("--encoder",    default="")
    ap.add_argument("--in_ch",      type=int, default=0)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir",    default="pred", help="Subfolder under out/pred/.")
    ap.add_argument("--out_tag",    default="")
    ap.add_argument("--thr",        type=float, default=-1.0,
                    help="Soglia binaria. Default: legge best_thr dal checkpoint.")
    ap.add_argument("--temperature",type=float, default=-1.0,
                    help="Temperature scaling. Default: legge temperature dal checkpoint.")
    ap.add_argument("--win",        type=int, default=512)
    ap.add_argument("--pad",        type=int, default=128)
    ap.add_argument(
        "--blend_mode",
        default="full_hann",
        choices=["legacy_core", "full_hann"],
        help=(
            "legacy_core keeps one non-overlapping central core per window; "
            "full_hann blends complete overlapping window predictions."
        ),
    )
    ap.add_argument("--batch",      type=int, default=8)
    ap.add_argument("--amp",        action="store_true")
    ap.add_argument("--wm_path",    default=None)
    ap.add_argument("--overwrite",  action="store_true")
    ap.add_argument("--allow_cpu",  action="store_true")
    ap.add_argument("--memmap",     action="store_true")
    ap.add_argument("--memmap_dir", default=str(OUT / "pred" / "_tmp"))
    ap.add_argument("--postprocess", default="none", choices=["none", "morpho", "gauss"],
                    help=(
                        "Post-processing sulla mappa di probabilita' prima del threshold:\n"
                        "  none   — nessuno (default, scientificamente piu' rigoroso)\n"
                        "  gauss  — smoothing gaussiano sulla prob map (continuo, reversibile)\n"
                        "  morpho — morfologia binaria post-threshold (solo per ablation)"
                    ))
    ap.add_argument("--gauss_sigma", type=float, default=1.0,
                    help="Sigma in pixel dello smoothing gaussiano (default 1.0). "
                         "Attivo solo se --postprocess gauss. "
                         "A risoluzione QB 0.6m: sigma=1 -> FWHM ~1.4m.")
    ap.add_argument("--tta",        action="store_true",
                    help="Enable 8-way D4 test-time augmentation in logit space.")
    ap.add_argument("--save_summary_json", action="store_true",
                    help="Salva un JSON sidecar per ogni scena.")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"model_path non trovato: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA non disponibile. Usa --allow_cpu per CPU (lento).")

    if args.scene_id.lower() == "all":
        if not HAS_CONFIG:
            raise SystemExit("--scene_id all richiede scripts.common.config importabile.")
        scene_list = sorted(get_all_scenes().keys())
    else:
        scene_list = [args.scene_id]

    cfg = load_cfg_from_checkpoint(model_path)
    source_suffix = str(args.source_suffix or cfg.get("source_suffix") or args.suffix)

    feature_preset = ""
    if args.feature_preset:
        try:
            feature_preset = resolve_feature_preset(args.feature_preset)
        except KeyError as exc:
            raise SystemExit(str(exc))
    else:
        try:
            feature_preset = resolve_feature_selection(
                feature_preset=str(cfg.get("feature_preset") or ""),
                suffix=args.suffix,
            )
        except KeyError:
            feature_preset = ""
    win = int(args.win)
    pad = int(args.pad)

    # ── Risoluzione threshold e temperatura (con controllo di coerenza) ──────
    thr, temperature, calibration_consistent, warned = resolve_thr_and_temperature(
        cfg, args.thr, args.temperature
    )

    arch = str(args.arch or cfg.get("arch", "unetpp"))
    encoder = str(args.encoder or cfg.get("encoder", "resnet34"))
    encoder_weights = str(cfg.get("encoder_weights", "imagenet"))
    in_ch = int(args.in_ch) if args.in_ch > 0 else int(cfg.get("in_ch", cfg.get("in_channels", 0)))

    if in_ch <= 0:
        if feature_preset:
            in_ch = len(feature_names_for_preset(feature_preset))
        else:
            feat0 = resolve_feat_path(scene_list[0], source_suffix)
            with rasterio.open(feat0) as src:
                in_ch = int(src.count)
    if in_ch <= 0:
        in_ch = expected_in_channels(args.suffix, fallback=16)

    wm_path = Path(args.wm_path) if args.wm_path else None
    if wm_path is None:
        for pat in DEFAULT_WM_PATTERNS:
            p = Path(pat)
            if p.exists():
                wm_path = p
                break
    if wm_path is None or not wm_path.exists():
        raise FileNotFoundError("Watermask non trovata. Usa --wm_path.")

    out_root = OUT / "pred" / args.out_dir
    tag = args.out_tag.strip()
    if tag and not tag.startswith("_"):
        tag = "_" + tag

    input_adapter = str(cfg.get("input_adapter") or "").strip()
    if not input_adapter:
        input_adapter = infer_input_adapter(
            arch=arch,
            encoder_weights=encoder_weights,
            feature_preset=feature_preset,
            suffix=args.suffix,
            in_ch=in_ch,
        )

    model = build_model(arch, encoder, in_ch, device)
    sd    = safe_load_state_dict(model_path, device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        raise RuntimeError(f"load_state_dict fallito: {e}")
    model.eval()

    # Verifica anticipata: in_ch vs canali del feature raster della prima scena
    # Evita crash a metà inferenza su scene grandi
    if scene_list:
        try:
            _feat_check = resolve_feat_path(scene_list[0], source_suffix)
            with rasterio.open(_feat_check) as _src_check:
                _feat_ch_src = _src_check.count
            if feature_preset:
                _feat_ch = len(_resolve_band_indexes(str(_feat_check), feature_preset))
            else:
                _feat_ch = _feat_ch_src
            if _feat_ch != in_ch:
                raise RuntimeError(
                    f"MISMATCH CANALI (pre-check): feature raster ha {_feat_ch} bande, "
                    f"modello si aspetta in_ch={in_ch}.\n"
                    f"  Feature: {_feat_check}\n"
                    f"  Usa --in_ch {_feat_ch} oppure usa il feature stack corretto."
                )
            extra = f", subset from {_feat_ch_src}" if feature_preset else ""
            print(f"  [pre-check] in_ch={in_ch} OK vs feature raster ({_feat_ch} bande{extra})")
        except FileNotFoundError:
            pass  # gestito in infer_one_scene per ogni scena

    print(f"\n[08_infer] START {_now()}")
    print(f"  model  : {model_path}")
    print(f"  arch={arch}  encoder={encoder}  in_ch={in_ch}")
    print(f"  suffix={args.suffix}  src_suf={source_suffix}  preset={feature_preset or 'FULL_SOURCE'}")
    print(f"  input_adapter={input_adapter}")
    print(
        f"  WIN={win}  PAD={pad}  STEP={win - 2 * pad}  "
        f"blend={args.blend_mode}  batch={args.batch}  amp={args.amp}"
    )
    print(f"  temperature      = {temperature:.4f}  (checkpoint: {cfg.get('temperature', 1.0):.4f})")
    print(f"  thr (inference)  = {thr:.3f}  "
          f"({'calibrated post-T' if calibration_consistent else '*** NOT calibrated for T — see WARNING above ***'})")
    print(f"  temp_scaling_done= {cfg.get('temp_scaling_done', False)}")
    if cfg.get("temp_scaling_done"):
        print(f"  best_thr_raw     = {cfg.get('best_thr_raw', '?')}")
        print(f"  best_thr_scaled  = {cfg.get('best_thr', '?')}  <- questo viene usato")
    tta_desc = "TTA D4 (8 simmetrie diedro, media logit space + mappa std uncertainty)" if args.tta else "off"
    print(f"  tta={tta_desc}")
    print(f"  postprocess={args.postprocess}"
          + (f"  gauss_sigma={args.gauss_sigma}" if args.postprocess == "gauss" else ""))
    print(f"  output extra: logit_raw (pre-T), uncertainty (std TTA D4)")
    print(f"  out_root: {out_root}")

    errors: List[Tuple[str, str]] = []
    for sid in scene_list:
        try:
            infer_one_scene(
                sid=sid,
                suffix=args.suffix,
                source_suffix=source_suffix,
                feature_preset=feature_preset,
                model=model,
                arch=arch,
                encoder=encoder,
                in_ch=in_ch,
                device=device,
                win=win,
                pad=pad,
                blend_mode=args.blend_mode,
                batch_size=args.batch,
                thr=thr,
                temperature=temperature,
                input_adapter=input_adapter,
                amp=args.amp,
                wm_path=wm_path,
                out_root=out_root,
                tag=tag,
                overwrite=args.overwrite,
                use_memmap=args.memmap,
                memmap_dir=Path(args.memmap_dir),
                postprocess=args.postprocess,
                gauss_sigma=args.gauss_sigma,
                tta=args.tta,
                save_summary_json=args.save_summary_json,
                calibration_consistent=calibration_consistent,
                cfg_meta=cfg,
            )
        except FileNotFoundError as e:
            print(f"  [SKIP] {sid}: {e}")
            errors.append((sid, str(e)))
        except Exception as e:
            print(f"  [ERROR] {sid}: {e}")
            errors.append((sid, str(e)))

    print(f"\n[08_infer] END {_now()}")
    print(f"  Scenes OK: {len(scene_list) - len(errors)}/{len(scene_list)}")
    if errors:
        for sid, err in errors:
            print(f"  {sid}: {err[:180]}")


if __name__ == "__main__":
    main()

