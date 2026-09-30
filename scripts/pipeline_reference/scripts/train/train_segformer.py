#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts.train.train_segformer - canonical deep segmentation training entrypoint.

Training loop per segmentazione binaria SAV con architetture SMP.

Correzioni critiche v5.0 rispetto a v4:

  BUG #1 FIXED — best_thr ricalibrato DOPO temperature scaling:
    In v4, best_thr veniva ottimizzato su sigmoid(logit_raw) durante il
    training, poi la temperatura T veniva fittata su NLL. Il threshold
    salvato nel checkpoint non era coerente con la distribuzione di
    probabilità scalata: in inferenza sigmoid(logit/T) con T≫1 comprime
    tutto verso 0.5, rendendo il threshold pre-calibration (tipicamente
    0.30) inutilizzabile (sovrasegmentazione sistematica).
    FIX: dopo fit_temperature(), si esegue un nuovo sweep completo su
    sigmoid(logit/T) e si sovrascrive best_thr con il valore calibrato.
    Il checkpoint finale contiene SEMPRE best_thr coerente con la
    distribuzione di probabilità che verrà usata in inferenza.

  BUG #2 FIXED — sweep threshold esteso, denso in basso e raffinato:
    In v4/v5 lo sweep partiva troppo in alto per i casi in cui il modello
    separa bene ma calibra probabilità compresse verso zero.
    FIX: griglia base densa tra 0.001 e 0.04, poi griglia regolare fino
    a 0.90, seguita da raffinamento locale automatico attorno al best thr.

  BUG #3 FIXED — iou_at_05_scaled aggiunta al meta:
    Il leaderboard riporta ora sia iou_at_05_raw (logit grezzi, confronto
    interno training) sia iou_at_05_scaled (post-temperature, metrica
    confrontabile con l'inferenza) e best_metric_scaled.

  MIGLIORAMENTO — per-source validation split:
    Per dataset multi-sorgente (A3_multi), la split train/val ora
    stratifica PER source_id se la colonna è presente, garantendo che
    ogni sorgente (2006, 2016) sia rappresentata nel validation set.
    Questo evita che la val loss sia dominata dalla sorgente maggioritaria.

  MIGLIORAMENTO — training log più completo:
    Ogni epoch stampa anche iou_at_05_raw per monitorare la calibrazione
    durante il training prima del temperature scaling.

Uso:
    python -m scripts.train.train_segformer \\
        --patch_csv out/patches/patches_multi_st12s.csv \\
        --out_dir   out/runs/unetpp_r34_s123 \\
        --arch unetpp --encoder resnet34 --in_ch 12 \\
        --suffix _st12s --tile 256 \\
        --epochs 120 --batch 32 --lr 2e-4 --amp \\
        --use_temp_scaling
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

from scripts.common.config import expected_in_channels, p_pt_store
from scripts.common.feature_presets import feature_names_for_preset, resolve_feature_selection
from scripts.models.hf_segformer import HFSegformerBinary
from scripts.models.model_input import adapt_batch_torch, infer_input_adapter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"The epoch parameter in `scheduler\.step\(\)` was not necessary.*",
    category=UserWarning,
    module=r"torch\.optim\.lr_scheduler",
)

# ── Versione e changelog v5.1 ────────────────────────────────────────────────
# v5.1 rispetto a v5.0:
#   FIX    — rimosso decorator @torch.no_grad() duplicato su _collect_logits
#   FIX    — THR_SWEEP paper-grade: range denso fino a 0.001 + refine locale
#   PERF   — torch.compile opzionale (--compile): +10-20% throughput su PyTorch 2.x
#   PERF   — in-memory dataset cache opzionale (--cache_ram): elimina I/O HDF5 ripetuto
#   PERF   — CutMix schedulato: prob=cutmix_prob*2 nelle prime metà epoche,
#             poi decade linearmente a cutmix_prob/2 (migliora stabilità fine-tuning)
#   MIGL   — logging su file <out_dir>/train.log oltre a stdout
#   MIGL   — best_epoch aggiunto al checkpoint meta
#   MIGL   — balanced_accuracy e MCC aggiunti al checkpoint meta (utili per selezione modello)
#   MIGL   — in_ch leggibile anche da --in_ch nel JSON via 06_suite_train (già funzionava via CLI)
# ─────────────────────────────────────────────────────────────────────────────

# Sweep threshold paper-grade:
# - griglia base densa in basso, perché nei modelli SAV con Focal/Lovasz
#   o calibration imperfetta il threshold ottimo può scendere ben sotto 0.05
# - raffinamento locale automatico attorno al miglior threshold coarse
_THR_SWEEP_BASE = sorted(
    {
        0.001, 0.002, 0.005,
        0.010, 0.020, 0.030, 0.040,
        *np.round(np.arange(0.05, 0.91, 0.05), 4).tolist(),
    }
)


def _make_refined_thresholds(best_thr: float) -> List[float]:
    """
    Raffina localmente il threshold migliore trovato sulla griglia base.

    Regole:
    - se il best è molto basso, raffina con step finissimo
    - altrimenti usa una finestra progressivamente più larga ma ancora densa
    """
    best_thr = float(best_thr)
    if best_thr <= 0.02:
        lo, hi, step = max(0.001, best_thr - 0.010), min(0.050, best_thr + 0.010), 0.001
    elif best_thr <= 0.10:
        lo, hi, step = max(0.001, best_thr - 0.020), min(0.150, best_thr + 0.020), 0.0025
    elif best_thr <= 0.30:
        lo, hi, step = max(0.02, best_thr - 0.040), min(0.40, best_thr + 0.040), 0.005
    else:
        lo, hi, step = max(0.10, best_thr - 0.080), min(0.90, best_thr + 0.080), 0.010

    n_steps = int(round((hi - lo) / step)) + 1
    refined = np.round(np.linspace(lo, hi, max(2, n_steps)), 4).tolist()
    return sorted({float(t) for t in refined if 0.0 < float(t) < 1.0})

_AUGMENT_CFG = {
    "depth_aug_prob": 0.20,
    "cutout_prob": 0.10,
    "profile": "default",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    # Full deterministic cuDNN can crash on this Windows/CUDA stack for some SegFormer runs.
    torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    """Seed per-worker per DataLoader multiprocessing (Windows-safe)."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx(device: torch.device, enabled: bool):
    from contextlib import nullcontext
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    return nullcontext()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mib(n_bytes: int) -> int:
    return int(round(float(n_bytes) / (1024.0 * 1024.0)))


def _gpu_snapshot(device: torch.device) -> Dict[str, Any]:
    snap: Dict[str, Any] = {"device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        snap.update(
            {
                "cuda_index": int(idx),
                "cuda_name": torch.cuda.get_device_name(idx),
                "cuda_total_mib": _mib(props.total_memory),
                "torch_alloc_mib": _mib(torch.cuda.memory_allocated(idx)),
                "torch_reserved_mib": _mib(torch.cuda.memory_reserved(idx)),
                "torch_max_alloc_mib": _mib(torch.cuda.max_memory_allocated(idx)),
                "torch_max_reserved_mib": _mib(torch.cuda.max_memory_reserved(idx)),
            }
        )

    smi = shutil.which("nvidia-smi")
    if smi:
        fields = [
            "index",
            "utilization.gpu",
            "utilization.memory",
            "memory.used",
            "memory.total",
            "temperature.gpu",
            "pstate",
            "power.draw",
            "clocks.sm",
            "clocks.mem",
        ]
        try:
            cp = subprocess.run(
                [
                    smi,
                    f"--query-gpu={','.join(fields)}",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
                check=False,
            )
            if cp.returncode == 0 and cp.stdout.strip():
                vals = [v.strip() for v in cp.stdout.strip().splitlines()[0].split(",")]
                for k, v in zip(fields, vals):
                    snap[f"smi_{k.replace('.', '_')}"] = v
            elif cp.stderr.strip():
                snap["smi_error"] = cp.stderr.strip()[:300]
        except Exception as e:
            snap["smi_error"] = str(e)[:300]
    return snap


def _perf_emit(perf_path: Optional[Path], event: Dict[str, Any]) -> None:
    rec = {"time": _now(), **event}
    line = json.dumps(rec, ensure_ascii=True, sort_keys=True)
    print(f"[PERF_JSON] {line}")
    if perf_path is not None:
        perf_path.parent.mkdir(parents=True, exist_ok=True)
        with perf_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _round_perf(stats: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in stats.items():
        if isinstance(v, float):
            out[k] = round(v, 6)
        else:
            out[k] = v
    return out


def clip_grad_norm_fast(model: nn.Module, grad_clip: float) -> None:
    if grad_clip <= 0:
        return
    try:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, foreach=True)
    except TypeError:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


def make_scaler(enabled: bool):
    enabled = bool(enabled) and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        try:
            from torch.cuda.amp import GradScaler
            return GradScaler(enabled=enabled)
        except Exception:
            return None


def parse_loss_mix(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --loss_mix token '{part}'. Expected format: name:weight")
        k, v = part.split(":", 1)
        out[k.strip().lower()] = float(v)
    return out or {"bce": 1.0}


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def meta_path_for_csv(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + "_meta.json")


def parse_store_list(value: str) -> List[str]:
    parts: List[str] = []
    for chunk in str(value or "").replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def infer_radiometric_aug_channels(suffix: str, in_ch: int) -> int:
    """Infer how many leading channels are raw radiometric inputs."""
    raw_names = {"B", "G", "R", "NIR"}
    try:
        preset = resolve_feature_selection(suffix=suffix)
        feat_names = feature_names_for_preset(preset)
        n_raw = 0
        for name in feat_names:
            if name in raw_names:
                n_raw += 1
            else:
                break
        return max(0, min(int(n_raw), int(in_ch)))
    except Exception:
        return max(0, min(4, int(in_ch)))


def read_pos_weight(csv_path: Path) -> float:
    meta = read_json(meta_path_for_csv(csv_path))
    for k in ("pos_weight", "combined_pos_weight"):
        if k in meta:
            try:
                return float(meta[k])
            except Exception:
                pass
    print("  [WARN] pos_weight not found in the patch meta JSON. Falling back to 1.0. "
          "Rebuild the meta file with --export_only if needed.")
    return 1.0


def fmt_optional_float(value: Any, fmt: str = ".4f", default: str = "N/A") -> str:
    try:
        return format(float(value), fmt)
    except Exception:
        return default


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def split_train_val(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split train/val 80/20.

    Se esiste una colonna 'split'/'set': usa quella (tr* = train, va* = val).

    Se esiste una colonna 'source_id'/'scene_id'/'sensor': stratifica per
    sorgente, garantendo che ogni sorgente sia rappresentata nel val set.
    Questo è critico per A3_multi (2006 + 2016): senza stratificazione
    il val set potrebbe essere dominato dalla sorgente maggioritaria,
    gonfiando la best_metric di training rispetto all'effettiva
    performance cross-temporale.

    Altrimenti: split casuale seed-fissato.
    """
    # 1) Split esplicito nel CSV
    split_col = find_col(df, ["split", "set", "subset", "phase"])
    if split_col:
        s_str = df[split_col].astype(str).str.lower()
        tr = df[s_str.str.startswith("tr")]
        va = df[s_str.str.startswith("va")]
        if len(tr) and len(va):
            return tr.reset_index(drop=True), va.reset_index(drop=True)

    # 2) Stratificazione per sorgente/scena
    src_col = find_col(df, ["source_id", "scene_id", "sensor", "year", "dataset"])
    if src_col is not None:
        sources = df[src_col].unique()
        if len(sources) > 1:
            rng = np.random.default_rng(seed)
            tr_idx: List[int] = []
            va_idx: List[int] = []
            for src in sorted(sources):
                idx = np.where(df[src_col] == src)[0]
                rng.shuffle(idx)
                n_va = max(1, int(0.2 * len(idx)))
                va_idx.extend(idx[:n_va].tolist())
                tr_idx.extend(idx[n_va:].tolist())
            return (df.iloc[tr_idx].reset_index(drop=True),
                    df.iloc[va_idx].reset_index(drop=True))

    # 3) Split casuale semplice
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_va = max(1, int(0.2 * len(df)))
    return (df.iloc[idx[n_va:]].reset_index(drop=True),
            df.iloc[idx[:n_va]].reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════════
# Data augmentation avanzata (label-safe, online)
# ═══════════════════════════════════════════════════════════════════════════════

def _elastic_deformation(
    x: torch.Tensor,   # (C, H, W)
    y: torch.Tensor,   # (H, W)
    w: torch.Tensor,   # (H, W)
    alpha: float = 12.0,
    sigma: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deformazione elastica leggera (Simard et al. 2003, adattata per SAV).
    Genera un campo di spostamento gaussiano e lo applica con grid_sample.

    Motivazione SAV: Posidonia/Cymodocea ha bordi morfologicamente irregolari
    che variano tra anni. La deformazione elastica simula questa variabilita'
    e migliora la generalizzazione sui bordi, che sono i pixel piu' critici
    per IoU. sigma=5 px a risoluzione QB/WV (~3-4 m) produce deformazioni
    biologicamente plausibili.
    """
    _, H, W = x.shape
    # Campo di spostamento random gaussiano
    dx = torch.randn(1, 1, H, W) * alpha
    dy = torch.randn(1, 1, H, W) * alpha
    # Smoothing gaussiano del campo con average pooling come proxy
    k = max(3, int(sigma) * 2 + 1)
    pad_e = k // 2
    dx = F.avg_pool2d(F.pad(dx, [pad_e]*4, mode="reflect"), k, stride=1)
    dy = F.avg_pool2d(F.pad(dy, [pad_e]*4, mode="reflect"), k, stride=1)

    # Griglia normalizzata [-1, 1]
    base_y = torch.linspace(-1, 1, H).view(1, H, 1).expand(1, H, W)
    base_x = torch.linspace(-1, 1, W).view(1, 1, W).expand(1, H, W)
    grid_y = (base_y + dy.squeeze(1) / H).clamp(-1, 1)
    grid_x = (base_x + dx.squeeze(1) / W).clamp(-1, 1)
    grid   = torch.stack([grid_x, grid_y], dim=-1)  # (1, H, W, 2)

    x_d = F.grid_sample(x.unsqueeze(0).float(), grid, mode="bilinear",
                         padding_mode="reflection", align_corners=True).squeeze(0)
    y_d = F.grid_sample(y.float().unsqueeze(0).unsqueeze(0), grid,
                         mode="nearest", padding_mode="border",
                         align_corners=True).squeeze(0).squeeze(0).long()
    w_d = F.grid_sample(w.unsqueeze(0).unsqueeze(0), grid, mode="bilinear",
                         padding_mode="border", align_corners=True).squeeze(0).squeeze(0)
    return x_d, y_d, w_d


def _cutmix_patch(
    x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
    x2: torch.Tensor, y2: torch.Tensor, w2: torch.Tensor,
    min_ratio: float = 0.15,
    max_ratio: float = 0.45,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CutMix cross-patch (Yun et al. 2019, adattato per segmentazione).
    Incolla una regione rettangolare del secondo sample nel primo.
    Le label e i pesi sono sostituiti correttamente nella regione.

    Motivazione SAV cross-year: il modello non deve imparare le
    statistiche globali di radianza di una singola scena. Mescolare
    regioni di scene diverse (2006 QB vs 2016 WV) forza a imparare
    caratteristiche locali invarianti, riducendo il domain shift.
    Applicato solo se il dataset contiene piu' scene (source_id diversi).
    """
    _, H, W = x.shape
    ratio_h = random.uniform(min_ratio, max_ratio)
    ratio_w = random.uniform(min_ratio, max_ratio)
    cut_h   = max(1, int(H * ratio_h))
    cut_w   = max(1, int(W * ratio_w))
    y0 = random.randint(0, H - cut_h)
    x0 = random.randint(0, W - cut_w)
    xm, ym, wm = x.clone(), y.clone(), w.clone()
    xm[:, y0:y0+cut_h, x0:x0+cut_w] = x2[:, y0:y0+cut_h, x0:x0+cut_w]
    ym[  y0:y0+cut_h, x0:x0+cut_w]  = y2[  y0:y0+cut_h, x0:x0+cut_w]
    wm[  y0:y0+cut_h, x0:x0+cut_w]  = w2[  y0:y0+cut_h, x0:x0+cut_w]
    return xm, ym, wm


def _water_column_depth_aug(xb: torch.Tensor) -> torch.Tensor:
    """
    Attenuazione leggera depth-aware sui soli canali radiometrici.
    Più aggressiva su R/NIR, più lieve su B/G.
    """
    n_ch = int(xb.shape[0])
    coeffs = torch.tensor([0.30, 0.45, 0.80, 1.05], dtype=xb.dtype, device=xb.device)[:n_ch]
    delta_d = random.uniform(-0.18, 0.30)
    if delta_d >= 0:
        scale = torch.exp(-coeffs.view(n_ch, 1, 1) * delta_d)
    else:
        scale = torch.exp(coeffs.view(n_ch, 1, 1) * (abs(delta_d) * 0.25))
    return xb * scale


def _random_cutout(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    min_ratio: float = 0.05,
    max_ratio: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Occlusione lieve label-safe: azzera il contributo alla loss nella regione.
    Serve a simulare piccoli disturbi residui / missing data locali.
    """
    _, H, W = x.shape
    area_ratio = random.uniform(min_ratio, max_ratio)
    aspect = random.uniform(0.6, 1.8)
    cut_h = max(1, int((H * W * area_ratio / aspect) ** 0.5))
    cut_w = max(1, int((H * W * area_ratio * aspect) ** 0.5))
    cut_h = min(cut_h, H)
    cut_w = min(cut_w, W)
    y0 = random.randint(0, H - cut_h)
    x0 = random.randint(0, W - cut_w)

    x = x.clone()
    w = w.clone()
    fill = x.mean(dim=(-2, -1), keepdim=True)
    x[:, y0:y0 + cut_h, x0:x0 + cut_w] = fill
    w[y0:y0 + cut_h, x0:x0 + cut_w] = 0.0
    return x, y, w


def augment(
    x: torch.Tensor,   # (C, H, W)  float
    y: torch.Tensor,   # (H, W)     short/long
    w: torch.Tensor,   # (H, W)     float
    n_aug_channels: Optional[int] = None,
    # D4 geometric group (flip + rot90)
    p_flip_h: float = 0.5,
    p_flip_v: float = 0.5,
    p_rot90:  float = 0.5,    # alzato: ora copre tutti e 3 i rot non-identita'
    # Augmentazione spettrale canale-indipendente
    p_ch_jitter: float  = 0.4,
    ch_jitter_std: float = 0.08,   # std per canale (z-scored, quindi 0.08 = piccolo shift)
    p_gamma_blue:  float = 0.3,
    gamma_range: Tuple[float, float] = (0.8, 1.2),  # simula variazione profondita' apparente
    p_rad_gain: float = 0.45,
    rad_gain_std: float = 0.06,
    p_rad_bias: float = 0.35,
    rad_bias_std: float = 0.04,
    p_spectral_tilt: float = 0.35,
    spectral_tilt_std: float = 0.06,
    p_sensor_noise: float = 0.25,
    sensor_noise_std: float = 0.01,
    p_depth_aug: Optional[float] = None,
    p_cutout: Optional[float] = None,
    # Deformazione elastica
    p_elastic:   float = 0.25,
    elastic_alpha: float = 12.0,
    elastic_sigma: float = 5.0,
    # Legacy brightness (mantenuto per compatibilita')
    p_bright: float = 0.0,  # disabilitato, sostituito da ch_jitter
    bright_delta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Augmentazione avanzata per SAV subacqueo.

    Trasformazioni geometriche (D4):
    - Flip H, flip V: invarianti per immagini satellitari nadir
    - Rot90 {1,2,3}: campiona uniformemente le 3 rotazioni non-identita'

    Augmentazione spettrale (label-safe: solo su x, non su y/w):
    - Per-channel jitter gaussiano: simula variazioni radiometriche
      inter-sensoriali e inter-annuali (shift di calibrazione)
    - Gamma sul canale Blue: simula variazione della profondita' apparente
      (B e' il canale piu' sensibile alla colonna d'acqua)

    Deformazione elastica:
    - Piccola deformazione dei bordi SAV: simula variabilita' morfologica
      inter-annuale di Posidonia/Cymodocea
    """
    # ── Geometriche (D4 group) ─────────────────────────────────────────────
    profile = str(_AUGMENT_CFG.get("profile", "default")).strip().lower()
    if profile == "geometric_only":
        p_ch_jitter = 0.0
        p_gamma_blue = 0.0
        p_rad_gain = 0.0
        p_rad_bias = 0.0
        p_spectral_tilt = 0.0
        p_sensor_noise = 0.0
        p_depth_aug = 0.0
        p_cutout = 0.0
        p_elastic = 0.0
        p_bright = 0.0
    elif profile == "geometric_brightness":
        p_ch_jitter = 0.0
        p_gamma_blue = 0.0
        p_rad_gain = 0.0
        p_rad_bias = 0.0
        p_spectral_tilt = 0.0
        p_sensor_noise = 0.0
        p_depth_aug = 0.0
        p_cutout = 0.0
        p_elastic = 0.0
        p_bright = 0.5
        bright_delta = 0.10

    if p_depth_aug is None:
        p_depth_aug = float(_AUGMENT_CFG.get("depth_aug_prob", 0.0))
    if p_cutout is None:
        p_cutout = float(_AUGMENT_CFG.get("cutout_prob", 0.0))

    if random.random() < p_flip_h:
        x = x.flip(-1);  y = y.flip(-1);  w = w.flip(-1)
    if random.random() < p_flip_v:
        x = x.flip(-2);  y = y.flip(-2);  w = w.flip(-2)
    if random.random() < p_rot90:
        k = random.randint(1, 3)
        x = torch.rot90(x, k, [-2, -1])
        y = torch.rot90(y.unsqueeze(0), k, [-2, -1]).squeeze(0)
        w = torch.rot90(w.unsqueeze(0), k, [-2, -1]).squeeze(0)

    # ── Spettrali (label-safe) ─────────────────────────────────────────────
    n_base = int(n_aug_channels) if n_aug_channels is not None else min(4, int(x.shape[0]))
    n_base = max(0, min(n_base, int(x.shape[0])))

    if n_base > 0 and random.random() < p_ch_jitter:
        # Jitter solo sui canali radiometrici grezzi.
        jitter = torch.randn(n_base, 1, 1, dtype=x.dtype, device=x.device) * ch_jitter_std
        x = x.clone()
        x[:n_base] = (x[:n_base] + jitter).clamp(-6.0, 6.0)

    if n_base > 0 and random.random() < p_gamma_blue:
        # Gamma sul canale 0 (Blue): exponentiazione che schiaccia/amplifica
        # le riflessioni basse, simulando variazione della colonna d'acqua.
        # Operato su x z-scored: shift -> [0,1] -> gamma -> shift indietro
        gamma = random.uniform(*gamma_range)
        xB = x[0]
        xB_01 = (xB - xB.min()) / (xB.max() - xB.min() + 1e-6)
        xB_g  = xB_01 ** gamma
        # Riscala per mantenere la media
        x = x.clone()
        x[0] = (xB_g - xB_g.mean()) / (xB_g.std() + 1e-6) * xB.std() + xB.mean()
        x[0] = x[0].clamp(-6.0, 6.0)

    # Radiometria fisicamente plausibile: tocca solo il blocco iniziale di
    # canali radiometrici reali (B,G,R,[NIR]). I canali derivati restano fermi.
    if n_base > 0:
        x = x.clone()
        xb = x[:n_base]
        if random.random() < p_rad_gain:
            gain = 1.0 + torch.randn(n_base, 1, 1, dtype=xb.dtype, device=xb.device) * rad_gain_std
            xb = xb * gain
        if random.random() < p_rad_bias:
            bias = torch.randn(n_base, 1, 1, dtype=xb.dtype, device=xb.device) * rad_bias_std
            xb = xb + bias
        if n_base >= 2 and random.random() < p_spectral_tilt:
            tilt_axis = torch.linspace(-1.0, 1.0, n_base, dtype=xb.dtype, device=xb.device).view(n_base, 1, 1)
            tilt = random.gauss(0.0, spectral_tilt_std)
            xb = xb * (1.0 + tilt_axis * tilt)
        if random.random() < p_sensor_noise:
            xb = xb + torch.randn_like(xb) * sensor_noise_std
        if random.random() < p_depth_aug:
            xb = _water_column_depth_aug(xb)
        x[:n_base] = xb.clamp(-6.0, 6.0)

    # ── Legacy brightness (mantenuto ma default off) ───────────────────────
    if n_base > 0 and p_bright > 0 and random.random() < p_bright:
        factor = 1.0 + random.uniform(-bright_delta, bright_delta)
        x = x.clone()
        x[:n_base] = (x[:n_base] * factor).clamp(-6.0, 6.0)

    # ── Deformazione elastica (ultima: opera su geometria) ─────────────────
    if p_cutout > 0 and random.random() < p_cutout:
        x, y, w = _random_cutout(x, y, w)

    if random.random() < p_elastic:
        x, y, w = _elastic_deformation(x, y.long(), w,
                                        alpha=elastic_alpha, sigma=elastic_sigma)

    return x, y.short(), w


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class PatchDatasetH5(Dataset):
    """
    Carica patch da HDF5 store (prodotto da 04_make_patches --out_h5).
    Layout atteso:
      /patches/<scene_id>__y<y0>_x<x0>_t<tile>/x  float16 (C,T,T)
      /patches/<scene_id>__y<y0>_x<x0>_t<tile>/y  uint8   (T,T)
      /patches/<scene_id>__y<y0>_x<x0>_t<tile>/w  uint8   (T,T)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        h5_path: str,
        tile: int,
        seed: int,
        augment_online: bool = False,
        n_aug_channels: int = 4,
    ) -> None:
        self.h5_path        = str(h5_path)
        self.tile           = int(tile)
        self.augment_online = augment_online
        self.n_aug_channels = int(n_aug_channels)
        self._file          = None

        scene_col = find_col(df, ["scene_id", "scene", "id"])
        col_col   = find_col(df, ["x0", "x", "col", "col_off"])
        row_col   = find_col(df, ["y0", "y", "row", "row_off"])

        if scene_col is None or col_col is None or row_col is None:
            raise ValueError(f"CSV mancano colonne scene_id/x0/y0. Colonne: {list(df.columns)}")

        self.keys = [
            f"{sid}__y{y0}_x{x0}_t{tile}"
            for sid, y0, x0 in zip(
                df[scene_col].astype(str),
                df[row_col].astype(int),
                df[col_col].astype(int),
            )
        ]

        if not Path(self.h5_path).exists():
            raise FileNotFoundError(f"HDF5 store non trovato: {self.h5_path}")

    def __len__(self) -> int:
        return len(self.keys)

    def _get_file(self):
        if self._file is None:
            import h5py
            self._file = h5py.File(self.h5_path, "r", swmr=True)
        return self._file

    def reset_handles(self) -> None:
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass
        self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file = None

    def __getitem__(self, idx: int):
        for attempt in range(4):
            try:
                key = self.keys[idx]
                f   = self._get_file()
                grp = f["patches"][key]
                x   = torch.from_numpy(grp["x"][:]).float()
                y   = torch.from_numpy(grp["y"][:]).long()
                w   = torch.from_numpy(grp["w"][:]).float()
                if self.augment_online:
                    x, y, w = augment(x, y.short(), w, n_aug_channels=self.n_aug_channels)
                    y = y.long()
                return x, y, w
            except Exception as e:
                if attempt == 3:
                    key = self.keys[idx]
                    raise RuntimeError(
                        f"H5 read/augment failed for key={key} in {self.h5_path}: {e}"
                    )
                idx = (idx + 1) % max(1, len(self))

    def __del__(self):
        try:
            if self._file is not None:
                self._file.close()
        except Exception:
            pass


class PatchDatasetH5Multi(Dataset):
    """Concatenate channels from multiple HDF5 stores sharing the same patch keys."""

    def __init__(
        self,
        df: pd.DataFrame,
        h5_paths: Sequence[str],
        tile: int,
        seed: int,
        augment_online: bool = False,
        n_aug_channels: int = 4,
    ) -> None:
        self.h5_paths = [str(Path(p).resolve()) for p in h5_paths]
        self.tile = int(tile)
        self.augment_online = augment_online
        self.n_aug_channels = int(n_aug_channels)
        self._files: Optional[List[Any]] = None

        scene_col = find_col(df, ["scene_id", "scene", "id"])
        col_col = find_col(df, ["x0", "x", "col", "col_off"])
        row_col = find_col(df, ["y0", "y", "row", "row_off"])
        if scene_col is None or col_col is None or row_col is None:
            raise ValueError(f"CSV mancano colonne scene_id/x0/y0. Colonne: {list(df.columns)}")

        self.keys = [
            f"{sid}__y{y0}_x{x0}_t{tile}"
            for sid, y0, x0 in zip(
                df[scene_col].astype(str),
                df[row_col].astype(int),
                df[col_col].astype(int),
            )
        ]

        for path in self.h5_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"HDF5 store non trovato: {path}")

    def __len__(self) -> int:
        return len(self.keys)

    def _get_files(self):
        if self._files is None:
            import h5py
            self._files = [h5py.File(path, "r", swmr=True) for path in self.h5_paths]
        return self._files

    def reset_handles(self) -> None:
        if self._files is None:
            return
        for handle in self._files:
            try:
                handle.close()
            except Exception:
                pass
        self._files = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_files"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._files = None

    def __getitem__(self, idx: int):
        for attempt in range(4):
            try:
                key = self.keys[idx]
                files = self._get_files()
                xs: List[torch.Tensor] = []
                y = None
                w = None
                for file_idx, handle in enumerate(files):
                    grp = handle["patches"][key]
                    xs.append(torch.from_numpy(grp["x"][:]).float())
                    if file_idx == 0:
                        y = torch.from_numpy(grp["y"][:]).long()
                        w = torch.from_numpy(grp["w"][:]).float()
                if y is None or w is None:
                    raise RuntimeError(f"Missing labels/weights for key={key}")
                x = torch.cat(xs, dim=0)
                if self.augment_online:
                    x, y, w = augment(x, y.short(), w, n_aug_channels=self.n_aug_channels)
                    y = y.long()
                return x, y, w
            except Exception as e:
                if attempt == 3:
                    key = self.keys[idx]
                    raise RuntimeError(
                        f"Multi-H5 read/augment failed for key={key} in {self.h5_paths}: {e}"
                    )
                idx = (idx + 1) % max(1, len(self))

    def __del__(self):
        self.reset_handles()


class PatchDatasetPTStore(Dataset):
    """
    Carica patch da store .pt:
      <pt_root>/<scene_id>__y{y0}_x{x0}_t{tile}.pt
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pt_root: Path,
        tile: int,
        seed: int,
        augment_online: bool = False,
        n_aug_channels: int = 4,
    ) -> None:
        self.df             = df.reset_index(drop=True)
        self.pt_root        = Path(pt_root)
        self.tile           = int(tile)
        self.augment_online = augment_online
        self.n_aug_channels = int(n_aug_channels)

        scene_col = find_col(df, ["scene_id", "scene", "id"])
        col_col   = find_col(df, ["x0", "x", "col", "col_off"])
        row_col   = find_col(df, ["y0", "y", "row", "row_off"])

        if scene_col is None or col_col is None or row_col is None:
            raise ValueError(f"CSV mancano colonne scene_id/x0/y0. Colonne: {list(df.columns)}")

        self.scene_ids = df[scene_col].astype(str).tolist()
        self.x0s       = df[col_col].astype(int).tolist()
        self.y0s       = df[row_col].astype(int).tolist()

        if not self.pt_root.exists():
            raise FileNotFoundError(f"pt_root non trovato: {self.pt_root}")

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _pt_path(self, i: int) -> Path:
        return self.pt_root / f"{self.scene_ids[i]}__y{self.y0s[i]}_x{self.x0s[i]}_t{self.tile}.pt"

    def __getitem__(self, idx: int):
        for attempt in range(4):
            try:
                p = self._pt_path(idx)
                if not p.exists():
                    raise FileNotFoundError(str(p))
                d = torch.load(p, map_location="cpu", weights_only=True)
                x = d["x"].float()
                y = d["y"].long()
                w = d["w"].float()
                if self.augment_online:
                    x, y, w = augment(x, y.short(), w, n_aug_channels=self.n_aug_channels)
                    y = y.long()
                return x, y, w
            except Exception as e:
                if attempt == 3:
                    p = self._pt_path(idx)
                    raise RuntimeError(f"PT read/augment failed for {p}: {e}")
                idx = (idx + 1) % max(1, len(self))


class PatchDatasetPTStoreMulti(Dataset):
    """Concatenate channels from multiple PT stores sharing the same patch keys."""

    def __init__(
        self,
        df: pd.DataFrame,
        pt_roots: Sequence[Path],
        tile: int,
        seed: int,
        augment_online: bool = False,
        n_aug_channels: int = 4,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.pt_roots = [Path(p) for p in pt_roots]
        self.tile = int(tile)
        self.augment_online = augment_online
        self.n_aug_channels = int(n_aug_channels)

        scene_col = find_col(df, ["scene_id", "scene", "id"])
        col_col = find_col(df, ["x0", "x", "col", "col_off"])
        row_col = find_col(df, ["y0", "y", "row", "row_off"])
        if scene_col is None or col_col is None or row_col is None:
            raise ValueError(f"CSV mancano colonne scene_id/x0/y0. Colonne: {list(df.columns)}")

        self.scene_ids = df[scene_col].astype(str).tolist()
        self.x0s = df[col_col].astype(int).tolist()
        self.y0s = df[row_col].astype(int).tolist()

        for root in self.pt_roots:
            if not root.exists():
                raise FileNotFoundError(f"pt_root non trovato: {root}")

    def __len__(self) -> int:
        return len(self.scene_ids)

    def _pt_path(self, root: Path, i: int) -> Path:
        return root / f"{self.scene_ids[i]}__y{self.y0s[i]}_x{self.x0s[i]}_t{self.tile}.pt"

    def __getitem__(self, idx: int):
        for attempt in range(4):
            try:
                xs: List[torch.Tensor] = []
                y = None
                w = None
                for root_idx, root in enumerate(self.pt_roots):
                    p = self._pt_path(root, idx)
                    if not p.exists():
                        raise FileNotFoundError(str(p))
                    d = torch.load(p, map_location="cpu", weights_only=True)
                    xs.append(d["x"].float())
                    if root_idx == 0:
                        y = d["y"].long()
                        w = d["w"].float()
                if y is None or w is None:
                    raise RuntimeError(f"Missing labels/weights for idx={idx}")
                x = torch.cat(xs, dim=0)
                if self.augment_online:
                    x, y, w = augment(x, y.short(), w, n_aug_channels=self.n_aug_channels)
                    y = y.long()
                return x, y, w
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"Multi-PT read/augment failed for idx={idx}: {e}")
                idx = (idx + 1) % max(1, len(self))




# ═══════════════════════════════════════════════════════════════════════════════
# In-memory cache dataset (opzionale, --cache_ram)
# ═══════════════════════════════════════════════════════════════════════════════

class CachedDataset(Dataset):
    """
    Wrapper che carica l'intero dataset in RAM al primo accesso.
    Elimina completamente l'I/O HDF5/PT durante il training,
    riducendo il tempo di epoca del 10-30% su sistemi con HDD o NVMe lento.

    Uso tipico: dataset <= 5000 patch × 256×256×16 fp16 ≈ 1.3 GB RAM.
    Non usare se la RAM disponibile è inferiore a 2× la dimensione del dataset.

    Il caricamento avviene al primo __len__/__getitem__ (lazy al primo epoch),
    oppure esplicitamente chiamando .preload().
    """

    def __init__(
        self,
        base_dataset: Dataset,
        augment_online: bool = False,
        n_aug_channels: int = 4,
        precompute_boundary_weight: bool = False,
        boundary_radius: int = 3,
        boundary_factor: float = 3.0,
    ) -> None:
        self.base = base_dataset
        self.augment_online = bool(augment_online)
        self.n_aug_channels = int(n_aug_channels)
        self.precompute_boundary_weight = bool(precompute_boundary_weight)
        self.boundary_radius = int(boundary_radius)
        self.boundary_factor = float(boundary_factor)
        self._cache: Optional[list] = None

    def preload(self) -> None:
        if self._cache is not None:
            return
        n = len(self.base)
        print(f"  [cache_ram] Pre-caricamento {n} patch in RAM ...", flush=True)
        self._cache = []
        for i in range(n):
            x, y, w = self.base[i]
            if self.precompute_boundary_weight:
                bw = compute_boundary_weights(
                    y.long().unsqueeze(0),
                    radius=self.boundary_radius,
                    boundary_factor=self.boundary_factor,
                ).squeeze(0)
                w = w.float() * bw
            # share_memory_() permette ai DataLoader worker (num_workers>0)
            # di accedere agli stessi tensori senza pickle/copia per processo.
            self._cache.append((x.share_memory_(), y.share_memory_(), w.share_memory_()))
        # Stima uso RAM
        if self._cache:
            x, _, _ = self._cache[0]
            mb = x.element_size() * x.nelement() * n / 1e6
            print(f"  [cache_ram] OK - {n} patch, ~{mb:.0f} MB RAM stimati")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        if self._cache is None:
            self.preload()
        item = self._cache[idx]
        if item is None:
            return None
        x, y, w = item
        # FIX (perf, 2026-04-29): clone solo se augment puo' realmente mutare i tensori.
        # Senza augment_online il pipeline a valle (train_one_epoch / _evaluate_streaming)
        #   1. fa x.float()/y.long()/w.float() seguito da .to(device, non_blocking=...) ->
        #      questo produce un nuovo tensore su GPU, lasciando il cache CPU intatto;
        #   2. l'unica eventuale mutazione in-place e' adapt_batch_torch sul x GPU
        #      (vedi model_input.py rgb_imagenet) o cutmix_prob>0 in train_one_epoch,
        #      entrambi su tensori GPU gia' separati.
        # Quindi il clone CPU era spreco puro: ~1.1MB x N_patch x N_epoch di memcpy
        # piu' la perdita del beneficio di share_memory_() sui worker (i tensori
        # clonati non sono shared -> pickling cross-process per ogni batch).
        # Se in futuro qualcuno aggiunge mutazione CPU pre-.to(device), forzare il
        # clone esplicitamente settando augment_online=True.
        if self.augment_online:
            x = x.clone()
            y = y.clone()
            w = w.clone()
            x, y, w = augment(x, y.short(), w, n_aug_channels=self.n_aug_channels)
            y = y.long()
        return x, y, w

# ═══════════════════════════════════════════════════════════════════════════════
# Model factory con inflate ImageNet weights
# ═══════════════════════════════════════════════════════════════════════════════

_ARCH_MAP = {
    "unetpp":       "unetplusplus",
    "unet++":       "unetplusplus",
    "unetplusplus": "unetplusplus",
    "unet":         "unet",
    "fpn":          "fpn",
    "deeplabv3p":   "deeplabv3plus",
    "deeplabv3+":   "deeplabv3plus",
}


def make_model(
    arch: str,
    encoder: str,
    in_ch: int,
    encoder_weights: Optional[str] = "imagenet",
) -> nn.Module:
    arch_l = arch.lower()
    if arch_l in {"segformer_hf", "segformerhf"}:
        pretrained = bool(encoder_weights and encoder_weights.lower() != "none")
        return HFSegformerBinary(
            encoder=encoder,
            in_ch=int(in_ch),
            pretrained=pretrained,
        )

    arch_smp = _ARCH_MAP.get(arch_l, arch_l)

    if encoder_weights and encoder_weights.lower() != "none" and in_ch != 3:
        model = smp.create_model(
            arch=arch_smp,
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )
        _inflate_first_conv(model, in_ch)
    else:
        model = smp.create_model(
            arch=arch_smp,
            encoder_name=encoder,
            encoder_weights=(encoder_weights if (encoder_weights and encoder_weights.lower() != "none") else None),
            in_channels=in_ch,
            classes=1,
            activation=None,
        )
    return model


def _inflate_first_conv(model: nn.Module, in_ch: int) -> None:
    """
    Sostituisce il primo layer conv (3 canali) con uno a in_ch canali.
    I pesi vengono espansi per media-ripetizione (He et al. / Howard et al.).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            old_w     = module.weight.data.clone()
            new_w     = torch.zeros(old_w.shape[0], in_ch, old_w.shape[2], old_w.shape[3])
            repeats   = in_ch // 3
            remainder = in_ch % 3
            new_w[:, :3 * repeats] = old_w.repeat(1, repeats, 1, 1)[:, :3 * repeats]
            if remainder > 0:
                new_w[:, 3 * repeats:] = old_w[:, :remainder]
            new_w = new_w / max(1, repeats + (1 if remainder else 0)) * 1.0
            new_conv = nn.Conv2d(
                in_ch, module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=(module.bias is not None),
            )
            new_conv.weight = nn.Parameter(new_w)
            if module.bias is not None:
                new_conv.bias = nn.Parameter(module.bias.data.clone())
            parent = model
            parts  = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
            print(f"  [inflate] {name}: 3ch -> {in_ch}ch (pesi ImageNet espansi)")
            break


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Boundary weight map
# ═══════════════════════════════════════════════════════════════════════════════

def compute_boundary_weights(
    target: torch.Tensor,   # (B, H, W) long {0, 1, 2}
    radius: int = 3,
    boundary_factor: float = 3.0,
) -> torch.Tensor:
    """
    Calcola una mappa di pesi che amplifica i pixel di bordo della GT.

    I bordi SAV sono i pixel piu' critici per IoU: un modello che sbaglia
    sistematicamente i bordi perde molto piu' di uno che sbaglia pixel interni.
    Questa mappa amplifica i pixel entro `radius` pixel dal bordo della classe
    vegetazione (label=2), senza modificare i pixel interni o non-annotati.

    Implementazione: dilatazione morfologica binaria con max_pool2d.
    boundary_mask = dilate(veg) XOR veg
    weights[boundary_mask] *= boundary_factor

    Motivazione: equivalente a campionare piu' densamente i pixel di bordo
    durante il training — difendibile come forma di focal sampling spaziale.
    Compatibile con il pos_weight gia' presente in LossMix (effetti moltiplicativi).
    """
    veg = (target == 2).float().unsqueeze(1)   # (B, 1, H, W)
    k   = 2 * radius + 1
    # Dilatazione: max_pool2d binario
    dilated = F.max_pool2d(veg, kernel_size=k, stride=1, padding=radius)
    boundary = (dilated - veg).squeeze(1) > 0.5  # (B, H, W) bool
    # Pesi base: 1.0 su tutti i pixel annotati
    w_extra = torch.ones_like(target, dtype=torch.float32)
    w_extra[boundary] = boundary_factor
    return w_extra


# ═══════════════════════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════════════════════

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1.0)
    if gt_sorted.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels[perm].float()
    grad = _lovasz_grad(gt_sorted).to(errors_sorted.device)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    logits = logits.squeeze(1)
    labels = labels.squeeze(1)
    for logit_i, label_i, valid_i in zip(logits, labels, valid_mask):
        keep = valid_i.bool()
        if keep.any():
            losses.append(_lovasz_hinge_flat(logit_i[keep], label_i[keep]))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def tversky_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1.0,
) -> torch.Tensor:
    tp = (probs * target * weight).sum()
    fp = (probs * (1.0 - target) * weight).sum()
    fn = ((1.0 - probs) * target * weight).sum()
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score


def compute_boundary_target(
    target: torch.Tensor,
    radius: int = 3,
) -> torch.Tensor:
    veg = (target == 2).float().unsqueeze(1)
    k = 2 * radius + 1
    dilated = F.max_pool2d(veg, kernel_size=k, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - veg, kernel_size=k, stride=1, padding=radius)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_aux_loss(
    logits: torch.Tensor,
    boundary_target: torch.Tensor,
    valid_weight: torch.Tensor,
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, boundary_target, reduction="none")
    num = (raw * valid_weight).sum()
    den = valid_weight.sum().clamp_min(1.0)
    return num / den


class LossMix(nn.Module):
    def __init__(
        self,
        mix: Dict[str, float],
        pos_weight: float = 1.0,
        label_smooth: float = 0.0,
        boundary_radius: int = 3,
        boundary_factor: float = 3.0,
        use_boundary_weight: bool = True,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ):
        super().__init__()
        self.mix = mix
        self.pw  = pos_weight
        self.ls  = label_smooth
        self.boundary_radius = boundary_radius
        self.boundary_factor = boundary_factor
        self.use_boundary_weight = use_boundary_weight
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)

    def forward(
        self,
        logits: torch.Tensor,   # (B, 1, H, W)
        target: torch.Tensor,   # (B, H, W) long {0,1,2}
        weight: torch.Tensor,   # (B, H, W) float
    ) -> torch.Tensor:
        t_hard = ((target == 2).float()).unsqueeze(1)
        t = t_hard.clone()

        # Boundary-weighted: amplifica pixel di bordo SAV
        if self.use_boundary_weight:
            bw = compute_boundary_weights(
                target,
                radius=self.boundary_radius,
                boundary_factor=self.boundary_factor,
            )
            # Combina i pesi del campionamento con i pesi di bordo
            w = (weight * bw).unsqueeze(1)
        else:
            w = weight.unsqueeze(1)

        if self.ls > 0:
            t = t * (1 - 2 * self.ls) + self.ls
        logits = logits.float()
        loss   = torch.tensor(0.0, device=logits.device)
        # FIX A (perf): calcola sigmoid una sola volta, riutilizzato dai branch
        # dice/focal/tversky. Prima erano fino a 3 sigmoid identici sullo stesso
        # tensore. Matematicamente identico al codice precedente.
        _need_prob = ("dice" in self.mix) or ("focal" in self.mix) or ("tversky" in self.mix)
        prob = torch.sigmoid(logits) if _need_prob else None
        if "bce" in self.mix:
            pw_t = torch.tensor([self.pw], device=logits.device)
            bce  = F.binary_cross_entropy_with_logits(logits, t, weight=w, pos_weight=pw_t)
            loss = loss + self.mix["bce"] * bce
        if "dice" in self.mix:
            num  = (2 * prob * t * w).sum()
            den  = ((prob + t) * w).sum()
            dice = 1.0 - (num + 1) / (den + 1)
            loss = loss + self.mix["dice"] * dice
        if "focal" in self.mix:
            gamma = 2.0
            bce_e = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
            p_t   = prob * t + (1 - prob) * (1 - t)
            focal = bce_e * ((1 - p_t) ** gamma) * w
            loss  = loss + self.mix["focal"] * focal.mean()
        if "tversky" in self.mix:
            tv = tversky_loss(
                prob,
                t,
                w,
                alpha=self.tversky_alpha,
                beta=self.tversky_beta,
            )
            loss = loss + self.mix["tversky"] * tv
        if "lovasz" in self.mix:
            valid_mask = (weight > 0)
            lovasz = lovasz_hinge(logits, t_hard, valid_mask)
            loss  = loss + self.mix["lovasz"] * lovasz
        if "boundary" in self.mix:
            boundary_target = compute_boundary_target(target, radius=max(1, self.boundary_radius))
            boundary_valid = (weight > 0).float().unsqueeze(1)
            boundary_term = boundary_aux_loss(logits, boundary_target, boundary_valid)
            loss = loss + self.mix["boundary"] * boundary_term
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# Collate
# ═══════════════════════════════════════════════════════════════════════════════

def collate_pad(batch, tile: int):
    xs, ys, ws = [], [], []
    for x, y, w in batch:
        if x is None:
            continue
        C, H, W_ = x.shape
        if H < tile or W_ < tile:
            x = F.pad(x, [0, max(0, tile - W_), 0, max(0, tile - H)])
            y = F.pad(y.unsqueeze(0).float(), [0, max(0, tile - W_), 0, max(0, tile - H)]).squeeze(0).long()
            w = F.pad(w.unsqueeze(0), [0, max(0, tile - W_), 0, max(0, tile - H)]).squeeze(0)
        xs.append(x[:, :tile, :tile])
        ys.append(y[:tile, :tile])
        ws.append(w[:tile, :tile])
    if not xs:
        return None
    return torch.stack(xs), torch.stack(ys), torch.stack(ws)


# ═══════════════════════════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

def make_scheduler(opt, epochs: int, warmup_epochs: int, name: str = "cosine"):
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        if name == "cosine":
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
            )
        else:
            main_sched = torch.optim.lr_scheduler.StepLR(
                opt, step_size=max(1, (epochs - warmup_epochs) // 3), gamma=0.5
            )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, main_sched], milestones=[warmup_epochs]
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Train / Evaluate
# ═══════════════════════════════════════════════════════════════════════════════



def _scheduled_cutmix_prob(cutmix_prob: float, epoch: int, total_epochs: int) -> float:
    """
    CutMix schedulato: alta probabilità nella prima metà, decade nella seconda.
    Prima metà:  prob = cutmix_prob * 1.5  (regularizzazione aggressiva)
    Seconda metà: decade linearmente a cutmix_prob * 0.3 (fine-tuning stabile)
    Motivazione: nelle prime epoche si vuole forzare invarianza cross-scene;
    nelle ultime epoche si vuole stabilità della convergenza.
    """
    if cutmix_prob <= 0.0:
        return 0.0
    midpoint = total_epochs / 2.0
    if epoch <= midpoint:
        return min(1.0, cutmix_prob * 1.5)
    else:
        t = (epoch - midpoint) / max(1.0, total_epochs - midpoint)
        return cutmix_prob * 1.5 * (1.0 - t) + cutmix_prob * 0.3 * t

def train_one_epoch(
    model, loader, loss_fn, device, amp: bool, opt, scaler, grad_clip: float,
    input_adapter: str = "identity",
    cutmix_prob: float = 0.0,
    epoch: int = 1,
    total_epochs: int = 1,
    grad_accum_steps: int = 1,
    perf_diag: bool = False,
    perf_diag_batches: int = 0,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    model.train()
    total_loss = torch.zeros((), device=device)
    n     = 0
    step_in_accum = 0
    grad_accum_steps = max(1, int(grad_accum_steps))
    opt.zero_grad(set_to_none=True)

    perf: Optional[Dict[str, Any]] = None
    if perf_diag:
        perf = {
            "batches": 0,
            "profiled_batches": 0,
            "opt_steps": 0,
            "data_wait_s": 0.0,
            "h2d_adapt_s": 0.0,
            "cutmix_s": 0.0,
            "forward_loss_s": 0.0,
            "backward_s": 0.0,
            "step_clip_opt_s": 0.0,
            "other_s": 0.0,
        }
    limit = max(0, int(perf_diag_batches))

    it = iter(loader)
    while True:
        t_data = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        prof = perf is not None and (limit <= 0 or int(perf["profiled_batches"]) < limit)
        if prof:
            perf["data_wait_s"] += time.perf_counter() - t_data

        if batch is None:
            continue
        x, y, w = batch

        if prof:
            sync_if_cuda(device)
            t_phase = time.perf_counter()
        x = x.float().to(device, non_blocking=(device.type == "cuda"))
        y = y.long().to(device, non_blocking=(device.type == "cuda"))
        w = w.float().to(device, non_blocking=(device.type == "cuda"))
        x = adapt_batch_torch(x, input_adapter)
        if prof:
            sync_if_cuda(device)
            perf["h2d_adapt_s"] += time.perf_counter() - t_phase

        # CutMix batch-level: mescola coppie di sample nel batch
        # Riduce il domain shift cross-year (A3_multi) e migliora generalizzazione
        if prof:
            sync_if_cuda(device)
            t_phase = time.perf_counter()
        _eff_cutmix = _scheduled_cutmix_prob(cutmix_prob, epoch, total_epochs)
        if _eff_cutmix > 0.0 and random.random() < _eff_cutmix and x.shape[0] >= 2:
            B = x.shape[0]
            # Indice permutato (accoppia sample in modo casuale)
            perm = torch.randperm(B, device=device)
            x2, y2, w2 = x[perm], y[perm], w[perm]
            for bi in range(B):
                if random.random() < 0.5:
                    xi, yi, wi = _cutmix_patch(
                        x[bi], y[bi], w[bi],
                        x2[bi], y2[bi], w2[bi],
                    )
                    x[bi], y[bi], w[bi] = xi.to(device), yi.to(device), wi.to(device)
        if prof:
            sync_if_cuda(device)
            perf["cutmix_s"] += time.perf_counter() - t_phase

        if prof:
            sync_if_cuda(device)
            t_phase = time.perf_counter()
        with autocast_ctx(device, amp):
            logits = model(x)
            loss   = loss_fn(logits, y, w)
        if prof:
            sync_if_cuda(device)
            perf["forward_loss_s"] += time.perf_counter() - t_phase
        total_loss = total_loss + loss.detach()
        loss = loss / float(grad_accum_steps)
        step_in_accum += 1

        if prof:
            sync_if_cuda(device)
            t_phase = time.perf_counter()
        if scaler is not None and amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if prof:
            sync_if_cuda(device)
            perf["backward_s"] += time.perf_counter() - t_phase

        if step_in_accum >= grad_accum_steps:
            if prof:
                sync_if_cuda(device)
                t_phase = time.perf_counter()
            if scaler is not None and amp:
                scaler.unscale_(opt)
                clip_grad_norm_fast(model, grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                clip_grad_norm_fast(model, grad_clip)
                opt.step()
            opt.zero_grad(set_to_none=True)
            step_in_accum = 0
            if prof:
                sync_if_cuda(device)
                perf["step_clip_opt_s"] += time.perf_counter() - t_phase
                perf["opt_steps"] += 1
        n     += 1
        if perf is not None:
            perf["batches"] += 1
            if prof:
                perf["profiled_batches"] += 1
    if step_in_accum > 0:
        prof = perf is not None and (limit <= 0 or int(perf["profiled_batches"]) < limit)
        if prof:
            sync_if_cuda(device)
            t_phase = time.perf_counter()
        if scaler is not None and amp:
            scaler.unscale_(opt)
            clip_grad_norm_fast(model, grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            clip_grad_norm_fast(model, grad_clip)
            opt.step()
        opt.zero_grad(set_to_none=True)
        if prof:
            sync_if_cuda(device)
            perf["step_clip_opt_s"] += time.perf_counter() - t_phase
            perf["opt_steps"] += 1
    if perf is not None:
        measured = sum(
            float(perf[k])
            for k in (
                "data_wait_s",
                "h2d_adapt_s",
                "cutmix_s",
                "forward_loss_s",
                "backward_s",
                "step_clip_opt_s",
            )
        )
        perf["measured_s"] = measured
        perf["avg_profiled_batch_s"] = measured / max(1, int(perf["profiled_batches"]))
        perf = _round_perf(perf)
    return float((total_loss / max(1, n)).item()), perf


@torch.no_grad()
def _collect_logits(
    model, loader, loss_fn, device, amp: bool, input_adapter: str = "identity"
) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raccoglie logit grezzi, labels e weights dal loader. Se loss_fn=None, val_loss=0.0.

    FIX B+ (perf, 2026-04-28):
      Prima: per ogni batch faceva 3 trasferimenti D2H di tensori 2D (B,H,W) completi
      (~7.3 MB/batch x 333 batch = 2.4 GB D2H per epoch); torch.cat su CPU su tensori
      pesanti; sweep_threshold poi girava su tensori CPU enormi (140M pixel) con un
      loop Python che faceva 4 .sum().item() per iterazione.

      Adesso: filtriamo i pixel labeled (w > 0) ON GPU e accumuliamo i tensori 1D
      filtrati direttamente su GPU. Questo:
        1. Azzera il D2H transfer dentro _collect_logits (saving stimato ~24s/epoch).
        2. Permette a _sweep_threshold e _expected_calibration_error di girare su GPU
           senza modifiche al loro codice: entrambi usano `weights > 0` per filtrare
           (con tensori già filtrati la maschera e' tutto-True quindi il risultato
           e' identico) e `device=p.device` per le linspace dell'ECE.
        3. dtype compression: targets viene compresso a uint8 (i valori sono solo
           0/1/2) per ridurre l'occupazione VRAM da ~980 MB a ~120 MB su 122M pixel.
           Tutti i caller usano `(targets == 2).float()` o `.long()`, semantica preservata.

      Saving osservato in profile_real.py: collect 34.1s -> ~10s, sweep 27.8s -> ~3-5s.
      Saving totale stimato per evaluate(): ~45-50s/epoch.

      Guardrail 2026-04-29: su GPU con meno di 12 GB di VRAM i tensori filtrati
      vengono spostati subito su CPU. Il keep-on-GPU saturava la RTX 8 GB e poteva
      bloccare la run dopo la prima evaluation.
    """
    model.eval()
    total = 0.0
    n     = 0
    all_logits:  List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    all_weights: List[torch.Tensor] = []

    keep_eval_tensors_on_gpu = False
    if device.type == "cuda":
        try:
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()
            total_vram = torch.cuda.get_device_properties(dev_idx).total_memory
            keep_eval_tensors_on_gpu = total_vram >= 12 * 1024**3
        except Exception:
            keep_eval_tensors_on_gpu = False

    for batch in loader:
        if batch is None:
            continue
        x, y, w = batch
        x = x.float().to(device, non_blocking=(device.type == "cuda"))
        y = y.long().to(device, non_blocking=(device.type == "cuda"))
        w = w.float().to(device, non_blocking=(device.type == "cuda"))
        x = adapt_batch_torch(x, input_adapter)
        with autocast_ctx(device, amp):
            logits = model(x)
            loss   = (loss_fn(logits, y, w) if loss_fn is not None else None)
        if loss is not None:
            total += float(loss.item())
            n     += 1
        # Filtra labeled pixels ON GPU. Su GPU da 8 GB non tenere tutta la
        # validation in VRAM: il training successivo puo' restare senza margine.
        logits_sq = logits.squeeze(1)              # (B,H,W) fp16 sotto AMP
        mask = w > 0                                # (B,H,W) bool
        logits_l = logits_sq[mask]
        targets_l = y[mask].to(torch.uint8)
        weights_l = w[mask]
        if keep_eval_tensors_on_gpu:
            all_logits.append(logits_l)
            all_targets.append(targets_l)
            all_weights.append(weights_l)
        else:
            all_logits.append(logits_l.float().cpu())
            all_targets.append(targets_l.cpu())
            all_weights.append(weights_l.float().cpu())
        del x, y, w, logits, logits_sq, mask, logits_l, targets_l, weights_l

    if not all_logits:
        # Mantengo dummy su CPU come prima: il caller verifica solo numel().
        dummy = torch.zeros(1)
        return 0.0, dummy, dummy, dummy

    return (
        total / max(1, n) if n > 0 else 0.0,
        torch.cat(all_logits),
        torch.cat(all_targets),
        torch.cat(all_weights),
    )


def _safe_metric_ratio(num: float, den: float) -> float:
    den_f = float(den)
    return float(num) / den_f if den_f > 0 else 0.0


def _binary_class_metrics(pred_pos: torch.Tensor, target_pos: torch.Tensor) -> Dict[str, float]:
    """Metrics for class-2 (veg) and class-1 (sand/non-veg) from a binary prediction."""
    tp = float((pred_pos * target_pos).sum().item())
    fp = float((pred_pos * (1.0 - target_pos)).sum().item())
    fn = float(((1.0 - pred_pos) * target_pos).sum().item())
    tn = float(((1.0 - pred_pos) * (1.0 - target_pos)).sum().item())

    iou_veg = _safe_metric_ratio(tp, tp + fp + fn)
    f1_veg = _safe_metric_ratio(2.0 * tp, 2.0 * tp + fp + fn)
    precision_veg = _safe_metric_ratio(tp, tp + fp)
    recall_veg = _safe_metric_ratio(tp, tp + fn)

    iou_sand = _safe_metric_ratio(tn, tn + fp + fn)
    f1_sand = _safe_metric_ratio(2.0 * tn, 2.0 * tn + fp + fn)
    precision_sand = _safe_metric_ratio(tn, tn + fn)
    recall_sand = _safe_metric_ratio(tn, tn + fp)

    miou = 0.5 * (iou_veg + iou_sand)
    mf1 = 0.5 * (f1_veg + f1_sand)
    min_iou = min(iou_veg, iou_sand)
    harmonic_iou = _safe_metric_ratio(2.0 * iou_veg * iou_sand, iou_veg + iou_sand)
    bal_acc = 0.5 * (recall_veg + recall_sand)
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_metric_ratio(tp * tn - fp * fn, mcc_den)

    return {
        "iou_veg": float(iou_veg),
        "iou_sand": float(iou_sand),
        "f1_veg": float(f1_veg),
        "f1_sand": float(f1_sand),
        "precision_veg": float(precision_veg),
        "precision_sand": float(precision_sand),
        "recall_veg": float(recall_veg),
        "recall_sand": float(recall_sand),
        "miou": float(miou),
        "mf1": float(mf1),
        "min_iou": float(min_iou),
        "harmonic_iou": float(harmonic_iou),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
    }


def _binary_class_metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    iou_veg = _safe_metric_ratio(tp, tp + fp + fn)
    f1_veg = _safe_metric_ratio(2.0 * tp, 2.0 * tp + fp + fn)
    precision_veg = _safe_metric_ratio(tp, tp + fp)
    recall_veg = _safe_metric_ratio(tp, tp + fn)

    iou_sand = _safe_metric_ratio(tn, tn + fp + fn)
    f1_sand = _safe_metric_ratio(2.0 * tn, 2.0 * tn + fp + fn)
    precision_sand = _safe_metric_ratio(tn, tn + fn)
    recall_sand = _safe_metric_ratio(tn, tn + fp)

    miou = 0.5 * (iou_veg + iou_sand)
    mf1 = 0.5 * (f1_veg + f1_sand)
    min_iou = min(iou_veg, iou_sand)
    harmonic_iou = _safe_metric_ratio(2.0 * iou_veg * iou_sand, iou_veg + iou_sand)
    bal_acc = 0.5 * (recall_veg + recall_sand)
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_metric_ratio(tp * tn - fp * fn, mcc_den)

    return {
        "iou_veg": float(iou_veg),
        "iou_sand": float(iou_sand),
        "f1_veg": float(f1_veg),
        "f1_sand": float(f1_sand),
        "precision_veg": float(precision_veg),
        "precision_sand": float(precision_sand),
        "recall_veg": float(recall_veg),
        "recall_sand": float(recall_sand),
        "miou": float(miou),
        "mf1": float(mf1),
        "min_iou": float(min_iou),
        "harmonic_iou": float(harmonic_iou),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
    }


def _resolve_threshold_metric_name(thr_metric: str) -> str:
    metric_map = {
        "iou": "miou",
        "f1": "mf1",
        "min_iou": "min_iou",
        "harmonic_iou": "harmonic_iou",
    }
    if thr_metric not in metric_map:
        raise ValueError(f"Unsupported thr_metric: {thr_metric}")
    return metric_map[thr_metric]


def _resolve_selection_metric_key(thr_metric: str, selection_metric_mode: str) -> str:
    base = _resolve_threshold_metric_name(thr_metric)
    if str(selection_metric_mode) == "at_05":
        return f"{base}_at_05"
    return "best_metric"


def _expected_calibration_error(
    probs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    n_bins: int = 15,
) -> float:
    labeled = weights > 0
    if int(labeled.sum().item()) == 0:
        return 0.0
    p = probs[labeled].float()
    y = (targets[labeled] == 2).float()
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    ece = torch.tensor(0.0, device=p.device)
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = ((p >= lo) if i == 0 else (p > lo)) & (p <= hi)
        if mask.any():
            frac = mask.float().mean()
            conf = p[mask].mean()
            acc = y[mask].mean()
            ece = ece + torch.abs(conf - acc) * frac
    return float(ece.item())


def _sweep_threshold(
    probs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    thr_metric: str,
) -> Dict[str, float]:
    """
    Sweep threshold su _THR_SWEEP con probabilità già sigmoid-ate.
    Restituisce (iou_at_05, f1_at_05, best_thr, best_score).
    Tutti i calcoli su pixel labeled (weight > 0).
    """
    labeled = weights > 0
    t_bin   = (targets[labeled] == 2).float()
    p_all   = probs[labeled]

    # Metriche fisse a 0.5
    metrics_05 = _binary_class_metrics((p_all >= 0.5).float(), t_bin)
    ece = _expected_calibration_error(probs, targets, weights, n_bins=15)
    brier = float(torch.mean((p_all - t_bin) ** 2).item()) if p_all.numel() > 0 else 0.0

    # Sweep threshold in due step:
    # 1) coarse globale, con densità alta vicino a zero
    # 2) refine locale attorno al best coarse
    best_thr = 0.5
    best_score = -1.0
    best_metric_name = _resolve_threshold_metric_name(thr_metric)
    best_metrics = metrics_05
    tested_thresholds = set()

    def _eval_thr(thr: float) -> None:
        nonlocal best_thr, best_score, best_metrics
        thr = float(thr)
        if thr in tested_thresholds:
            return
        tested_thresholds.add(thr)
        p_t = (p_all >= thr).float()
        metrics_t = _binary_class_metrics(p_t, t_bin)
        s = float(metrics_t[best_metric_name])
        if s > best_score:
            best_score = s
            best_thr = thr
            best_metrics = metrics_t

    for thr in _THR_SWEEP_BASE:
        _eval_thr(thr)

    for thr in _make_refined_thresholds(best_thr):
        _eval_thr(thr)

    return {
        "best_thr": float(best_thr),
        "best_metric": float(best_score),
        "best_metric_name": best_metric_name,
        "thr_sweep_min": float(min(tested_thresholds)) if tested_thresholds else 0.0,
        "thr_sweep_max": float(max(tested_thresholds)) if tested_thresholds else 0.0,
        "thr_sweep_points": int(len(tested_thresholds)),
        "miou_at_05": float(metrics_05["miou"]),
        "mf1_at_05": float(metrics_05["mf1"]),
        "min_iou_at_05": float(metrics_05["min_iou"]),
        "harmonic_iou_at_05": float(metrics_05["harmonic_iou"]),
        "iou_veg_at_05": float(metrics_05["iou_veg"]),
        "iou_sand_at_05": float(metrics_05["iou_sand"]),
        "f1_veg_at_05": float(metrics_05["f1_veg"]),
        "f1_sand_at_05": float(metrics_05["f1_sand"]),
        "precision_veg_at_05": float(metrics_05["precision_veg"]),
        "precision_sand_at_05": float(metrics_05["precision_sand"]),
        "recall_veg_at_05": float(metrics_05["recall_veg"]),
        "recall_sand_at_05": float(metrics_05["recall_sand"]),
        "miou_best": float(best_metrics["miou"]),
        "mf1_best": float(best_metrics["mf1"]),
        "min_iou_best": float(best_metrics["min_iou"]),
        "harmonic_iou_best": float(best_metrics["harmonic_iou"]),
        "iou_veg_best": float(best_metrics["iou_veg"]),
        "iou_sand_best": float(best_metrics["iou_sand"]),
        "f1_veg_best": float(best_metrics["f1_veg"]),
        "f1_sand_best": float(best_metrics["f1_sand"]),
        "precision_veg_best": float(best_metrics["precision_veg"]),
        "precision_sand_best": float(best_metrics["precision_sand"]),
        "recall_veg_best": float(best_metrics["recall_veg"]),
        "recall_sand_best": float(best_metrics["recall_sand"]),
        "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
        "mcc": float(best_metrics["mcc"]),
        "ece": float(ece),
        "brier": float(brier),
    }


def _zero_eval_metrics(thr_metric: str) -> Dict[str, float]:
    best_metric_name = _resolve_threshold_metric_name(thr_metric)
    return {
        "best_thr": 0.5,
        "best_metric": 0.0,
        "best_metric_name": best_metric_name,
        "thr_sweep_min": 0.0,
        "thr_sweep_max": 0.0,
        "thr_sweep_points": 0,
        "miou_at_05": 0.0,
        "mf1_at_05": 0.0,
        "min_iou_at_05": 0.0,
        "harmonic_iou_at_05": 0.0,
        "iou_veg_at_05": 0.0,
        "iou_sand_at_05": 0.0,
        "f1_veg_at_05": 0.0,
        "f1_sand_at_05": 0.0,
        "precision_veg_at_05": 0.0,
        "precision_sand_at_05": 0.0,
        "recall_veg_at_05": 0.0,
        "recall_sand_at_05": 0.0,
        "miou_best": 0.0,
        "mf1_best": 0.0,
        "min_iou_best": 0.0,
        "harmonic_iou_best": 0.0,
        "iou_veg_best": 0.0,
        "iou_sand_best": 0.0,
        "f1_veg_best": 0.0,
        "f1_sand_best": 0.0,
        "precision_veg_best": 0.0,
        "precision_sand_best": 0.0,
        "recall_veg_best": 0.0,
        "recall_sand_best": 0.0,
        "balanced_accuracy": 0.0,
        "mcc": 0.0,
        "ece": 0.0,
        "brier": 0.0,
    }


def _stream_threshold_values() -> List[float]:
    vals = set(float(t) for t in _THR_SWEEP_BASE)
    vals.add(0.5)
    for t in _THR_SWEEP_BASE:
        vals.update(float(v) for v in _make_refined_thresholds(float(t)))
    return sorted(v for v in vals if 0.0 < v < 1.0)


@torch.inference_mode()
def _evaluate_at05_fast(
    model,
    loader,
    loss_fn,
    device,
    amp: bool,
    thr_metric: str,
    input_adapter: str,
    compute_loss: bool,
    perf_diag: bool = False,
    perf_diag_batches: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Fast validation for selection_metric_mode=at_05.

    For threshold 0.5, sigmoid(logit) >= 0.5 is exactly logit >= 0, so this
    path avoids sigmoid, threshold sweep, calibration bins, brier, and logits
    accumulation. It is intended for per-epoch model selection only; final
    calibrated metrics are still computed by fit_temperature().
    """
    model.eval()
    tp = torch.zeros((), device=device, dtype=torch.int64)
    fp = torch.zeros((), device=device, dtype=torch.int64)
    fn = torch.zeros((), device=device, dtype=torch.int64)
    tn = torch.zeros((), device=device, dtype=torch.int64)

    total_loss = 0.0
    n_loss = 0
    perf: Optional[Dict[str, Any]] = None
    if perf_diag:
        perf = {
            "batches": 0,
            "profiled_batches": 0,
            "data_wait_s": 0.0,
            "h2d_adapt_s": 0.0,
            "forward_loss_s": 0.0,
            "metrics_s": 0.0,
        }
    limit = max(0, int(perf_diag_batches))

    it = iter(loader)
    while True:
        t_data = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        prof = perf is not None and (limit <= 0 or int(perf["profiled_batches"]) < limit)
        if prof:
            perf["data_wait_s"] += time.perf_counter() - t_data
        if batch is None:
            continue
        x, y, w = batch
        if prof:
            sync_if_cuda(device)
            t_phase = time.perf_counter()
        x = x.float().to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))
        if compute_loss and loss_fn is not None:
            w_loss = w.float().to(device, non_blocking=(device.type == "cuda"))
            labeled = w_loss > 0
        else:
            w_loss = None
            labeled = w.to(device, dtype=torch.bool, non_blocking=(device.type == "cuda"))
        x = adapt_batch_torch(x, input_adapter)
        if prof:
            sync_if_cuda(device)
            perf["h2d_adapt_s"] += time.perf_counter() - t_phase
            t_phase = time.perf_counter()
        with autocast_ctx(device, amp):
            logits = model(x)
            loss = (loss_fn(logits, y.long(), w_loss) if compute_loss and loss_fn is not None else None)
        if prof:
            sync_if_cuda(device)
            perf["forward_loss_s"] += time.perf_counter() - t_phase
        if loss is not None:
            total_loss += float(loss.item())
            n_loss += 1

        if prof:
            sync_if_cuda(device)
            t_phase = time.perf_counter()
        pred_pos = logits.squeeze(1) >= 0.0
        target_pos = y == 2

        tp += (pred_pos & target_pos & labeled).sum()
        fp += (pred_pos & (~target_pos) & labeled).sum()
        fn += ((~pred_pos) & target_pos & labeled).sum()
        tn += ((~pred_pos) & (~target_pos) & labeled).sum()
        if prof:
            sync_if_cuda(device)
            perf["metrics_s"] += time.perf_counter() - t_phase

        del x, y, w, w_loss, logits, labeled, pred_pos, target_pos
        if perf is not None:
            perf["batches"] += 1
            if prof:
                perf["profiled_batches"] += 1

    total = tp + fp + fn + tn
    if float(total.item()) <= 0.0:
        metrics = _zero_eval_metrics(thr_metric)
        metrics["fast_at05"] = True
        metrics["val_loss_computed"] = bool(n_loss > 0)
        if perf is not None:
            measured = sum(float(perf[k]) for k in ("data_wait_s", "h2d_adapt_s", "forward_loss_s", "metrics_s"))
            perf["measured_s"] = measured
            perf["avg_profiled_batch_s"] = measured / max(1, int(perf["profiled_batches"]))
            metrics["_perf"] = _round_perf(perf)
        return total_loss / max(1, n_loss) if n_loss > 0 else 0.0, metrics

    metrics_05 = _binary_class_metrics_from_counts(
        float(tp.item()), float(fp.item()), float(fn.item()), float(tn.item())
    )
    best_metric_name = _resolve_threshold_metric_name(thr_metric)
    best_score = float(metrics_05[best_metric_name])
    val_loss = total_loss / max(1, n_loss) if n_loss > 0 else 0.0

    out = {
        "best_thr": 0.5,
        "best_metric": best_score,
        "best_metric_name": best_metric_name,
        "thr_sweep_min": 0.5,
        "thr_sweep_max": 0.5,
        "thr_sweep_points": 1,
        "miou_at_05": float(metrics_05["miou"]),
        "mf1_at_05": float(metrics_05["mf1"]),
        "min_iou_at_05": float(metrics_05["min_iou"]),
        "harmonic_iou_at_05": float(metrics_05["harmonic_iou"]),
        "iou_veg_at_05": float(metrics_05["iou_veg"]),
        "iou_sand_at_05": float(metrics_05["iou_sand"]),
        "f1_veg_at_05": float(metrics_05["f1_veg"]),
        "f1_sand_at_05": float(metrics_05["f1_sand"]),
        "precision_veg_at_05": float(metrics_05["precision_veg"]),
        "precision_sand_at_05": float(metrics_05["precision_sand"]),
        "recall_veg_at_05": float(metrics_05["recall_veg"]),
        "recall_sand_at_05": float(metrics_05["recall_sand"]),
        "miou_best": float(metrics_05["miou"]),
        "mf1_best": float(metrics_05["mf1"]),
        "min_iou_best": float(metrics_05["min_iou"]),
        "harmonic_iou_best": float(metrics_05["harmonic_iou"]),
        "iou_veg_best": float(metrics_05["iou_veg"]),
        "iou_sand_best": float(metrics_05["iou_sand"]),
        "f1_veg_best": float(metrics_05["f1_veg"]),
        "f1_sand_best": float(metrics_05["f1_sand"]),
        "precision_veg_best": float(metrics_05["precision_veg"]),
        "precision_sand_best": float(metrics_05["precision_sand"]),
        "recall_veg_best": float(metrics_05["recall_veg"]),
        "recall_sand_best": float(metrics_05["recall_sand"]),
        "balanced_accuracy": float(metrics_05["balanced_accuracy"]),
        "mcc": float(metrics_05["mcc"]),
        "ece": 0.0,
        "brier": 0.0,
        "fast_at05": True,
        "val_loss_computed": bool(n_loss > 0),
    }
    if perf is not None:
        measured = sum(float(perf[k]) for k in ("data_wait_s", "h2d_adapt_s", "forward_loss_s", "metrics_s"))
        perf["measured_s"] = measured
        perf["avg_profiled_batch_s"] = measured / max(1, int(perf["profiled_batches"]))
        out["_perf"] = _round_perf(perf)
    return val_loss, out


@torch.no_grad()
def _evaluate_streaming(
    model,
    loader,
    loss_fn,
    device,
    amp: bool,
    thr_metric: str,
    input_adapter: str,
    sweep_thresholds: bool,
) -> Tuple[float, Dict[str, float]]:
    """Validation metrics without materializing validation logits."""
    model.eval()
    threshold_values = _stream_threshold_values() if sweep_thresholds else [0.5]
    threshold_values = sorted(set(float(t) for t in threshold_values))
    thresholds = torch.tensor(threshold_values, device=device, dtype=torch.float32)
    n_thr = int(thresholds.numel())

    tp = torch.zeros(n_thr, device=device, dtype=torch.float64)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    tn = torch.zeros_like(tp)

    n_bins = 15
    bin_total = torch.zeros(n_bins, device=device, dtype=torch.float64)
    bin_conf = torch.zeros_like(bin_total)
    bin_acc = torch.zeros_like(bin_total)
    brier_sum = torch.tensor(0.0, device=device, dtype=torch.float64)
    labeled_total = torch.tensor(0.0, device=device, dtype=torch.float64)

    total_loss = 0.0
    n_loss = 0
    chunk = 64

    for batch in loader:
        if batch is None:
            continue
        x, y, w = batch
        x = x.float().to(device, non_blocking=(device.type == "cuda"))
        y = y.long().to(device, non_blocking=(device.type == "cuda"))
        w = w.float().to(device, non_blocking=(device.type == "cuda"))
        x = adapt_batch_torch(x, input_adapter)
        with autocast_ctx(device, amp):
            logits = model(x)
            loss = (loss_fn(logits, y, w) if loss_fn is not None else None)
        if loss is not None:
            total_loss += float(loss.item())
            n_loss += 1

        labeled = w > 0
        if not bool(labeled.any().item()):
            continue

        probs = torch.sigmoid(logits.squeeze(1)).float()
        p = probs[labeled]
        target = (y[labeled] == 2)
        target_f = target.float()
        n_pix = int(p.numel())
        pos_count = target.sum().to(torch.float64)
        neg_count = torch.tensor(float(n_pix), device=device, dtype=torch.float64) - pos_count

        brier_sum += ((p - target_f) ** 2).to(torch.float64).sum()
        labeled_total += float(n_pix)

        bin_idx = torch.clamp(torch.ceil(p * n_bins).to(torch.long) - 1, 0, n_bins - 1)
        bin_total += torch.bincount(bin_idx, minlength=n_bins).to(device=device, dtype=torch.float64)
        bin_conf += torch.bincount(bin_idx, weights=p.to(torch.float64), minlength=n_bins).to(device=device, dtype=torch.float64)
        bin_acc += torch.bincount(bin_idx, weights=target_f.to(torch.float64), minlength=n_bins).to(device=device, dtype=torch.float64)

        target_row = target.unsqueeze(0)
        p_row = p.unsqueeze(0)
        for start in range(0, n_thr, chunk):
            end = min(start + chunk, n_thr)
            pred = p_row >= thresholds[start:end].view(-1, 1)
            tp_c = (pred & target_row).sum(dim=1).to(torch.float64)
            pred_c = pred.sum(dim=1).to(torch.float64)
            fp_c = pred_c - tp_c
            tp[start:end] += tp_c
            fp[start:end] += fp_c
            fn[start:end] += pos_count - tp_c
            tn[start:end] += neg_count - fp_c

        del x, y, w, logits, probs, p, target, target_f, labeled

    if float(labeled_total.item()) <= 0.0:
        return total_loss / max(1, n_loss) if n_loss > 0 else 0.0, _zero_eval_metrics(thr_metric)

    best_metric_name = _resolve_threshold_metric_name(thr_metric)
    metrics_by_thr = [
        _binary_class_metrics_from_counts(
            float(tp[i].item()), float(fp[i].item()), float(fn[i].item()), float(tn[i].item())
        )
        for i in range(n_thr)
    ]
    idx_05 = threshold_values.index(0.5)
    metrics_05 = metrics_by_thr[idx_05]

    scores = [float(m[best_metric_name]) for m in metrics_by_thr]
    best_idx = max(range(n_thr), key=lambda i: scores[i])
    best_thr = float(threshold_values[best_idx])
    best_score = float(scores[best_idx])
    best_metrics = metrics_by_thr[best_idx]

    nonempty = bin_total > 0
    ece_t = torch.tensor(0.0, device=device, dtype=torch.float64)
    if bool(nonempty.any().item()):
        frac = bin_total[nonempty] / labeled_total
        conf = bin_conf[nonempty] / bin_total[nonempty]
        acc = bin_acc[nonempty] / bin_total[nonempty]
        ece_t = torch.abs(conf - acc).mul(frac).sum()
    brier = float((brier_sum / labeled_total).item())
    ece = float(ece_t.item())

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return total_loss / max(1, n_loss) if n_loss > 0 else 0.0, {
        "best_thr": float(best_thr),
        "best_metric": float(best_score),
        "best_metric_name": best_metric_name,
        "thr_sweep_min": float(min(threshold_values)) if threshold_values else 0.0,
        "thr_sweep_max": float(max(threshold_values)) if threshold_values else 0.0,
        "thr_sweep_points": int(len(threshold_values)),
        "miou_at_05": float(metrics_05["miou"]),
        "mf1_at_05": float(metrics_05["mf1"]),
        "min_iou_at_05": float(metrics_05["min_iou"]),
        "harmonic_iou_at_05": float(metrics_05["harmonic_iou"]),
        "iou_veg_at_05": float(metrics_05["iou_veg"]),
        "iou_sand_at_05": float(metrics_05["iou_sand"]),
        "f1_veg_at_05": float(metrics_05["f1_veg"]),
        "f1_sand_at_05": float(metrics_05["f1_sand"]),
        "precision_veg_at_05": float(metrics_05["precision_veg"]),
        "precision_sand_at_05": float(metrics_05["precision_sand"]),
        "recall_veg_at_05": float(metrics_05["recall_veg"]),
        "recall_sand_at_05": float(metrics_05["recall_sand"]),
        "miou_best": float(best_metrics["miou"]),
        "mf1_best": float(best_metrics["mf1"]),
        "min_iou_best": float(best_metrics["min_iou"]),
        "harmonic_iou_best": float(best_metrics["harmonic_iou"]),
        "iou_veg_best": float(best_metrics["iou_veg"]),
        "iou_sand_best": float(best_metrics["iou_sand"]),
        "f1_veg_best": float(best_metrics["f1_veg"]),
        "f1_sand_best": float(best_metrics["f1_sand"]),
        "precision_veg_best": float(best_metrics["precision_veg"]),
        "precision_sand_best": float(best_metrics["precision_sand"]),
        "recall_veg_best": float(best_metrics["recall_veg"]),
        "recall_sand_best": float(best_metrics["recall_sand"]),
        "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
        "mcc": float(best_metrics["mcc"]),
        "ece": float(ece),
        "brier": float(brier),
    }


@torch.no_grad()
def evaluate(
    model, loader, loss_fn, device, amp: bool, thr_metric: str = "iou",
    input_adapter: str = "identity", sweep_thresholds: bool = True,
    compute_loss: bool = True,
    perf_diag: bool = False,
    perf_diag_batches: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """
    Restituisce (val_loss, metrics_dict).
    Le metriche '_raw' sono calcolate su sigmoid(logit) senza temperature scaling.
    Usate durante il training per monitorare la convergenza.
    """
    if not sweep_thresholds:
        return _evaluate_at05_fast(
            model, loader, loss_fn, device, amp, thr_metric, input_adapter, compute_loss,
            perf_diag=perf_diag, perf_diag_batches=perf_diag_batches,
        )
    return _evaluate_streaming(
        model, loader, loss_fn, device, amp, thr_metric, input_adapter, sweep_thresholds
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Temperature scaling
# ═══════════════════════════════════════════════════════════════════════════════

def fit_temperature(
    model: nn.Module,
    loader,
    device: torch.device,
    amp: bool,
    pos_weight: float,
    thr_metric: str,
    input_adapter: str = "identity",
    n_iter: int = 50,
) -> Tuple[float, Dict[str, float]]:
    """
    Calibra temperatura T tramite minimizzazione NLL su validation set.

    Dopo aver fittato T, esegue uno sweep threshold su sigmoid(logit/T)
    per trovare best_thr_scaled, che è il threshold coerente con la
    distribuzione di probabilità usata in inferenza.

    Restituisce:
        T, metrics_dict

    NOTA SCIENTIFICA:
        Il best_thr_scaled è l'unico threshold scientificamente corretto
        da salvare nel checkpoint e usare in 08_infer.py quando temperature≠1.
        Il threshold pre-scaling (best_thr_raw) è utile solo per comparare
        metriche di training interno, non per l'inferenza.
    """
    _, logits, targets, weights = _collect_logits(model, loader, None, device, False, input_adapter=input_adapter)

    if logits.numel() <= 1:
        return 1.0, evaluate(model, loader, None, device, amp, thr_metric, input_adapter=input_adapter)[1]

    # _collect_logits puo' restituire CPU o GPU in base alla VRAM disponibile.
    # T e pw_t devono stare sullo stesso device dei logits.
    device_l = logits.device
    pw_t   = torch.tensor([pos_weight], device=device_l)

    # Flatten per BCE
    logits_f = logits.reshape(1, -1)
    t_bin_f  = (targets.reshape(-1) == 2).float().unsqueeze(0)
    w_t_f    = weights.reshape(-1).unsqueeze(0)

    T = nn.Parameter(torch.ones(1, device=device_l))
    opt_t = torch.optim.LBFGS([T], lr=0.1, max_iter=n_iter)

    def closure():
        opt_t.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            logits_f / T.clamp(min=0.05),
            t_bin_f, weight=w_t_f, pos_weight=pw_t,
        )
        loss.backward()
        return loss

    opt_t.step(closure)
    T_val = float(T.clamp(min=0.05).item())

    # Calcola metriche SULLE PROBABILITÀ SCALATE
    probs_scaled = torch.sigmoid(logits / T_val)
    metrics = _sweep_threshold(probs_scaled, targets, weights, thr_metric)
    if device.type == "cuda":
        del probs_scaled, logits, targets, weights, logits_f, t_bin_f, w_t_f
        torch.cuda.empty_cache()
    return T_val, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Save / Load checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

def save_best(out_dir: Path, model: nn.Module, meta: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta},
               out_dir / "model_best.pt")
    (out_dir / "model_best.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser("05_train.py - SAV segmentation training")
    ap.add_argument("--patch_csv",  required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--arch",       default="unetpp")
    ap.add_argument("--encoder",    default="resnet34")
    ap.add_argument("--encoder_weights", default="imagenet")
    ap.add_argument(
        "--input_adapter",
        default="auto",
        choices=[
            "auto",
            "identity",
            "rgb_imagenet",
            "rgb_aux_identity",
            "rgb_nir_centered",
            "rgb_nir_aux_centered",
        ],
        help="Model-aware input normalization. 'auto' is recommended.",
    )
    ap.add_argument("--in_ch",      type=int, default=0,
                    help="Input channels. Default: inferred from suffix.")
    ap.add_argument("--epochs",     type=int, default=60)
    ap.add_argument("--batch",      type=int, default=32)
    ap.add_argument("--eval_batch", type=int, default=0,
                    help="Batch size validation. 0 = usa --batch.")
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--wd",         type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--scheduler",  default="cosine")
    ap.add_argument("--early_patience", type=int, default=15)
    ap.add_argument("--thr_metric", default="min_iou",
                    choices=["iou", "f1", "min_iou", "harmonic_iou"])
    ap.add_argument("--selection_metric_mode", default="best",
                    choices=["best", "at_05"],
                    help="Metrica usata per early stopping / model selection. best=legacy threshold-swept, at_05=fixed threshold 0.5.")
    ap.add_argument("--label_smooth", type=float, default=0.05)
    ap.add_argument("--tversky_alpha", type=float, default=0.3)
    ap.add_argument("--tversky_beta", type=float, default=0.7)
    ap.add_argument("--grad_clip",  type=float, default=1.0)
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                    help="Numero di micro-batch da accumulare prima di optimizer.step().")
    ap.add_argument("--amp",        action="store_true")
    ap.add_argument("--use_temp_scaling", action="store_true",
                    help=(
                        "Applica temperature scaling dopo training. "
                        "IMPORTANTE: quando abilitato, best_thr nel checkpoint "
                        "e' ricalibrato su sigmoid(logit/T), coerente con 08_infer.py."
                    ))
    ap.add_argument("--seed",       type=int, default=123)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--loss_mix",   default="dice:1.0,focal:1.0,lovasz:0.5")
    ap.add_argument("--boundary_weight",    action="store_true", default=True,
                    help="Amplifica pixel di bordo SAV nella loss (default: attivo).")
    ap.add_argument("--no_boundary_weight", action="store_true",
                    help="Disabilita boundary weighting (per ablation study).")
    ap.add_argument("--boundary_radius",    type=int,   default=3,
                    help="Raggio in pixel del bordo amplificato (default 3).")
    ap.add_argument("--boundary_factor",    type=float, default=3.0,
                    help="Fattore moltiplicativo sui pixel di bordo (default 3.0).")
    ap.add_argument("--cutmix_prob",        type=float, default=0.0,
                    help="Probabilita' di CutMix tra patch dello stesso batch "
                         "(default 0.0 = off). Raccomandato 0.3 per A3_multi_fair "
                         "per ridurre domain shift cross-year.")
    ap.add_argument("--depth_aug_prob",     type=float, default=0.20,
                    help="Probabilita' di depth-aware radiometric augmentation sui canali raw.")
    ap.add_argument("--cutout_prob",        type=float, default=0.10,
                    help="Probabilita' di cutout label-safe con weight=0 nella regione occlusa.")
    ap.add_argument("--aug_profile", default="default",
                    choices=["default", "geometric_only", "geometric_brightness"],
                    help="Profilo di augmentation. 'default' mantiene il comportamento attuale.")
    ap.add_argument("--tile",       type=int, default=256)
    ap.add_argument("--suffix",     default="_st16s")
    ap.add_argument("--h5_path",    default="",
                    help="Path HDF5 singolo o lista separata da virgole/; per concatenare component stores.")
    ap.add_argument("--pt_dir",     default="",
                    help="Path PT singolo o lista separata da virgole/; per concatenare component stores.")
    ap.add_argument("--pos_weight", type=float, default=None)
    aug_group = ap.add_mutually_exclusive_group()
    aug_group.add_argument("--augment", dest="augment", action="store_true",
                           help="Abilita augment online (default).")
    aug_group.add_argument("--no_augment", dest="augment", action="store_false",
                           help="Disabilita augment online.")
    ap.set_defaults(augment=True)
    # Ottimizzazioni performance
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile il modello (PyTorch 2.x). +10-20%% throughput. "
                         "Primo epoch piu' lento (warm-up JIT). Default: off.")
    ap.add_argument("--cache_ram", action="store_true",
                    help="Pre-carica l'intero dataset in RAM prima del training. "
                         "Elimina I/O HDF5 ripetuto. Richiede ~2x la dimensione del dataset in RAM. "
                         "Raccomandato se dataset < 3 GB e RAM disponibile > 6 GB.")
    ap.add_argument("--perf_diag_epochs", type=int, default=0,
                    help="Numero di epoche iniziali da profilare con sync CUDA e breakdown interno.")
    ap.add_argument("--perf_diag_batches", type=int, default=0,
                    help="Batch da profilare per epoch. 0 = tutti i batch delle epoche diagnostiche.")
    ap.add_argument("--perf_diag_path", default="",
                    help="Path JSONL per diagnostica performance. Default: <out_dir>/perf_diag.jsonl.")
    args = ap.parse_args()
    if args.in_ch <= 0:
        args.in_ch = expected_in_channels(args.suffix, fallback=16)
    _AUGMENT_CFG["depth_aug_prob"] = float(max(0.0, args.depth_aug_prob))
    _AUGMENT_CFG["cutout_prob"] = float(max(0.0, args.cutout_prob))
    _AUGMENT_CFG["profile"] = str(args.aug_profile)

    set_seed(args.seed)
    patch_csv   = Path(args.patch_csv).resolve()
    out_dir     = Path(args.out_dir).resolve()
    patch_meta  = read_json(meta_path_for_csv(patch_csv))
    use_augment = bool(args.augment)
    n_aug_channels = infer_radiometric_aug_channels(args.suffix, args.in_ch)
    try:
        train_feature_preset = resolve_feature_selection(suffix=args.suffix)
        train_feature_channels = feature_names_for_preset(train_feature_preset)
    except Exception:
        train_feature_preset = patch_meta.get("feature_preset")
        train_feature_channels = patch_meta.get("feature_channels")

    ew = None if (args.encoder_weights.lower() in ("none", "")) else args.encoder_weights
    input_adapter = (
        infer_input_adapter(
            arch=args.arch,
            encoder_weights=ew,
            feature_preset=str(train_feature_preset or ""),
            suffix=args.suffix,
            in_ch=int(args.in_ch),
        )
        if args.input_adapter == "auto"
        else args.input_adapter
    )

    # ── File logging (oltre a stdout) ────────────────────────────────────────
    import logging as _logging
    out_dir.mkdir(parents=True, exist_ok=True)
    _log_path = out_dir / "train.log"
    _file_handler = _logging.FileHandler(str(_log_path), encoding="utf-8")
    _file_handler.setFormatter(_logging.Formatter("%(message)s"))
    _root_logger = _logging.getLogger()
    _root_logger.addHandler(_file_handler)

    class _TeeLogger:
        """Redirige stdout sia al terminale sia al file di log."""
        def __init__(self, log_path: Path):
            import sys as _sys
            self._orig = _sys.stdout
            self._f    = open(log_path, "a", encoding="utf-8", buffering=1)
        def write(self, msg):
            self._orig.write(msg)
            self._f.write(msg)
        def flush(self):
            self._orig.flush()
            self._f.flush()
        def fileno(self):
            return self._orig.fileno()
        def isatty(self):
            return bool(getattr(self._orig, "isatty", lambda: False)())
        def __getattr__(self, name):
            return getattr(self._orig, name)

    import sys as _sys
    _tee = _TeeLogger(_log_path)
    _sys.stdout = _tee
    perf_diag_epochs = max(0, int(args.perf_diag_epochs))
    perf_diag_batches = max(0, int(args.perf_diag_batches))
    perf_diag_path: Optional[Path] = None
    if perf_diag_epochs > 0:
        perf_diag_path = Path(args.perf_diag_path).resolve() if args.perf_diag_path else (out_dir / "perf_diag.jsonl")
        if perf_diag_path.exists():
            perf_diag_path.unlink()

    print(f"\n[05_train] AVVIO {_now()}")
    print(f"  patch_csv: {patch_csv}")
    print(f"  out_dir  : {out_dir}")
    print(f"  arch={args.arch}  encoder={args.encoder}  encoder_weights={args.encoder_weights}")
    eff_batch = int(args.batch) * max(1, int(args.grad_accum_steps))
    eval_batch_log = int(args.eval_batch) if int(args.eval_batch) > 0 else int(args.batch)
    print(f"  in_ch={args.in_ch}  epochs={args.epochs}  batch={args.batch}  eval_batch={eval_batch_log}  grad_accum={int(args.grad_accum_steps)}  eff_batch={eff_batch}  lr={args.lr}")
    print(f"  amp={args.amp}  augment={use_augment}  early_patience={args.early_patience}")
    print(f"  use_temp_scaling={args.use_temp_scaling}")
    print(f"  selection_metric_mode={args.selection_metric_mode}")
    print(f"  radiometric_aug_channels={n_aug_channels}")
    print(f"  input_adapter={input_adapter}")
    print(f"  aug_profile={args.aug_profile}")
    print(f"  depth_aug_prob={_AUGMENT_CFG['depth_aug_prob']:.2f}  cutout_prob={_AUGMENT_CFG['cutout_prob']:.2f}")
    print(f"  tversky_alpha={args.tversky_alpha:.2f}  tversky_beta={args.tversky_beta:.2f}")
    if perf_diag_epochs > 0:
        print(f"  perf_diag=on epochs={perf_diag_epochs} batches={perf_diag_batches or 'all'} path={perf_diag_path}")
    if args.selection_metric_mode == "at_05":
        print("  eval_thresholds=fixed_0.5_fast (logit>=0, no per-epoch sweep/loss/ece/brier)")
    else:
        print(
            f"  thr_sweep=[{_THR_SWEEP_BASE[0]:.3f}..{_THR_SWEEP_BASE[-1]:.2f}] "
            f"({len(_THR_SWEEP_BASE)} punti base + refine)"
        )

    if not patch_csv.exists():
        raise FileNotFoundError(f"patch_csv non trovato: {patch_csv}")

    df           = pd.read_csv(patch_csv)
    tr_df, va_df = split_train_val(df, seed=args.seed)
    device       = get_device()

    # Stampa composizione split per diagnostica
    src_col = find_col(df, ["source_id", "scene_id", "sensor", "year", "dataset"])
    if src_col:
        tr_src = tr_df[src_col].value_counts().to_dict()
        va_src = va_df[src_col].value_counts().to_dict()
        print(f"  device={device.type}  train={len(tr_df)} {tr_src}  val={len(va_df)} {va_src}")
    else:
        print(f"  device={device.type}  train={len(tr_df)}  val={len(va_df)}")

    loss_mix_cfg = parse_loss_mix(args.loss_mix)

    # pos_weight: usato solo se la loss contiene BCE con peso > 0.
    uses_bce = float(loss_mix_cfg.get("bce", 0.0)) > 0.0
    if not uses_bce:
        pos_weight = 1.0
        print("  pos_weight=1.000  (non usato: BCE assente da loss_mix)")
    elif args.pos_weight is not None:
        pos_weight = float(args.pos_weight)
        print(f"  pos_weight={pos_weight:.3f}  (da --pos_weight CLI)")
    else:
        pos_weight = read_pos_weight(patch_csv)
        print(f"  pos_weight={pos_weight:.3f}  (da meta JSON)")

    # Store: HDF5 (priorità) o PT. Supporta liste separate da virgole per component stores.
    h5_paths = [Path(p).resolve() for p in parse_store_list(args.h5_path)]
    pt_roots = [Path(p).resolve() for p in parse_store_list(args.pt_dir)] if args.pt_dir else []
    use_h5 = len(h5_paths) > 0
    if use_h5:
        for h5_path in h5_paths:
            if not h5_path.exists():
                raise FileNotFoundError(f"HDF5 store non trovato: {h5_path}")
        if len(h5_paths) == 1:
            print(f"  h5_path: {h5_paths[0]}")
            ds_tr = PatchDatasetH5(tr_df, str(h5_paths[0]), args.tile, args.seed,     augment_online=use_augment, n_aug_channels=n_aug_channels)
            ds_va = PatchDatasetH5(va_df, str(h5_paths[0]), args.tile, args.seed + 1, augment_online=False, n_aug_channels=n_aug_channels)
        else:
            print("  h5_paths:")
            for h5_path in h5_paths:
                print(f"    - {h5_path}")
            ds_tr = PatchDatasetH5Multi(tr_df, [str(p) for p in h5_paths], args.tile, args.seed,     augment_online=use_augment, n_aug_channels=n_aug_channels)
            ds_va = PatchDatasetH5Multi(va_df, [str(p) for p in h5_paths], args.tile, args.seed + 1, augment_online=False, n_aug_channels=n_aug_channels)
    else:
        if not pt_roots:
            pt_roots = [p_pt_store(args.suffix, args.tile)]
        for pt_root in pt_roots:
            if not pt_root.exists():
                raise FileNotFoundError(
                    f"PT store non trovato: {pt_root}\n"
                    "Esegui 04_make_patches.py con --export_pt oppure usa --h5_path."
                )
        if len(pt_roots) == 1:
            print(f"  pt_root: {pt_roots[0]}")
            ds_tr = PatchDatasetPTStore(tr_df, pt_roots[0], args.tile, args.seed,     augment_online=use_augment, n_aug_channels=n_aug_channels)
            ds_va = PatchDatasetPTStore(va_df, pt_roots[0], args.tile, args.seed + 1, augment_online=False, n_aug_channels=n_aug_channels)
        else:
            print("  pt_roots:")
            for pt_root in pt_roots:
                print(f"    - {pt_root}")
            ds_tr = PatchDatasetPTStoreMulti(tr_df, pt_roots, args.tile, args.seed,     augment_online=use_augment, n_aug_channels=n_aug_channels)
            ds_va = PatchDatasetPTStoreMulti(va_df, pt_roots, args.tile, args.seed + 1, augment_online=False, n_aug_channels=n_aug_channels)

    # Safety: mismatch canali
    probe = None
    for i in range(min(64, len(ds_tr))):
        probe = ds_tr[i]
        if probe is not None:
            break
    if probe is None:
        raise RuntimeError("Impossibile caricare campioni dallo store.")
    _x, _y, _w = probe
    if int(_x.shape[0]) != args.in_ch:
        raise RuntimeError(
            f"MISMATCH CANALI: store produce {_x.shape[0]} canali, --in_ch={args.in_ch}."
        )

    # Windows/spawn: non serializzare handle aperti nei worker
    if hasattr(ds_tr, "reset_handles"):
        ds_tr.reset_handles()
    if hasattr(ds_va, "reset_handles"):
        ds_va.reset_handles()

    use_boundary_weight = args.boundary_weight and not args.no_boundary_weight
    boundary_weights_precomputed = bool(
        getattr(args, "cache_ram", False)
        and use_boundary_weight
        and not use_augment
        and float(args.cutmix_prob) <= 0.0
    )

    # In-memory cache opzionale (--cache_ram)
    if getattr(args, 'cache_ram', False):
        print("  [cache_ram] attivo - pre-caricamento dataset in RAM...")
        ds_tr.augment_online = False
        ds_tr_wrapped = CachedDataset(
            ds_tr,
            augment_online=use_augment,
            n_aug_channels=n_aug_channels,
            precompute_boundary_weight=boundary_weights_precomputed,
            boundary_radius=args.boundary_radius,
            boundary_factor=args.boundary_factor,
        )
        ds_tr_wrapped.preload()
        ds_tr = ds_tr_wrapped
        ds_va.augment_online = False
        ds_va_wrapped = CachedDataset(
            ds_va,
            augment_online=False,
            n_aug_channels=n_aug_channels,
            precompute_boundary_weight=boundary_weights_precomputed,
            boundary_radius=args.boundary_radius,
            boundary_factor=args.boundary_factor,
        )
        ds_va_wrapped.preload()
        ds_va = ds_va_wrapped
        if boundary_weights_precomputed:
            print("  [cache_ram] boundary weights precomputati nella cache")

    model = make_model(args.arch, args.encoder, args.in_ch, encoder_weights=ew).to(device)
    hf_checkpoint_name = getattr(model, "checkpoint", None)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params={n_params:.1f}M")
    if hf_checkpoint_name:
        print(f"  hf_checkpoint={hf_checkpoint_name}")

    # torch.compile opzionale (--compile): richiede PyTorch >= 2.0
    if getattr(args, 'compile', False):
        try:
            from torch.utils import _triton as _torch_triton
            has_triton = bool(getattr(_torch_triton, "has_triton", lambda: False)())
        except Exception:
            has_triton = False
        try:
            if not has_triton:
                raise RuntimeError("no working Triton backend detected on this machine")
            import torch._dynamo as _dynamo
            _dynamo.config.suppress_errors = True
            model = torch.compile(model)
            print("  [torch.compile] attivo - primo epoch piu' lento (JIT warm-up), poi +10-20% throughput")
        except Exception as e:
            print(f"  [torch.compile] non disponibile: {e} - training senza compile")

    loss_fn = LossMix(
        mix=loss_mix_cfg,
        pos_weight=pos_weight,
        label_smooth=args.label_smooth,
        boundary_radius=args.boundary_radius,
        boundary_factor=args.boundary_factor,
        use_boundary_weight=(use_boundary_weight and not boundary_weights_precomputed),
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
    )
    print(f"  boundary_weight={use_boundary_weight}  "
          f"radius={args.boundary_radius}  factor={args.boundary_factor}"
          f"  cutmix_prob={args.cutmix_prob}"
          f"  precomputed={boundary_weights_precomputed}")

    eval_batch = int(args.eval_batch) if int(args.eval_batch) > 0 else int(args.batch)

    _common = dict(
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=partial(collate_pad, tile=args.tile),
        worker_init_fn=seed_worker,
    )
    if args.num_workers > 0:
        _common["persistent_workers"] = True
        _common["prefetch_factor"] = 2
    _common_va = dict(_common)
    _common_va["batch_size"] = eval_batch

    g_tr = torch.Generator()
    g_tr.manual_seed(args.seed)
    g_va = torch.Generator()
    g_va.manual_seed(args.seed + 1)

    dl_tr = DataLoader(ds_tr, shuffle=True, generator=g_tr, **_common)
    dl_va = DataLoader(ds_va, shuffle=False, generator=g_va, **_common_va)

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched  = make_scheduler(opt, args.epochs, args.warmup_epochs, args.scheduler)
    scaler = make_scaler(enabled=args.amp)

    if perf_diag_epochs > 0:
        script_path = Path(__file__).resolve()
        _perf_emit(
            perf_diag_path,
            {
                "event": "env",
                "pid": os.getpid(),
                "ppid": os.getppid() if hasattr(os, "getppid") else None,
                "python": sys.executable,
                "python_version": sys.version.replace("\n", " "),
                "cwd": str(Path.cwd()),
                "script": str(script_path),
                "script_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(script_path.stat().st_mtime)),
                "torch": torch.__version__,
                "cuda": str(torch.version.cuda),
                "cudnn": str(torch.backends.cudnn.version()),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)) if torch.cuda.is_available() else None,
                "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)) if torch.cuda.is_available() else None,
                "device": str(device),
                "gpu": _gpu_snapshot(device),
                "num_workers": int(args.num_workers),
                "pin_memory": bool(_common.get("pin_memory")),
                "persistent_workers": bool(_common.get("persistent_workers", False)),
                "prefetch_factor": int(_common.get("prefetch_factor", 0)),
                "batch": int(args.batch),
                "eval_batch": int(eval_batch),
                "grad_accum_steps": int(args.grad_accum_steps),
                "cache_ram": bool(args.cache_ram),
                "boundary_weight_precomputed": bool(boundary_weights_precomputed),
                "train_len": int(len(ds_tr)),
                "val_len": int(len(ds_va)),
                "out_dir": str(out_dir),
            },
        )

    best_score = -1.0
    best_thr   = 0.5
    best_epoch = -1
    patience   = 0

    for ep in range(1, args.epochs + 1):
        diag_this_epoch = perf_diag_epochs > 0 and ep <= perf_diag_epochs
        if diag_this_epoch and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if diag_this_epoch:
            _perf_emit(perf_diag_path, {"event": "epoch_start", "epoch": ep, "gpu": _gpu_snapshot(device)})
        sync_if_cuda(device)
        t0      = time.time()
        tr_loss, train_perf = train_one_epoch(
            model, dl_tr, loss_fn, device, args.amp, opt, scaler, args.grad_clip,
            input_adapter=input_adapter,
            cutmix_prob=args.cutmix_prob,
            epoch=ep,
            total_epochs=args.epochs,
            grad_accum_steps=args.grad_accum_steps,
            perf_diag=diag_this_epoch,
            perf_diag_batches=perf_diag_batches,
        )
        sync_if_cuda(device)
        train_elapsed = int(time.time() - t0)
        if diag_this_epoch:
            _perf_emit(
                perf_diag_path,
                {
                    "event": "train_breakdown",
                    "epoch": ep,
                    "wall_s": train_elapsed,
                    "stats": train_perf or {},
                    "gpu": _gpu_snapshot(device),
                },
            )
        if sched is not None:
            sched.step()

        sync_if_cuda(device)
        t_eval0 = time.time()
        va_loss, eval_raw = evaluate(
            model, dl_va, loss_fn, device, args.amp, args.thr_metric,
            input_adapter=input_adapter,
            sweep_thresholds=(args.selection_metric_mode != "at_05"),
            compute_loss=(args.selection_metric_mode != "at_05"),
            perf_diag=diag_this_epoch,
            perf_diag_batches=perf_diag_batches,
        )
        sync_if_cuda(device)
        eval_elapsed = int(time.time() - t_eval0)
        eval_perf = eval_raw.pop("_perf", None)
        if diag_this_epoch:
            _perf_emit(
                perf_diag_path,
                {
                    "event": "eval_breakdown",
                    "epoch": ep,
                    "wall_s": eval_elapsed,
                    "stats": eval_perf or {},
                    "gpu": _gpu_snapshot(device),
                },
            )
        selection_metric_key = _resolve_selection_metric_key(args.thr_metric, args.selection_metric_mode)
        if args.selection_metric_mode == "at_05":
            thr_star = 0.5
            score_star = float(eval_raw[selection_metric_key])
            selection_label = selection_metric_key
            veg_key = "iou_veg_at_05"
            sand_key = "iou_sand_at_05"
        else:
            thr_star = float(eval_raw["best_thr"])
            score_star = float(eval_raw["best_metric"])
            selection_label = f"{eval_raw['best_metric_name']}*"
            veg_key = "iou_veg_best"
            sand_key = "iou_sand_best"

        lr_now   = opt.param_groups[0]["lr"]
        improved = score_star > best_score
        elapsed  = int(time.time() - t0)
        va_text = f"{va_loss:.4f}" if eval_raw.get("val_loss_computed", True) else "skip"

        msg = (
            f"[E{ep:03d}/{args.epochs}] "
            f"tr={tr_loss:.4f}  va={va_text}  "
            f"thr_sel={thr_star:.3f}  "
            f"veg_sel={eval_raw[veg_key]:.4f}  "
            f"sand_sel={eval_raw[sand_key]:.4f}  "
            f"{selection_label}={score_star:.4f}  "
            f"lr={lr_now:.2e}  {elapsed}s  "
            f"(train={train_elapsed}s eval={eval_elapsed}s)"
        )

        if improved:
            best_score = score_star
            best_thr   = thr_star
            best_epoch = ep
            patience   = 0
            # Nota: best_thr qui è raw (pre-temp-scaling).
            # Verrà sovrascritto dopo fit_temperature se use_temp_scaling=True.
            meta = {
                "timestamp":         _now(),
                "arch":              args.arch,
                "encoder":           args.encoder,
                "encoder_weights":   str(ew),
                "hf_checkpoint":     hf_checkpoint_name,
                "in_ch":             args.in_ch,
                "batch":            int(args.batch),
                "grad_accum_steps": int(args.grad_accum_steps),
                "effective_batch":  int(args.batch) * max(1, int(args.grad_accum_steps)),
                "tile":              args.tile,
                "suffix":            args.suffix,
                "source_suffix":     patch_meta.get("source_suffix", args.suffix),
                "feature_preset":    train_feature_preset,
                "feature_channels":  train_feature_channels,
                "input_adapter":     input_adapter,
                "n_radiometric_aug_channels": int(n_aug_channels),
                "h5_paths":          [str(p) for p in h5_paths] if use_h5 else None,
                "pt_roots":          [str(p) for p in pt_roots] if not use_h5 else None,
                "pos_weight":        float(pos_weight),
                "loss_mix":          loss_mix_cfg,
                "lr":                args.lr,
                "wd":                args.wd,
                "scheduler":         args.scheduler,
                "warmup_epochs":     args.warmup_epochs,
                "best_epoch":        int(best_epoch),
                # Metriche aggiuntive paper-grade alla soglia ottimale
                "balanced_accuracy_raw": float(eval_raw["balanced_accuracy"]),
                "mcc_raw":               float(eval_raw["mcc"]),
                "ece_raw":               float(eval_raw["ece"]),
                "brier_raw":             float(eval_raw["brier"]),
                # Metriche RAW (macro sulle due classi annotate)
                "best_thr_raw":      float(best_thr),
                "best_metric_raw":   float(best_score),
                "selection_metric_mode": str(args.selection_metric_mode),
                "selection_metric_key_raw": str(selection_metric_key),
                "fast_eval_at05_raw": bool(eval_raw.get("fast_at05", False)),
                "val_loss_computed_raw": bool(eval_raw.get("val_loss_computed", True)),
                "thr_metric":        str(args.thr_metric),
                "thr_sweep_min":     float(eval_raw["thr_sweep_min"]),
                "thr_sweep_max":     float(eval_raw["thr_sweep_max"]),
                "thr_sweep_points":  int(eval_raw["thr_sweep_points"]),
                "miou_at_05_raw":    float(eval_raw["miou_at_05"]),
                "mf1_at_05_raw":     float(eval_raw["mf1_at_05"]),
                "min_iou_at_05_raw": float(eval_raw["min_iou_at_05"]),
                "harmonic_iou_at_05_raw": float(eval_raw["harmonic_iou_at_05"]),
                "iou_veg_at_05_raw": float(eval_raw["iou_veg_at_05"]),
                "iou_sand_at_05_raw": float(eval_raw["iou_sand_at_05"]),
                "f1_veg_at_05_raw":  float(eval_raw["f1_veg_at_05"]),
                "f1_sand_at_05_raw": float(eval_raw["f1_sand_at_05"]),
                "min_iou_best_raw":  float(eval_raw["min_iou_best"]),
                "harmonic_iou_best_raw": float(eval_raw["harmonic_iou_best"]),
                "precision_veg_best_raw": float(eval_raw["precision_veg_best"]),
                "precision_sand_best_raw": float(eval_raw["precision_sand_best"]),
                "recall_veg_best_raw": float(eval_raw["recall_veg_best"]),
                "recall_sand_best_raw": float(eval_raw["recall_sand_best"]),
                "iou_veg_best_raw":  float(eval_raw["iou_veg_best"]),
                "iou_sand_best_raw": float(eval_raw["iou_sand_best"]),
                "f1_veg_best_raw":   float(eval_raw["f1_veg_best"]),
                "f1_sand_best_raw":  float(eval_raw["f1_sand_best"]),
                # Legacy fields kept for compatibility with old reports/tools
                "iou_at_05_raw":     float(eval_raw["iou_veg_at_05"]),
                "f1_at_05_raw":      float(eval_raw["f1_veg_at_05"]),
                # best_thr è inizialmente = best_thr_raw.
                # Verrà aggiornato a best_thr_scaled se use_temp_scaling=True.
                "best_thr":          float(best_thr),
                "best_metric":       float(best_score),
                "temperature":       1.0,   # sarà aggiornato se use_temp_scaling=True
                "temp_scaling_done": False,
                "threshold_source":   "raw",
                "calibration_enabled": bool(args.use_temp_scaling),
                "augment":           use_augment,
                "aug_profile":       str(args.aug_profile),
                "boundary_weight":   use_boundary_weight,
                "boundary_weight_precomputed": bool(boundary_weights_precomputed),
                "boundary_radius":   args.boundary_radius,
                "boundary_factor":   args.boundary_factor,
                "tversky_alpha":     float(args.tversky_alpha),
                "tversky_beta":      float(args.tversky_beta),
                "cutmix_prob":       args.cutmix_prob,
                "depth_aug_prob":    float(_AUGMENT_CFG["depth_aug_prob"]),
                "cutout_prob":       float(_AUGMENT_CFG["cutout_prob"]),
                "amp":               args.amp,
            }
            save_best(out_dir, model, meta)
            msg += "  BEST"
        else:
            patience += 1
            msg += f"  (pat {patience}/{args.early_patience})"

        print(msg)
        if diag_this_epoch:
            _perf_emit(
                perf_diag_path,
                {
                    "event": "epoch_end",
                    "epoch": ep,
                    "elapsed_s": elapsed,
                    "train_s": train_elapsed,
                    "eval_s": eval_elapsed,
                    "tr_loss": float(tr_loss),
                    "selection_metric": str(selection_label),
                    "selection_score": float(score_star),
                    "lr": float(lr_now),
                    "improved": bool(improved),
                    "gpu": _gpu_snapshot(device),
                },
            )

        if patience >= args.early_patience:
            print(f"  [INFO] Early stopping all'epoca {ep}.")
            break

    # ─────────────────────────────────────────────────────────────────────────
    # Temperature scaling post-training
    # ─────────────────────────────────────────────────────────────────────────
    best_pt = out_dir / "model_best.pt"

    if best_pt.exists() and args.use_temp_scaling:
        print("\n[INFO] Fitting temperature scaling su validation set...")
        ck = torch.load(str(best_pt), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()

        eval_raw_full = None
        if bool(ck.get("meta", {}).get("fast_eval_at05_raw", False)):
            print("[INFO] Recomputing full raw validation metrics once for checkpoint metadata...")
            _, eval_raw_full = evaluate(
                model,
                dl_va,
                None,
                device,
                args.amp,
                args.thr_metric,
                input_adapter=input_adapter,
                sweep_thresholds=True,
                compute_loss=False,
            )

        T_val, eval_scaled = fit_temperature(
            model,
            dl_va,
            device,
            args.amp,
            pos_weight,
            args.thr_metric,
            input_adapter=input_adapter,
        )
        best_thr_scaled = float(eval_scaled["best_thr"])
        best_score_scaled = float(eval_scaled["best_metric"])

        meta_upd = ck.get("meta", {})
        meta_upd["temperature"]      = T_val
        meta_upd["temp_scaling_done"] = True
        meta_upd["threshold_source"]  = "calibrated"
        # best_thr e best_metric vengono SOVRASCRITTI con i valori calibrati
        # su sigmoid(logit/T). Questo è il threshold che verrà usato in inferenza.
        meta_upd["best_thr"]          = best_thr_scaled
        meta_upd["best_metric"]       = best_score_scaled
        meta_upd["best_metric_scaled"] = best_score_scaled
        meta_upd["thr_sweep_min"]      = float(eval_scaled["thr_sweep_min"])
        meta_upd["thr_sweep_max"]      = float(eval_scaled["thr_sweep_max"])
        meta_upd["thr_sweep_points"]   = int(eval_scaled["thr_sweep_points"])
        meta_upd["thr_metric"]              = eval_scaled["best_metric_name"]
        meta_upd["miou_at_05_scaled"]       = float(eval_scaled["miou_at_05"])
        meta_upd["mf1_at_05_scaled"]        = float(eval_scaled["mf1_at_05"])
        meta_upd["min_iou_at_05_scaled"]    = float(eval_scaled["min_iou_at_05"])
        meta_upd["harmonic_iou_at_05_scaled"] = float(eval_scaled["harmonic_iou_at_05"])
        meta_upd["iou_veg_at_05_scaled"]    = float(eval_scaled["iou_veg_at_05"])
        meta_upd["iou_sand_at_05_scaled"]   = float(eval_scaled["iou_sand_at_05"])
        meta_upd["f1_veg_at_05_scaled"]     = float(eval_scaled["f1_veg_at_05"])
        meta_upd["f1_sand_at_05_scaled"]    = float(eval_scaled["f1_sand_at_05"])
        meta_upd["min_iou_best_scaled"]     = float(eval_scaled["min_iou_best"])
        meta_upd["harmonic_iou_best_scaled"] = float(eval_scaled["harmonic_iou_best"])
        meta_upd["precision_veg_best_scaled"] = float(eval_scaled["precision_veg_best"])
        meta_upd["precision_sand_best_scaled"] = float(eval_scaled["precision_sand_best"])
        meta_upd["recall_veg_best_scaled"]  = float(eval_scaled["recall_veg_best"])
        meta_upd["recall_sand_best_scaled"] = float(eval_scaled["recall_sand_best"])
        meta_upd["iou_veg_best_scaled"]     = float(eval_scaled["iou_veg_best"])
        meta_upd["iou_sand_best_scaled"]    = float(eval_scaled["iou_sand_best"])
        meta_upd["f1_veg_best_scaled"]      = float(eval_scaled["f1_veg_best"])
        meta_upd["f1_sand_best_scaled"]     = float(eval_scaled["f1_sand_best"])
        meta_upd["balanced_accuracy_scaled"] = float(eval_scaled["balanced_accuracy"])
        meta_upd["mcc_scaled"]               = float(eval_scaled["mcc"])
        meta_upd["ece_scaled"]               = float(eval_scaled["ece"])
        meta_upd["brier_scaled"]             = float(eval_scaled["brier"])
        meta_upd["iou_at_05_scaled"]         = float(eval_scaled["iou_veg_at_05"])
        meta_upd["f1_at_05_scaled"]          = float(eval_scaled["f1_veg_at_05"])
        if eval_raw_full is not None:
            meta_upd["balanced_accuracy_raw"] = float(eval_raw_full["balanced_accuracy"])
            meta_upd["mcc_raw"] = float(eval_raw_full["mcc"])
            meta_upd["ece_raw"] = float(eval_raw_full["ece"])
            meta_upd["brier_raw"] = float(eval_raw_full["brier"])
            meta_upd["fast_eval_at05_raw"] = False
            meta_upd["raw_metrics_recomputed_posthoc"] = True
        # Le metriche raw sono preservate per trasparenza scientifica
        # (già presenti come best_thr_raw, best_metric_raw)

        save_best(out_dir, model, meta_upd)
        print(f"  [OK] Temperature T={T_val:.4f}")
        print(f"  best_thr_scaled   = {best_thr_scaled:.3f}  (USATO in inferenza)")
        print(f"  best_metric_scaled= {best_score_scaled:.4f}")
        print(f"  ece_scaled        = {float(eval_scaled['ece']):.4f}")
        print(f"  miou_at_05_scaled = {float(eval_scaled['miou_at_05']):.4f}  "
              f"veg={float(eval_scaled['iou_veg_at_05']):.4f}  sand={float(eval_scaled['iou_sand_at_05']):.4f}")
        print(f"  [vs raw] best_thr_raw={meta_upd.get('best_thr_raw', '?')}  "
              f"best_metric_raw={fmt_optional_float(meta_upd.get('best_metric_raw'))}")

    if best_pt.exists():
        ck   = torch.load(str(best_pt), map_location="cpu", weights_only=False)
        meta = ck.get("meta", {})
        print(f"\n{'='*60}")
        print(f"  Best model        : {best_pt}")
        print(f"  temp_scaling_done : {meta.get('temp_scaling_done', False)}")
        print(f"  temperature       : {meta.get('temperature', 1.0):.4f}")
        print(f"  best_thr          : {float(meta.get('best_thr', 0.5)):.3f}  "
              f"(calibrated={'yes' if meta.get('temp_scaling_done') else 'no, raw'})")
        print(f"  best_metric       : {float(meta.get('best_metric', -1)):.4f}  "
              f"({meta.get('thr_metric', '')})")
        if meta.get("temp_scaling_done"):
            print(f"  miou_at_05_scaled : {fmt_optional_float(meta.get('miou_at_05_scaled'))}  "
                  f"veg={fmt_optional_float(meta.get('iou_veg_at_05_scaled'))}  "
                  f"sand={fmt_optional_float(meta.get('iou_sand_at_05_scaled'))}")
            print(f"  ece_scaled        : {fmt_optional_float(meta.get('ece_scaled'))}")
            print(f"  [raw] best_thr_raw: {meta.get('best_thr_raw', '?')}  "
                  f"best_metric_raw: {fmt_optional_float(meta.get('best_metric_raw'))}")
        else:
            print(f"  miou_at_05_raw    : {fmt_optional_float(meta.get('miou_at_05_raw'))}  "
                  f"veg={fmt_optional_float(meta.get('iou_veg_at_05_raw'))}  "
                  f"sand={fmt_optional_float(meta.get('iou_sand_at_05_raw'))}")
            print(f"  ece_raw           : {fmt_optional_float(meta.get('ece_raw'))}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()









