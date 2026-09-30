#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts.patches.make_patches - canonical patch CSV/H5 builder.

Estrae patch (coordinate) dal raster feature + maschera di training.
Novità v5:
  • pos_frac DINAMICO: min(0.5, n_nonveg/(n_veg+n_nonveg+1)) — non più hardcoded.
  • --export_only per rigenerare solo i meta JSON (ripristina pos_weight).
  • --out_h5 <path>: scrive direttamente in HDF5 durante l'estrazione (raccomandato).
    Elimina il doppio I/O del workflow precedente (make_patches → .pt → 04b_export_hdf5).
  • export .pt opzionale (legacy), con PT-sanitize (clamp ±6, nan→0, fp16).
  • Spatial block split con min_block_size check (>=2*tile per evitare leakage).
  • VALID-FILTER su valid_* (WM==1 & finite).
  • Meta JSON include combined_pos_weight (usato da 05_train.py).

Storage raccomandato:
  --out_h5 out/patches_bin/STORE_st12s_t256.h5    # scrive HDF5 direttamente
  Nessun bisogno di 04b_export_hdf5.py in questo caso.

Storage legacy (compatibilità):
  --export_pt                                       # scrive .pt atomici in STORE_*/
  poi facoltativamente: 04b_export_hdf5.py per conversione

Uso tipico (HDF5 diretto):
    python -m scripts.patches.make_patches \
        --scene_id 20060408_qb 20160825_wv --mask_year 2006 2016 \
        --suffix _st12s --out_csv out/patches/patches_multi_st12s.csv \
        --n_patches 5000 --tile 256 \
        --out_h5 out/patches_bin/STORE_st12s_t256.h5

Uso tipico (multi bilanciato, totale 6000 = 2000+2000+2000):
    python -m scripts.patches.make_patches \
        --scene_id 20060408_qb 20160825_wv 20230812_pl --mask_year 2006 2016 2023 \
        --suffix _st3b --source_suffix _st7m --feature_preset CMP_BASE3 \
        --out_csv out/patches/patches_multi3_balanced_papertrack_v2.csv \
        --n_patches_total 6000 --tile 256 \
        --out_h5 out/patches_bin/STORE_m3_st3b_t256.h5

Uso tipico (dataset singolo, HDF5):
    python -m scripts.patches.make_patches \
        --scene_id 20060408_qb --mask_year 2006 \
        --suffix _st12s --out_csv out/patches/patches_2006_st12s.csv \
        --n_patches 5000 --tile 256 \
        --out_h5 out/patches_bin/STORE_st12s_t256.h5

    python -m scripts.patches.make_patches \
        --scene_id 20160825_wv --mask_year 2016 \
        --suffix _st12s --out_csv out/patches/patches_2016_st12s.csv \
        --n_patches 5000 --tile 256 \
        --out_h5 out/patches_bin/STORE_st12s_t256.h5  --h5_append

Rigenerazione solo meta (ripristina pos_weight):
    python -m scripts.patches.make_patches \
        --export_only --out_csv out/patches/patches_2006_st12s.csv \
        --suffix _st12s --tile 256
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import h5py
    HAS_H5 = True
except ImportError:
    HAS_H5 = False

from scripts.common.config import OUT, expected_in_channels, p_feat, p_valid, p_pt_store, p_training_mask
from scripts.common.feature_presets import (
    band_indexes_for_channels,
    feature_names_for_preset,
    preset_for_suffix,
    resolve_feature_preset,
    resolve_feature_selection,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Costanti
# ═══════════════════════════════════════════════════════════════════════════════

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

FEAT_CLIP = 6.0


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_scene_patch_targets(n_scenes: int, n_patches: int, n_patches_total: int) -> List[int]:
    if n_scenes <= 0:
        return []
    if int(n_patches_total) > 0:
        if int(n_patches_total) < n_scenes:
            raise ValueError("--n_patches_total deve essere >= numero di scene.")
        base = int(n_patches_total) // n_scenes
        rem = int(n_patches_total) % n_scenes
        return [base + (1 if i < rem else 0) for i in range(n_scenes)]
    return [int(n_patches)] * n_scenes


def _valid_frac_window(vmask: np.ndarray, y0: int, x0: int, tile: int) -> float:
    vw = vmask[y0: y0 + tile, x0: x0 + tile]
    if vw.shape != (tile, tile):
        return 0.0
    return float((vw == 1).mean())


def _sanitize(x: np.ndarray) -> np.ndarray:
    """Clamp, nan→0, cast float16."""
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -FEAT_CLIP, FEAT_CLIP)
    return x.astype(np.float16)


@lru_cache(maxsize=64)
def _read_available_feature_names(feat_tif: str) -> Tuple[str, ...]:
    path = Path(feat_tif)
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
def _resolve_band_indexes(feat_tif: str, feature_preset: str) -> Tuple[int, ...]:
    requested = feature_names_for_preset(feature_preset)
    available = _read_available_feature_names(feat_tif)
    return tuple(band_indexes_for_channels(available, requested))


def _read_feature_patch(
    feat_tif: str,
    window: Window,
    feature_preset: str = "",
) -> np.ndarray:
    with rasterio.open(feat_tif) as src:
        if feature_preset:
            indexes = list(_resolve_band_indexes(feat_tif, feature_preset))
            return src.read(indexes=indexes, window=window).astype("float32")
        return src.read(window=window).astype("float32")


# ═══════════════════════════════════════════════════════════════════════════════
# Imbalance analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_imbalance(mask: np.ndarray, pos_frac_cap: float = 0.5) -> Dict[str, Any]:
    """
    Training mask convention:
      0 = unknown / ignore
      1 = non-veg  (negativo)
      2 = veg      (positivo / classe target)
    """
    n_veg    = int((mask == 2).sum())
    n_nonveg = int((mask == 1).sum())
    n_unk    = int((mask == 0).sum())

    if n_veg == 0 or n_nonveg == 0:
        return {
            "n_veg": n_veg, "n_nonveg": n_nonveg, "n_unknown": n_unk,
            "imbalance_ratio": float("inf"),
            "minority": "unknown",
            "pos_weight": 1.0,
            "pos_frac": 0.5,
        }

    pos_weight      = float(n_nonveg) / float(n_veg)
    imbalance_ratio = max(n_veg, n_nonveg) / min(n_veg, n_nonveg)
    minority        = "veg" if n_veg < n_nonveg else "nonveg"
    pos_frac = min(float(pos_frac_cap), float(n_nonveg) / float(n_veg + n_nonveg + 1))

    return {
        "n_veg":           n_veg,
        "n_nonveg":        n_nonveg,
        "n_unknown":       n_unk,
        "imbalance_ratio": float(imbalance_ratio),
        "minority":        minority,
        "pos_weight":      float(pos_weight),
        "pos_frac":        float(pos_frac),
    }


def window_ok(
    mask_win: np.ndarray, pos_frac: float, boundary_frac: float,
    rng: Optional[random.Random] = None,
) -> bool:
    veg    = int((mask_win == 2).sum())
    nonveg = int((mask_win == 1).sum())
    tot    = veg + nonveg
    if tot == 0:
        return False
    rng = rng or random
    mixed = (veg > 0) and (nonveg > 0)
    if mixed:
        return rng.random() < boundary_frac
    want_pos = rng.random() < pos_frac
    return (veg > 0) if want_pos else (nonveg > 0)



# ═══════════════════════════════════════════════════════════════════════════════
# Spatial block split
# ═══════════════════════════════════════════════════════════════════════════════

def make_holdout_blocks(
    n_blocks_x: int,
    n_blocks_y: int,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[set, set]:
    rng = random.Random(seed)
    blocks = list(range(n_blocks_x * n_blocks_y))
    rng.shuffle(blocks)
    n_total = len(blocks)
    n_test = int(round(test_frac * n_total))
    n_test = max(0, min(n_total, n_test))
    remaining = blocks[n_test:]
    n_val = int(round(val_frac * n_total))
    n_val = max(0, min(len(remaining), n_val))
    test_blocks = set(blocks[:n_test])
    val_blocks = set(remaining[:n_val])
    return val_blocks, test_blocks


def block_class_stats(mask: np.ndarray, n_blocks_x: int, n_blocks_y: int) -> List[Dict[str, float]]:
    height, width = mask.shape
    out: List[Dict[str, float]] = []
    for block_id in range(n_blocks_x * n_blocks_y):
        xs, xe, ys, ye = _block_bounds(width, height, n_blocks_x, n_blocks_y, block_id)
        block = mask[ys:ye, xs:xe]
        veg = int(np.sum(block == 2))
        nonveg = int(np.sum(block == 1))
        n = veg + nonveg
        frac = float(veg / n) if n else float("nan")
        out.append({"block_id": float(block_id), "veg": float(veg), "nonveg": float(nonveg), "n": float(n), "veg_frac": frac})
    return out


def _subset_veg_fraction(stats: List[Dict[str, float]], block_ids: set[int], fallback: float) -> float:
    veg = sum(float(stats[i]["veg"]) for i in block_ids)
    n = sum(float(stats[i]["n"]) for i in block_ids)
    return float(veg / n) if n > 0 else float(fallback)


def make_balanced_holdout_blocks(
    mask: np.ndarray,
    n_blocks_x: int,
    n_blocks_y: int,
    val_frac: float,
    test_frac: float,
    seed: int,
    search: int,
    max_spread: float,
) -> Tuple[set, set, Dict[str, Any]]:
    stats = block_class_stats(mask, n_blocks_x, n_blocks_y)
    all_blocks = list(range(n_blocks_x * n_blocks_y))
    n_total = len(all_blocks)
    n_test = max(0, min(n_total, int(round(test_frac * n_total))))
    n_val = max(0, min(n_total - n_test, int(round(val_frac * n_total))))
    total_veg = sum(float(s["veg"]) for s in stats)
    total_n = sum(float(s["n"]) for s in stats)
    global_frac = float(total_veg / total_n) if total_n > 0 else 0.5
    best: Tuple[float, int, set, set, Dict[str, Any]] | None = None

    n_search = max(1, int(search))
    for offset in range(n_search):
        trial_seed = int(seed) + offset
        rng = random.Random(trial_seed)
        blocks = all_blocks[:]
        rng.shuffle(blocks)
        test_blocks = set(blocks[:n_test])
        val_blocks = set(blocks[n_test:n_test + n_val])
        train_blocks = set(all_blocks) - test_blocks - val_blocks
        fracs = {
            "train": _subset_veg_fraction(stats, train_blocks, global_frac),
            "val": _subset_veg_fraction(stats, val_blocks, global_frac),
            "test": _subset_veg_fraction(stats, test_blocks, global_frac),
        }
        spread = max(fracs.values()) - min(fracs.values())
        score = spread + 0.25 * abs(fracs["train"] - global_frac)
        diag = {
            "strategy": "balanced",
            "selected_seed": trial_seed,
            "search": n_search,
            "global_veg_fraction": global_frac,
            "train_veg_fraction": fracs["train"],
            "val_veg_fraction": fracs["val"],
            "test_veg_fraction": fracs["test"],
            "veg_fraction_spread": spread,
            "max_spread_requested": float(max_spread),
            "n_train_blocks": len(train_blocks),
            "n_val_blocks": len(val_blocks),
            "n_test_blocks": len(test_blocks),
        }
        candidate = (score, trial_seed, val_blocks, test_blocks, diag)
        if best is None or candidate[0] < best[0]:
            best = candidate
            if spread <= max_spread:
                break

    assert best is not None
    _, _, val_blocks, test_blocks, diag = best
    return val_blocks, test_blocks, diag


def make_shared_holdout_blocks(
    masks: List[np.ndarray],
    n_blocks_x: int,
    n_blocks_y: int,
    val_frac: float,
    test_frac: float,
    seed: int,
    search: int,
    max_spread: float,
) -> Tuple[set, set, Dict[str, Any]]:
    """
    Compute ONE val/test block assignment shared across all scenes.

    Optimises class balance and holdout compactness over all scenes so that
    the chosen val and test blocks are geographically identical across dates.
    Compact holdouts preserve more usable training area when spatial buffers
    are enabled.
    """
    all_stats = [block_class_stats(m, n_blocks_x, n_blocks_y) for m in masks]
    all_blocks = list(range(n_blocks_x * n_blocks_y))
    n_total = len(all_blocks)
    n_test  = max(0, min(n_total, int(round(test_frac * n_total))))
    n_val   = max(0, min(n_total - n_test, int(round(val_frac * n_total))))

    global_fracs = []
    for stats in all_stats:
        tv = sum(float(s["veg"]) for s in stats)
        tn = sum(float(s["n"])   for s in stats)
        global_fracs.append(float(tv / tn) if tn > 0 else 0.5)

    best: Tuple[float, int, set, set, Dict[str, Any]] | None = None
    n_search = max(1, int(search))

    for offset in range(n_search):
        trial_seed = int(seed) + offset
        rng = random.Random(trial_seed)
        blocks = all_blocks[:]
        rng.shuffle(blocks)
        test_blocks  = set(blocks[:n_test])
        val_blocks   = set(blocks[n_test:n_test + n_val])
        train_blocks = set(all_blocks) - test_blocks - val_blocks

        per_scene_spreads = []
        per_scene_fracs = []
        per_scene_train_retention = []
        for stats, gf in zip(all_stats, global_fracs):
            fracs = {
                "train": _subset_veg_fraction(stats, train_blocks, gf),
                "val":   _subset_veg_fraction(stats, val_blocks,   gf),
                "test":  _subset_veg_fraction(stats, test_blocks,  gf),
            }
            per_scene_spreads.append(max(fracs.values()) - min(fracs.values()))
            per_scene_fracs.append({key: round(value, 4) for key, value in fracs.items()})
            total_n = sum(float(s["n"]) for s in stats)
            train_n = sum(float(stats[block_id]["n"]) for block_id in train_blocks)
            per_scene_train_retention.append(float(train_n / total_n) if total_n > 0 else 0.0)

        worst_spread = max(per_scene_spreads)
        holdout_blocks = val_blocks | test_blocks
        train_boundary_edges = 0
        for block_id in holdout_blocks:
            by, bx = divmod(block_id, n_blocks_x)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = by + dy, bx + dx
                if 0 <= ny < n_blocks_y and 0 <= nx < n_blocks_x:
                    neighbor = ny * n_blocks_x + nx
                    if neighbor not in holdout_blocks:
                        train_boundary_edges += 1
        compactness_penalty = (
            float(train_boundary_edges) / float(max(1, 4 * len(holdout_blocks)))
        )
        min_train_retention = min(per_scene_train_retention)
        retention_penalty = max(0.0, 0.55 - min_train_retention)
        score = worst_spread + 0.04 * compactness_penalty + 0.50 * retention_penalty
        diag = {
            "strategy": "shared_balanced",
            "selected_seed": trial_seed,
            "search": n_search,
            "per_scene_spreads": [round(s, 4) for s in per_scene_spreads],
            "per_scene_fractions": per_scene_fracs,
            "per_scene_train_retention": [round(v, 4) for v in per_scene_train_retention],
            "worst_spread": round(worst_spread, 4),
            "train_boundary_edges": int(train_boundary_edges),
            "compactness_penalty": round(compactness_penalty, 4),
            "selection_score": round(score, 6),
            "max_spread_requested": float(max_spread),
            "n_train_blocks": len(train_blocks),
            "n_val_blocks":   len(val_blocks),
            "n_test_blocks":  len(test_blocks),
        }
        candidate = (score, trial_seed, val_blocks, test_blocks, diag)
        if best is None or score < best[0]:
            best = candidate

    assert best is not None
    _, _, val_blocks, test_blocks, diag = best
    return val_blocks, test_blocks, diag


def _block_bounds(width: int, height: int, n_blocks_x: int, n_blocks_y: int, block_id: int) -> Tuple[int, int, int, int]:
    by, bx = divmod(block_id, n_blocks_x)
    xs = int(np.floor(bx * width / n_blocks_x))
    xe = int(np.floor((bx + 1) * width / n_blocks_x))
    ys = int(np.floor(by * height / n_blocks_y))
    ye = int(np.floor((by + 1) * height / n_blocks_y))
    return xs, xe, ys, ye


def _window_intersects_buffered_blocks(
    x0: int,
    y0: int,
    tile: int,
    width: int,
    height: int,
    n_blocks_x: int,
    n_blocks_y: int,
    blocks: set,
    buffer_px: int,
) -> bool:
    px0, px1 = x0, x0 + tile
    py0, py1 = y0, y0 + tile
    for block_id in blocks:
        xs, xe, ys, ye = _block_bounds(
            width, height, n_blocks_x, n_blocks_y, block_id
        )
        xs -= buffer_px
        xe += buffer_px
        ys -= buffer_px
        ye += buffer_px
        if (px0 < xe) and (px1 > xs) and (py0 < ye) and (py1 > ys):
            return True
    return False


def assign_split_with_buffer(
    x0: int, y0: int, tile: int,
    width: int, height: int,
    n_blocks_x: int, n_blocks_y: int,
    val_blocks: set,
    test_blocks: set,
    val_buffer_px: int = 0,
    test_buffer_px: int = 0,
) -> str:
    if val_blocks & test_blocks:
        raise ValueError("Validation and test block sets must be disjoint")

    xc = x0 + tile / 2.0
    yc = y0 + tile / 2.0
    bx = min(n_blocks_x - 1, int(xc / max(1, width) * n_blocks_x))
    by = min(n_blocks_y - 1, int(yc / max(1, height) * n_blocks_y))
    block_id = by * n_blocks_x + bx

    # Held-out splits must also be spatially independent from each other.
    # Previously these early returns bypassed the reciprocal buffer checks.
    if block_id in test_blocks:
        if _window_intersects_buffered_blocks(
            x0, y0, tile, width, height, n_blocks_x, n_blocks_y,
            val_blocks, val_buffer_px,
        ):
            return "ignore"
        return "test"
    if block_id in val_blocks:
        if _window_intersects_buffered_blocks(
            x0, y0, tile, width, height, n_blocks_x, n_blocks_y,
            test_blocks, test_buffer_px,
        ):
            return "ignore"
        return "val"

    if _window_intersects_buffered_blocks(
        x0, y0, tile, width, height, n_blocks_x, n_blocks_y,
        test_blocks, test_buffer_px,
    ):
        return "ignore"
    if _window_intersects_buffered_blocks(
        x0, y0, tile, width, height, n_blocks_x, n_blocks_y,
        val_blocks, val_buffer_px,
    ):
        return "ignore"
    return "train"


# ═══════════════════════════════════════════════════════════════════════════════
# Row sampling
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Row:
    scene_id: str
    feat_tif: str
    mask_tif: str
    y0: int
    x0: int
    tile: int
    split: str


def dedupe_rows(rows: List[Row]) -> Tuple[List[Row], int]:
    """
    Deduplica le patch per chiave geometrica.
    Mantiene la prima occorrenza per avere CSV/HDF5 coerenti e cardinalita' auditabile.
    """
    seen = set()
    out: List[Row] = []
    dup = 0
    for r in rows:
        key = (r.scene_id, int(r.y0), int(r.x0), int(r.tile))
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        out.append(r)
    return out, dup


def build_rows(
    scene_id: str,
    feat_path: Path,
    mask_path: Path,
    valid_path: Optional[Path],
    tile: int, n_patches: int,
    pos_frac: float, boundary_frac: float,
    jitter: int,
    n_blocks_x: int, n_blocks_y: int,
    val_frac: float, seed: int,
    min_valid_frac: float = 0.95,
    val_buffer_px: int = 0,
    test_frac: float = 0.0,
    test_buffer_px: int = 0,
    scene_role: str = "auto",
    split_strategy: str = "random",
    split_seed_search: int = 1,
    split_balance_max_spread: float = 0.10,
    fixed_val_blocks: Optional[set] = None,
    fixed_test_blocks: Optional[set] = None,
) -> List[Row]:
    rng = random.Random(seed)

    vmask = None
    if valid_path is not None and valid_path.exists():
        with rasterio.open(valid_path) as vs:
            vmask = vs.read(1).astype("uint8")
    else:
        print(f"  [WARN] valid_path non trovato: {valid_path} — filtro VALID disabilitato")

    with rasterio.open(mask_path) as ms:
        mask = ms.read(1)
    H, W = mask.shape

    max_x0 = max(0, W - tile)
    max_y0 = max(0, H - tile)

    rows: List[Row] = []
    if fixed_val_blocks is not None and fixed_test_blocks is not None:
        # Blocks locked externally (--shared_blocks mode): use as-is, no per-scene search.
        val_blocks  = set(fixed_val_blocks)
        test_blocks = set(fixed_test_blocks)
        print(
            "    split_strategy=shared_locked "
            f"val_blocks={sorted(val_blocks)} "
            f"test_blocks={sorted(test_blocks)}"
        )
    elif split_strategy == "balanced" and scene_role == "auto":
        val_blocks, test_blocks, split_diag = make_balanced_holdout_blocks(
            mask,
            n_blocks_x,
            n_blocks_y,
            val_frac,
            test_frac,
            seed,
            split_seed_search,
            split_balance_max_spread,
        )
        print(
            "    split_strategy=balanced "
            f"seed={split_diag['selected_seed']} "
            f"veg_frac train/val/test="
            f"{split_diag['train_veg_fraction']:.3f}/"
            f"{split_diag['val_veg_fraction']:.3f}/"
            f"{split_diag['test_veg_fraction']:.3f} "
            f"spread={split_diag['veg_fraction_spread']:.3f}"
        )
        if float(split_diag["veg_fraction_spread"]) > float(split_balance_max_spread):
            print(
                f"    [WARN] balanced split spread above target: "
                f"{split_diag['veg_fraction_spread']:.3f} > {split_balance_max_spread:.3f}"
            )
    else:
        val_blocks, test_blocks = make_holdout_blocks(n_blocks_x, n_blocks_y, val_frac, test_frac, seed)

    # no-replacement sulle finestre (y0, x0)
    seen: set[tuple[int, int]] = set()
    max_unique = (max_x0 + 1) * (max_y0 + 1)
    max_tries = min(max_unique, n_patches * 200)
    tries = 0

    if jitter > 0:
        print("  [WARN] jitter ignorato in modalità no-replacement su x0/y0")

    while len(rows) < n_patches and tries < max_tries and len(seen) < max_unique:
        tries += 1

        x0 = rng.randint(0, max_x0)
        y0 = rng.randint(0, max_y0)

        key = (y0, x0)
        if key in seen:
            continue
        seen.add(key)

        if vmask is not None:
            if _valid_frac_window(vmask, y0, x0, tile) < min_valid_frac:
                continue

        mw = mask[y0:y0 + tile, x0:x0 + tile]
        if mw.shape != (tile, tile):
            continue

        if not window_ok(mw, pos_frac, boundary_frac, rng=rng):
            continue

        if scene_role in ("train", "val", "test"):
            sp = scene_role
        else:
            sp = assign_split_with_buffer(
                x0, y0, tile, W, H, n_blocks_x, n_blocks_y,
                val_blocks, test_blocks,
                val_buffer_px, test_buffer_px
            )
            if sp == "ignore":
                continue

        rows.append(Row(
            scene_id=scene_id,
            feat_tif=str(feat_path),
            mask_tif=str(mask_path),
            y0=y0,
            x0=x0,
            tile=tile,
            split=sp,
        ))

    if len(rows) < n_patches:
        print(
            f"  [WARN] {scene_id}: ottenute {len(rows)}/{n_patches} patch "
            f"in {tries} tentativi unici. "
            f"Probabile spazio valido troppo stretto: riduci --min_valid_frac o --n_patches."
        )

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# CSV I/O
# ═══════════════════════════════════════════════════════════════════════════════

FIELDNAMES = ["scene_id", "feat_tif", "mask_tif", "y0", "x0", "tile", "split"]


def write_csv(rows: List[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def rows_from_csv(path: Path) -> List[Row]:
    df = pd.read_csv(path)
    return [
        Row(
            scene_id=str(r["scene_id"]),
            feat_tif=str(r["feat_tif"]),
            mask_tif=str(r["mask_tif"]),
            y0=int(r["y0"]), x0=int(r["x0"]),
            tile=int(r["tile"]),
            split=str(r["split"]),
        )
        for _, r in df.iterrows()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

def imbalance_from_rows(rows: List[Row]) -> Dict[str, Any]:
    by_scene: Dict[str, List[Row]] = {}
    for r in rows:
        by_scene.setdefault(r.scene_id, []).append(r)
    result = {}
    for sid, srows in by_scene.items():
        mask_path = srows[0].mask_tif
        tile      = srows[0].tile
        veg = nv  = unk = 0
        seen: set = set()
        for r in srows:
            key = (r.y0, r.x0)
            if key in seen:
                continue
            seen.add(key)
            try:
                with rasterio.open(mask_path) as ms:
                    w = ms.read(1, window=Window(r.x0, r.y0, tile, tile))
                veg += int((w == 2).sum())
                nv  += int((w == 1).sum())
                unk += int((w == 0).sum())
            except Exception:
                pass
        result[sid] = {
            "n_veg": veg, "n_nonveg": nv, "n_unknown": unk,
            "pos_weight": round(nv / max(1, veg), 4),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HDF5 export diretto  (NUOVO — raccomandato)
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_expected_channels(x: np.ndarray, in_ch: int, key: str) -> np.ndarray:
    """Fail fast on feature/channel mismatches instead of padding or truncating silently."""
    if x.shape[0] != in_ch:
        raise ValueError(
            f"{key}: feature channels = {x.shape[0]}, expected = {in_ch}. "
            "Regenerate the feature stack or align --in_ch/--suffix before exporting patches."
        )
    return x


def export_h5_store(
    rows: List[Row],
    h5_path: Path,
    in_ch: int,
    feature_preset: str = "",
    append: bool = False,
    compression: str = "lzf",
) -> Dict[str, Any]:
    """
    Scrive patch direttamente in HDF5 durante make_patches.
    Struttura identica a quella di 04b_export_hdf5.py per piena compatibilità
    con 05_train.py.

    Layout HDF5:
      /patches/<scene_id>__y<y0>_x<x0>_t<tile>/x  → float16 (C, T, T)
      /patches/<scene_id>__y<y0>_x<x0>_t<tile>/y  → uint8   (T, T)
      /patches/<scene_id>__y<y0>_x<x0>_t<tile>/w  → uint8   (T, T)
      /index   → JSON-encoded list of keys (per shuffle veloce)

    Args:
        rows:        lista Row da scrivere
        h5_path:     path file HDF5 di output
        in_ch:       numero di canali feature attesi
        append:      se True, aggiunge a HDF5 esistente (utile per multi-CSV)
        compression: "lzf" (default, veloce), "gzip", "none"
    """
    if not HAS_H5:
        raise ImportError(
            "h5py non installato. Esegui:\n"
            "  pip install h5py --break-system-packages\n"
            "oppure:  conda install h5py"
        )

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Determina keys già presenti se append
    existing_keys: set = set()
    if append and h5_path.exists():
        with h5py.File(str(h5_path), "r") as f:
            if "patches" in f:
                existing_keys = set(f["patches"].keys())
        print(f"  [HDF5] append: {len(existing_keys)} patch già presenti")

    mode = "a" if (append and h5_path.exists()) else "w"
    compress_opts: Dict[str, Any] = (
        {"compression": compression} if compression != "none"
        else {}
    )

    written = skipped = errors = 0
    keys_list: List[str] = list(existing_keys)
    t0 = time.time()

    with h5py.File(str(h5_path), mode) as f:
        grp = f.require_group("patches")

        for i, r in enumerate(rows):
            key = f"{r.scene_id}__y{r.y0}_x{r.x0}_t{r.tile}"

            if key in existing_keys:
                skipped += 1
                continue

            try:
                x = _read_feature_patch(
                    r.feat_tif,
                    Window(r.x0, r.y0, r.tile, r.tile),
                    feature_preset=feature_preset,
                )
                with rasterio.open(r.mask_tif) as ms:
                    m = ms.read(
                        1, window=Window(r.x0, r.y0, r.tile, r.tile)
                    ).astype(np.uint8)
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  [WARN] {key}: {e}")
                continue

            # Sanitize + cast fp16
            x = _sanitize(x)

            # Strict channel check: never hide feature-stack mismatches.
            x = _ensure_expected_channels(x, in_ch, key)

            # Weight mask: 1 dove labeled (1 o 2), 0 altrove
            w = ((m == 1) | (m == 2)).astype(np.uint8)

            patch = grp.require_group(key)
            existed_full = ("x" in patch) and ("y" in patch) and ("w" in patch)
            if existed_full:
                keys_list.append(key)
                skipped += 1
                continue

            if "x" not in patch:
                patch.create_dataset("x", data=x, dtype=np.float16,
                                     chunks=True, **compress_opts)
            if "y" not in patch:
                patch.create_dataset("y", data=m, dtype=np.uint8,
                                     chunks=True, **compress_opts)
            if "w" not in patch:
                patch.create_dataset("w", data=w, dtype=np.uint8,
                                     chunks=True, **compress_opts)

            patch.attrs["scene_id"] = r.scene_id
            patch.attrs["split"]    = r.split
            if feature_preset:
                patch.attrs["feature_preset"] = feature_preset

            keys_list.append(key)
            written += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate    = max(written, 1) / elapsed
                eta     = (len(rows) - i - 1) / rate
                print(
                    f"  [{i+1}/{len(rows)}]  "
                    f"written={written}  skip={skipped}  err={errors}  "
                    f"rate={rate:.0f}/s  ETA={eta:.0f}s"
                )

        # Aggiorna indice globale (permette shuffle senza scansione HDF5)
        if "index" in f:
            del f["index"]
        f.create_dataset(
            "index",
            data=np.frombuffer(
                json.dumps(keys_list).encode(), dtype=np.uint8
            ),
        )

    elapsed = time.time() - t0
    size_mb = h5_path.stat().st_size / 1e6
    print(f"\n  [HDF5] completato in {elapsed:.0f}s")
    print(f"  [HDF5] written={written}  skipped={skipped}  errors={errors}")
    print(f"  [HDF5] file size: {size_mb:.1f} MB  →  {h5_path}")

    return {
        "enabled":   True,
        "h5_path":   str(h5_path),
        "written":   written,
        "skipped":   skipped,
        "errors":    errors,
        "size_mb":   round(size_mb, 1),
        "n_total":   len(keys_list),
        "append":    append,
        "compression": compression,
        "feature_preset": feature_preset or None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PT export (legacy — mantenuto per compatibilità)
# ═══════════════════════════════════════════════════════════════════════════════

def export_pt_store(
    rows: List[Row],
    store_dir: Path,
    in_ch: int,
    feature_preset: str = "",
    overwrite: bool = False,
    x_dtype: str = "float16",
) -> Dict[str, Any]:
    if not HAS_TORCH:
        raise ImportError("torch non installato. Usa --out_h5 per HDF5 diretto.")

    store_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = errors = 0

    for r in rows:
        fname = f"{r.scene_id}__y{r.y0}_x{r.x0}_t{r.tile}.pt"
        fpath = store_dir / fname
        if fpath.exists() and not overwrite:
            skipped += 1
            continue

        try:
            x = _read_feature_patch(
                r.feat_tif,
                Window(r.x0, r.y0, r.tile, r.tile),
                feature_preset=feature_preset,
            )
            with rasterio.open(r.mask_tif) as ms:
                m = ms.read(1, window=Window(r.x0, r.y0, r.tile, r.tile)).astype("uint8")
        except Exception as e:
            errors += 1
            continue

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -FEAT_CLIP, FEAT_CLIP)

        x = _ensure_expected_channels(x, in_ch, fpath.name)

        w = ((m == 1) | (m == 2)).astype("uint8")

        xt = torch.from_numpy(x).to(torch.float16 if x_dtype == "float16" else torch.float32)
        yt = torch.from_numpy(m)
        wt = torch.from_numpy(w)

        torch.save({"x": xt, "y": yt, "w": wt}, fpath)
        written += 1

    return {
        "enabled":  True,
        "out_dir":  str(store_dir),
        "written":  written,
        "skipped":  skipped,
        "errors":   errors,
        "feature_preset": feature_preset or None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Estrae patch CSV + export store (HDF5 o .pt) per training ST_SAV.\n"
            "RACCOMANDATO: --out_h5 per scrittura HDF5 diretta (nessun .pt intermedio)."
        )
    )
    ap.add_argument("--out_csv",    required=True,
                    help="Path CSV di output (coordinate patch).")
    ap.add_argument("--suffix",     default="_st16s",
                    help="Logical experiment suffix (for example: _st5d or _st10f).")
    ap.add_argument("--source_suffix", default="",
                    help="Optional source feature-stack suffix. Use this to subset channels "
                         "from a richer master stack without rewriting raster copies.")
    ap.add_argument("--feature_preset", default="",
                    help="Optional feature preset to subset from the source stack. "
                         "If omitted, it is inferred from --suffix when possible.")
    ap.add_argument("--tile",       type=int, default=256)
    ap.add_argument("--scene_id",   nargs="*", default=None)
    ap.add_argument("--mask_year",  nargs="*", default=None,
                    help="Anno maschera (es. 2006 o 2016); uno per scena o uno solo (replicato).")
    ap.add_argument(
        "--training_mask",
        nargs="*",
        default=None,
        help="Optional explicit mask path per scene; overrides paths inferred from --mask_year.",
    )
    ap.add_argument("--n_patches",  type=int, default=5000)
    ap.add_argument("--n_patches_total", type=int, default=0,
                    help="Totale patch da ripartire in modo bilanciato tra le scene.")
    ap.add_argument("--boundary_frac", type=float, default=0.5)
    ap.add_argument("--pos_frac_cap", type=float, default=0.5,
                    help="Cap massimo per oversampling patch positive (default compatibile: 0.5).")
    ap.add_argument("--jitter",     type=int, default=32)
    ap.add_argument("--n_blocks_x", type=int, default=6)
    ap.add_argument("--n_blocks_y", type=int, default=6)
    ap.add_argument("--val_frac",   type=float, default=0.2)
    ap.add_argument("--test_frac",  type=float, default=0.0,
                    help="Frazione di blocchi tenuta come test held-out completamente unseen.")
    ap.add_argument("--val_buffer_px", type=int, default=0,
                    help="Buffer spaziale attorno ai blocchi di validation; patch train nel buffer vengono escluse.")
    ap.add_argument("--test_buffer_px", type=int, default=0,
                    help="Buffer spaziale attorno ai blocchi di test; patch train/val nel buffer vengono escluse.")
    ap.add_argument("--min_valid_frac", type=float, default=0.95)
    ap.add_argument("--scene_role", nargs="*", default=None,
                    help="Ruolo per scena: auto/train/val/test. Utile per split cross-scene reviewer-grade.")
    ap.add_argument("--seed",       type=int, default=123)
    ap.add_argument("--split_strategy", default="random", choices=["random", "balanced"],
                    help="Block split strategy. balanced searches seeds for similar veg/nonveg fractions.")
    ap.add_argument("--split_seed_search", type=int, default=1,
                    help="Number of consecutive seeds tested when --split_strategy balanced.")
    ap.add_argument("--split_balance_max_spread", type=float, default=0.10,
                    help="Accepted max train/val/test vegetation-fraction spread for balanced block split.")
    ap.add_argument("--shared_blocks", action="store_true",
                    help="Compute ONE shared val/test block assignment across all scenes "
                         "(requires --split_strategy balanced). Eliminates cross-date spatial leakage.")
    ap.add_argument("--fixed_val_blocks", nargs="+", type=int, default=None,
                    help="Explicit validation block IDs shared by all scenes.")
    ap.add_argument("--fixed_test_blocks", nargs="+", type=int, default=None,
                    help="Explicit test block IDs shared by all scenes.")
    ap.add_argument("--in_ch",      type=int, default=0,
                    help="Expected feature channels. Default: inferred from suffix.")

    # ── HDF5 diretto (RACCOMANDATO) ──────────────────────────────────────────
    h5_grp = ap.add_argument_group("HDF5 export (raccomandato)")
    h5_grp.add_argument(
        "--out_h5", default="",
        help="Path file HDF5 di output. Se specificato, scrive direttamente in HDF5."
    )
    h5_grp.add_argument(
        "--h5_append", action="store_true",
        help=(
            "Modalità append: aggiunge patch a HDF5 esistente. "
            "Usare quando si vuole un unico HDF5 per più dataset "
            "(es. 2006 e 2016 separati ma stesso store)."
        )
    )
    h5_grp.add_argument(
        "--h5_compression", default="lzf",
        choices=["lzf", "gzip", "none"],
        help="Compressione HDF5: lzf=veloce (default), gzip=più compatta, none=senza."
    )

    # ── PT store (legacy) ────────────────────────────────────────────────────
    pt_grp = ap.add_argument_group("PT store (legacy)")
    pt_grp.add_argument(
        "--export_pt", action="store_true",
        help="Scrive .pt atomici (uno per patch). Più lento e pesante di HDF5."
    )
    pt_grp.add_argument("--pt_dir",      default="",
                        help="Override directory PT store.")
    pt_grp.add_argument("--pt_overwrite", action="store_true")

    # ── Export only ──────────────────────────────────────────────────────────
    ap.add_argument(
        "--export_only", action="store_true",
        help=(
            "Legge CSV esistente, rigenera meta JSON. "
            "Non richiede --scene_id. Utile per ripristinare pos_weight."
        )
    )
    args = ap.parse_args()
    source_suffix = args.source_suffix or args.suffix

    feature_preset = ""
    if args.feature_preset:
        try:
            feature_preset = resolve_feature_preset(args.feature_preset)
        except KeyError as exc:
            raise SystemExit(str(exc))
    else:
        try:
            feature_preset = resolve_feature_selection(suffix=args.suffix)
        except KeyError:
            feature_preset = ""

    if args.in_ch <= 0:
        if feature_preset:
            args.in_ch = len(feature_names_for_preset(feature_preset))
        else:
            args.in_ch = expected_in_channels(args.suffix, fallback=16)

    set_seed(args.seed)
    out_csv   = Path(args.out_csv)
    store_dir = (
        Path(args.pt_dir).resolve() if args.pt_dir
        else p_pt_store(args.suffix, args.tile)
    )

    print(f"\n[04_make_patches] AVVIO")
    print(f"  out_csv   : {out_csv}")
    print(f"  suffix    : {args.suffix}")
    print(f"  src_suf   : {source_suffix}")
    print(f"  preset    : {feature_preset or 'FULL_SOURCE'}")
    if args.n_patches_total > 0:
        print(f"  tile={args.tile}  target_total={args.n_patches_total}  in_ch={args.in_ch}")
        print("  allocation_mode=balanced_total")
    else:
        print(f"  tile={args.tile}  target_per_scene={args.n_patches}  in_ch={args.in_ch}")
    print(f"  export_only={args.export_only}")
    if args.out_h5:
        print(f"  out_h5    : {args.out_h5}  (append={args.h5_append}  compression={args.h5_compression})")
    elif args.export_pt:
        print(f"  store_dir : {store_dir}  [PT legacy]")

    # ── EXPORT ONLY ────────────────────────────────────────────────────────
    if args.export_only:
        if not out_csv.exists():
            raise FileNotFoundError(f"CSV non trovato: {out_csv}")
        rows_all = rows_from_csv(out_csv)
        print(f"  [INFO] export_only: {len(rows_all)} righe da CSV esistente")

        h5_meta: Dict[str, Any] = {"enabled": False}
        pt_meta: Dict[str, Any] = {"enabled": False}

        if args.out_h5:
            h5_meta = export_h5_store(
                rows_all, Path(args.out_h5), args.in_ch,
                feature_preset=feature_preset,
                append=args.h5_append, compression=args.h5_compression
            )
        elif args.export_pt:
            pt_meta = export_pt_store(
                rows_all, store_dir, args.in_ch,
                feature_preset=feature_preset,
                overwrite=args.pt_overwrite,
            )

        tile    = int(rows_all[0].tile) if rows_all else args.tile
        imb     = imbalance_from_rows(rows_all)
        all_pw  = [v["pos_weight"] for v in imb.values() if v["pos_weight"] > 0]
        combined_pw = float(np.mean(all_pw)) if all_pw else 1.0
        split_counts = {
            "train": int(sum(1 for r in rows_all if r.split == "train")),
            "val": int(sum(1 for r in rows_all if r.split == "val")),
            "test": int(sum(1 for r in rows_all if r.split == "test")),
        }

        meta_path = out_csv.with_name(out_csv.stem + "_meta.json")
        meta_path.write_text(json.dumps({
            "csv": str(out_csv),
            "suffix": args.suffix,
            "source_suffix": source_suffix,
            "feature_preset": feature_preset or None,
            "feature_channels": feature_names_for_preset(feature_preset) if feature_preset else None,
            "tile": tile,
            "combined_pos_weight": combined_pw,
            "per_scene": imb,
            "split_counts": split_counts,
            "h5_export": h5_meta,
            "pt_export": pt_meta,
            "export_only": True,
            "note": "Meta rigenerato da CSV esistente via --export_only.",
        }, indent=2), encoding="utf-8")

        print(f"\n  CSV      : {out_csv}")
        print(f"  Meta     : {meta_path}")
        print(f"  pos_weight (combined): {combined_pw:.3f}")
        return

    # ── NORMAL MODE ────────────────────────────────────────────────────────
    if not args.scene_id or not args.mask_year:
        raise ValueError(
            "Devi specificare --scene_id e --mask_year (oppure usa --export_only)."
        )

    scene_ids  = list(args.scene_id)
    mask_years = list(args.mask_year)
    if len(mask_years) == 1:
        mask_years = mask_years * len(scene_ids)
    if len(mask_years) != len(scene_ids):
        raise ValueError("--mask_year deve essere 1 o len(--scene_id).")
    if args.training_mask:
        training_masks = [Path(path).resolve() for path in args.training_mask]
        if len(training_masks) == 1:
            training_masks = training_masks * len(scene_ids)
        if len(training_masks) != len(scene_ids):
            raise ValueError("--training_mask must be provided once or once per scene_id.")
    else:
        training_masks = [p_training_mask(year) for year in mask_years]
    scene_patch_targets = compute_scene_patch_targets(
        n_scenes=len(scene_ids),
        n_patches=int(args.n_patches),
        n_patches_total=int(args.n_patches_total),
    )

    rows_all: List[Row] = []
    imbalance_info: Dict[str, Any] = {}

    scene_roles = list(args.scene_role) if args.scene_role else ["auto"] * len(scene_ids)
    if len(scene_roles) == 1:
        scene_roles = scene_roles * len(scene_ids)
    if len(scene_roles) != len(scene_ids):
        raise ValueError("--scene_role must be provided once or once per scene_id.")
    scene_roles = [str(s).lower() for s in scene_roles]
    for sr in scene_roles:
        if sr not in ("auto", "train", "val", "test"):
            raise ValueError("--scene_role accepts only: auto, train, val, test")

    scene_patch_plan: Dict[str, int] = dict(zip(scene_ids, scene_patch_targets))

    # -- Shared blocks precomputation ------------------------------------------
    shared_val_blocks: Optional[set] = None
    shared_test_blocks: Optional[set] = None
    if args.fixed_val_blocks is not None or args.fixed_test_blocks is not None:
        if args.fixed_val_blocks is None or args.fixed_test_blocks is None:
            raise ValueError("--fixed_val_blocks and --fixed_test_blocks must be provided together.")
        shared_val_blocks = set(int(v) for v in args.fixed_val_blocks)
        shared_test_blocks = set(int(v) for v in args.fixed_test_blocks)
        n_grid_blocks = int(args.n_blocks_x) * int(args.n_blocks_y)
        invalid = sorted(
            block_id
            for block_id in (shared_val_blocks | shared_test_blocks)
            if block_id < 0 or block_id >= n_grid_blocks
        )
        if invalid:
            raise ValueError(f"Fixed block IDs outside [0, {n_grid_blocks - 1}]: {invalid}")
        overlap = sorted(shared_val_blocks & shared_test_blocks)
        if overlap:
            raise ValueError(f"Validation/test fixed blocks overlap: {overlap}")
        print(
            f"\n  [fixed_blocks] val={sorted(shared_val_blocks)}  "
            f"test={sorted(shared_test_blocks)}"
        )
    elif getattr(args, "shared_blocks", False) and args.split_strategy == "balanced" and len(scene_ids) > 1:
        print("\n  [shared_blocks] Loading masks for joint block optimisation...")
        all_masks = []
        for sid_sb, mt_sb in zip(scene_ids, training_masks):
            if not mt_sb.exists():
                raise FileNotFoundError(f"Training mask non trovata: {mt_sb}")
            with rasterio.open(mt_sb) as ms_sb:
                all_masks.append(ms_sb.read(1))
        shared_val_blocks, shared_test_blocks, sb_diag = make_shared_holdout_blocks(
            masks=all_masks,
            n_blocks_x=args.n_blocks_x,
            n_blocks_y=args.n_blocks_y,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            search=args.split_seed_search,
            max_spread=args.split_balance_max_spread,
        )
        print(
            f"  [shared_blocks] selected seed={sb_diag['selected_seed']}  "
            f"worst_spread={sb_diag['worst_spread']:.4f}  "
            f"val={sorted(shared_val_blocks)}  test={sorted(shared_test_blocks)}"
        )
        if sb_diag["worst_spread"] > args.split_balance_max_spread:
            print(
                f"  [WARN] shared_blocks spread {sb_diag['worst_spread']:.4f} "
                f"> max_spread {args.split_balance_max_spread:.4f} — continuing anyway"
            )
    # --------------------------------------------------------------------------

    for sid, my, mt, sr, n_scene_patches in zip(
        scene_ids, mask_years, training_masks, scene_roles, scene_patch_targets
    ):
        ft = p_feat(sid, source_suffix)
        vt = p_valid(sid, source_suffix)

        print(f"\n  [{sid}]")
        if not ft.exists():
            raise FileNotFoundError(f"Feature non trovate: {ft}  (esegui 02_prepare_features)")
        if not mt.exists():
            raise FileNotFoundError(f"Training mask non trovata: {mt}")

        with rasterio.open(mt) as ms:
            m = ms.read(1)
        imb = analyze_imbalance(m, pos_frac_cap=args.pos_frac_cap)
        imbalance_info[sid] = imb

        print(
            f"    veg={imb['n_veg']:,}  nonveg={imb['n_nonveg']:,}  "
            f"imbalance={imb['imbalance_ratio']:.1f}x  "
            f"pos_weight={imb['pos_weight']:.3f}  pos_frac={imb['pos_frac']:.3f}"
        )
        print(f"    target_patches={int(n_scene_patches)}")

        rows = build_rows(
            scene_id=sid, feat_path=ft, mask_path=mt, valid_path=vt,
            tile=args.tile, n_patches=int(n_scene_patches),
            pos_frac=imb["pos_frac"], boundary_frac=args.boundary_frac,
            jitter=args.jitter,
            n_blocks_x=args.n_blocks_x, n_blocks_y=args.n_blocks_y,
            val_frac=args.val_frac, seed=args.seed,
            min_valid_frac=args.min_valid_frac,
            val_buffer_px=args.val_buffer_px,
            test_frac=args.test_frac,
            test_buffer_px=args.test_buffer_px,
            scene_role=sr,
            split_strategy=args.split_strategy,
            split_seed_search=args.split_seed_search,
            split_balance_max_spread=args.split_balance_max_spread,
            fixed_val_blocks=shared_val_blocks,
            fixed_test_blocks=shared_test_blocks,
            )
        
        rows_all.extend(rows)

        # dedup finale di sicurezza
        seen_keys = set()
        rows_unique = []
        dup_count = 0
        for r in rows_all:
            k = (r.scene_id, r.y0, r.x0, r.tile)
            if k in seen_keys:
                dup_count += 1
                continue
            seen_keys.add(k)
            rows_unique.append(r)

        if dup_count > 0:
            print(f"  [WARN] duplicate patch keys rimosse: {dup_count}")

        rows_all = rows_unique
        write_csv(rows_all, out_csv)

    # Unit check: every sampled row respects the shared geographic partition.
    if shared_val_blocks is not None and len(scene_ids) > 1:
        import csv as _csv
        scene_split_blocks_check: Dict[str, Dict[str, set]] = {}
        scene_shapes: Dict[str, Tuple[int, int]] = {}
        for _sid, _mask_path in zip(scene_ids, training_masks):
            with rasterio.open(_mask_path) as _mask_ds:
                scene_shapes[_sid] = (int(_mask_ds.width), int(_mask_ds.height))
        with open(out_csv, newline="") as _f:
            for _row in _csv.DictReader(_f):
                _sid = _row["scene_id"]
                _split = _row["split"]
                _width, _height = scene_shapes[_sid]
                xc = float(_row["x0"]) + args.tile / 2.0
                yc = float(_row["y0"]) + args.tile / 2.0
                bx = min(args.n_blocks_x - 1, int(xc / max(1, _width) * args.n_blocks_x))
                by = min(args.n_blocks_y - 1, int(yc / max(1, _height) * args.n_blocks_y))
                blk = by * args.n_blocks_x + bx
                if _sid not in scene_split_blocks_check:
                    scene_split_blocks_check[_sid] = {"train": set(), "val": set(), "test": set()}
                scene_split_blocks_check[_sid][_split].add(blk)
                if _split == "val":
                    assert blk in shared_val_blocks, f"[ASSERTION] val row in block {blk}, expected {shared_val_blocks}"
                elif _split == "test":
                    assert blk in shared_test_blocks, f"[ASSERTION] test row in block {blk}, expected {shared_test_blocks}"
                elif _split == "train":
                    assert blk not in (shared_val_blocks | shared_test_blocks), (
                        f"[ASSERTION] train row in held-out block {blk}"
                    )
        print(
            f"\n  [shared_blocks] UNIT CHECK PASSED: all rows respect "
            f"val={sorted(shared_val_blocks)} and test={sorted(shared_test_blocks)} "
            f"across {len(scene_ids)} scenes."
        )

    # pos_weight combinato
    all_pw = [imbalance_info[s]["pos_weight"] for s in imbalance_info
              if imbalance_info[s]["pos_weight"] > 0]
    combined_pw = float(np.mean(all_pw)) if all_pw else 1.0

    # ── Export store ──────────────────────────────────────────────────────
    h5_meta: Dict[str, Any] = {"enabled": False}
    pt_meta: Dict[str, Any] = {"enabled": False}

    if args.out_h5:
        h5_meta = export_h5_store(
            rows_all, Path(args.out_h5), args.in_ch,
            feature_preset=feature_preset,
            append=args.h5_append, compression=args.h5_compression
        )
    elif args.export_pt:
        pt_meta = export_pt_store(
            rows_all, store_dir, args.in_ch,
            feature_preset=feature_preset,
            overwrite=args.pt_overwrite,
        )

    # ── Meta JSON ─────────────────────────────────────────────────────────
    n_tr = sum(1 for r in rows_all if r.split == "train")
    n_va = sum(1 for r in rows_all if r.split == "val")
    n_te = sum(1 for r in rows_all if r.split == "test")
    split_counts = {
        "train": int(n_tr),
        "val": int(n_va),
        "test": int(n_te),
    }
    split_fractions_realized = {
        name: (count / len(rows_all) if rows_all else 0.0)
        for name, count in split_counts.items()
    }

    scene_patch_values = sorted({int(v) for v in scene_patch_plan.values()})
    effective_n_patches_per_scene = scene_patch_values[0] if len(scene_patch_values) == 1 else None
    generated_n_patches_total = int(len(rows_all))

    meta_path = out_csv.with_name(out_csv.stem + "_meta.json")
    meta_path.write_text(json.dumps({
        "csv":                 str(out_csv),
        "suffix":              args.suffix,
        "source_suffix":       source_suffix,
        "feature_preset":      feature_preset or None,
        "feature_channels":    feature_names_for_preset(feature_preset) if feature_preset else None,
        "tile":                int(args.tile),
        "combined_pos_weight": combined_pw,
        "per_scene":           imbalance_info,
        "n_patches_per_scene": effective_n_patches_per_scene,
        "n_patches_total":     generated_n_patches_total,
        "requested_n_patches_per_scene_arg": int(args.n_patches),
        "requested_n_patches_total_arg": int(args.n_patches_total),
        "scene_patch_plan":    scene_patch_plan,
        "training_masks":      {sid: str(path) for sid, path in zip(scene_ids, training_masks)},
        "boundary_frac":       float(args.boundary_frac),
        "pos_frac_cap":        float(args.pos_frac_cap),
        "jitter":              int(args.jitter),
        "n_blocks_x":          int(args.n_blocks_x),
        "n_blocks_y":          int(args.n_blocks_y),
        "val_frac":            float(args.val_frac),
        "split_fraction_semantics": (
            "val_frac and test_frac guide block selection; they are not "
            "row-level sampling quotas"
        ),
        "val_buffer_px":       int(args.val_buffer_px),
        "test_frac":           float(args.test_frac),
        "test_buffer_px":      int(args.test_buffer_px),
        "split_strategy":      str(args.split_strategy),
        "split_seed_search":   int(args.split_seed_search),
        "split_balance_max_spread": float(args.split_balance_max_spread),
        "fixed_val_blocks":    sorted(shared_val_blocks) if shared_val_blocks is not None else None,
        "fixed_test_blocks":   sorted(shared_test_blocks) if shared_test_blocks is not None else None,
        "scene_roles":         scene_roles,
        "split_counts":        split_counts,
        "split_fractions_realized": split_fractions_realized,
        "min_valid_frac":      float(args.min_valid_frac),
        "seed":                int(args.seed),
        "h5_export":           h5_meta,
        "pt_export":           pt_meta,
        "export_only":         False,
    }, indent=2), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"  CSV      : {out_csv}")
    print(f"  Meta     : {meta_path}")
    print(f"  Totale   : {len(rows_all)}  (train={n_tr}, val={n_va}, test={n_te})")
    print(f"  pos_weight (combined): {combined_pw:.3f}")
    if args.out_h5:
        print(f"  HDF5     : {args.out_h5}  "
              f"(written={h5_meta.get('written',0)}, skip={h5_meta.get('skipped',0)})")
    elif args.export_pt:
        print(f"  PT store : {store_dir}  "
              f"(written={pt_meta.get('written',0)}, skip={pt_meta.get('skipped',0)})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()





