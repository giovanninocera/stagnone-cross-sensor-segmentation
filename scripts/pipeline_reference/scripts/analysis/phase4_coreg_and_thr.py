"""
Phase 4: co-registration RMSE (between scene pairs) + threshold sensitivity.

Outputs:
  out/runs/analysis/power_v3/coreg_rmse.csv
  out/runs/analysis/power_v3/threshold_sensitivity.csv
  out/runs/analysis/power_v3/PHASE4_REPORT.md
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from scripts.common.config import DATA, OUT, ROOT

# ── constants ─────────────────────────────────────────────────────────────────
POWER_V3   = OUT / "runs" / "analysis" / "power_v3"
RBOA_DIR   = DATA / "rboa"
WM_PATH    = DATA / "geom" / "wm_ALL_f.tif"
MASK_ROOT  = OUT / "training_mask"
PRED_ROOT  = OUT / "pred" / "replacement_2003_paper_v2_matched"

TEST_BLOCKS = (15, 18, 19, 22)
N_BLOCKS_X  = 6
N_BLOCKS_Y  = 6

# co-reg params
REF_SCENE      = "20060408_qb"
TARGET_SCENES  = ["20030709_qb", "20160825_wv", "20230812_pl"]
PATCH_PX       = 32    # NCC template size
SEARCH_PX      = 8     # ±8 px search radius
N_PATCHES      = 80    # patches per scene pair
RNG_SEED       = 42
MIN_NCC        = 0.5   # discard low-confidence matches

# threshold sweep
SCENES = ["20160825_wv", "20230812_pl"]   # test scenes only
BRANCHES = ["control06", "replacement03"]
SEEDS    = [123, 231, 312]
THR_SWEEP = [round(t, 2) for t in np.arange(0.10, 0.91, 0.05)]


# ── helpers ───────────────────────────────────────────────────────────────────
def _load_rboa_band1(scene_id: str) -> np.ndarray:
    date = scene_id[:8]
    p = RBOA_DIR / f"RBOA_{date}.tif"
    with rasterio.open(str(p)) as ds:
        arr = ds.read(1).astype(np.float32)
    # normalize to [0,1]
    lo, hi = np.percentile(arr[arr > 0], [2, 98]) if (arr > 0).any() else (0, 1)
    hi = max(hi, lo + 1e-6)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _make_block_mask(height: int, width: int) -> np.ndarray:
    bh = height // N_BLOCKS_Y
    bw = width  // N_BLOCKS_X
    bm = np.zeros((height, width), dtype=np.int32)
    for by in range(N_BLOCKS_Y):
        for bx in range(N_BLOCKS_X):
            bid = by * N_BLOCKS_X + bx
            y0 = by * bh
            y1 = (by + 1) * bh if by < N_BLOCKS_Y - 1 else height
            x0 = bx * bw
            x1 = (bx + 1) * bw if bx < N_BLOCKS_X - 1 else width
            bm[y0:y1, x0:x1] = bid
    return bm


def _ncc_offset(tmpl: np.ndarray, search: np.ndarray, r: int) -> tuple[float, float, float]:
    """Brute-force NCC: return (dy, dx, peak_ncc). tmpl shape (P,P), search (P+2r, P+2r)."""
    P = tmpl.shape[0]
    H, W = search.shape
    t = tmpl - tmpl.mean()
    t_norm = np.sqrt((t * t).sum())
    if t_norm < 1e-6:
        return float("nan"), float("nan"), float("nan")
    t /= t_norm

    best, best_dy, best_dx = -2.0, 0, 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y0 = r + dy
            x0 = r + dx
            if y0 < 0 or x0 < 0 or y0 + P > H or x0 + P > W:
                continue
            patch = search[y0:y0 + P, x0:x0 + P].copy()
            patch -= patch.mean()
            pn = np.sqrt((patch * patch).sum())
            if pn < 1e-6:
                continue
            ncc_val = float(np.dot(t.ravel(), patch.ravel() / pn))
            if ncc_val > best:
                best, best_dy, best_dx = ncc_val, dy, dx

    return float(best_dy), float(best_dx), best


# ── Phase 4a: co-registration RMSE ───────────────────────────────────────────
def phase4a_coreg() -> pd.DataFrame:
    out_csv = POWER_V3 / "coreg_rmse.csv"
    if out_csv.exists():
        print(f"[4a] Co-reg CSV exists: {out_csv}")
        return pd.read_csv(out_csv)

    print("\n[Phase 4a] Co-registration RMSE via NCC template matching")
    rng = np.random.default_rng(RNG_SEED)

    ref_arr = _load_rboa_band1(REF_SCENE)
    H, W = ref_arr.shape

    # stable sampling region: NONVEG pixels in 2006 training mask (saltpan/dike edges)
    # These are permanent structures that appear in all sensors
    stable_mask_path = MASK_ROOT / "training_2006__FULL.tif"
    with rasterio.open(str(stable_mask_path)) as ds:
        train_mask = ds.read(1)
    nonveg = (train_mask == 1)  # NONVEG = saltpans, dikes — stable between dates

    # gradient magnitude for hard-feature sampling (edges of saltpan walls)
    gy = np.gradient(ref_arr, axis=0)
    gx = np.gradient(ref_arr, axis=1)
    grad = np.sqrt(gy ** 2 + gx ** 2)
    grad_thr = float(np.percentile(grad[nonveg], 70))

    margin = PATCH_PX // 2 + SEARCH_PX + 4
    candidate = nonveg & (grad >= grad_thr)
    candidate[:margin, :] = False
    candidate[-margin:, :] = False
    candidate[:, :margin] = False
    candidate[:, -margin:] = False

    ys, xs = np.where(candidate)
    n_pick = min(N_PATCHES, len(ys))
    chosen = rng.choice(len(ys), size=n_pick, replace=False)
    sample_ys = ys[chosen]
    sample_xs = xs[chosen]

    all_rows: list[dict] = []

    for tgt_id in TARGET_SCENES:
        try:
            tgt_arr = _load_rboa_band1(tgt_id)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        print(f"  {REF_SCENE} vs {tgt_id} ({n_pick} patches)...")
        rows: list[dict] = []

        for cy, cx in zip(sample_ys.tolist(), sample_xs.tolist()):
            P = PATCH_PX
            r = SEARCH_PX

            tmpl = ref_arr[cy - P//2 : cy + P//2, cx - P//2 : cx + P//2]
            if tmpl.shape != (P, P):
                continue

            y0 = cy - P//2 - r
            y1 = cy + P//2 + r
            x0 = cx - P//2 - r
            x1 = cx + P//2 + r
            if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
                continue
            search = tgt_arr[y0:y1, x0:x1]

            dy, dx, ncc_val = _ncc_offset(tmpl, search, r)
            if np.isnan(dy) or ncc_val < MIN_NCC:
                continue

            rows.append({
                "ref_scene":  REF_SCENE,
                "tgt_scene":  tgt_id,
                "cy": cy, "cx": cx,
                "dy_px": dy, "dx_px": dx,
                "offset_px": float(np.sqrt(dy**2 + dx**2)),
                "ncc_peak": ncc_val,
            })

        if rows:
            df_pair = pd.DataFrame(rows)
            rmse  = float(np.sqrt((df_pair["dy_px"]**2 + df_pair["dx_px"]**2).mean()))
            med   = float(df_pair["offset_px"].median())
            print(f"    n={len(df_pair)}  RMSE={rmse:.3f} px ({rmse*2:.2f} m)  "
                  f"median={med:.3f} px  bias=(dy={df_pair['dy_px'].mean():.2f}, dx={df_pair['dx_px'].mean():.2f})")
            all_rows.extend(rows)
        else:
            print(f"    [WARN] No valid matches for {tgt_id}")

    result = pd.DataFrame(all_rows)
    result.to_csv(out_csv, index=False)
    print(f"  Written: {out_csv}")
    return result


# ── Phase 4b: threshold sensitivity ──────────────────────────────────────────
def _find_prob_tif(branch: str, seed: int, scene_id: str) -> Path | None:
    pred_dir = PRED_ROOT / f"{branch}_s{seed}"
    candidates = sorted(pred_dir.glob(f"prob_{scene_id}_*.tif"))
    return candidates[0] if candidates else None


def _eval_at_thr(prob: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                 thr: float) -> dict[str, float]:
    pred = (prob >= thr)
    gt_veg  = (labels == 2) & valid
    gt_nov  = (labels == 1) & valid
    vld     = valid
    tp = int((pred & gt_veg).sum())
    fp = int((pred & gt_nov).sum())
    fn = int((~pred & gt_veg).sum())
    tn = int((~pred & gt_nov).sum())
    n  = int(vld.sum())
    iou_v = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    iou_n = tn / (tn + fp + fn) if (tn + fp + fn) else float("nan")
    cover = (tp + fp) / n if n else float("nan")
    return {
        "iou_veg": iou_v, "iou_nonveg": iou_n,
        "min_iou": min(iou_v, iou_n),
        "cover_fraction": cover,
        "n_pixels": n,
    }


def phase4b_threshold() -> pd.DataFrame:
    out_csv = POWER_V3 / "threshold_sensitivity.csv"
    if out_csv.exists():
        print(f"[4b] Threshold CSV exists: {out_csv}")
        return pd.read_csv(out_csv)

    print("\n[Phase 4b] Threshold sensitivity sweep")
    rows: list[dict] = []

    for scene_id in SCENES:
        mask_path = MASK_ROOT / f"training_{scene_id[:4]}__FULL.tif"
        if not mask_path.exists():
            print(f"  [SKIP] mask not found: {mask_path}")
            continue

        with rasterio.open(str(mask_path)) as ds:
            labels = ds.read(1)
            height, width = labels.shape

        block_mask = _make_block_mask(height, width)
        test_mask  = np.isin(block_mask, TEST_BLOCKS)
        labeled    = np.isin(labels, (1, 2))
        valid      = test_mask & labeled

        for branch in BRANCHES:
            for seed in SEEDS:
                prob_tif = _find_prob_tif(branch, seed, scene_id)
                if prob_tif is None:
                    print(f"  [SKIP] no prob TIF for {branch} s{seed} {scene_id}")
                    continue

                with rasterio.open(str(prob_tif)) as ds:
                    prob = ds.read(1).astype(np.float32)

                for thr in THR_SWEEP:
                    m = _eval_at_thr(prob, labels, valid, thr)
                    rows.append({
                        "scene_id": scene_id,
                        "branch": branch,
                        "seed": seed,
                        "threshold": thr,
                        **m,
                    })

                print(f"  {scene_id} {branch} s{seed}: done ({len(THR_SWEEP)} thresholds)")

    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"  Written: {out_csv}")
    return result


# ── Phase 4 report ────────────────────────────────────────────────────────────
def write_report(coreg_df: pd.DataFrame, thr_df: pd.DataFrame) -> None:
    now = time.strftime("%Y-%m-%d")
    lines = [
        "# Phase 4 — Co-registration RMSE and Threshold Sensitivity",
        "",
        f"**Date**: {now}  ",
        "**Status**: COMPLETE  ",
        "",
        "---",
        "",
        "## 1. Co-registration RMSE",
        "",
        "**Method**: Normalized cross-correlation (NCC) template matching between RBOA band-1 "
        f"rasters. Reference: {REF_SCENE}. Template {PATCH_PX}×{PATCH_PX} px, search "
        f"±{SEARCH_PX} px, n={N_PATCHES} high-gradient patches per pair, NCC threshold ≥ {MIN_NCC}.",
        "",
    ]

    if coreg_df.empty:
        lines.append("*No co-registration results available.*")
    else:
        summary_rows = []
        for tgt in TARGET_SCENES:
            sub = coreg_df[coreg_df["tgt_scene"] == tgt]
            if sub.empty:
                continue
            rmse  = float(np.sqrt((sub["dy_px"]**2 + sub["dx_px"]**2).mean()))
            med   = float(sub["offset_px"].median())
            bias_y = float(sub["dy_px"].mean())
            bias_x = float(sub["dx_px"].mean())
            n     = len(sub)
            summary_rows.append({
                "pair": f"{REF_SCENE} vs {tgt}",
                "n": n,
                "RMSE (px)": round(rmse, 3),
                "RMSE (m)": round(rmse * 2, 2),
                "median (px)": round(med, 3),
                "bias dy (px)": round(bias_y, 3),
                "bias dx (px)": round(bias_x, 3),
            })

        if summary_rows:
            sum_df = pd.DataFrame(summary_rows)
            lines.append("| Scene pair | n | RMSE (px) | RMSE (m) | Median (px) | Bias dy | Bias dx |")
            lines.append("|------------|---|-----------|----------|-------------|---------|---------|")
            for _, r in sum_df.iterrows():
                lines.append(
                    f"| {r['pair']} | {r['n']} | {r['RMSE (px)']} | "
                    f"{r['RMSE (m)']} | {r['median (px)']} | "
                    f"{r['bias dy (px)']} | {r['bias dx (px)']} |"
                )

        lines += [
            "",
            "**Interpretation**: a co-registration RMSE < 1 px (< 2 m at 2 m GSD) indicates "
            "sub-pixel alignment. This bounds the expected label noise in the 1-5 px boundary "
            "band (Phase 2): if RMSE << boundary band width, boundary errors are dominated by "
            "photo-interpretation uncertainty, not co-registration error.",
            "",
        ]

    lines += [
        "---",
        "",
        "## 2. Threshold Sensitivity",
        "",
        "**Method**: For each probability raster (paper_v2 matched branches, 3 seeds, "
        "test scenes 2016 WV and 2023 PL), threshold was swept from 0.10 to 0.90 in 0.05 "
        "steps. Both min-IoU (accuracy) and cover fraction (science product) are reported.",
        "",
    ]

    if thr_df.empty:
        lines.append("*No threshold sensitivity results available.*")
    else:
        # Calibrated threshold for paper_v2 is 0.65 for ctrl06, 0.65 for repl03
        calib_thr = 0.65

        for scene in SCENES:
            lines.append(f"### {scene}")
            lines.append("")
            lines.append("Mean across 3 seeds (min_iou and cover_fraction at each threshold):")
            lines.append("")
            lines.append("| threshold | ctrl06 min_iou | ctrl06 cover | repl03 min_iou | repl03 cover |")
            lines.append("|-----------|----------------|--------------|----------------|--------------|")
            for thr in THR_SWEEP:
                sub = thr_df[(thr_df["scene_id"] == scene) & (thr_df["threshold"] == thr)]
                c = sub[sub["branch"] == "control06"]["min_iou"].mean()
                c_cov = sub[sub["branch"] == "control06"]["cover_fraction"].mean()
                r = sub[sub["branch"] == "replacement03"]["min_iou"].mean()
                r_cov = sub[sub["branch"] == "replacement03"]["cover_fraction"].mean()
                marker = " **<- calibrated**" if abs(thr - calib_thr) < 0.01 else ""
                lines.append(
                    f"| {thr:.2f}{marker} | {c:.4f} | {c_cov:.4f} | {r:.4f} | {r_cov:.4f} |"
                )
            lines.append("")

        # Cover sensitivity: delta cover over full thr range
        lines.append("### Cover Fraction Sensitivity")
        lines.append("")
        lines.append("Range of cover fraction estimates (mean over 3 seeds) when threshold "
                      "varies from 0.10 to 0.90:")
        lines.append("")
        lines.append("| scene | branch | cover @ thr=0.10 | cover @ calib | "
                      "cover @ thr=0.90 | range |")
        lines.append("|-------|--------|------------------|---------------|"
                      "------------------|-------|")
        for scene in SCENES:
            for branch in BRANCHES:
                sub = thr_df[(thr_df["scene_id"] == scene) & (thr_df["branch"] == branch)]
                c_lo = sub[sub["threshold"] == 0.10]["cover_fraction"].mean()
                c_cal = sub[abs(sub["threshold"] - calib_thr) < 0.01]["cover_fraction"].mean()
                c_hi = sub[sub["threshold"] == 0.90]["cover_fraction"].mean()
                rng = c_lo - c_hi
                lines.append(
                    f"| {scene} | {branch} | {c_lo:.4f} | {c_cal:.4f} | "
                    f"{c_hi:.4f} | {rng:.4f} |"
                )
        lines += [
            "",
            "**Interpretation**: a cover sensitivity range < 0.05 (5 percentage points) across "
            "the full threshold sweep indicates the cover estimate is robust to threshold choice. "
            "The calibrated threshold (thr=0.65) is the one used in the paper_v2 benchmark.",
        ]

    lines += ["", "---", ""]

    report_path = POWER_V3 / "PHASE4_REPORT.md"
    POWER_V3.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {report_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 80)
    print("Phase 4: co-registration RMSE + threshold sensitivity")
    print("=" * 80)
    POWER_V3.mkdir(parents=True, exist_ok=True)

    coreg_df = phase4a_coreg()
    thr_df   = phase4b_threshold()
    write_report(coreg_df, thr_df)


if __name__ == "__main__":
    main()
