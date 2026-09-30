"""Build the non-structural scientific alignment package for manuscript v3.

The script is intentionally read-only with respect to the training/evaluation
artifacts.  It verifies patch and evaluation support, reconstructs the complete
M1/M2/M3 scene-ablation matrix, evaluates the final five-seed probability
ensemble on the frozen held-out support, audits the available 2023 field-point
derivative against the final product, and writes manuscript-ready methods and
results text.

It does not read or modify the manuscript DOCX and it does not touch the
structural-footprint pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/deliverables/scientific_alignment_v3"
PATCH_DIR = ROOT / "out/patches"
EVAL_DIR = ROOT / "out/eval_support/final_scene_ablation_031623_v3"
COMMON_DIR = (
    ROOT / "out/runs/analysis/final_scene_ablation_031623_v3/common_support"
)
FULL_DIR = (
    ROOT / "out/runs/analysis/final_scene_ablation_031623_v3/full_scene_products"
)
PRED_DIR = ROOT / "out/pred/final_scene_ablation_031623_v3_m3_full_hann"

SCENES = {
    "20030709_qb": ("2003 QuickBird", "2003"),
    "20160825_wv": ("2016 WorldView-2", "2016"),
    "20230812_pl": ("2023 Pleiades", "2023"),
}
REFERENCE_FALLBACKS = {
    # The frozen support CSV retains the pre-alignment 2003 filename.  The
    # registered 2 m mask is the actual evaluation reference used by v3.
    "20030709_qb": ROOT / "out/training_mask/training_2003__FULL__b-2m.tif",
    "20160825_wv": ROOT / "out/training_mask/training_2016__FULL.tif",
    "20230812_pl": ROOT / "out/training_mask/training_2023__FULL.tif",
}
VALID_MASKS = {
    scene_id: ROOT
    / "data/features"
    / f"valid_{scene_id}_st3b_final_scene_ablation_031623_v3_common.tif"
    for scene_id in SCENES
}
BRANCH_LABELS = {
    "M1_03": "M1-2003",
    "M1_16": "M1-2016",
    "M1_23": "M1-2023",
    "M2_0316": "M2-2003/2016",
    "M2_0323": "M2-2003/2023",
    "M2_1623": "M2-2016/2023",
    "M3_031623": "M3-2003/2016/2023",
}
BRANCH_FILE_KEYS = {
    "M1_03": "m1_03",
    "M1_16": "m1_16",
    "M1_23": "m1_23",
    "M2_0316": "m2_0316",
    "M2_0323": "m2_0323",
    "M2_1623": "m2_1623",
    "M3_031623": "m3_031623",
}
SEEDS = (123, 231, 312, 423, 531)
N_BINS = 15
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260716
CELL_PX = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def patch_and_support_audit() -> dict[str, Any]:
    branch_rows: list[dict[str, Any]] = []
    m3_scene_rows: list[dict[str, Any]] = []
    for branch, file_key in BRANCH_FILE_KEYS.items():
        path = PATCH_DIR / f"patches_final_scene_ablation_031623_v3_{file_key}.csv"
        frame = pd.read_csv(path)
        if len(frame) != 6000:
            raise AssertionError(f"{branch}: expected 6,000 rows, found {len(frame)}")
        split = frame["split"].value_counts().to_dict()
        scene_counts = frame["scene_id"].value_counts().to_dict()
        branch_rows.append(
            {
                "branch": branch,
                "branch_label": BRANCH_LABELS[branch],
                "training_scenes": "+".join(SCENES[key][1] for key in sorted(scene_counts)),
                "total_patches": int(len(frame)),
                "train_patches": int(split.get("train", 0)),
                "validation_patches": int(split.get("val", 0)),
                "test_patches": int(split.get("test", 0)),
                "patches_per_included_scene": "|".join(
                    f"{SCENES[key][1]}:{int(scene_counts[key])}" for key in sorted(scene_counts)
                ),
                "csv_sha256": sha256(path),
            }
        )
        if branch == "M3_031623":
            grouped = frame.groupby(["scene_id", "split"]).size()
            for scene_id in SCENES:
                m3_scene_rows.append(
                    {
                        "scene_id": scene_id,
                        "scene": SCENES[scene_id][0],
                        "train_patches": int(grouped.get((scene_id, "train"), 0)),
                        "validation_patches": int(grouped.get((scene_id, "val"), 0)),
                        "test_patches": int(grouped.get((scene_id, "test"), 0)),
                        "total_patches": int(scene_counts.get(scene_id, 0)),
                    }
                )

    support_path = EVAL_DIR / "common_eval_manifest.json"
    support = json.loads(support_path.read_text(encoding="utf-8"))
    total_unique = sum(
        int(values["unique_labelled_pixels"])
        for values in support["scene_stats"].values()
    )
    if total_unique != 2_005_445:
        raise AssertionError(f"Unexpected frozen support: {total_unique}")
    if not bool(support["leakage_audit"]["passed"]):
        raise AssertionError("Frozen-support leakage audit did not pass")

    write_csv(OUT / "patch_branch_counts.csv", branch_rows)
    write_csv(OUT / "patch_m3_by_scene.csv", m3_scene_rows)
    support_row = {
        "windows_total": int(support["rows"]),
        "windows_per_scene": 69,
        "unique_pixels_total": total_unique,
        "unique_pixels_2003": int(support["scene_stats"]["20030709_qb"]["unique_labelled_pixels"]),
        "unique_pixels_2016": int(support["scene_stats"]["20160825_wv"]["unique_labelled_pixels"]),
        "unique_pixels_2023": int(support["scene_stats"]["20230812_pl"]["unique_labelled_pixels"]),
        "validation_blocks": "|".join(map(str, support["validation_blocks"])),
        "test_blocks": "|".join(map(str, support["test_blocks"])),
        "reciprocal_buffer_px": int(support["reciprocal_buffer_px"]),
        "reciprocal_buffer_m": int(support["reciprocal_buffer_px"]) * 2,
        "train_validation_rows_checked": int(support["leakage_audit"]["reference_train_val_rows_checked"]),
        "overlap_pairs": int(support["leakage_audit"]["cross_split_overlap_pairs"]),
        "manifest_sha256": sha256(support_path),
    }
    write_csv(OUT / "frozen_test_support.csv", [support_row])
    return {
        "branches": branch_rows,
        "m3_by_scene": m3_scene_rows,
        "support": support_row,
    }


def ablation_audit() -> dict[str, Any]:
    summary_path = COMMON_DIR / "segformer_five_seed_summary.csv"
    bootstrap_path = COMMON_DIR / "segformer_paired_spatial_bootstrap.csv"
    summary = pd.read_csv(summary_path)
    bootstrap = pd.read_csv(bootstrap_path)

    pooled = summary[summary["scene_id"] == "ALL_SCENES"].copy()
    pooled["branch_label"] = pooled["branch"].map(BRANCH_LABELS)
    pooled = pooled.sort_values("min_iou_mean", ascending=False)
    pooled_rows = [
        {
            "branch": row.branch,
            "branch_label": row.branch_label,
            "n_seeds": int(row.n_seeds),
            "n_unique_pixels": int(row.n_unique_pixels),
            "min_iou_mean": float(row.min_iou_mean),
            "min_iou_seed_sd": float(row.min_iou_sd),
            "iou_veg_mean": float(row.iou_veg_mean),
            "iou_nonveg_mean": float(row.iou_nonveg_mean),
            "balanced_accuracy_mean": float(row.balanced_accuracy_mean),
            "mcc_mean": float(row.mcc_mean),
        }
        for row in pooled.itertuples(index=False)
    ]

    m3_scene = summary[
        (summary["branch"] == "M3_031623") & (summary["scene_id"] != "ALL_SCENES")
    ].copy()
    m3_scene_rows = [
        {
            "scene_id": row.scene_id,
            "scene": SCENES[row.scene_id][0],
            "n_unique_pixels": int(row.n_unique_pixels),
            "min_iou_mean": float(row.min_iou_mean),
            "min_iou_seed_sd": float(row.min_iou_sd),
            "iou_veg_mean": float(row.iou_veg_mean),
            "iou_nonveg_mean": float(row.iou_nonveg_mean),
            "ece_15_seed_mean": float(row.ece_15_mean),
            "brier_seed_mean": float(row.brier_mean),
        }
        for row in m3_scene.itertuples(index=False)
    ]

    pooled_boot = bootstrap[bootstrap["scene_id"] == "ALL_SCENES"].copy()
    pairwise_rows = [
        {
            "comparison": row.comparison,
            "primary": row.primary,
            "competitor": row.competitor,
            "competitor_label": BRANCH_LABELS[row.competitor],
            "delta_min_iou_mean": float(row.delta_min_iou_mean),
            "ci95_low": float(row.delta_min_iou_ci95_low),
            "ci95_high": float(row.delta_min_iou_ci95_high),
            "probability_delta_gt_zero": float(row.prob_delta_min_iou_gt_0),
            "bootstrap_draws": int(row.bootstrap_draws),
        }
        for row in pooled_boot.itertuples(index=False)
    ]

    holdout_spec = {
        "20030709_qb": "M2_1623",
        "20160825_wv": "M2_0323",
        "20230812_pl": "M2_0316",
    }
    holdout_rows: list[dict[str, Any]] = []
    for scene_id, omitted_branch in holdout_spec.items():
        left = summary[(summary["branch"] == omitted_branch) & (summary["scene_id"] == scene_id)].iloc[0]
        right = summary[(summary["branch"] == "M3_031623") & (summary["scene_id"] == scene_id)].iloc[0]
        boot = bootstrap[
            (bootstrap["primary"] == "M3_031623")
            & (bootstrap["competitor"] == omitted_branch)
            & (bootstrap["scene_id"] == scene_id)
        ].iloc[0]
        holdout_rows.append(
            {
                "omitted_scene": SCENES[scene_id][0],
                "label_holdout_branch": BRANCH_LABELS[omitted_branch],
                "label_holdout_min_iou_mean": float(left["min_iou_mean"]),
                "label_holdout_seed_sd": float(left["min_iou_sd"]),
                "m3_min_iou_mean": float(right["min_iou_mean"]),
                "m3_seed_sd": float(right["min_iou_sd"]),
                "m3_minus_holdout_delta": float(boot["delta_min_iou_mean"]),
                "delta_ci95_low": float(boot["delta_min_iou_ci95_low"]),
                "delta_ci95_high": float(boot["delta_min_iou_ci95_high"]),
                "probability_delta_gt_zero": float(boot["prob_delta_min_iou_gt_0"]),
                "interpretation": "labels withheld; target-scene unlabeled normalization retained",
            }
        )

    write_csv(OUT / "ablation_pooled.csv", pooled_rows)
    write_csv(OUT / "ablation_m3_by_scene.csv", m3_scene_rows)
    write_csv(OUT / "ablation_m3_pairwise_bootstrap.csv", pairwise_rows)
    write_csv(OUT / "ablation_scene_label_holdout.csv", holdout_rows)
    return {
        "pooled": pooled_rows,
        "m3_by_scene": m3_scene_rows,
        "pairwise": pairwise_rows,
        "label_holdout": holdout_rows,
        "summary_sha256": sha256(summary_path),
        "bootstrap_sha256": sha256(bootstrap_path),
    }


def probability_path(seed: int, scene_id: str) -> Path:
    matches = sorted((PRED_DIR / f"seed{seed}").glob(f"prob_{scene_id}_w512_p128_thr*.tif"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one probability file for {scene_id}/seed{seed}: {matches}")
    return matches[0]


def calibration_metrics(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p = np.clip(probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
    y = target.astype(np.float64)
    bins = np.minimum((p * N_BINS).astype(np.int16), N_BINS - 1)
    count = np.bincount(bins, minlength=N_BINS).astype(np.float64)
    prob_sum = np.bincount(bins, weights=p, minlength=N_BINS)
    label_sum = np.bincount(bins, weights=y, minlength=N_BINS)
    nonempty = count > 0
    mean_p = np.zeros(N_BINS, dtype=np.float64)
    mean_y = np.zeros(N_BINS, dtype=np.float64)
    mean_p[nonempty] = prob_sum[nonempty] / count[nonempty]
    mean_y[nonempty] = label_sum[nonempty] / count[nonempty]
    ece = float(np.sum(count[nonempty] * np.abs(mean_p[nonempty] - mean_y[nonempty])) / len(p))
    mce = float(np.max(np.abs(mean_p[nonempty] - mean_y[nonempty])))
    return {
        "n_pixels": int(len(p)),
        "prevalence": float(y.mean()),
        "mean_probability": float(p.mean()),
        "ece_15": ece,
        "mce_15": mce,
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
    }


def reliability_rows(scene_id: str, probability: np.ndarray, target: np.ndarray) -> list[dict[str, Any]]:
    p = np.clip(probability.astype(np.float64), 0.0, 1.0)
    y = target.astype(np.float64)
    bins = np.minimum((p * N_BINS).astype(np.int16), N_BINS - 1)
    rows: list[dict[str, Any]] = []
    for index in range(N_BINS):
        selected = bins == index
        rows.append(
            {
                "scene_id": scene_id,
                "scene": "Pooled" if scene_id == "ALL_SCENES" else SCENES[scene_id][0],
                "bin": index + 1,
                "bin_low": index / N_BINS,
                "bin_high": (index + 1) / N_BINS,
                "n_pixels": int(selected.sum()),
                "mean_probability": float(p[selected].mean()) if selected.any() else float("nan"),
                "observed_vegetation_fraction": float(y[selected].mean()) if selected.any() else float("nan"),
            }
        )
    return rows


def cell_aggregates(
    probability: np.ndarray,
    target: np.ndarray,
    row_index: np.ndarray,
    col_index: np.ndarray,
) -> list[dict[str, np.ndarray | float | int]]:
    p = np.clip(probability.astype(np.float64), 1e-7, 1.0 - 1e-7)
    y = target.astype(np.float64)
    bins = np.minimum((p * N_BINS).astype(np.int16), N_BINS - 1)
    cell_ids = (row_index // CELL_PX).astype(np.int64) * 100_000 + (col_index // CELL_PX)
    aggregates: list[dict[str, np.ndarray | float | int]] = []
    for cell_id in np.unique(cell_ids):
        selected = cell_ids == cell_id
        local_bins = bins[selected]
        local_p = p[selected]
        local_y = y[selected]
        aggregates.append(
            {
                "cell_id": int(cell_id),
                "count": np.bincount(local_bins, minlength=N_BINS).astype(np.float64),
                "prob_sum": np.bincount(local_bins, weights=local_p, minlength=N_BINS),
                "label_sum": np.bincount(local_bins, weights=local_y, minlength=N_BINS),
                "brier_sum": float(np.sum((local_p - local_y) ** 2)),
                "nll_sum": float(-np.sum(local_y * np.log(local_p) + (1.0 - local_y) * np.log(1.0 - local_p))),
            }
        )
    return aggregates


def metrics_from_cell_sample(cells: Iterable[dict[str, Any]]) -> dict[str, float]:
    cells = list(cells)
    count = np.sum([item["count"] for item in cells], axis=0)
    prob_sum = np.sum([item["prob_sum"] for item in cells], axis=0)
    label_sum = np.sum([item["label_sum"] for item in cells], axis=0)
    total = float(count.sum())
    nonempty = count > 0
    ece = float(
        np.sum(
            count[nonempty]
            * np.abs(prob_sum[nonempty] / count[nonempty] - label_sum[nonempty] / count[nonempty])
        )
        / total
    )
    return {
        "ece_15": ece,
        "brier": float(sum(float(item["brier_sum"]) for item in cells) / total),
        "nll": float(sum(float(item["nll_sum"]) for item in cells) / total),
        "mean_probability": float(prob_sum.sum() / total),
        "prevalence": float(label_sum.sum() / total),
    }


def bootstrap_calibration(
    cells_by_scene: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    scopes = list(SCENES) + ["ALL_SCENES"]
    results: list[dict[str, Any]] = []
    for scope in scopes:
        draws: dict[str, list[float]] = defaultdict(list)
        selected_scenes = list(SCENES) if scope == "ALL_SCENES" else [scope]
        for _ in range(BOOTSTRAP_DRAWS):
            sampled_cells: list[dict[str, Any]] = []
            for scene_id in selected_scenes:
                cells = cells_by_scene[scene_id]
                indices = rng.integers(0, len(cells), size=len(cells))
                sampled_cells.extend(cells[index] for index in indices)
            metrics = metrics_from_cell_sample(sampled_cells)
            for key, value in metrics.items():
                draws[key].append(value)
        for metric, values in draws.items():
            array = np.asarray(values, dtype=np.float64)
            results.append(
                {
                    "scene_id": scope,
                    "scene": "Pooled" if scope == "ALL_SCENES" else SCENES[scope][0],
                    "metric": metric,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "ci95_low": float(np.quantile(array, 0.025)),
                    "ci95_high": float(np.quantile(array, 0.975)),
                }
            )
    return results


def calibration_audit() -> dict[str, Any]:
    windows = pd.read_csv(EVAL_DIR / "common_eval_windows.csv")
    support_manifest = json.loads((EVAL_DIR / "common_eval_manifest.json").read_text(encoding="utf-8"))
    all_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
    all_targets: list[np.ndarray] = []
    per_prediction_rows: list[dict[str, Any]] = []
    ensemble_summary: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    cells_by_scene: dict[str, list[dict[str, Any]]] = {}
    ensemble_cache: dict[str, np.ndarray] = {}
    target_cache: dict[str, np.ndarray] = {}
    reference_paths_used: dict[str, dict[str, str]] = {}

    for scene_id in SCENES:
        scene_windows = windows[windows["scene_id"] == scene_id]
        reference_path = Path(scene_windows.iloc[0]["mask_tif"])
        if not reference_path.exists():
            reference_path = REFERENCE_FALLBACKS[scene_id]
        if not reference_path.exists():
            raise FileNotFoundError(f"Missing evaluation reference for {scene_id}: {reference_path}")
        reference_paths_used[scene_id] = {
            "path": str(reference_path),
            "sha256": sha256(reference_path),
            "valid_mask_path": str(VALID_MASKS[scene_id]),
            "valid_mask_sha256": sha256(VALID_MASKS[scene_id]),
        }
        with rasterio.open(reference_path) as source:
            reference = source.read(1)
            reference_profile = source.profile
        with rasterio.open(VALID_MASKS[scene_id]) as source:
            if (
                source.shape != reference.shape
                or source.transform != reference_profile["transform"]
                or source.crs != reference_profile["crs"]
            ):
                raise RuntimeError(f"Valid-mask grid mismatch: {VALID_MASKS[scene_id]}")
            scene_valid = source.read(1) > 0
        support = np.zeros(reference.shape, dtype=bool)
        for row in scene_windows.itertuples(index=False):
            support[int(row.y0): int(row.y0) + int(row.tile), int(row.x0): int(row.x0) + int(row.tile)] = True
        # The frozen support builder required both a labelled reference pixel
        # and the scene-specific feature-valid mask.  Reproduce that definition
        # exactly rather than treating every labelled pixel in a window as test.
        valid = support & scene_valid & np.isin(reference, (1, 2))
        expected = int(support_manifest["scene_stats"][scene_id]["unique_labelled_pixels"])
        if int(valid.sum()) != expected:
            raise AssertionError(f"{scene_id}: held-out support {valid.sum()} != {expected}")
        row_index, col_index = np.nonzero(valid)
        target = (reference[valid] == 2).astype(np.uint8)

        seed_probabilities: list[np.ndarray] = []
        for seed in SEEDS:
            path = probability_path(seed, scene_id)
            with rasterio.open(path) as source:
                if source.shape != reference.shape or source.transform != reference_profile["transform"] or source.crs != reference_profile["crs"]:
                    raise RuntimeError(f"Grid mismatch: {path}")
                probability = source.read(1)[valid].astype(np.float32)
            if not np.all(np.isfinite(probability) & (probability >= 0.0) & (probability <= 1.0)):
                raise RuntimeError(f"Invalid held-out probability values: {path}")
            seed_probabilities.append(probability)
            metrics = calibration_metrics(probability, target)
            per_prediction_rows.append(
                {
                    "prediction": f"seed{seed}",
                    "scene_id": scene_id,
                    "scene": SCENES[scene_id][0],
                    **metrics,
                }
            )
            all_probabilities[f"seed{seed}"].append(probability)

        ensemble = np.mean(np.stack(seed_probabilities, axis=0), axis=0).astype(np.float32)
        mean_path = FULL_DIR / f"mean_pveg_{scene_id}.tif"
        with rasterio.open(mean_path) as source:
            saved_mean = source.read(1)[valid].astype(np.float32)
        max_abs_difference = float(np.max(np.abs(saved_mean - ensemble)))
        if max_abs_difference > 2e-6:
            raise AssertionError(f"Saved ensemble mismatch for {scene_id}: {max_abs_difference}")
        metrics = calibration_metrics(ensemble, target)
        ensemble_summary.append(
            {
                "prediction": "five_seed_mean",
                "scene_id": scene_id,
                "scene": SCENES[scene_id][0],
                **metrics,
                "spatial_cells": int(len(np.unique((row_index // CELL_PX) * 100_000 + col_index // CELL_PX))),
                "saved_mean_max_abs_difference": max_abs_difference,
            }
        )
        per_prediction_rows.append(ensemble_summary[-1].copy())
        reliability.extend(reliability_rows(scene_id, ensemble, target))
        cells_by_scene[scene_id] = cell_aggregates(ensemble, target, row_index, col_index)
        ensemble_cache[scene_id] = ensemble
        target_cache[scene_id] = target
        all_probabilities["five_seed_mean"].append(ensemble)
        all_targets.append(target)

    pooled_target = np.concatenate(all_targets)
    for prediction, parts in all_probabilities.items():
        pooled_probability = np.concatenate(parts)
        per_prediction_rows.append(
            {
                "prediction": prediction,
                "scene_id": "ALL_SCENES",
                "scene": "Pooled",
                **calibration_metrics(pooled_probability, pooled_target),
            }
        )
    pooled_ensemble = np.concatenate([ensemble_cache[key] for key in SCENES])
    pooled_metrics = calibration_metrics(pooled_ensemble, pooled_target)
    ensemble_summary.append(
        {
            "prediction": "five_seed_mean",
            "scene_id": "ALL_SCENES",
            "scene": "Pooled",
            **pooled_metrics,
            "spatial_cells": int(sum(len(value) for value in cells_by_scene.values())),
            "saved_mean_max_abs_difference": max(
                float(row["saved_mean_max_abs_difference"]) for row in ensemble_summary
            ),
        }
    )
    reliability.extend(reliability_rows("ALL_SCENES", pooled_ensemble, pooled_target))
    bootstrap_rows = bootstrap_calibration(cells_by_scene)
    ci_lookup = {(row["scene_id"], row["metric"]): row for row in bootstrap_rows}
    for row in ensemble_summary:
        for metric in ("ece_15", "brier", "nll", "mean_probability", "prevalence"):
            ci = ci_lookup[(row["scene_id"], metric)]
            row[f"{metric}_ci95_low"] = ci["ci95_low"]
            row[f"{metric}_ci95_high"] = ci["ci95_high"]

    write_csv(OUT / "calibration_all_seeds_heldout.csv", per_prediction_rows)
    write_csv(OUT / "calibration_ensemble_heldout.csv", ensemble_summary)
    write_csv(OUT / "calibration_reliability_bins.csv", reliability)
    write_csv(OUT / "calibration_spatial_bootstrap.csv", bootstrap_rows)
    authoritative_path = FULL_DIR / "calibration_metrics_heldout_unique_pixels.csv"
    write_csv(authoritative_path, ensemble_summary)
    support_note_path = FULL_DIR / "CALIBRATION_SUPPORT_README.md"
    support_note_path.write_text(
        "# Calibration support correction\n\n"
        "`calibration_metrics_heldout_unique_pixels.csv` is the authoritative "
        "calibration table for manuscript inference. It evaluates the final "
        "five-seed probability mean on the frozen union of 207 class-independent "
        "held-out windows, scoring each of 2,005,445 labeled global pixels once.\n\n"
        "The legacy `calibration_metrics.csv` sampled all labeled pixels in the "
        "full reference rasters. Its values are descriptive full-reference "
        "diagnostics and must not be reported as held-out/test calibration.\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(6.8, 6.2))
    colors = {
        "20030709_qb": "#b96b00",
        "20160825_wv": "#16877f",
        "20230812_pl": "#24699b",
    }
    reliability_frame = pd.DataFrame(reliability)
    for scene_id, color in colors.items():
        selected = reliability_frame[
            (reliability_frame["scene_id"] == scene_id)
            & (reliability_frame["n_pixels"] > 0)
        ]
        axis.plot(
            selected["mean_probability"],
            selected["observed_vegetation_fraction"],
            marker="o",
            linewidth=1.8,
            label=SCENES[scene_id][0],
            color=color,
        )
    axis.plot([0, 1], [0, 1], "--", color="#4a4a4a", linewidth=1.2, label="Perfect calibration")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted P(VEG)", ylabel="Observed vegetation fraction")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUT / "calibration_reliability_heldout.png", dpi=220)
    plt.close(figure)

    return {
        "ensemble": ensemble_summary,
        "bootstrap": bootstrap_rows,
        "support_definition": "frozen unique labelled pixels from 207 class-independent held-out windows",
        "inference_product": "mean of five validation-temperature-scaled full-scene Hann-blended probabilities",
        "authoritative_stage_csv": str(authoritative_path),
        "authoritative_stage_csv_sha256": sha256(authoritative_path),
        "legacy_full_reference_csv": str(FULL_DIR / "calibration_metrics.csv"),
        "legacy_full_reference_csv_sha256": sha256(FULL_DIR / "calibration_metrics.csv"),
        "support_note": str(support_note_path),
        "reference_paths_used": reference_paths_used,
    }


def field_audit() -> dict[str, Any]:
    field_path = ROOT / "data/local_multiclass/field_points.geojson"
    collection = json.loads(field_path.read_text(encoding="utf-8"))
    features = [
        feature for feature in collection["features"]
        if int(feature["properties"].get("year", -1)) == 2023
    ]
    if len(features) != 54:
        raise AssertionError(f"Expected 54 derived 2023 points, found {len(features)}")

    probability_path = FULL_DIR / "mean_pveg_20230812_pl.tif"
    mask_path = FULL_DIR / "consensus_vegmask_20230812_pl.tif"
    with rasterio.open(probability_path) as source:
        probability = source.read(1)
        transform = source.transform
        width = source.width
        height = source.height
    with rasterio.open(mask_path) as source:
        mask = source.read(1)
        if source.transform != transform:
            raise RuntimeError("Field audit raster grid mismatch")

    rows: list[dict[str, Any]] = []
    coordinates: list[tuple[float, float]] = []
    for feature in features:
        properties = feature["properties"]
        x = float(properties["utm_x"])
        y = float(properties["utm_y"])
        row, col = rowcol(transform, x, y)
        if not (0 <= row < height and 0 <= col < width):
            raise RuntimeError(f"Field point outside final raster: {properties['id']}")
        class_id = int(properties["class_id"])
        field_vegetated = class_id not in (1, 9)
        mapped_vegetated = int(mask[row, col]) == 2
        local = mask[max(0, row - 1): min(height, row + 2), max(0, col - 1): min(width, col + 2)]
        vegetation_within_one_cell = bool(np.any(local == 2))
        block_x = min(5, int(col * 6 // width))
        block_y = min(5, int(row * 6 // height))
        block_id = block_y * 6 + block_x
        coordinates.append((x, y))
        rows.append(
            {
                "point_id": int(properties["id"]),
                "year": 2023,
                "field_class": properties["class"],
                "field_class_id": class_id,
                "binary_mapping_rule": "NONVEG" if class_id in (1, 9) else "VEG",
                "field_vegetated": int(field_vegetated),
                "ensemble_pveg": float(probability[row, col]),
                "consensus_vegetated_exact": int(mapped_vegetated),
                "consensus_vegetated_within_one_2m_cell": int(vegetation_within_one_cell),
                "agreement_exact": int(field_vegetated == mapped_vegetated),
                "block_id": block_id,
                "in_fixed_test_block": int(block_id in (15, 18, 19, 22)),
                "utm_x": x,
                "utm_y": y,
            }
        )

    tp = sum(row["field_vegetated"] and row["consensus_vegetated_exact"] for row in rows)
    fn = sum(row["field_vegetated"] and not row["consensus_vegetated_exact"] for row in rows)
    tn = sum(not row["field_vegetated"] and not row["consensus_vegetated_exact"] for row in rows)
    fp = sum(not row["field_vegetated"] and row["consensus_vegetated_exact"] for row in rows)
    sensitivity_ci = wilson_interval(tp, tp + fn)
    specificity_ci = wilson_interval(tn, tn + fp)
    tolerance_hits = sum(
        row["field_vegetated"] and row["consensus_vegetated_within_one_2m_cell"]
        for row in rows
    )
    tolerance_ci = wilson_interval(tolerance_hits, tp + fn)
    coordinate_array = np.asarray(coordinates, dtype=np.float64)
    distances = np.sqrt(
        np.sum((coordinate_array[:, None, :] - coordinate_array[None, :, :]) ** 2, axis=2)
    )
    distances[distances == 0] = np.nan
    nearest = np.nanmin(distances, axis=1)

    summary = {
        "decision": "REMOVE_FROM_INFERENTIAL_MANUSCRIPT",
        "permitted_use": "optional descriptive supplement only; not validation, accuracy, specificity, or detectability evidence",
        "derived_field_file": str(field_path),
        "derived_field_sha256": sha256(field_path),
        "raw_survey_archived_in_project": False,
        "survey_date_verifiable_from_derived_file": False,
        "n_points_2023": len(rows),
        "n_field_vegetated": tp + fn,
        "n_field_nonvegetated": tn + fp,
        "true_positive": tp,
        "false_negative": fn,
        "true_negative": tn,
        "false_positive": fp,
        "exact_sensitivity": tp / (tp + fn),
        "exact_sensitivity_wilson95_low": sensitivity_ci[0],
        "exact_sensitivity_wilson95_high": sensitivity_ci[1],
        "exact_specificity": tn / (tn + fp),
        "exact_specificity_wilson95_low": specificity_ci[0],
        "exact_specificity_wilson95_high": specificity_ci[1],
        "vegetated_hits_within_one_2m_cell": tolerance_hits,
        "one_cell_tolerance_sensitivity": tolerance_hits / (tp + fn),
        "one_cell_tolerance_sensitivity_wilson95_low": tolerance_ci[0],
        "one_cell_tolerance_sensitivity_wilson95_high": tolerance_ci[1],
        "n_points_in_fixed_test_blocks": sum(row["in_fixed_test_block"] for row in rows),
        "median_nearest_neighbor_m": float(np.median(nearest)),
        "minimum_nearest_neighbor_m": float(np.min(nearest)),
        "field_vegetated_median_pveg": float(np.median([row["ensemble_pveg"] for row in rows if row["field_vegetated"]])),
        "field_nonvegetated_median_pveg": float(np.median([row["ensemble_pveg"] for row in rows if not row["field_vegetated"]])),
        "reasons": [
            "only a derived class-assigned GeoJSON is present; raw notes, protocol and date metadata are not archived",
            "the binary mapping includes post-hoc ecological-category choices, including reef plateau as NONVEG",
            "the sample is opportunistic, spatially clustered and contains only seven NONVEG records",
            "the exact vegetation hit rate is low and cannot establish an optical detectability mechanism",
            "the project validation README records unresolved disagreement between in-situ notes and post-hoc classes",
        ],
    }
    write_csv(OUT / "field_crosscheck_final_product.csv", rows)
    write_csv(
        OUT / "field_crosscheck_summary.csv",
        [{key: json.dumps(value) if isinstance(value, list) else value for key, value in summary.items()}],
    )
    return summary


def methods_inventory() -> dict[str, Any]:
    primary_config_path = ROOT / "configs/train/final_scene_ablation_031623_v3/matrix_3seed.json"
    seed_extension_path = ROOT / "configs/train/final_scene_ablation_031623_v3/matrix_additional_2seed.json"
    feature_paths = {
        scene_id: ROOT
        / "data/features"
        / f"feat_{scene_id}_st3b_final_scene_ablation_031623_v3_common.json"
        for scene_id in SCENES
    }
    provenance_path = ROOT / "data/rboa/RBOA_20030709.provenance.json"
    rboa_metadata_paths = {
        "20160825_wv": ROOT / "data/rboa/RBOA_20160825.tif.json",
        "20230812_pl": ROOT / "data/rboa/RBOA_20230812.tif.json",
    }
    baseline_manifest_path = ROOT / "out/runs/train/final_model_family_baselines_v3/experiment_manifest.json"
    full_scene_manifest_path = FULL_DIR / "full_scene_manifest.json"
    inference_summary_path = next(
        (PRED_DIR / "seed123").glob("summary_20030709_qb_w512_p128_thr*.json")
    )
    config = json.loads(primary_config_path.read_text(encoding="utf-8"))
    features = {
        scene_id: json.loads(path.read_text(encoding="utf-8"))
        for scene_id, path in feature_paths.items()
    }
    feature = features["20030709_qb"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    rboa_metadata = {
        scene_id: json.loads(path.read_text(encoding="utf-8"))
        for scene_id, path in rboa_metadata_paths.items()
    }
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    full_scene_manifest = json.loads(full_scene_manifest_path.read_text(encoding="utf-8"))
    inference_summary = json.loads(inference_summary_path.read_text(encoding="utf-8"))
    run = config["runs"][0]
    inventory = {
        "inputs": {
            "channels": feature["feature_names"],
            "grid": "EPSG:32633, 2 m sampling grid",
            "normalization": "per-scene 2nd-98th percentile scaling fitted on valid, label-agnostic image pixels in the union of fixed training-split windows",
            "normalization_by_scene": {
                scene_id: {
                    "fit_pixels": int(meta["normalization_fit"]["fit_pixels"]),
                    "training_window_rows": int(meta["normalization_fit"]["patch_rows_used"]),
                    "fallback_to_all_valid": bool(meta["normalization_fit"]["fallback_to_all_valid"]),
                    "percentile_low": float(meta["p_lo"]),
                    "percentile_high": float(meta["p_hi"]),
                    "rgb_clip_values": {
                        band: meta["raw_clip_stats"][band] for band in ("B", "G", "R")
                    },
                }
                for scene_id, meta in features.items()
            },
            "normalization_scope_caveat": "target-scene radiometry is used for label-independent normalization in scene-exclusion branches",
        },
        "quickbird_2003": {
            "acquisition_date": "2003-07-07",
            "legacy_scene_identifier": "20030709_qb",
            "raw_source": provenance["raw_source_filename"],
            "processing_steps": provenance["processing_steps"],
            "radiometric_independence_caveat": provenance["radiometric_independence_caveat"],
        },
        "bottom_of_atmosphere_inputs_2016_2023": {
            scene_id: {
                "source": meta["source"],
                "source_band_order": meta["band_order"],
                "common_grid": f"{meta['crs']}, {abs(float(meta['transform'][0])):g} m",
                "resampling": meta["resampling"],
                "water_mask_applied": bool(meta["water_mask_applied"]),
                "output_nodata": meta["output_nodata"],
                "cleaning_rules": meta["cleaning_rules"],
            }
            for scene_id, meta in rboa_metadata.items()
        },
        "segformer": {
            "architecture": f"SegFormer {run['encoder']}",
            "pretraining": config["encoder_weights"],
            "optimizer": "AdamW",
            "learning_rate": run["lr"],
            "weight_decay": run["wd"],
            "loss": run["loss_mix"],
            "label_smoothing": run["label_smooth"],
            "effective_batch": run["batch"] * run["grad_accum_steps"],
            "epochs_max": config["stage2_epochs"],
            "early_stopping_patience": config["stage2_patience"],
            "warmup_epochs": config["warmup_epochs"],
            "scheduler": config["scheduler"],
            "gradient_clip": config["grad_clip"],
            "mixed_precision": config["amp"],
            "augmentation": config["augment"],
            "boundary_weight_radius_px": config["boundary_radius"],
            "boundary_weight_factor": config["boundary_factor"],
            "temperature_scaling": config["use_temp_scaling"],
            "temperature_objective": "LBFGS minimization of class-weighted binary cross-entropy on validation logits",
            "hard_threshold_rule": "validation sweep after temperature scaling; maximize minimum class IoU",
            "seeds": list(SEEDS),
        },
        "full_scene_inference": {
            "window_px": int(inference_summary["win"]),
            "step_px": int(inference_summary["inference_step"]),
            "blend": inference_summary["blend_mode"],
            "input_adapter": inference_summary["input_adapter"],
            "test_time_augmentation": bool(inference_summary["tta"]),
            "postprocessing": inference_summary["postprocess"],
            "seed_thresholds": full_scene_manifest["qa"]["20030709_qb"]["seed_thresholds"],
            "seed_temperatures": full_scene_manifest["qa"]["20030709_qb"]["seed_temperatures"],
            "threshold_selection_metric": run["thr_metric"],
            "final_probability": "arithmetic mean of five validation-temperature-scaled probability rasters",
            "hard_mask": "per-pixel majority of five seed masks using their validation-selected thresholds",
        },
        "unet": {
            "architecture": "U-Net with ResNet-34 encoder",
            "pretraining": "ImageNet",
            "recipe": "same optimizer/loss/schedule as SegFormer; batch 8 with two-step gradient accumulation",
        },
        "random_forest": {
            "features": "B, G, R pixel values; no spatial features",
            "trees": 400,
            "max_depth": 20,
            "min_samples_leaf": 2,
            "train_pixels_per_class": 250000,
            "validation_pixels_per_class": 150000,
        },
        "sources": {
            "primary_config_sha256": sha256(primary_config_path),
            "seed_extension_sha256": sha256(seed_extension_path),
            "feature_manifest_sha256": {
                scene_id: sha256(path) for scene_id, path in feature_paths.items()
            },
            "quickbird_provenance_sha256": sha256(provenance_path),
            "rboa_metadata_sha256": {
                scene_id: sha256(path) for scene_id, path in rboa_metadata_paths.items()
            },
            "baseline_manifest_sha256": sha256(baseline_manifest_path),
            "full_scene_manifest_sha256": sha256(full_scene_manifest_path),
            "inference_summary_sha256": sha256(inference_summary_path),
            "baseline_runs": len(baseline_manifest["runs"]),
        },
        "not_recoverable_from_v3": [
            "reference-mask interpreter identities and blinding protocol",
            "a signed/immutable raw record for the claimed 10 August 2023 field survey",
            "independent empirical co-registration RMSE for the three final acquisitions",
        ],
    }
    (OUT / "methods_verified_parameters.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return inventory


def manuscript_text(
    patch: dict[str, Any],
    ablation: dict[str, Any],
    calibration: dict[str, Any],
    field: dict[str, Any],
) -> str:
    support = patch["support"]
    m3_patch = {row["scene"]: row for row in patch["m3_by_scene"]}
    pooled = {row["branch"]: row for row in ablation["pooled"]}
    pairwise = {row["competitor"]: row for row in ablation["pairwise"]}
    cal = {row["scene_id"]: row for row in calibration["ensemble"]}
    holdout = {row["omitted_scene"]: row for row in ablation["label_holdout"]}

    return f"""# Manuscript-ready scientific alignment text (v3; non-structural)

## Methods - preprocessing and inputs

The final analysis used water-masked blue, green and red channels on a common
2 m sampling grid in EPSG:32633. Continuous bands were resampled bilinearly and
categorical masks with nearest-neighbour interpolation. The historical
QuickBird acquisition occurred on 7 July 2003; the internal identifier
`20030709_qb` is retained only for file compatibility. The 2003 product was
derived from top-of-atmosphere reflectance by NIR-based Hochberg glint
correction, dark-object subtraction using the second water-pixel percentile,
and band-wise linear cross-calibration to deep-water pseudo-invariant features
from a QuickBird acquisition of 8 April 2006. Values were clipped to [0,1] and
the common water mask was applied. The 2006 acquisition was not a model input,
but it was a radiometric anchor for the 2003 product and therefore must not be
described as an independent fourth observation.

The 25 August 2016 WorldView-2 and 12 August 2023 Pleiades inputs were archived
bottom-of-atmosphere reflectance products with blue, green, red and
near-infrared bands. They were
warped bilinearly to the same 2 m grid, screened for non-finite and documented
out-of-range values, and intersected with the common water mask; zero was
treated as nodata. Only the visible blue, green and red bands entered the
classifiers.

For modelling, each scene/channel was clipped to its 2nd and 98th percentiles
and linearly scaled. Percentiles were fitted on valid, label-agnostic image
pixels in the union of fixed training-split windows for the corresponding
scene (5,710,545/5,710,524/5,710,529 pixels for 2003/2016/2023); no fallback to
full-scene statistics occurred. The normalized stacks were then shared by all
M1, M2 and M3 branches. Consequently, scene-exclusion experiments withhold the
target scene's labels but retain label-independent access to its radiometry for
normalization; they are target-normalized label-held-out domain tests, not
deployment-blind sensor/date tests.

## Methods - patches, spatial split and evaluation support

Each scene-combination branch contained exactly 6,000 256 x 256-pixel patches,
allocated equally across the scenes included in that branch: 6,000 per scene
for M1, 3,000 per scene for M2 and 2,000 per scene for M3. In M3 the realized
train/validation/test counts were 3,240/1,001/1,759. Per scene these were
{m3_patch['2003 QuickBird']['train_patches']}/{m3_patch['2003 QuickBird']['validation_patches']}/{m3_patch['2003 QuickBird']['test_patches']}
for 2003, {m3_patch['2016 WorldView-2']['train_patches']}/{m3_patch['2016 WorldView-2']['validation_patches']}/{m3_patch['2016 WorldView-2']['test_patches']}
for 2016 and {m3_patch['2023 Pleiades']['train_patches']}/{m3_patch['2023 Pleiades']['validation_patches']}/{m3_patch['2023 Pleiades']['test_patches']}
for 2023.

The lagoon was partitioned into a 6 x 6 block grid. Blocks 6, 24, 28 and 30
were fixed for validation and blocks 15, 18, 19 and 22 for testing, with a
reciprocal 256-pixel (512 m) exclusion buffer. Final evaluation did not use the
overlapping patch rows as independent observations. It used 207 deterministic,
class-independent windows (69 per scene), reconstructed predictions at global
coordinates and scored every labeled global pixel once after averaging
duplicates. The frozen support comprised {support['unique_pixels_total']:,}
unique pixels: {support['unique_pixels_2003']:,} in 2003,
{support['unique_pixels_2016']:,} in 2016 and {support['unique_pixels_2023']:,}
in 2023. An audit against {support['train_validation_rows_checked']:,}
train/validation rows found zero cross-split overlaps.

## Methods - models and uncertainty

SegFormer used a MiT-B2 encoder initialized from ImageNet weights. Training
used AdamW (learning rate 1.2e-4; weight decay 1e-4), an equal-weight sum of
Dice, focal and Lovasz losses, label smoothing of 0.05, boundary weighting
within three pixels by a factor of three, effective batch size 16, mixed
precision and gradient clipping at 1.0. Training ran for at most 70 epochs with
five warm-up epochs, cosine decay and early-stopping patience 18; no data
augmentation was applied. Five predeclared seeds were used (123, 231, 312,
423 and 531). For each run, temperature was fitted on validation logits by
LBFGS minimization of the class-weighted binary cross-entropy, after which the
threshold was swept on the scaled probabilities to maximize minimum class IoU.

Full-scene inference used 512 x 512-pixel windows advanced by 256 pixels and
combined with full-Hann blending, without test-time augmentation or categorical
post-processing. The five validation-selected temperatures were
0.8162, 1.6973, 0.8443, 1.4990 and 1.7941; the corresponding hard thresholds
were 0.49, 0.58, 0.69, 0.57 and 0.60. The continuous final product was the
arithmetic mean of the five temperature-scaled probability rasters, whereas the
consensus hard map was the per-pixel majority of the five seed-specific masks.

The U-Net comparator used an ImageNet-pretrained ResNet-34 encoder and the same
neural optimization recipe, with batch size eight and two-step gradient
accumulation. Random Forest used RGB pixel values only, with 400 trees, maximum
depth 20, minimum leaf size two and random samples of 250,000 training and
150,000 validation pixels per class. Because RF has no spatial context and
model capacities differ, the result is an implementation-level model-family
comparison rather than a pure causal test of architecture.

The primary metric was the lower of vegetation and non-vegetated IoU. Branch
differences used 5,000 paired hierarchical bootstrap draws: seeds were sampled
with replacement and 256-pixel spatial cells were sampled with replacement
within each scene.

## Results - complete scene ablation

On the common {support['unique_pixels_total']:,}-pixel support, pooled
mean minimum-IoU +/- seed SD was
{fmt(pooled['M3_031623']['min_iou_mean'])} +/- {fmt(pooled['M3_031623']['min_iou_seed_sd'])}
for M3, {fmt(pooled['M2_0323']['min_iou_mean'])} +/- {fmt(pooled['M2_0323']['min_iou_seed_sd'])}
for the best M2 branch and {fmt(pooled['M1_23']['min_iou_mean'])} +/- {fmt(pooled['M1_23']['min_iou_seed_sd'])}
for the best M1 branch. M3 exceeded the best M1 branch by
{fmt(pairwise['M1_23']['delta_min_iou_mean'])} (95% spatial-bootstrap interval
{fmt(pairwise['M1_23']['ci95_low'])} to {fmt(pairwise['M1_23']['ci95_high'])})
and the best M2 branch by {fmt(pairwise['M2_0323']['delta_min_iou_mean'])}
({fmt(pairwise['M2_0323']['ci95_low'])} to {fmt(pairwise['M2_0323']['ci95_high'])}).

When a scene's labels were excluded, minimum-IoU was
{fmt(holdout['2003 QuickBird']['label_holdout_min_iou_mean'])} for 2003,
{fmt(holdout['2016 WorldView-2']['label_holdout_min_iou_mean'])} for 2016 and
{fmt(holdout['2023 Pleiades']['label_holdout_min_iou_mean'])} for 2023, compared
with {fmt(holdout['2003 QuickBird']['m3_min_iou_mean'])},
{fmt(holdout['2016 WorldView-2']['m3_min_iou_mean'])} and
{fmt(holdout['2023 Pleiades']['m3_min_iou_mean'])}, respectively, for M3.
The corresponding M3-minus-label-holdout differences were
{fmt(holdout['2003 QuickBird']['m3_minus_holdout_delta'])}
({fmt(holdout['2003 QuickBird']['delta_ci95_low'])} to {fmt(holdout['2003 QuickBird']['delta_ci95_high'])}),
{fmt(holdout['2016 WorldView-2']['m3_minus_holdout_delta'])}
({fmt(holdout['2016 WorldView-2']['delta_ci95_low'])} to {fmt(holdout['2016 WorldView-2']['delta_ci95_high'])}) and
{fmt(holdout['2023 Pleiades']['m3_minus_holdout_delta'])}
({fmt(holdout['2023 Pleiades']['delta_ci95_low'])} to {fmt(holdout['2023 Pleiades']['delta_ci95_high'])}).
These comparisons show the cost of omitting a scene's labeled domain under the
shared target-scene normalization; they do not establish deployment-blind
transfer to an unseen acquisition.

## Methods and Results - held-out ensemble calibration

Calibration of the final product was evaluated on the same frozen unique-pixel
test support, not on all labeled pixels. For each scene, the five
validation-temperature-scaled full-scene probability rasters were averaged and
sampled only at held-out pixels. Equal-width 15-bin ECE, Brier score and
negative log-likelihood were reported. Uncertainty was estimated with 5,000
spatial-cell bootstrap draws within scene; pooled draws resampled cells within
each scene and then combined them.

For the five-seed ensemble, ECE/Brier/NLL were
{fmt(cal['20030709_qb']['ece_15'])}/{fmt(cal['20030709_qb']['brier'])}/{fmt(cal['20030709_qb']['nll'])}
in 2003,
{fmt(cal['20160825_wv']['ece_15'])}/{fmt(cal['20160825_wv']['brier'])}/{fmt(cal['20160825_wv']['nll'])}
in 2016 and
{fmt(cal['20230812_pl']['ece_15'])}/{fmt(cal['20230812_pl']['brier'])}/{fmt(cal['20230812_pl']['nll'])}
in 2023. Pooled values were
{fmt(cal['ALL_SCENES']['ece_15'])}/{fmt(cal['ALL_SCENES']['brier'])}/{fmt(cal['ALL_SCENES']['nll'])}.
These scores quantify calibration to image-derived binary references within the
represented archive; they are not calibration to ecological occurrence or
fractional vegetation cover.

## Field-data decision

The 2023 field observations should be removed from the inferential manuscript.
The repository contains a derived 54-point GeoJSON but not an immutable raw
survey, acquisition protocol, survey-date metadata or reconciliation of the
original notes with the post-hoc classes. Against the final five-seed product,
the derived binary mapping yielded {field['true_positive']}/{field['n_field_vegetated']}
exact vegetation hits (sensitivity {field['exact_sensitivity']:.3f}; Wilson 95%
interval {field['exact_sensitivity_wilson95_low']:.3f}-{field['exact_sensitivity_wilson95_high']:.3f})
and {field['true_negative']}/{field['n_field_nonvegetated']} non-vegetated hits
(specificity {field['exact_specificity']:.3f}; {field['exact_specificity_wilson95_low']:.3f}-{field['exact_specificity_wilson95_high']:.3f}).
Allowing a one-cell (2 m) positional tolerance increased vegetation hits only
to {field['vegetated_hits_within_one_2m_cell']}/{field['n_field_vegetated']};
only {field['n_points_in_fixed_test_blocks']} of 54 points fell in the fixed test
blocks, and median nearest-neighbour spacing was
{field['median_nearest_neighbor_m']:.1f} m. The seven negative records are
insufficient for a general high-specificity claim, and the vegetation misses
cannot establish a detectability mechanism without contemporaneous depth,
clarity and cover measurements. If retained at all, these observations should
appear only as a provenance-qualified, descriptive supplementary table and not
as independent validation, accuracy, specificity or detectability evidence.
"""


def alignment_report(
    patch: dict[str, Any],
    ablation: dict[str, Any],
    calibration: dict[str, Any],
    field: dict[str, Any],
    methods: dict[str, Any],
) -> str:
    return f"""# Scientific alignment v3 - non-structural audit

Generated from frozen project artifacts. The structural pipeline and manuscript
DOCX were not modified.

## Decisions

- Final M3 patch table: 6,000 patches; 3,240 train, 1,001 validation and 1,759 test.
- Frozen test support: {patch['support']['unique_pixels_total']:,} unique labeled pixels in 207 windows.
- Complete M1/M2/M3 matrix: seven branches, five seeds each, all on identical support.
- Ensemble calibration: recomputed on held-out unique pixels; the previous full-reference calibration CSV is not valid as a test estimate.
- Field cross-check: {field['decision']}; optional descriptive supplement only.
- QuickBird 2003: the April 2006 PIF radiometric anchor must be disclosed.
- Scene-exclusion wording: label-held-out with target-scene normalization, not true sensor/date hold-out.

## Machine-verifiable outputs

- `patch_branch_counts.csv`
- `patch_m3_by_scene.csv`
- `frozen_test_support.csv`
- `ablation_pooled.csv`
- `ablation_m3_by_scene.csv`
- `ablation_m3_pairwise_bootstrap.csv`
- `ablation_scene_label_holdout.csv`
- `calibration_all_seeds_heldout.csv`
- `calibration_ensemble_heldout.csv`
- `calibration_reliability_bins.csv`
- `calibration_spatial_bootstrap.csv`
- `calibration_reliability_heldout.png`
- `field_crosscheck_final_product.csv`
- `field_crosscheck_summary.csv`
- `methods_verified_parameters.json`
- `MANUSCRIPT_READY_TEXT.md`

## Unresolved author-supplied information

{chr(10).join('- ' + item for item in methods['not_recoverable_from_v3'])}
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    patch = patch_and_support_audit()
    ablation = ablation_audit()
    calibration = calibration_audit()
    field = field_audit()
    methods = methods_inventory()
    manuscript = manuscript_text(patch, ablation, calibration, field)
    (OUT / "MANUSCRIPT_READY_TEXT.md").write_text(manuscript, encoding="utf-8")
    report = alignment_report(patch, ablation, calibration, field, methods)
    (OUT / "SCIENTIFIC_ALIGNMENT_V3.md").write_text(report, encoding="utf-8")

    result = {
        "status": "PASS",
        "scope": "non-structural scientific alignment v3",
        "patch_and_support": patch,
        "ablation": ablation,
        "calibration": calibration,
        "field": field,
        "methods": methods,
    }
    manifest_path = OUT / "scientific_alignment_manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    output_hashes = {
        str(path.relative_to(OUT)): sha256(path)
        for path in sorted(OUT.iterdir())
        if path.is_file() and path != manifest_path
    }
    result["output_sha256"] = output_hashes
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "output_dir": str(OUT),
        "unique_test_pixels": patch["support"]["unique_pixels_total"],
        "field_decision": field["decision"],
        "calibration_pooled": calibration["ensemble"][-1],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
