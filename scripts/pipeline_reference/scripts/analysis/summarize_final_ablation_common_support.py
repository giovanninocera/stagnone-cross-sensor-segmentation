"""Summarize five-seed v3 results with paired spatial bootstrap."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(r"${PROJECT_ROOT}")
EXPERIMENT = "final_scene_ablation_031623_v3"
IN_DIR = ROOT / "out/runs/analysis" / EXPERIMENT / "common_support"
METRICS_CSV = IN_DIR / "all_segformer_metrics.csv"
CELLS_CSV = IN_DIR / "all_segformer_cells.csv"
BRANCHES = (
    "M1_03",
    "M1_16",
    "M1_23",
    "M2_0316",
    "M2_0323",
    "M2_1623",
    "M3_031623",
)
SEEDS = (123, 231, 312, 423, 531)
SCENES = ("20030709_qb", "20160825_wv", "20230812_pl", "ALL_SCENES")
METRIC_COLUMNS = (
    "iou_veg",
    "iou_nonveg",
    "min_iou",
    "macro_iou",
    "balanced_accuracy",
    "mcc",
    "ece_15",
    "brier",
)
PRIMARY_COMPARATOR = "M3_031623"


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    iou_veg = tp / max(1, tp + fp + fn)
    iou_nonveg = tn / max(1, tn + fp + fn)
    recall_veg = tp / max(1, tp + fn)
    recall_nonveg = tn / max(1, tn + fp)
    denominator = math.sqrt(
        max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    return {
        "min_iou": min(iou_veg, iou_nonveg),
        "macro_iou": (iou_veg + iou_nonveg) / 2,
        "balanced_accuracy": (recall_veg + recall_nonveg) / 2,
        "mcc": (tp * tn - fp * fn) / denominator,
    }


def validate_metrics(frame: pd.DataFrame) -> None:
    expected = {
        (branch, seed, scene)
        for branch in BRANCHES
        for seed in SEEDS
        for scene in SCENES
    }
    actual = {
        (str(row.branch), int(row.seed), str(row.scene_id))
        for row in frame.itertuples()
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Metrics table is incomplete. Missing={missing[:10]} extra={extra[:10]}"
        )
    duplicated = frame.duplicated(["branch", "seed", "scene_id"], keep=False)
    if duplicated.any():
        raise RuntimeError("Duplicate branch/seed/scene rows in metrics table.")
    for scene, group in frame.groupby("scene_id"):
        counts = group["n_unique_pixels"].astype(int).unique()
        if len(counts) != 1:
            raise RuntimeError(f"Unequal support pixel counts for {scene}: {counts}")


def summarize_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (branch, scene_id), group in frame.groupby(["branch", "scene_id"], sort=True):
        if tuple(sorted(group["seed"].astype(int))) != SEEDS:
            raise RuntimeError(f"Expected five fixed seeds for {branch}/{scene_id}")
        record: dict[str, Any] = {
            "branch": branch,
            "scene_id": scene_id,
            "n_seeds": len(group),
            "n_unique_pixels": int(group["n_unique_pixels"].iloc[0]),
        }
        for metric in METRIC_COLUMNS:
            values = group[metric].astype(float).to_numpy()
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
            record[f"{metric}_min"] = float(values.min())
            record[f"{metric}_max"] = float(values.max())
            record[f"{metric}_seed_ci95_low"] = mean - 2.776 * sd / math.sqrt(5)
            record[f"{metric}_seed_ci95_high"] = mean + 2.776 * sd / math.sqrt(5)
        records.append(record)
    return pd.DataFrame(records)


def aligned_cells(
    cells: pd.DataFrame, branch_a: str, branch_b: str, seed: int, scene: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["scene_id", "cell_x", "cell_y"]
    left = cells[(cells.branch == branch_a) & (cells.seed == seed)]
    right = cells[(cells.branch == branch_b) & (cells.seed == seed)]
    if scene != "ALL_SCENES":
        left = left[left.scene_id == scene]
        right = right[right.scene_id == scene]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if not left[keys].equals(right[keys]):
        raise RuntimeError(
            f"Spatial cell mismatch for {branch_a} vs {branch_b}, seed {seed}, {scene}"
        )
    if len(left) == 0:
        raise RuntimeError(f"No cells for seed {seed}, scene {scene}")
    return left, right


def sum_confusion(frame: pd.DataFrame) -> tuple[int, int, int, int]:
    return tuple(
        int(frame[column].astype(np.int64).sum())
        for column in ("tp", "fp", "fn", "tn")
    )


def sample_paired_confusions(
    left: pd.DataFrame,
    right: pd.DataFrame,
    rng: np.random.Generator,
    scene: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if scene == "ALL_SCENES":
        left_parts = []
        right_parts = []
        for scene_id in sorted(left["scene_id"].unique()):
            left_group = left[left.scene_id == scene_id].reset_index(drop=True)
            right_group = right[right.scene_id == scene_id].reset_index(drop=True)
            if len(left_group) != len(right_group):
                raise RuntimeError(f"Paired cell count mismatch for {scene_id}")
            chosen = rng.integers(0, len(left_group), size=len(left_group))
            left_parts.append(left_group.iloc[chosen])
            right_parts.append(right_group.iloc[chosen])
        left_sampled = pd.concat(left_parts, ignore_index=True)
        right_sampled = pd.concat(right_parts, ignore_index=True)
    else:
        if len(left) != len(right):
            raise RuntimeError(f"Paired cell count mismatch for {scene}")
        chosen = rng.integers(0, len(left), size=len(left))
        left_sampled = left.iloc[chosen]
        right_sampled = right.iloc[chosen]
    return sum_confusion(left_sampled), sum_confusion(right_sampled)


def confusion_metrics_array(confusion: np.ndarray) -> dict[str, np.ndarray]:
    tp, fp, fn, tn = (
        confusion[:, index].astype(np.float64) for index in range(4)
    )
    iou_veg = tp / np.maximum(1.0, tp + fp + fn)
    iou_nonveg = tn / np.maximum(1.0, tn + fp + fn)
    recall_veg = tp / np.maximum(1.0, tp + fn)
    recall_nonveg = tn / np.maximum(1.0, tn + fp)
    denominator = np.sqrt(
        np.maximum(
            1.0,
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn),
        )
    )
    return {
        "min_iou": np.minimum(iou_veg, iou_nonveg),
        "macro_iou": (iou_veg + iou_nonveg) / 2,
        "balanced_accuracy": (recall_veg + recall_nonveg) / 2,
        "mcc": (tp * tn - fp * fn) / denominator,
    }


def cell_confusion_groups(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    columns = ["tp", "fp", "fn", "tn"]
    return {
        str(scene_id): group[columns].to_numpy(dtype=np.int64)
        for scene_id, group in frame.groupby("scene_id", sort=True)
    }


def resample_paired_confusion_arrays(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    draws: int,
    rng: np.random.Generator,
    scene: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_result = np.zeros((draws, 4), dtype=np.int64)
    right_result = np.zeros((draws, 4), dtype=np.int64)
    scenes = sorted(left) if scene == "ALL_SCENES" else [scene]
    for scene_id in scenes:
        left_values = left[scene_id]
        right_values = right[scene_id]
        if left_values.shape != right_values.shape:
            raise RuntimeError(f"Paired cell shape mismatch for {scene_id}")
        indices = rng.integers(
            0, len(left_values), size=(draws, len(left_values))
        )
        left_result += left_values[indices].sum(axis=1)
        right_result += right_values[indices].sum(axis=1)
    return left_result, right_result


def paired_bootstrap(
    cells: pd.DataFrame, draws: int, random_seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, Any]] = []
    comparison_branches = [branch for branch in BRANCHES if branch != PRIMARY_COMPARATOR]
    for competitor in comparison_branches:
        for scene in SCENES:
            pairs = {
                seed: tuple(
                    cell_confusion_groups(frame)
                    for frame in aligned_cells(
                        cells, PRIMARY_COMPARATOR, competitor, seed, scene
                    )
                )
                for seed in SEEDS
            }
            distributions = {
                metric: np.zeros(draws, dtype=np.float64)
                for metric in ("min_iou", "macro_iou", "balanced_accuracy", "mcc")
            }
            sampled_seeds = rng.choice(
                SEEDS, size=(draws, len(SEEDS)), replace=True
            )
            for slot in range(len(SEEDS)):
                for seed in SEEDS:
                    selector = sampled_seeds[:, slot] == seed
                    selected_draws = int(selector.sum())
                    if selected_draws == 0:
                        continue
                    primary_groups, competitor_groups = pairs[seed]
                    primary_confusion, competitor_confusion = (
                        resample_paired_confusion_arrays(
                            primary_groups,
                            competitor_groups,
                            selected_draws,
                            rng,
                            scene,
                        )
                    )
                    primary_metrics = confusion_metrics_array(primary_confusion)
                    competitor_metrics = confusion_metrics_array(
                        competitor_confusion
                    )
                    for metric in distributions:
                        distributions[metric][selector] += (
                            primary_metrics[metric] - competitor_metrics[metric]
                        )
            for metric in distributions:
                distributions[metric] /= len(SEEDS)
            record: dict[str, Any] = {
                "comparison": f"{PRIMARY_COMPARATOR}_minus_{competitor}",
                "primary": PRIMARY_COMPARATOR,
                "competitor": competitor,
                "scene_id": scene,
                "bootstrap_draws": draws,
                "random_seed": random_seed,
            }
            for metric, values in distributions.items():
                record[f"delta_{metric}_mean"] = float(values.mean())
                record[f"delta_{metric}_median"] = float(np.median(values))
                record[f"delta_{metric}_ci95_low"] = float(
                    np.quantile(values, 0.025)
                )
                record[f"delta_{metric}_ci95_high"] = float(
                    np.quantile(values, 0.975)
                )
                record[f"prob_delta_{metric}_gt_0"] = float(np.mean(values > 0))
            records.append(record)
            print(f"[BOOT] {PRIMARY_COMPARATOR} - {competitor}, {scene}", flush=True)
    return pd.DataFrame(records)


def write_atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260703)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise ValueError("Use at least 100 bootstrap draws.")
    metrics = pd.read_csv(METRICS_CSV)
    cells = pd.read_csv(CELLS_CSV)
    validate_metrics(metrics)
    summary = summarize_seeds(metrics)
    bootstrap = paired_bootstrap(cells, args.bootstrap_draws, args.random_seed)
    IN_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = IN_DIR / "segformer_five_seed_summary.csv"
    bootstrap_path = IN_DIR / "segformer_paired_spatial_bootstrap.csv"
    write_atomic_csv(summary, summary_path)
    write_atomic_csv(bootstrap, bootstrap_path)
    manifest = {
        "experiment": EXPERIMENT,
        "branches": list(BRANCHES),
        "seeds": list(SEEDS),
        "metrics_input": str(METRICS_CSV),
        "cells_input": str(CELLS_CSV),
        "seed_summary": str(summary_path),
        "paired_spatial_bootstrap": str(bootstrap_path),
        "bootstrap_draws": args.bootstrap_draws,
        "random_seed": args.random_seed,
        "bootstrap_design": (
            "paired hierarchical resampling: five seeds with replacement; "
            "spatial 256-pixel cells with replacement within each scene"
        ),
    }
    manifest_path = IN_DIR / "summary_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
