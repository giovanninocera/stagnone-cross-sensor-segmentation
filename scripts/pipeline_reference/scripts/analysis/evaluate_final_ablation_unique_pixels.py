"""Evaluate all v3 SegFormer branches on one deduplicated spatial support.

The script intentionally fails on incomplete calibration, changed support files,
missing predictions, conflicting labels, or unequal pixel coverage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import rasterio
import torch

from scripts.models.model_input import adapt_batch_torch
from scripts.train.train_segformer import make_model


ROOT = Path(r"${PROJECT_ROOT}")
EXPERIMENT = "final_scene_ablation_031623_v3"
SUPPORT_DIR = ROOT / "out" / "eval_support" / EXPERIMENT
SUPPORT_CSV = SUPPORT_DIR / "common_eval_windows.csv"
SUPPORT_H5 = SUPPORT_DIR / "common_eval_windows.h5"
SUPPORT_MANIFEST = SUPPORT_DIR / "common_eval_manifest.json"
RUN_ROOTS = {
    123: ROOT / "out/runs/train/final_scene_ablation_031623_v3/stage2",
    231: ROOT / "out/runs/train/final_scene_ablation_031623_v3/stage2",
    312: ROOT / "out/runs/train/final_scene_ablation_031623_v3/stage2",
    423: ROOT / "out/runs/train/final_scene_ablation_031623_v3_seed_extension/stage2",
    531: ROOT / "out/runs/train/final_scene_ablation_031623_v3_seed_extension/stage2",
}
BRANCHES = {
    "M1_03": {"directory": "FINALABL_M1_03", "training_scenes": "2003"},
    "M1_16": {"directory": "FINALABL_M1_16", "training_scenes": "2016"},
    "M1_23": {"directory": "FINALABL_M1_23", "training_scenes": "2023"},
    "M2_0316": {"directory": "FINALABL_M2_0316", "training_scenes": "2003+2016"},
    "M2_0323": {"directory": "FINALABL_M2_0323", "training_scenes": "2003+2023"},
    "M2_1623": {"directory": "FINALABL_M2_1623", "training_scenes": "2016+2023"},
    "M3_031623": {
        "directory": "FINALABL_M3_031623",
        "training_scenes": "2003+2016+2023",
    },
}
ALL_SEEDS = (123, 231, 312, 423, 531)
CELL_SIZE = 256
MIN_CELL_PIXELS = 512
OUT_DIR = ROOT / "out/runs/analysis/final_scene_ablation_031623_v3/common_support"


@dataclass(frozen=True)
class SceneSupport:
    scene_id: str
    width: int
    height: int
    flat_indices: np.ndarray
    labels: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_support_files() -> dict[str, Any]:
    manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
    expected = {
        SUPPORT_CSV: manifest["csv_sha256"],
        SUPPORT_H5: manifest["h5_sha256"],
    }
    for path, expected_hash in expected.items():
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(
                f"Support hash mismatch for {path}: {actual} != {expected_hash}"
            )
    audit = manifest.get("leakage_audit", {})
    if not audit.get("passed") or int(audit.get("cross_split_overlap_pairs", -1)) != 0:
        raise RuntimeError("The common support did not pass its leakage audit.")
    return manifest


def patch_key(row: pd.Series) -> str:
    return (
        f"{row['scene_id']}__y{int(row['y0'])}_x{int(row['x0'])}_"
        f"t{int(row['tile'])}"
    )


def _unique_labels(flat: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    labels_sorted = labels[order]
    unique_flat, first = np.unique(flat_sorted, return_index=True)
    unique_labels = labels_sorted[first]
    last = np.r_[first[1:] - 1, len(flat_sorted) - 1]
    if np.any(labels_sorted[last] != unique_labels):
        raise RuntimeError("Conflicting labels found for duplicate global pixels.")
    return unique_flat.astype(np.int64), unique_labels.astype(np.uint8)


def build_scene_support(
    frame: pd.DataFrame,
) -> dict[str, SceneSupport]:
    output: dict[str, SceneSupport] = {}
    with h5py.File(SUPPORT_H5, "r") as store:
        for scene_id, group in frame.groupby("scene_id", sort=True):
            mask_path = Path(str(group.iloc[0]["mask_tif"]))
            with rasterio.open(mask_path) as source:
                width, height = source.width, source.height
            flat_parts: list[np.ndarray] = []
            label_parts: list[np.ndarray] = []
            for _, row in group.iterrows():
                patch = store["patches"][patch_key(row)]
                labels = patch["y"][...]
                weights = patch["w"][...]
                valid = (weights > 0) & np.isin(labels, (1, 2))
                yy, xx = np.nonzero(valid)
                flat_parts.append(
                    (yy.astype(np.int64) + int(row["y0"])) * width
                    + xx.astype(np.int64)
                    + int(row["x0"])
                )
                label_parts.append(labels[valid].astype(np.uint8))
            flat, labels = _unique_labels(
                np.concatenate(flat_parts), np.concatenate(label_parts)
            )
            output[str(scene_id)] = SceneSupport(
                scene_id=str(scene_id),
                width=width,
                height=height,
                flat_indices=flat,
                labels=labels,
            )
    return output


def find_run_dir(branch: str, seed: int, require_complete: bool = True) -> Path:
    branch_root = RUN_ROOTS[seed] / str(BRANCHES[branch]["directory"])
    matches = sorted(branch_root.glob(f"*_s{seed}_augoff"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one run for {branch} seed {seed}; found {len(matches)}"
        )
    directory = matches[0]
    if require_complete:
        missing = [
            name
            for name in ("model_best.pt", "model_best.json")
            if not (directory / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Incomplete run {branch} seed {seed}; missing {', '.join(missing)}"
            )
        metadata = json.loads(
            (directory / "model_best.json").read_text(encoding="utf-8")
        )
        if not bool(metadata.get("temp_scaling_done", False)):
            raise RuntimeError(
                f"Incomplete run {branch} seed {seed}; temperature scaling pending"
            )
        if str(metadata.get("threshold_source")) != "calibrated":
            raise RuntimeError(
                f"Incomplete run {branch} seed {seed}; calibrated threshold pending"
            )
    return directory


def discover_runs(
    branches: Iterable[str], seeds: Iterable[int]
) -> list[dict[str, Any]]:
    rows = []
    for branch in branches:
        for seed in seeds:
            try:
                directory = find_run_dir(branch, seed, require_complete=False)
                files_complete = all(
                    (directory / name).is_file()
                    for name in ("model_best.pt", "model_best.json")
                )
                metadata = (
                    json.loads(
                        (directory / "model_best.json").read_text(encoding="utf-8")
                    )
                    if files_complete
                    else {}
                )
                complete = (
                    files_complete
                    and bool(metadata.get("temp_scaling_done", False))
                    and str(metadata.get("threshold_source")) == "calibrated"
                )
                error = ""
            except Exception as exc:
                directory = None
                complete = False
                error = str(exc)
            rows.append(
                {
                    "branch": branch,
                    "seed": seed,
                    "training_scenes": BRANCHES[branch]["training_scenes"],
                    "run_dir": str(directory) if directory else "",
                    "complete": complete,
                    "error": error,
                }
            )
    return rows


def load_model(
    directory: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], float, float]:
    metadata = json.loads((directory / "model_best.json").read_text(encoding="utf-8"))
    if metadata.get("source_suffix") != "_st3b_final_scene_ablation_031623_v3_common":
        raise RuntimeError(f"Unexpected feature suffix in {directory}")
    if not bool(metadata.get("temp_scaling_done", False)):
        raise RuntimeError(f"Temperature scaling is incomplete in {directory}")
    if str(metadata.get("threshold_source")) != "calibrated":
        raise RuntimeError(f"Threshold is not calibrated in {directory}")
    temperature = float(metadata["temperature"])
    threshold = float(metadata["best_thr"])
    if not (math.isfinite(temperature) and temperature > 0):
        raise RuntimeError(f"Invalid temperature in {directory}: {temperature}")
    if not (0.0 < threshold < 1.0):
        raise RuntimeError(f"Invalid threshold in {directory}: {threshold}")
    model = make_model(
        str(metadata["arch"]),
        str(metadata["encoder"]),
        int(metadata["in_ch"]),
        encoder_weights=str(metadata.get("encoder_weights", "none")),
    ).to(device)
    checkpoint = torch.load(
        directory / "model_best.pt", map_location=device, weights_only=False
    )
    state = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata, temperature, threshold


def add_patch_probabilities(
    support: SceneSupport,
    row: pd.Series,
    labels: np.ndarray,
    weights: np.ndarray,
    probability: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    valid = (weights > 0) & np.isin(labels, (1, 2)) & np.isfinite(probability)
    if not valid.any():
        return
    yy, xx = np.nonzero(valid)
    flat = (
        (yy.astype(np.int64) + int(row["y0"])) * support.width
        + xx.astype(np.int64)
        + int(row["x0"])
    )
    positions = np.searchsorted(support.flat_indices, flat)
    if np.any(positions >= support.flat_indices.size):
        raise RuntimeError("Predicted a pixel outside the frozen support.")
    if np.any(support.flat_indices[positions] != flat):
        raise RuntimeError("Predicted a pixel not present in the frozen support.")
    if np.any(support.labels[positions] != labels[valid]):
        raise RuntimeError("Prediction labels disagree with the frozen support.")
    np.add.at(sums, positions, probability[valid].astype(np.float64))
    np.add.at(counts, positions, 1)


def infer_model(
    directory: Path,
    frame: pd.DataFrame,
    support: dict[str, SceneSupport],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float, float]:
    model, metadata, temperature, threshold = load_model(directory, device)
    scene_sums = {
        scene_id: np.zeros(item.flat_indices.size, dtype=np.float64)
        for scene_id, item in support.items()
    }
    scene_counts = {
        scene_id: np.zeros(item.flat_indices.size, dtype=np.uint16)
        for scene_id, item in support.items()
    }
    adapter = str(metadata["input_adapter"])
    with h5py.File(SUPPORT_H5, "r") as store:
        for start in range(0, len(frame), batch_size):
            rows = frame.iloc[start : start + batch_size]
            xs: list[np.ndarray] = []
            labels: list[np.ndarray] = []
            weights: list[np.ndarray] = []
            for _, row in rows.iterrows():
                patch = store["patches"][patch_key(row)]
                x = patch["x"][...].astype(np.float32)
                if x.shape[0] != int(metadata["in_ch"]):
                    raise RuntimeError(
                        f"Input channels {x.shape[0]} != model channels "
                        f"{metadata['in_ch']} for {directory}"
                    )
                xs.append(x)
                labels.append(patch["y"][...])
                weights.append(patch["w"][...])
            tensor = adapt_batch_torch(
                torch.from_numpy(np.stack(xs)).to(device, non_blocking=True), adapter
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(tensor)
                probabilities = torch.sigmoid(logits / temperature)
            probability_array = probabilities.detach().float().cpu().numpy()
            if probability_array.ndim == 4:
                probability_array = probability_array[:, 0]
            if probability_array.shape != (
                len(rows),
                int(rows.iloc[0]["tile"]),
                int(rows.iloc[0]["tile"]),
            ):
                raise RuntimeError(
                    f"Unexpected model output shape {probability_array.shape}"
                )
            if not np.isfinite(probability_array).all():
                raise RuntimeError(f"Non-finite probability from {directory}")
            for (_, row), label, weight, probability in zip(
                rows.iterrows(), labels, weights, probability_array
            ):
                scene_id = str(row["scene_id"])
                add_patch_probabilities(
                    support[scene_id],
                    row,
                    label,
                    weight,
                    probability,
                    scene_sums[scene_id],
                    scene_counts[scene_id],
                )
    output: dict[str, np.ndarray] = {}
    for scene_id in sorted(support):
        counts = scene_counts[scene_id]
        if np.any(counts == 0):
            raise RuntimeError(
                f"{int(np.count_nonzero(counts == 0))} frozen pixels were not "
                f"predicted for {scene_id}"
            )
        output[scene_id] = (scene_sums[scene_id] / counts).astype(np.float32)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, metadata, temperature, threshold


def confusion(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> tuple[int, int, int, int]:
    target = labels == 2
    prediction = probabilities >= threshold
    return (
        int(np.count_nonzero(prediction & target)),
        int(np.count_nonzero(prediction & ~target)),
        int(np.count_nonzero(~prediction & target)),
        int(np.count_nonzero(~prediction & ~target)),
    )


def metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    tp, fp, fn, tn = confusion(labels, probabilities, threshold)
    iou_veg = tp / max(1, tp + fp + fn)
    iou_nonveg = tn / max(1, tn + fp + fn)
    precision_veg = tp / max(1, tp + fp)
    recall_veg = tp / max(1, tp + fn)
    f1_veg = 2 * tp / max(1, 2 * tp + fp + fn)
    precision_nonveg = tn / max(1, tn + fn)
    recall_nonveg = tn / max(1, tn + fp)
    f1_nonveg = 2 * tn / max(1, 2 * tn + fp + fn)
    denominator = math.sqrt(
        max(1, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    binary = (labels == 2).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selector = (
            (probabilities >= low) & (probabilities <= high)
            if index == len(edges) - 2
            else (probabilities >= low) & (probabilities < high)
        )
        if selector.any():
            ece += float(selector.mean()) * abs(
                float(probabilities[selector].mean())
                - float(binary[selector].mean())
            )
    return {
        "n_unique_pixels": int(labels.size),
        "n_nonveg": int(np.count_nonzero(labels == 1)),
        "n_veg": int(np.count_nonzero(labels == 2)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou_veg": iou_veg,
        "iou_nonveg": iou_nonveg,
        "min_iou": min(iou_veg, iou_nonveg),
        "macro_iou": (iou_veg + iou_nonveg) / 2,
        "precision_veg": precision_veg,
        "recall_veg": recall_veg,
        "f1_veg": f1_veg,
        "precision_nonveg": precision_nonveg,
        "recall_nonveg": recall_nonveg,
        "f1_nonveg": f1_nonveg,
        "balanced_accuracy": (recall_veg + recall_nonveg) / 2,
        "mcc": (tp * tn - fp * fn) / denominator,
        "ece_15": ece,
        "brier": float(np.mean((probabilities - binary) ** 2)),
    }


def calibration_cells(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, float | int]:
    binary = (labels == 2).astype(np.float64)
    result: dict[str, float | int] = {
        "brier_sum": float(np.sum((probabilities - binary) ** 2))
    }
    edges = np.linspace(0.0, 1.0, 16)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selector = (
            (probabilities >= low) & (probabilities <= high)
            if index == len(edges) - 2
            else (probabilities >= low) & (probabilities < high)
        )
        result[f"ece_bin_{index:02d}_n"] = int(selector.sum())
        result[f"ece_bin_{index:02d}_prob_sum"] = float(
            probabilities[selector].sum()
        )
        result[f"ece_bin_{index:02d}_target_sum"] = float(binary[selector].sum())
    return result


def model_rows(
    branch: str,
    seed: int,
    directory: Path,
    support: dict[str, SceneSupport],
    probabilities: dict[str, np.ndarray],
    temperature: float,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    pooled_labels: list[np.ndarray] = []
    pooled_probabilities: list[np.ndarray] = []
    base = {
        "branch": branch,
        "seed": seed,
        "training_scenes": BRANCHES[branch]["training_scenes"],
        "run_dir": str(directory),
        "temperature": temperature,
        "checkpoint_threshold": threshold,
    }
    for scene_id in sorted(support):
        scene = support[scene_id]
        labels = scene.labels
        probs = probabilities[scene_id]
        pooled_labels.append(labels)
        pooled_probabilities.append(probs)
        checkpoint_metrics = metrics(labels, probs, threshold)
        fixed_metrics = metrics(labels, probs, 0.5)
        metric_rows.append(
            {
                **base,
                "scene_id": scene_id,
                **checkpoint_metrics,
                **{
                    f"{key}_at_05": value
                    for key, value in fixed_metrics.items()
                    if key
                    not in {"n_unique_pixels", "n_nonveg", "n_veg", "tp", "fp", "fn", "tn"}
                },
            }
        )
        yy = scene.flat_indices // scene.width
        xx = scene.flat_indices % scene.width
        cell_y = yy // CELL_SIZE
        cell_x = xx // CELL_SIZE
        encoded = cell_y * math.ceil(scene.width / CELL_SIZE) + cell_x
        for cell_id in np.unique(encoded):
            selector = encoded == cell_id
            if int(selector.sum()) < MIN_CELL_PIXELS:
                continue
            local_labels = labels[selector]
            local_probs = probs[selector]
            local = metrics(local_labels, local_probs, threshold)
            cell_rows.append(
                {
                    **base,
                    "scene_id": scene_id,
                    "cell_x": int(cell_x[selector][0]),
                    "cell_y": int(cell_y[selector][0]),
                    **local,
                    **calibration_cells(local_labels, local_probs),
                }
            )
    labels_all = np.concatenate(pooled_labels)
    probabilities_all = np.concatenate(pooled_probabilities)
    checkpoint_metrics = metrics(labels_all, probabilities_all, threshold)
    fixed_metrics = metrics(labels_all, probabilities_all, 0.5)
    metric_rows.append(
        {
            **base,
            "scene_id": "ALL_SCENES",
            **checkpoint_metrics,
            **{
                f"{key}_at_05": value
                for key, value in fixed_metrics.items()
                if key
                not in {"n_unique_pixels", "n_nonveg", "n_veg", "tp", "fp", "fn", "tn"}
            },
        }
    )
    return metric_rows, cell_rows


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_one(
    branch: str,
    seed: int,
    frame: pd.DataFrame,
    support: dict[str, SceneSupport],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    directory = find_run_dir(branch, seed, require_complete=True)
    probabilities, metadata, temperature, threshold = infer_model(
        directory, frame, support, device, batch_size
    )
    metric_rows, cell_rows = model_rows(
        branch,
        seed,
        directory,
        support,
        probabilities,
        temperature,
        threshold,
    )
    stem = f"{branch.lower()}_s{seed}"
    metric_path = OUT_DIR / "per_model" / f"metrics_{stem}.csv"
    cell_path = OUT_DIR / "per_model" / f"cells_{stem}.csv"
    manifest_path = OUT_DIR / "per_model" / f"manifest_{stem}.json"
    write_csv_atomic(metric_path, metric_rows)
    write_csv_atomic(cell_path, cell_rows)
    run_manifest = {
        "experiment": EXPERIMENT,
        "branch": branch,
        "seed": seed,
        "training_scenes": BRANCHES[branch]["training_scenes"],
        "run_dir": str(directory),
        "checkpoint_sha256": sha256_file(directory / "model_best.pt"),
        "metadata_sha256": sha256_file(directory / "model_best.json"),
        "support_csv_sha256": sha256_file(SUPPORT_CSV),
        "support_h5_sha256": sha256_file(SUPPORT_H5),
        "temperature": temperature,
        "threshold": threshold,
        "input_adapter": metadata["input_adapter"],
        "scene_pixel_counts": {
            scene_id: int(scene.flat_indices.size)
            for scene_id, scene in support.items()
        },
        "metric_csv": str(metric_path),
        "cell_csv": str(cell_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return {
        "branch": branch,
        "seed": seed,
        "metrics": str(metric_path),
        "cells": str(cell_path),
        "manifest": str(manifest_path),
    }


def consolidate(expected_runs: list[tuple[str, int]]) -> None:
    metric_frames = []
    cell_frames = []
    for branch, seed in expected_runs:
        stem = f"{branch.lower()}_s{seed}"
        metric_path = OUT_DIR / "per_model" / f"metrics_{stem}.csv"
        cell_path = OUT_DIR / "per_model" / f"cells_{stem}.csv"
        if not metric_path.is_file() or not cell_path.is_file():
            raise RuntimeError(f"Cannot consolidate missing result {branch} seed {seed}")
        metric_frames.append(pd.read_csv(metric_path))
        cell_frames.append(pd.read_csv(cell_path))
    metrics_frame = pd.concat(metric_frames, ignore_index=True)
    cells_frame = pd.concat(cell_frames, ignore_index=True)
    metrics_tmp = OUT_DIR / "all_segformer_metrics.csv.tmp"
    cells_tmp = OUT_DIR / "all_segformer_cells.csv.tmp"
    metrics_frame.to_csv(metrics_tmp, index=False)
    cells_frame.to_csv(cells_tmp, index=False)
    metrics_tmp.replace(OUT_DIR / "all_segformer_metrics.csv")
    cells_tmp.replace(OUT_DIR / "all_segformer_cells.csv")


def parse_items(raw: str, allowed: Iterable[str]) -> list[str]:
    allowed_list = list(allowed)
    if raw.lower() == "all":
        return allowed_list
    result = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(result) - set(allowed_list))
    if unknown:
        raise ValueError(f"Unknown values: {unknown}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()

    support_manifest = verify_support_files()
    branches = parse_items(args.branches, BRANCHES)
    seeds = [
        int(value)
        for value in parse_items(args.seeds, [str(seed) for seed in ALL_SEEDS])
    ]
    discovery = discover_runs(branches, seeds)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "support": support_manifest,
                    "runs": discovery,
                    "complete": sum(bool(row["complete"]) for row in discovery),
                    "expected": len(discovery),
                },
                indent=2,
            )
        )
        return 0
    incomplete = [row for row in discovery if not row["complete"]]
    if incomplete:
        raise RuntimeError(
            "Refusing evaluation because runs are incomplete:\n"
            + json.dumps(incomplete, indent=2)
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    frame = pd.read_csv(SUPPORT_CSV)
    support = build_scene_support(frame)
    expected_counts = {
        scene_id: int(stats["unique_labelled_pixels"])
        for scene_id, stats in support_manifest["scene_stats"].items()
    }
    actual_counts = {
        scene_id: int(scene.flat_indices.size)
        for scene_id, scene in support.items()
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Frozen pixel counts changed: {actual_counts} != {expected_counts}"
        )
    device = torch.device(args.device)
    completed = []
    for branch in branches:
        for seed in seeds:
            print(f"[EVAL] {branch} seed {seed}", flush=True)
            completed.append(
                evaluate_one(
                    branch, seed, frame, support, device, args.batch_size
                )
            )
    if args.consolidate:
        consolidate([(branch, seed) for branch in branches for seed in seeds])
    print(json.dumps({"completed": completed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
