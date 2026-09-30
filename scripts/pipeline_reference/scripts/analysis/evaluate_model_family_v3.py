"""Evaluate v3 SegFormer, U-Net and Random Forest on the frozen support."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from scripts.analysis import evaluate_final_ablation_unique_pixels as core
from scripts.train.train_tabular import build_feature_matrix


ROOT = Path(r"${PROJECT_ROOT}")
EXPERIMENT = "final_model_family_v3"
BASELINE_ROOT = ROOT / "out/runs/train/final_model_family_baselines_v3"
OUT_DIR = ROOT / "out/runs/analysis/final_scene_ablation_031623_v3/model_family"
MODELS = ("segformer", "unet", "rf")
SCENES = ("20030709_qb", "20160825_wv", "20230812_pl")
SEEDS = (123, 231, 312, 423, 531)


def run_dir(model: str, seed: int) -> Path:
    if model == "segformer":
        return core.find_run_dir("M3_031623", seed, require_complete=True)
    if model in {"unet", "rf"}:
        return BASELINE_ROOT / model / f"seed{seed}"
    raise KeyError(model)


def validate_run(model: str, seed: int) -> dict[str, Any]:
    directory = run_dir(model, seed)
    metadata_path = directory / "model_best.json"
    artifact = directory / ("model.pkl" if model == "rf" else "model_best.pt")
    missing = [path for path in (metadata_path, artifact) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete {model} seed {seed}: {missing}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if model != "rf":
        if not bool(metadata.get("temp_scaling_done", False)):
            raise RuntimeError(f"Temperature scaling missing: {directory}")
        if str(metadata.get("threshold_source")) != "calibrated":
            raise RuntimeError(f"Calibrated threshold missing: {directory}")
    threshold = float(metadata["best_thr"])
    if not 0.0 < threshold < 1.0:
        raise RuntimeError(f"Invalid threshold for {directory}: {threshold}")
    return {
        "model": model,
        "seed": seed,
        "run_dir": str(directory),
        "metadata": str(metadata_path),
        "artifact": str(artifact),
        "threshold": threshold,
        "complete": True,
    }


def unique_rf_features(
    frame: pd.DataFrame,
    support: dict[str, core.SceneSupport],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    filled: dict[str, np.ndarray] = {}
    with h5py.File(core.SUPPORT_H5, "r") as store:
        first = store["patches"][core.patch_key(frame.iloc[0])]["x"]
        channels = int(first.shape[0])
        for scene_id, scene in support.items():
            arrays[scene_id] = np.empty(
                (scene.flat_indices.size, channels), dtype=np.float32
            )
            filled[scene_id] = np.zeros(scene.flat_indices.size, dtype=bool)

        for _, row in frame.iterrows():
            scene_id = str(row["scene_id"])
            scene = support[scene_id]
            patch = store["patches"][core.patch_key(row)]
            x = patch["x"][...].astype(np.float32)
            labels = patch["y"][...]
            weights = patch["w"][...]
            valid = (weights > 0) & np.isin(labels, (1, 2))
            yy, xx = np.nonzero(valid)
            flat = (
                (yy.astype(np.int64) + int(row["y0"])) * scene.width
                + xx.astype(np.int64)
                + int(row["x0"])
            )
            positions = np.searchsorted(scene.flat_indices, flat)
            if np.any(positions >= scene.flat_indices.size):
                raise RuntimeError("RF feature pixel outside frozen support.")
            if np.any(scene.flat_indices[positions] != flat):
                raise RuntimeError("RF feature pixel not in frozen support.")
            values = np.moveaxis(x, 0, -1)[valid]
            new = ~filled[scene_id][positions]
            arrays[scene_id][positions[new]] = values[new]
            filled[scene_id][positions[new]] = True

    for scene_id in sorted(support):
        missing = int(np.count_nonzero(~filled[scene_id]))
        if missing:
            raise RuntimeError(f"Missing {missing} RF features for {scene_id}")
        if not np.isfinite(arrays[scene_id]).all():
            raise RuntimeError(f"Non-finite RF features for {scene_id}")
    return arrays


def infer_rf(
    directory: Path,
    support: dict[str, core.SceneSupport],
    features: dict[str, np.ndarray],
    chunk_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], float, float]:
    metadata = json.loads(
        (directory / "model_best.json").read_text(encoding="utf-8")
    )
    with (directory / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    names = [str(value) for value in metadata["feature_names"]]
    stats = metadata.get("feature_stats") or {}
    output: dict[str, np.ndarray] = {}
    for scene_id in sorted(support):
        matrix, used, _ = build_feature_matrix(
            features[scene_id], names, stats=stats
        )
        if used != names:
            raise RuntimeError(f"RF feature order changed: {used} != {names}")
        probabilities = np.empty(matrix.shape[0], dtype=np.float32)
        for start in range(0, matrix.shape[0], chunk_size):
            stop = min(start + chunk_size, matrix.shape[0])
            probabilities[start:stop] = model.predict_proba(
                matrix[start:stop]
            )[:, 1].astype(np.float32)
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"Non-finite RF probabilities for {scene_id}")
        output[scene_id] = probabilities
    del model
    gc.collect()
    return output, metadata, 1.0, float(metadata["best_thr"])


def evaluate_one(
    model_name: str,
    seed: int,
    frame: pd.DataFrame,
    support: dict[str, core.SceneSupport],
    rf_features: dict[str, np.ndarray] | None,
    device: torch.device,
    batch_size: int,
    rf_chunk_size: int,
) -> dict[str, Any]:
    directory = run_dir(model_name, seed)
    if model_name == "rf":
        if rf_features is None:
            raise RuntimeError("RF features were not prepared.")
        probabilities, metadata, temperature, threshold = infer_rf(
            directory, support, rf_features, rf_chunk_size
        )
    else:
        probabilities, metadata, temperature, threshold = core.infer_model(
            directory, frame, support, device, batch_size
        )

    metric_rows, cell_rows = core.model_rows(
        "M3_031623",
        seed,
        directory,
        support,
        probabilities,
        temperature,
        threshold,
    )
    for row in metric_rows:
        row.pop("branch")
        row["model"] = model_name
        row["training_scenes"] = "2003+2016+2023"
    for row in cell_rows:
        row.pop("branch")
        row["model"] = model_name
        row["training_scenes"] = "2003+2016+2023"

    stem = f"{model_name}_s{seed}"
    metric_path = OUT_DIR / "per_model" / f"metrics_{stem}.csv"
    cell_path = OUT_DIR / "per_model" / f"cells_{stem}.csv"
    manifest_path = OUT_DIR / "per_model" / f"manifest_{stem}.json"
    core.write_csv_atomic(metric_path, metric_rows)
    core.write_csv_atomic(cell_path, cell_rows)
    artifact = directory / ("model.pkl" if model_name == "rf" else "model_best.pt")
    manifest = {
        "experiment": EXPERIMENT,
        "model": model_name,
        "seed": seed,
        "run_dir": str(directory),
        "artifact_sha256": core.sha256_file(artifact),
        "metadata_sha256": core.sha256_file(directory / "model_best.json"),
        "support_csv_sha256": core.sha256_file(core.SUPPORT_CSV),
        "support_h5_sha256": core.sha256_file(core.SUPPORT_H5),
        "temperature": temperature,
        "threshold": threshold,
        "input_adapter": metadata.get("input_adapter", "tabular"),
        "scene_pixel_counts": {
            scene_id: int(scene.flat_indices.size)
            for scene_id, scene in support.items()
        },
        "metric_csv": str(metric_path),
        "cell_csv": str(cell_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest


def consolidate() -> None:
    metric_frames = []
    cell_frames = []
    for model_name in MODELS:
        for seed in SEEDS:
            stem = f"{model_name}_s{seed}"
            metric_frames.append(
                pd.read_csv(OUT_DIR / "per_model" / f"metrics_{stem}.csv")
            )
            cell_frames.append(
                pd.read_csv(OUT_DIR / "per_model" / f"cells_{stem}.csv")
            )
    metrics = pd.concat(metric_frames, ignore_index=True)
    cells = pd.concat(cell_frames, ignore_index=True)
    expected_metrics = {
        (model_name, seed, scene_id)
        for model_name in MODELS
        for seed in SEEDS
        for scene_id in (*SCENES, "ALL_SCENES")
    }
    actual_metrics = {
        (str(row.model), int(row.seed), str(row.scene_id))
        for row in metrics.itertuples()
    }
    if actual_metrics != expected_metrics:
        raise RuntimeError(
            "Model-family metric labels are incomplete or inconsistent: "
            f"missing={sorted(expected_metrics - actual_metrics)[:10]}, "
            f"extra={sorted(actual_metrics - expected_metrics)[:10]}"
        )
    if metrics.duplicated(["model", "seed", "scene_id"], keep=False).any():
        raise RuntimeError("Duplicate model/seed/scene rows in model-family metrics.")
    expected_cell_groups = {
        (model_name, seed)
        for model_name in MODELS
        for seed in SEEDS
    }
    actual_cell_groups = {
        (str(row.model), int(row.seed))
        for row in cells[["model", "seed"]].drop_duplicates().itertuples()
    }
    if actual_cell_groups != expected_cell_groups:
        raise RuntimeError(
            "Model-family cell labels are incomplete or inconsistent: "
            f"missing={sorted(expected_cell_groups - actual_cell_groups)}, "
            f"extra={sorted(actual_cell_groups - expected_cell_groups)}"
        )
    metrics_tmp = OUT_DIR / "all_model_family_metrics.csv.tmp"
    cells_tmp = OUT_DIR / "all_model_family_cells.csv.tmp"
    metrics.to_csv(metrics_tmp, index=False)
    cells.to_csv(cells_tmp, index=False)
    metrics_tmp.replace(OUT_DIR / "all_model_family_metrics.csv")
    cells_tmp.replace(OUT_DIR / "all_model_family_cells.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rf-chunk-size", type=int, default=200_000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    support_manifest = core.verify_support_files()
    discovery = []
    for model_name in MODELS:
        for seed in SEEDS:
            try:
                discovery.append(validate_run(model_name, seed))
            except Exception as exc:
                discovery.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "complete": False,
                        "error": str(exc),
                    }
                )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "support": support_manifest,
                    "runs": discovery,
                    "expected": 15,
                },
                indent=2,
            )
        )
        return 0
    incomplete = [row for row in discovery if not bool(row["complete"])]
    if incomplete:
        raise RuntimeError(
            "Refusing model-family evaluation with incomplete runs:\n"
            + json.dumps(incomplete, indent=2)
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    if args.batch_size < 1 or args.rf_chunk_size < 1:
        raise ValueError("Batch and RF chunk sizes must be positive.")

    frame = pd.read_csv(core.SUPPORT_CSV)
    support = core.build_scene_support(frame)
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
    rf_features: dict[str, np.ndarray] | None = None
    completed = []
    for model_name in MODELS:
        if model_name == "rf":
            rf_features = unique_rf_features(frame, support)
        for seed in SEEDS:
            stem = f"{model_name}_s{seed}"
            expected = [
                OUT_DIR / "per_model" / f"metrics_{stem}.csv",
                OUT_DIR / "per_model" / f"cells_{stem}.csv",
                OUT_DIR / "per_model" / f"manifest_{stem}.json",
            ]
            if args.resume and all(path.is_file() for path in expected):
                metric_models = set(
                    pd.read_csv(expected[0], usecols=["model"])["model"].astype(str)
                )
                cell_models = set(
                    pd.read_csv(expected[1], usecols=["model"])["model"].astype(str)
                )
                if metric_models == {model_name} and cell_models == {model_name}:
                    print(f"[SKIP] {model_name} seed {seed}", flush=True)
                    continue
                print(
                    f"[REBUILD] {model_name} seed {seed}: stale model labels",
                    flush=True,
                )
            print(f"[EVAL] {model_name} seed {seed}", flush=True)
            completed.append(
                evaluate_one(
                    model_name,
                    seed,
                    frame,
                    support,
                    rf_features,
                    device,
                    args.batch_size,
                    args.rf_chunk_size,
                )
            )
    consolidate()
    print(json.dumps({"completed": completed, "consolidated": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
