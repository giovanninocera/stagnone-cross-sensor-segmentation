"""Train final RF and U-Net baselines on the corrected v3 M3 dataset."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import torch

from scripts.common.config import OUT, ROOT


EXPERIMENT = "final_model_family_baselines_v3"
PATCH_CSV = (
    OUT / "patches" / "patches_final_scene_ablation_031623_v3_m3_031623.csv"
)
H5_PATH = (
    OUT
    / "patches_bin"
    / "STORE_final_scene_ablation_031623_v3_m3_031623_st3b_t256.h5"
)
OUT_ROOT = OUT / "runs" / "train" / EXPERIMENT
FEATURE_SUFFIX = "_st3b_final_scene_ablation_031623_v3_common"
SEEDS = (123, 231, 312, 423, 531)


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_items(raw: str, allowed: tuple[str, ...]) -> list[str]:
    if raw.lower() == "all":
        return list(allowed)
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown values: {unknown}")
    return values


def parse_seeds(raw: str) -> list[int]:
    if raw.lower() == "all":
        return list(SEEDS)
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    unknown = sorted(set(values) - set(SEEDS))
    if unknown:
        raise ValueError(f"Unknown seeds: {unknown}")
    return values


def write_log(message: str) -> None:
    print(message, flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "baseline_master.log").open(
        "a", encoding="utf-8", errors="replace"
    ) as handle:
        handle.write(message + "\n")


def run_stream(command: list[str], title: str) -> None:
    write_log("")
    write_log("=" * 100)
    write_log(f"[{now()}] {title}")
    write_log("CMD: " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        write_log(line.rstrip("\r\n"))
    return_code = process.wait()
    write_log(f"[{now()}] return_code={return_code}")
    if return_code != 0:
        raise RuntimeError(f"{title} failed with return code {return_code}")


def validate_inputs() -> dict[str, Any]:
    if not PATCH_CSV.is_file() or not H5_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset input: {PATCH_CSV} or {H5_PATH}")
    frame = pd.read_csv(PATCH_CSV)
    required = {"scene_id", "x0", "y0", "tile", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Patch CSV missing columns: {missing}")
    if len(frame) != 6000:
        raise RuntimeError(f"Expected 6000 patches, found {len(frame)}")
    if frame.duplicated(["scene_id", "x0", "y0", "tile"]).any():
        raise RuntimeError("Duplicate patch coordinates in corrected M3 CSV")
    expected_scenes = {"20030709_qb", "20160825_wv", "20230812_pl"}
    if set(frame["scene_id"]) != expected_scenes:
        raise RuntimeError(f"Unexpected scenes: {sorted(set(frame['scene_id']))}")
    scene_counts = frame.groupby("scene_id").size().to_dict()
    if any(int(value) != 2000 for value in scene_counts.values()):
        raise RuntimeError(f"Expected 2000 patches per scene: {scene_counts}")
    expected_keys = {
        f"{row.scene_id}__y{int(row.y0)}_x{int(row.x0)}_t{int(row.tile)}"
        for row in frame.itertuples()
    }
    with h5py.File(H5_PATH, "r") as store:
        actual_keys = set(store["patches"].keys())
        first = store["patches"][next(iter(actual_keys))]["x"]
        shape = tuple(int(value) for value in first.shape)
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"H5/CSV key mismatch: missing={len(expected_keys-actual_keys)}, "
            f"extra={len(actual_keys-expected_keys)}"
        )
    if shape != (3, 256, 256):
        raise RuntimeError(f"Unexpected patch tensor shape: {shape}")
    return {
        "patch_rows": len(frame),
        "scene_counts": scene_counts,
        "split_counts": frame.groupby("split").size().to_dict(),
        "h5_keys": len(actual_keys),
        "tensor_shape": shape,
    }


def run_directory(model: str, seed: int) -> Path:
    return OUT_ROOT / model / f"seed{seed}"


def run_complete(model: str, seed: int) -> bool:
    directory = run_directory(model, seed)
    metadata_path = directory / "model_best.json"
    if model == "rf":
        return metadata_path.is_file() and (directory / "model.pkl").is_file()
    if not metadata_path.is_file() or not (directory / "model_best.pt").is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(metadata.get("temp_scaling_done", False)) and (
        metadata.get("threshold_source") == "calibrated"
    )


def rf_command(seed: int) -> list[str]:
    directory = run_directory("rf", seed)
    return [
        sys.executable,
        "-u",
        "-m",
        "scripts.train.train_tabular",
        "--patch_csv",
        str(PATCH_CSV),
        "--h5_path",
        str(H5_PATH),
        "--out_dir",
        str(directory),
        "--model",
        "rf",
        "--feature_names",
        "B,G,R",
        "--tile",
        "256",
        "--seed",
        str(seed),
        "--thr_metric",
        "min_iou",
        "--train_pixels_per_class",
        "250000",
        "--val_pixels_per_class",
        "150000",
        "--n_jobs",
        "-1",
        "--rf_n_estimators",
        "400",
        "--rf_max_depth",
        "20",
        "--rf_min_samples_leaf",
        "2",
    ]


def unet_command(seed: int) -> list[str]:
    directory = run_directory("unet", seed)
    return [
        sys.executable,
        "-u",
        "-m",
        "scripts.train.train_segformer",
        "--patch_csv",
        str(PATCH_CSV),
        "--h5_path",
        str(H5_PATH),
        "--out_dir",
        str(directory),
        "--arch",
        "unet",
        "--encoder",
        "resnet34",
        "--encoder_weights",
        "imagenet",
        "--input_adapter",
        "auto",
        "--in_ch",
        "3",
        "--tile",
        "256",
        "--suffix",
        FEATURE_SUFFIX,
        "--epochs",
        "70",
        "--early_patience",
        "18",
        "--warmup_epochs",
        "5",
        "--scheduler",
        "cosine",
        "--batch",
        "8",
        "--eval_batch",
        "8",
        "--grad_accum_steps",
        "2",
        "--lr",
        "0.00012",
        "--wd",
        "0.0001",
        "--loss_mix",
        "dice:1.0,focal:1.0,lovasz:1.0",
        "--label_smooth",
        "0.05",
        "--thr_metric",
        "min_iou",
        "--selection_metric_mode",
        "at_05",
        "--grad_clip",
        "1.0",
        "--num_workers",
        "2",
        "--seed",
        str(seed),
        "--amp",
        "--no_augment",
        "--use_temp_scaling",
        "--boundary_radius",
        "3",
        "--boundary_factor",
        "3.0",
    ]


def collect_status(models: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        for seed in seeds:
            directory = run_directory(model, seed)
            row: dict[str, Any] = {
                "model": model,
                "seed": seed,
                "complete": run_complete(model, seed),
                "run_dir": str(directory),
            }
            metadata_path = directory / "model_best.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                row["best_metric"] = metadata.get("best_metric")
                row["best_thr"] = metadata.get("best_thr")
                row["temperature"] = metadata.get("temperature", 1.0)
            rows.append(row)
    return rows


def write_status(rows: list[dict[str, Any]], inputs: dict[str, Any]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (OUT_ROOT / "baseline_status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "experiment": EXPERIMENT,
        "updated": now(),
        "patch_csv": str(PATCH_CSV),
        "h5_path": str(H5_PATH),
        "feature_suffix": FEATURE_SUFFIX,
        "inputs": inputs,
        "runs": rows,
    }
    (OUT_ROOT / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    models = parse_items(args.models, ("rf", "unet"))
    seeds = parse_seeds(args.seeds)
    inputs = validate_inputs()
    commands = []
    for model in models:
        for seed in seeds:
            command = rf_command(seed) if model == "rf" else unet_command(seed)
            commands.append(
                {
                    "model": model,
                    "seed": seed,
                    "skip": args.resume and run_complete(model, seed),
                    "command": command,
                }
            )
    if args.dry_run:
        print(json.dumps({"inputs": inputs, "runs": commands}, indent=2))
        return 0
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "baseline_master.pid").write_text(
        str(os.getpid()), encoding="ascii"
    )
    if "unet" in models and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the U-Net baseline.")
    write_log(f"[{now()}] Starting {EXPERIMENT}")
    write_log(f"models={models} seeds={seeds}")
    for item in commands:
        model = str(item["model"])
        seed = int(item["seed"])
        if bool(item["skip"]):
            write_log(f"[skip] {model} seed {seed}: complete")
            continue
        run_directory(model, seed).mkdir(parents=True, exist_ok=True)
        run_stream(list(item["command"]), f"{model.upper()} seed {seed}")
        if not run_complete(model, seed):
            raise RuntimeError(f"{model} seed {seed} did not produce a valid run")
        write_status(collect_status(models, seeds), inputs)
    rows = collect_status(models, seeds)
    write_status(rows, inputs)
    if not all(bool(row["complete"]) for row in rows):
        raise RuntimeError("Baseline matrix ended with incomplete runs")
    write_log(f"[{now()}] COMPLETE {len(rows)}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
