#!/usr/bin/env python3
"""Deduplicated unique-pixel evaluation for SegFormer, U-Net, and Random Forest."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import rasterio
import torch

from scripts.models.model_input import adapt_batch_torch
from scripts.train.train_segformer import make_model


ROOT = Path(r"${PROJECT_ROOT}")
PATCH_CSV = (
    ROOT
    / "out/patches/patches_replacement_2003_paper_v2_replacement03_matched.csv"
)
H5_PATH = (
    ROOT
    / "out/patches_bin/"
    "STORE_replacement_2003_paper_v2_replacement03_matched_st3b_t256.h5"
)
SEGFORMER_ROOT = (
    ROOT
    / "out/runs/train/replacement_2003_paper_v2_matched/stage2/"
    "PAPERV2_M3_REPLACEMENT03_MATCHED"
)
BASELINE_ROOT = ROOT / "out/runs/train/baseline_paper_v2_final"
OUT = ROOT / "out/runs/analysis/paper_v2_final/model_family"
TEST_BLOCKS = (15, 18, 19, 22)
N_BLOCKS = 6
NODATA = np.nan


def find_single(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one match for {pattern} under {root}, found {len(matches)}")
    return matches[0]


def run_dir(model: str, seed: int) -> Path:
    if model == "segformer":
        return find_single(SEGFORMER_ROOT, f"*b8_s{seed}_ga2_augoff")
    if model == "unet":
        return BASELINE_ROOT / "unet/paper_v2_final" / f"seed{seed}"
    if model == "rf":
        return BASELINE_ROOT / "tabular/rf/paper_v2_final" / f"seed{seed}"
    raise KeyError(model)


def block_bounds(width: int, height: int, block_id: int) -> tuple[int, int, int, int]:
    by, bx = divmod(block_id, N_BLOCKS)
    return (
        math.floor(bx * width / N_BLOCKS),
        math.floor((bx + 1) * width / N_BLOCKS),
        math.floor(by * height / N_BLOCKS),
        math.floor((by + 1) * height / N_BLOCKS),
    )


def test_mask(shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for block in TEST_BLOCKS:
        x0, x1, y0, y1 = block_bounds(shape[1], shape[0], block)
        result[y0:y1, x0:x1] = True
    return result


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    target = labels == 2
    prediction = probabilities >= threshold
    tp = int(np.count_nonzero(prediction & target))
    fp = int(np.count_nonzero(prediction & ~target))
    fn = int(np.count_nonzero(~prediction & target))
    tn = int(np.count_nonzero(~prediction & ~target))
    iou_veg = tp / max(1, tp + fp + fn)
    iou_nonveg = tn / max(1, tn + fp + fn)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    binary = target.astype(np.float64)
    for low, high in zip(edges[:-1], edges[1:]):
        selector = (
            (probabilities >= low) & (probabilities <= high)
            if high == 1.0
            else (probabilities >= low) & (probabilities < high)
        )
        if selector.any():
            ece += selector.mean() * abs(
                float(probabilities[selector].mean()) - float(binary[selector].mean())
            )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou_veg": float(iou_veg),
        "iou_nonveg": float(iou_nonveg),
        "min_iou": float(min(iou_veg, iou_nonveg)),
        "macro_iou": float((iou_veg + iou_nonveg) / 2),
        "balanced_accuracy": float(
            (
                tp / max(1, tp + fn)
                + tn / max(1, tn + fp)
            )
            / 2
        ),
        "ece_15": float(ece),
        "brier": float(np.mean((probabilities - binary) ** 2)),
    }


def deep_model(directory: Path) -> tuple[torch.nn.Module, dict[str, Any], torch.device]:
    metadata = json.loads((directory / "model_best.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(
        str(metadata["arch"]),
        str(metadata["encoder"]),
        int(metadata["in_ch"]),
        encoder_weights=str(metadata.get("encoder_weights", "none")),
    ).to(device)
    checkpoint = torch.load(directory / "model_best.pt", map_location=device, weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, metadata, device


def patch_key(row: pd.Series) -> str:
    return (
        f"{row['scene_id']}__y{int(row['y0'])}_x{int(row['x0'])}_"
        f"t{int(row['tile'])}"
    )


def initialize_scene_arrays(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output = {}
    for scene_id, group in df.groupby("scene_id"):
        mask_path = Path(str(group.iloc[0]["mask_tif"]))
        with rasterio.open(mask_path) as source:
            shape = (source.height, source.width)
        output[str(scene_id)] = {
            "sum": np.zeros(shape, dtype=np.float32),
            "count": np.zeros(shape, dtype=np.uint16),
            "label": np.zeros(shape, dtype=np.uint8),
            "test": test_mask(shape),
        }
    return output


def add_patch(
    arrays: dict[str, dict[str, Any]],
    row: pd.Series,
    probability: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> None:
    scene = arrays[str(row["scene_id"])]
    y0, x0, tile = int(row["y0"]), int(row["x0"]), int(row["tile"])
    y1, x1 = y0 + tile, x0 + tile
    valid = (
        (weights > 0)
        & np.isin(labels, (1, 2))
        & scene["test"][y0:y1, x0:x1]
        & np.isfinite(probability)
    )
    view_sum = scene["sum"][y0:y1, x0:x1]
    view_count = scene["count"][y0:y1, x0:x1]
    view_label = scene["label"][y0:y1, x0:x1]
    view_sum[valid] += probability[valid].astype(np.float32)
    view_count[valid] += 1
    existing = view_label[valid]
    incoming = labels[valid].astype(np.uint8)
    if np.any((existing > 0) & (existing != incoming)):
        raise RuntimeError("Conflicting labels for a duplicated global pixel.")
    view_label[valid] = incoming


def infer_deep(
    model_name: str,
    seed: int,
    df: pd.DataFrame,
    arrays: dict[str, dict[str, Any]],
    batch_size: int,
) -> tuple[float, float]:
    directory = run_dir(model_name, seed)
    model, metadata, device = deep_model(directory)
    temperature = (
        float(metadata.get("temperature", 1.0))
        if bool(metadata.get("temp_scaling_done", False))
        else 1.0
    )
    threshold = float(metadata.get("best_thr", 0.5))
    adapter = str(metadata.get("input_adapter", "identity"))
    with h5py.File(H5_PATH, "r") as store:
        for start in range(0, len(df), batch_size):
            batch_rows = df.iloc[start : start + batch_size]
            xs, ys, ws = [], [], []
            for _, row in batch_rows.iterrows():
                group = store["patches"][patch_key(row)]
                xs.append(group["x"][...].astype(np.float32))
                ys.append(group["y"][...])
                ws.append(group["w"][...])
            tensor = torch.from_numpy(np.stack(xs)).to(device, non_blocking=True)
            tensor = adapt_batch_torch(tensor, adapter)
            with torch.no_grad(), torch.autocast(
                device_type="cuda", enabled=device.type == "cuda"
            ):
                logits = model(tensor)
                probs = torch.sigmoid(logits / max(temperature, 1e-6))
            probabilities = probs.detach().float().cpu().numpy()
            if probabilities.ndim == 4:
                probabilities = probabilities[:, 0]
            for (_, row), probability, labels, weights in zip(
                batch_rows.iterrows(), probabilities, ys, ws
            ):
                add_patch(arrays, row, probability, labels, weights)
    return threshold, temperature


def infer_rf(
    seed: int,
    df: pd.DataFrame,
    arrays: dict[str, dict[str, Any]],
) -> tuple[float, float]:
    directory = run_dir("rf", seed)
    metadata = json.loads((directory / "model_best.json").read_text(encoding="utf-8"))
    with (directory / "model.pkl").open("rb") as handle:
        model = pickle.load(handle)
    with h5py.File(H5_PATH, "r") as store:
        for _, row in df.iterrows():
            group = store["patches"][patch_key(row)]
            x = group["x"][...].astype(np.float32)
            labels = group["y"][...]
            weights = group["w"][...]
            valid = (weights > 0) & np.isin(labels, (1, 2))
            probability = np.full(labels.shape, np.nan, dtype=np.float32)
            if valid.any():
                features = np.moveaxis(x, 0, -1)[valid]
                probability[valid] = model.predict_proba(features)[:, 1]
            add_patch(arrays, row, probability, labels, weights)
    return float(metadata.get("best_thr", 0.5)), 1.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(model_name: str, seed: int, batch_size: int) -> None:
    df = pd.read_csv(PATCH_CSV)
    df = df[df["split"].astype(str).str.lower().str.startswith("te")].reset_index(drop=True)
    arrays = initialize_scene_arrays(df)
    if model_name == "rf":
        threshold, temperature = infer_rf(seed, df, arrays)
    else:
        threshold, temperature = infer_deep(model_name, seed, df, arrays, batch_size)

    rows = []
    cells = []
    pooled_labels = []
    pooled_probabilities = []
    for scene_id, scene in arrays.items():
        selector = (scene["count"] > 0) & np.isin(scene["label"], (1, 2))
        labels = scene["label"][selector]
        probabilities = scene["sum"][selector] / scene["count"][selector]
        pooled_labels.append(labels)
        pooled_probabilities.append(probabilities)
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "scene_id": scene_id,
                "threshold": threshold,
                "temperature": temperature,
                "n_unique_pixels": int(selector.sum()),
                **metrics(labels, probabilities, threshold),
            }
        )
        height, width = selector.shape
        for block in TEST_BLOCKS:
            x0, x1, y0, y1 = block_bounds(width, height, block)
            for yy in range(y0, y1, 256):
                for xx in range(x0, x1, 256):
                    y2, x2 = min(y1, yy + 256), min(x1, xx + 256)
                    local = selector[yy:y2, xx:x2]
                    if int(local.sum()) < 512:
                        continue
                    local_labels = scene["label"][yy:y2, xx:x2][local]
                    local_probabilities = (
                        scene["sum"][yy:y2, xx:x2][local]
                        / scene["count"][yy:y2, xx:x2][local]
                    )
                    cells.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "scene_id": scene_id,
                            "block_id": block,
                            "cell_x": xx // 256,
                            "cell_y": yy // 256,
                            "n_unique_pixels": int(local.sum()),
                            **metrics(local_labels, local_probabilities, threshold),
                        }
                    )
    pooled_labels_array = np.concatenate(pooled_labels)
    pooled_probability_array = np.concatenate(pooled_probabilities)
    rows.append(
        {
            "model": model_name,
            "seed": seed,
            "scene_id": "ALL_SCENES",
            "threshold": threshold,
            "temperature": temperature,
            "n_unique_pixels": int(pooled_labels_array.size),
            **metrics(pooled_labels_array, pooled_probability_array, threshold),
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / f"unique_pixel_metrics_{model_name}_s{seed}.csv", rows)
    write_csv(OUT / f"unique_pixel_cells_{model_name}_s{seed}.csv", cells)
    manifest = {
        "model": model_name,
        "seed": seed,
        "run_dir": str(run_dir(model_name, seed)),
        "patch_csv": str(PATCH_CSV),
        "h5_path": str(H5_PATH),
        "test_blocks": list(TEST_BLOCKS),
        "deduplication": "mean probability for duplicate global pixel coordinates",
        "threshold": threshold,
        "temperature": temperature,
    }
    (OUT / f"unique_pixel_manifest_{model_name}_s{seed}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows[-1], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("segformer", "unet", "rf"), required=True)
    parser.add_argument("--seed", type=int, choices=(123, 231, 312), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    evaluate(args.model, args.seed, args.batch_size)


if __name__ == "__main__":
    main()
