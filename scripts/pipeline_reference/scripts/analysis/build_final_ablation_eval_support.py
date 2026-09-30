"""Build deterministic, class-independent evaluation windows for the v3 ablation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio
from rasterio.windows import Window

from scripts.common.config import OUT, ROOT
from scripts.patches.make_patches import (
    _block_bounds,
    _sanitize,
    assign_split_with_buffer,
)


EXPERIMENT = "final_scene_ablation_031623_v3"
FEATURE_SUFFIX = "_st3b_final_scene_ablation_031623_v3_common"
SCENES = {
    "20030709_qb": "2003",
    "20160825_wv": "2016",
    "20230812_pl": "2023",
}
BRANCHES = (
    "m1_03",
    "m1_16",
    "m1_23",
    "m2_0316",
    "m2_0323",
    "m2_1623",
    "m3_031623",
)
VAL_BLOCKS = {6, 24, 28, 30}
TEST_BLOCKS = {15, 18, 19, 22}
N_BLOCKS_X = 6
N_BLOCKS_Y = 6
BUFFER_PX = 256
TILE = 256
STRIDE = 128

OUT_DIR = OUT / "eval_support" / EXPERIMENT
CSV_PATH = OUT_DIR / "common_eval_windows.csv"
H5_PATH = OUT_DIR / "common_eval_windows.h5"
MANIFEST_PATH = OUT_DIR / "common_eval_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def positions(start: int, end: int, window: int, stride: int) -> list[int]:
    if end - start < window:
        return []
    values = list(range(start, end - window + 1, stride))
    last = end - window
    if not values or values[-1] != last:
        values.append(last)
    return values


def candidate_windows(width: int, height: int) -> list[tuple[int, int]]:
    windows: set[tuple[int, int]] = set()
    for block_id in sorted(TEST_BLOCKS):
        xs, xe, ys, ye = _block_bounds(
            width, height, N_BLOCKS_X, N_BLOCKS_Y, block_id
        )
        for y0 in positions(ys, ye, TILE, STRIDE):
            for x0 in positions(xs, xe, TILE, STRIDE):
                split = assign_split_with_buffer(
                    x0=x0,
                    y0=y0,
                    tile=TILE,
                    width=width,
                    height=height,
                    n_blocks_x=N_BLOCKS_X,
                    n_blocks_y=N_BLOCKS_Y,
                    val_blocks=VAL_BLOCKS,
                    test_blocks=TEST_BLOCKS,
                    val_buffer_px=BUFFER_PX,
                    test_buffer_px=BUFFER_PX,
                )
                if split == "test":
                    windows.add((x0, y0))
    return sorted(windows, key=lambda item: (item[1], item[0]))


def rectangles_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["x0"] < right["x0"] + right["tile"]
        and left["x0"] + left["tile"] > right["x0"]
        and left["y0"] < right["y0"] + right["tile"]
        and left["y0"] + left["tile"] > right["y0"]
    )


def read_branch_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch in BRANCHES:
        path = OUT / "patches" / f"patches_{EXPERIMENT}_{branch}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw["split"] not in {"train", "val"}:
                    continue
                rows.append(
                    {
                        "branch": branch,
                        "scene_id": raw["scene_id"],
                        "split": raw["split"],
                        "x0": int(raw["x0"]),
                        "y0": int(raw["y0"]),
                        "tile": int(raw["tile"]),
                    }
                )
    return rows


def audit_against_training(
    support_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_rows = read_branch_rows()
    overlaps: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in reference_rows:
        by_scene.setdefault(row["scene_id"], []).append(row)
    for support in support_rows:
        for reference in by_scene.get(support["scene_id"], []):
            if rectangles_overlap(support, reference):
                overlaps.append({"support": support, "reference": reference})
                if len(overlaps) >= 20:
                    break
        if len(overlaps) >= 20:
            break
    if overlaps:
        raise RuntimeError(
            "Evaluation support overlaps train/validation windows: "
            + json.dumps(overlaps[:3], indent=2)
        )
    return {
        "reference_train_val_rows_checked": len(reference_rows),
        "cross_split_overlap_pairs": 0,
        "passed": True,
    }


def build() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if H5_PATH.exists():
        H5_PATH.unlink()

    rows: list[dict[str, Any]] = []
    scene_stats: dict[str, Any] = {}
    with h5py.File(H5_PATH, "w") as store:
        patch_group = store.create_group("patches")
        for scene_id, year in SCENES.items():
            feature_path = (
                ROOT / "data" / "features" / f"feat_{scene_id}{FEATURE_SUFFIX}.tif"
            )
            valid_path = (
                ROOT / "data" / "features" / f"valid_{scene_id}{FEATURE_SUFFIX}.tif"
            )
            mask_path = OUT / "training_mask" / f"training_{year}__FULL.tif"
            with (
                rasterio.open(feature_path) as feature,
                rasterio.open(valid_path) as valid_source,
                rasterio.open(mask_path) as mask_source,
            ):
                if not (
                    feature.width == valid_source.width == mask_source.width
                    and feature.height == valid_source.height == mask_source.height
                    and feature.transform
                    == valid_source.transform
                    == mask_source.transform
                    and feature.crs == valid_source.crs == mask_source.crs
                ):
                    raise RuntimeError(f"Grid mismatch for {scene_id}")

                windows = candidate_windows(feature.width, feature.height)
                unique_support = np.zeros(
                    (feature.height, feature.width), dtype=bool
                )
                label_counts = {1: 0, 2: 0}
                kept = 0
                for x0, y0 in windows:
                    window = Window(x0, y0, TILE, TILE)
                    # Match the training HDF5 path exactly: replace non-finite
                    # values, clip to the feature range, and store float16.
                    x = _sanitize(feature.read(window=window).astype(np.float32))
                    labels = mask_source.read(1, window=window).astype(np.uint8)
                    valid = valid_source.read(1, window=window) > 0
                    weights = (valid & np.isin(labels, (1, 2))).astype(np.uint8)
                    if not weights.any():
                        continue

                    key = f"{scene_id}__y{y0}_x{x0}_t{TILE}"
                    group = patch_group.create_group(key)
                    group.attrs["scene_id"] = scene_id
                    group.attrs["split"] = "test"
                    group.create_dataset("x", data=x, compression="lzf")
                    group.create_dataset("y", data=labels, compression="lzf")
                    group.create_dataset("w", data=weights, compression="lzf")
                    rows.append(
                        {
                            "scene_id": scene_id,
                            "feat_tif": str(feature_path),
                            "mask_tif": str(mask_path),
                            "y0": y0,
                            "x0": x0,
                            "tile": TILE,
                            "split": "test",
                        }
                    )
                    view = unique_support[y0 : y0 + TILE, x0 : x0 + TILE]
                    view |= weights > 0
                    kept += 1

                full_labels = mask_source.read(1)
                for label in label_counts:
                    label_counts[label] = int(
                        np.count_nonzero(unique_support & (full_labels == label))
                    )
                scene_stats[scene_id] = {
                    "candidate_windows": len(windows),
                    "kept_windows": kept,
                    "unique_labelled_pixels": int(unique_support.sum()),
                    "class_1_nonveg_pixels": label_counts[1],
                    "class_2_veg_pixels": label_counts[2],
                }

    fields = ["scene_id", "feat_tif", "mask_tif", "y0", "x0", "tile", "split"]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    audit = audit_against_training(rows)
    manifest = {
        "experiment": EXPERIMENT,
        "purpose": "common class-independent unique-pixel evaluation support",
        "scenes": SCENES,
        "tile": TILE,
        "stride": STRIDE,
        "validation_blocks": sorted(VAL_BLOCKS),
        "test_blocks": sorted(TEST_BLOCKS),
        "reciprocal_buffer_px": BUFFER_PX,
        "window_selection": (
            "regular grid; full footprint must be assigned to test after "
            "reciprocal validation buffering; no class-balance criterion"
        ),
        "rows": len(rows),
        "scene_stats": scene_stats,
        "leakage_audit": audit,
        "csv_path": str(CSV_PATH),
        "csv_sha256": sha256_file(CSV_PATH),
        "h5_path": str(H5_PATH),
        "h5_sha256": sha256_file(H5_PATH),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    manifest = build()
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
