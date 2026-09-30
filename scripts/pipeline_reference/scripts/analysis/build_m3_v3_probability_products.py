"""Aggregate five full-scene M3 predictions, validate them and publish canonically."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(r"${PROJECT_ROOT}")
PRED = ROOT / "out/pred/final_scene_ablation_031623_v3_m3_full_hann"
STAGE = (
    ROOT
    / "out/runs/analysis/final_scene_ablation_031623_v3/full_scene_products"
)
CANONICAL = ROOT / "out/runs/analysis/power_v3/prob_roughness_proxy"
EXPORT = Path(r"${LICENSED_EXPORT_ROOT}")
EXPORT_DATA = EXPORT / "dati"
SUPPORT_CSV = (
    ROOT
    / "out/eval_support/final_scene_ablation_031623_v3/common_eval_windows.csv"
)
MODEL_METRICS = (
    ROOT
    / "out/runs/analysis/final_scene_ablation_031623_v3/model_family/"
    "all_model_family_metrics.csv"
)
SCENES = {
    "20030709_qb": "QB 2003",
    "20160825_wv": "WV2 2016",
    "20230812_pl": "PL 2023",
}
REFERENCE_FALLBACKS = {
    # The frozen support CSV retains the historical pre-alignment filename for
    # 2003; this is the registered 2 m reference used by the completed v3 run.
    "20030709_qb": ROOT / "out/training_mask/training_2003__FULL__b-2m.tif",
    "20160825_wv": ROOT / "out/training_mask/training_2016__FULL.tif",
    "20230812_pl": ROOT / "out/training_mask/training_2023__FULL.tif",
}
SEEDS = (123, 231, 312, 423, 531)
NODATA_FLOAT = -9999.0
NODATA_MASK = 255
MAX_SEAM_RATIO = 1.25
EXPECTED_HELDOUT_PIXELS = {
    "20030709_qb": 768_261,
    "20160825_wv": 541_175,
    "20230812_pl": 696_009,
}


def probability_path(seed: int, scene_id: str) -> Path:
    matches = sorted(
        (PRED / f"seed{seed}").glob(
            f"prob_{scene_id}_w512_p128_thr*.tif"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one probability for {scene_id}/seed{seed}, found {matches}"
        )
    return matches[0]


def mask_path(seed: int, scene_id: str) -> Path:
    matches = sorted(
        (PRED / f"seed{seed}").glob(
            f"vegmask_{scene_id}_w512_p128_thr*.tif"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one mask for {scene_id}/seed{seed}, found {matches}"
        )
    return matches[0]


def summary_path(seed: int, scene_id: str) -> Path:
    matches = sorted(
        (PRED / f"seed{seed}").glob(
            f"summary_{scene_id}_w512_p128_thr*.json"
        )
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one summary for {scene_id}/seed{seed}, found {matches}"
        )
    return matches[0]


def read(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as source:
        return source.read(1), source.profile.copy()


def same_grid(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(
        left[key] == right[key]
        for key in ("width", "height", "transform", "crs")
    )


def positions(length: int, window: int = 512, step: int = 256) -> list[int]:
    if length <= window:
        return [0]
    values = list(range(0, length - window + 1, step))
    last = length - window
    if values[-1] != last:
        values.append(last)
    return values


def boundaries(length: int) -> list[int]:
    result: set[int] = set()
    for origin in positions(length):
        result.add(origin)
        result.add(min(length, origin + 512))
    return sorted(value for value in result if 0 < value < length)


def seam_metric(
    array: np.ndarray,
    valid: np.ndarray,
    axis: int,
    seam_boundaries: list[int],
) -> dict[str, float]:
    differences = np.abs(np.diff(array, axis=axis))
    valid_pairs = (
        valid[:, 1:] & valid[:, :-1]
        if axis == 1
        else valid[1:, :] & valid[:-1, :]
    )
    length = array.shape[1] if axis == 1 else array.shape[0]
    seams: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    for boundary in seam_boundaries:
        index = boundary - 1
        mask = valid_pairs[:, index] if axis == 1 else valid_pairs[index, :]
        values = differences[:, index] if axis == 1 else differences[index, :]
        if mask.any():
            seams.append(values[mask])
        for offset in (-64, -32, 32, 64):
            control_index = boundary + offset - 1
            if not 0 <= control_index < length - 1:
                continue
            control_mask = (
                valid_pairs[:, control_index]
                if axis == 1
                else valid_pairs[control_index, :]
            )
            control_values = (
                differences[:, control_index]
                if axis == 1
                else differences[control_index, :]
            )
            if control_mask.any():
                controls.append(control_values[control_mask])
    if not seams or not controls:
        raise RuntimeError("Insufficient pixels for seam diagnostics.")
    seam_values = np.concatenate(seams)
    control_values = np.concatenate(controls)
    control_mean = float(control_values.mean())
    return {
        "seam_mean": float(seam_values.mean()),
        "control_mean": control_mean,
        "ratio": float(seam_values.mean() / max(control_mean, 1e-12)),
    }


def write_float(
    path: Path, array: np.ndarray, profile: dict[str, Any], description: str
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        dtype="float32",
        count=1,
        nodata=NODATA_FLOAT,
        compress="deflate",
        predictor=3,
        tiled=True,
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(array.astype(np.float32), 1)
        destination.set_band_description(1, description)
        destination.update_tags(
            MODEL="SegFormer MiT-B2 M3 v3",
            SEEDS="123,231,312,423,531",
            BLEND_MODE="full_hann",
            WINDOW="512",
            STEP="256",
            STATUS="validated",
        )


def write_classes(
    path: Path,
    array: np.ndarray,
    profile: dict[str, Any],
    descriptions: dict[int, str],
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        dtype="uint8", count=1, nodata=0, compress="deflate", tiled=True
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(array.astype(np.uint8), 1)
        destination.write_colormap(
            1,
            {
                0: (0, 0, 0, 0),
                1: (217, 238, 242, 220),
                2: (130, 201, 160, 220),
                3: (22, 122, 69, 225),
            },
        )
        destination.update_tags(
            **{f"CLASS_{key}": value for key, value in descriptions.items()},
            MODEL="SegFormer MiT-B2 M3 v3",
            SEEDS="123,231,312,423,531",
            BLEND_MODE="full_hann",
            STATUS="validated",
        )


def write_mask(
    path: Path, array: np.ndarray, profile: dict[str, Any]
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        dtype="uint8",
        count=1,
        nodata=NODATA_MASK,
        compress="deflate",
        tiled=True,
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(array.astype(np.uint8), 1)
        destination.update_tags(
            CLASS_1="NONVEG",
            CLASS_2="VEG",
            CONSENSUS="majority of five validation-calibrated seed masks",
            MODEL="SegFormer MiT-B2 M3 v3",
            SEEDS="123,231,312,423,531",
            STATUS="validated",
        )


def reference_masks() -> dict[str, Path]:
    frame = pd.read_csv(SUPPORT_CSV)
    result = {}
    for scene_id, group in frame.groupby("scene_id"):
        values = sorted(set(group["mask_tif"].astype(str)))
        if len(values) != 1:
            raise RuntimeError(f"Reference mask ambiguity for {scene_id}: {values}")
        path = Path(values[0])
        if not path.exists():
            path = REFERENCE_FALLBACKS[str(scene_id)]
        if not path.exists():
            raise FileNotFoundError(f"Missing reference mask for {scene_id}: {path}")
        result[str(scene_id)] = path
    return result


def heldout_support(scene_id: str, shape: tuple[int, int]) -> tuple[np.ndarray, int]:
    """Return the union of the frozen, class-independent evaluation windows."""
    frame = pd.read_csv(SUPPORT_CSV)
    selected = frame[(frame["scene_id"] == scene_id) & (frame["split"] == "test")]
    if len(selected) != 69:
        raise RuntimeError(f"Expected 69 held-out windows for {scene_id}, found {len(selected)}")
    support = np.zeros(shape, dtype=bool)
    for row in selected.itertuples(index=False):
        y0, x0, tile = int(row.y0), int(row.x0), int(row.tile)
        support[y0 : y0 + tile, x0 : x0 + tile] = True
    return support, int(len(selected))


def ece(labels: np.ndarray, probabilities: np.ndarray) -> float:
    target = (labels == 2).astype(np.float64)
    result = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (
            (probabilities >= low) & (probabilities <= high)
            if index == len(edges) - 2
            else (probabilities >= low) & (probabilities < high)
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float(probabilities[selected].mean())
                - float(target[selected].mean())
            )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def publish(files: list[Path]) -> dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    local_backup = CANONICAL / "backups" / f"pre_m3_v3_{stamp}"
    export_backup = EXPORT / "backups" / f"pre_m3_v3_probability_{stamp}"
    local_backup.mkdir(parents=True, exist_ok=False)
    export_backup.mkdir(parents=True, exist_ok=False)

    published = []
    for source in files:
        for base, backup in (
            (CANONICAL, local_backup),
            (EXPORT_DATA, export_backup),
        ):
            target = base / source.name
            if target.is_file():
                shutil.copy2(target, backup / target.name)
            atomic_copy(source, target)
            published.append(str(target))
    return {
        "local_backup": str(local_backup),
        "export_backup": str(export_backup),
        "published": published,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    references = reference_masks()

    qa: dict[str, Any] = {}
    output_files: list[Path] = []
    calibration_rows: list[dict[str, Any]] = []
    cover_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    for scene_id, scene_label in SCENES.items():
        probabilities = []
        masks = []
        summaries = []
        profile: dict[str, Any] | None = None
        for seed in SEEDS:
            probability, candidate = read(probability_path(seed, scene_id))
            mask, mask_profile = read(mask_path(seed, scene_id))
            summary = json.loads(
                summary_path(seed, scene_id).read_text(encoding="utf-8")
            )
            if summary.get("blend_mode") != "full_hann":
                raise RuntimeError(f"Wrong blend mode: {summary_path(seed, scene_id)}")
            if not bool(summary.get("calibration_consistent", False)):
                raise RuntimeError(
                    f"Calibration mismatch: {summary_path(seed, scene_id)}"
                )
            if profile is None:
                profile = candidate
            elif not same_grid(profile, candidate):
                raise RuntimeError(f"Probability grid mismatch for {scene_id}")
            if not same_grid(candidate, mask_profile):
                raise RuntimeError(f"Mask grid mismatch for {scene_id}/seed{seed}")
            probabilities.append(probability.astype(np.float32))
            masks.append(mask.astype(np.uint8))
            summaries.append(summary)

        assert profile is not None
        stack = np.stack(probabilities)
        valid = np.all(np.isfinite(stack) & (stack >= 0) & (stack <= 1), axis=0)
        if not valid.any():
            raise RuntimeError(f"No valid ensemble pixels for {scene_id}")
        for seed_index, probability in enumerate(probabilities):
            seed_valid = (
                np.isfinite(probability)
                & (probability >= 0)
                & (probability <= 1)
            )
            if not np.array_equal(valid, seed_valid):
                raise RuntimeError(
                    f"Seed coverage mismatch for {scene_id}/seed{SEEDS[seed_index]}"
                )

        mean = np.full(valid.shape, NODATA_FLOAT, dtype=np.float32)
        sd = np.full(valid.shape, NODATA_FLOAT, dtype=np.float32)
        mean[valid] = stack[:, valid].mean(axis=0)
        sd[valid] = stack[:, valid].std(axis=0, ddof=1)

        vote_stack = np.stack([mask == 2 for mask in masks])
        consensus = np.full(valid.shape, NODATA_MASK, dtype=np.uint8)
        consensus[valid] = np.where(vote_stack[:, valid].sum(axis=0) >= 3, 2, 1)
        roughness = np.zeros(valid.shape, dtype=np.uint8)
        roughness[valid & (mean < 1.0 / 3.0)] = 1
        roughness[
            valid & (mean >= 1.0 / 3.0) & (mean < 2.0 / 3.0)
        ] = 2
        roughness[valid & (mean >= 2.0 / 3.0)] = 3

        mean_path = STAGE / f"mean_pveg_{scene_id}.tif"
        sd_path = STAGE / f"sd_pveg_{scene_id}.tif"
        mask_out = STAGE / f"consensus_vegmask_{scene_id}.tif"
        roughness_path = STAGE / f"roughness_binned_{scene_id}.tif"
        proxy_path = STAGE / f"roughness_proxy_{scene_id}.tif"
        write_float(mean_path, mean, profile, "Mean calibrated P(VEG), five seeds")
        write_float(sd_path, sd, profile, "P(VEG) sample SD across five seeds")
        write_mask(mask_out, consensus, profile)
        write_classes(
            roughness_path,
            roughness,
            profile,
            {1: "Low P(VEG)", 2: "Medium P(VEG)", 3: "High P(VEG)"},
        )
        write_float(
            proxy_path, mean, profile, "Relative roughness proxy: mean P(VEG)"
        )
        output_files.extend(
            [mean_path, sd_path, mask_out, roughness_path, proxy_path]
        )

        vertical = seam_metric(mean, valid, 1, boundaries(profile["width"]))
        horizontal = seam_metric(mean, valid, 0, boundaries(profile["height"]))
        if vertical["ratio"] > MAX_SEAM_RATIO:
            raise RuntimeError(
                f"Vertical seam ratio failed for {scene_id}: {vertical['ratio']}"
            )
        if horizontal["ratio"] > MAX_SEAM_RATIO:
            raise RuntimeError(
                f"Horizontal seam ratio failed for {scene_id}: {horizontal['ratio']}"
            )

        reference, reference_profile = read(references[scene_id])
        if not same_grid(profile, reference_profile):
            raise RuntimeError(f"Reference grid mismatch for {scene_id}")
        test_support, n_test_windows = heldout_support(scene_id, reference.shape)
        labelled = valid & test_support & np.isin(reference, (1, 2))
        if int(labelled.sum()) != EXPECTED_HELDOUT_PIXELS[scene_id]:
            raise RuntimeError(
                f"Frozen held-out support mismatch for {scene_id}: "
                f"{int(labelled.sum())} != {EXPECTED_HELDOUT_PIXELS[scene_id]}"
            )
        labels = reference[labelled]
        probs = mean[labelled]
        binary = (labels == 2).astype(np.float64)
        calibration_rows.append(
            {
                "scene_id": scene_id,
                "scene_label": scene_label,
                "n_test_px": int(labelled.sum()),
                "n_test_blocks": 4,
                "n_test_windows": n_test_windows,
                "support_definition": "union of frozen class-independent held-out windows; unique pixels",
                "ECE": round(ece(labels, probs), 6),
                "Brier": round(float(np.mean((probs - binary) ** 2)), 6),
                "mean_prob": round(float(probs.mean()), 6),
                "true_veg_frac": round(float(binary.mean()), 6),
            }
        )
        thresholds = [float(item["thr"]) for item in summaries]
        temperatures = [float(item["temperature"]) for item in summaries]
        seed_hard = [
            float(np.mean(mask[valid] == 2) * 100.0) for mask in masks
        ]
        operational = float(np.mean(consensus[valid] == 2) * 100.0)
        soft = float(mean[valid].mean() * 100.0)
        cover_rows.append(
            {
                "scene_id": scene_id,
                "scene_label": scene_label,
                "n_valid_px": int(valid.sum()),
                "seed_thresholds": "|".join(f"{value:.3f}" for value in thresholds),
                "seed_temperatures": "|".join(
                    f"{value:.3f}" for value in temperatures
                ),
                "hard_operational_mean_pct": round(operational, 4),
                "hard_operational_sd_pct": round(
                    float(np.std(seed_hard, ddof=1)), 4
                ),
                "meanp_ge_0p667_pct": round(
                    float(np.mean(mean[valid] >= 2.0 / 3.0) * 100.0), 4
                ),
                "soft_cover_pct": round(soft, 4),
                "soft_minus_operational_hard_pp": round(
                    soft - operational, 4
                ),
            }
        )
        for code, name in ((2, "VEG"), (1, "NONVEG")):
            selected = labelled & (reference == code)
            values = mean[selected]
            distribution_rows.append(
                {
                    "scene_id": scene_id,
                    "scene_label": scene_label,
                    "stratum": name,
                    "n_px": int(values.size),
                    "mean_pveg": round(float(values.mean()), 6),
                    "median_pveg": round(float(np.median(values)), 6),
                    "p10": round(float(np.quantile(values, 0.1)), 6),
                    "p90": round(float(np.quantile(values, 0.9)), 6),
                    "frac_above_0p5": round(float(np.mean(values >= 0.5)), 6),
                }
            )

        qa[scene_id] = {
            "valid_pixels": int(valid.sum()),
            "mean_pveg": float(mean[valid].mean()),
            "mean_seed_sd": float(sd[valid].mean()),
            "vertical_seam": vertical,
            "horizontal_seam": horizontal,
            "seed_probability_paths": [
                str(probability_path(seed, scene_id)) for seed in SEEDS
            ],
            "seed_thresholds": thresholds,
            "seed_temperatures": temperatures,
        }

    metrics = pd.read_csv(MODEL_METRICS)
    segformer_pooled = metrics[
        (metrics["model"] == "segformer")
        & (metrics["scene_id"] == "ALL_SCENES")
    ]
    if tuple(sorted(segformer_pooled["seed"].astype(int))) != SEEDS:
        raise RuntimeError("Five SegFormer model-family metrics are required.")
    qa["common_support_pooled_min_iou"] = {
        "mean": float(segformer_pooled["min_iou"].mean()),
        "sd": float(segformer_pooled["min_iou"].std(ddof=1)),
    }

    calibration_path = STAGE / "calibration_metrics.csv"
    cover_path = STAGE / "hard_vs_soft_cover.csv"
    distribution_path = STAGE / "sanity_pveg_distribution.csv"
    write_csv(calibration_path, calibration_rows)
    write_csv(cover_path, cover_rows)
    write_csv(distribution_path, distribution_rows)
    output_files.extend([calibration_path, cover_path, distribution_path])

    publish_result = publish(output_files) if args.publish else None
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": "final_scene_ablation_031623_v3",
        "model": "M3 SegFormer MiT-B2",
        "seeds": list(SEEDS),
        "scenes": list(SCENES),
        "seam_ratio_limit": MAX_SEAM_RATIO,
        "qa": qa,
        "outputs": [str(path) for path in output_files],
        "published": bool(args.publish),
        "publish_result": publish_result,
    }
    manifest_path = STAGE / "full_scene_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
