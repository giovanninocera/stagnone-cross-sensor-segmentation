#!/usr/bin/env python3
"""Build dual-support structural-selection products.

Diagnostic support:
  cleaned supervised traces, used for optical status and shift robustness.

Structural support:
  spectral-growth footprints, used only after the corresponding core object
  passes the endpoint-retention and relative-shift criteria.

The paper-facing structural support stays separate from P(VEG). Binned and
continuous combinations are diagnostic composites only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage


ROOT = Path(r"${PROJECT_ROOT}")
RUN = ROOT / "out" / "runs" / "analysis" / "paper_v3_hybrid"
DEFAULT_CONFIG = ROOT / "configs" / "analysis" / "hybrid_core_footprint.json"

CLASSIFICATION_CSV = RUN / "hybrid_core_classification.csv"
SCENARIO_CSV = RUN / "hybrid_shift_scenarios.csv"
SELECTED_IDS_RASTER = RUN / "selected_structural_footprints_2023.tif"
DIAGNOSTIC_2003 = RUN / "diagnostic_binned_composite_2003.tif"
DIAGNOSTIC_2023 = RUN / "diagnostic_binned_composite_2023.tif"
SUMMARY_JSON = RUN / "hybrid_summary.json"
SUMMARY_CSV = RUN / "hybrid_summary.csv"
REPORT = RUN / "HYBRID_METHOD_REPORT.md"
STRUCTURAL_MANIFEST = RUN / "structural_selection_manifest.json"

PIXEL_SIZE_M = 2.0
R_EXCL = 2
R1_OUT = 6
R2_OUT = 12
STRUCTURE = np.ones((3, 3), dtype=np.uint8)
RAW_DIRECTIONS = (
    ("E", 0.0, 1.0),
    ("W", 0.0, -1.0),
    ("S", 1.0, 0.0),
    ("N", -1.0, 0.0),
    ("SE", 1.0, 1.0),
    ("SW", 1.0, -1.0),
    ("NE", -1.0, 1.0),
    ("NW", -1.0, -1.0),
)

RETAINED_CLASSES = frozenset(
    {"retained_vegetated_endpoint", "retained_bare_endpoint"}
)
CLASS_ORDER = (
    "retained_vegetated_endpoint",
    "retained_bare_endpoint",
    "homogenized_vegetated",
    "homogenized_bare",
    "unclassified",
)


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


D_EXCL = disk(R_EXCL)
D1 = disk(R1_OUT)
D2 = disk(R2_OUT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="JSON file containing inputs, thresholds and shift-gate settings.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 2:
        raise ValueError("Expected hybrid configuration schema_version=2.")
    for section in ("inputs", "classification", "coregistration", "diagnostic_bins"):
        if section not in payload or not isinstance(payload[section], dict):
            raise ValueError(f"Missing configuration section: {section}")

    rules = payload["classification"]
    collapse_ratio = float(rules["collapse_ratio"])
    minimum_signals = int(rules["minimum_collapsed_signals"])
    endpoint_threshold = float(rules["endpoint_pveg_threshold"])
    minimum_decline = float(rules["retained_bare_minimum_pveg_decline"])
    minimum_c1 = float(rules["retained_vegetated_minimum_modern_c1"])
    if not 0 < collapse_ratio < 1:
        raise ValueError("collapse_ratio must lie strictly between 0 and 1.")
    if minimum_signals not in (1, 2, 3):
        raise ValueError("minimum_collapsed_signals must be 1, 2 or 3.")
    if not 0 < endpoint_threshold < 1:
        raise ValueError("endpoint_pveg_threshold must lie between 0 and 1.")
    if not 0 <= minimum_decline <= 1:
        raise ValueError("retained_bare_minimum_pveg_decline must be in [0, 1].")
    if minimum_c1 < 0:
        raise ValueError("retained_vegetated_minimum_modern_c1 must be non-negative.")

    gate = payload["coregistration"]
    if gate.get("direction_vector_normalization") != "euclidean_unit":
        raise ValueError("Only Euclidean-unit direction vectors are supported.")
    primary = tuple(float(value) for value in gate["primary_shift_magnitudes_m"])
    stress = tuple(float(value) for value in gate["stress_shift_magnitudes_m"])
    if not primary or any(value <= 0 for value in primary + stress):
        raise ValueError("All configured nonzero shift magnitudes must be positive.")
    if len(set(primary + stress)) != len(primary + stress):
        raise ValueError("Primary and stress shift magnitudes must be unique.")
    required_fraction = float(gate["required_primary_fraction"])
    if not 0 < required_fraction <= 1:
        raise ValueError("required_primary_fraction must lie in (0, 1].")
    sensitivities = tuple(float(value) for value in gate["sensitivity_fractions"])
    if not sensitivities or any(value <= 0 or value > 1 for value in sensitivities):
        raise ValueError("sensitivity_fractions must contain values in (0, 1].")

    bins = payload["diagnostic_bins"]
    low = float(bins["low_upper_probability"])
    high = float(bins["medium_upper_probability"])
    if not 0 < low < high < 1:
        raise ValueError("Diagnostic probability boundaries must satisfy 0 < low < high < 1.")
    return payload


def input_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = {key: resolve_path(value) for key, value in config["inputs"].items()}
    required = {
        "core_ids",
        "footprint_ids",
        "water",
        "rboa_2003",
        "rboa_2023",
        "pveg_2003",
        "pveg_2023",
    }
    missing_keys = required - set(paths)
    if missing_keys:
        raise ValueError(f"Missing configured input paths: {sorted(missing_keys)}")
    missing_files = [str(paths[key]) for key in sorted(required) if not paths[key].is_file()]
    if missing_files:
        raise FileNotFoundError(f"Missing configured structural inputs: {missing_files}")
    return paths


def normalized_directions() -> tuple[tuple[str, float, float], ...]:
    output: list[tuple[str, float, float]] = []
    for name, dy, dx in RAW_DIRECTIONS:
        norm = math.hypot(dy, dx)
        if norm <= 0:
            raise ValueError(f"Direction {name} has zero length.")
        output.append((name, dy / norm, dx / norm))
    return tuple(output)


def build_shift_scenarios(
    gate: dict[str, Any], pixel_size_m: float = PIXEL_SIZE_M
) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = [
        {
            "scenario": "shift_0m",
            "scenario_group": "zero",
            "shift_m": 0.0,
            "direction": "NONE",
            "shift_dy_m": 0.0,
            "shift_dx_m": 0.0,
            "dy_px": 0.0,
            "dx_px": 0.0,
            "primary_scenario": True,
        }
    ]
    groups = (
        ("primary", gate["primary_shift_magnitudes_m"], True),
        ("stress", gate["stress_shift_magnitudes_m"], False),
    )
    for group, magnitudes, primary in groups:
        for shift_m_raw in magnitudes:
            shift_m = float(shift_m_raw)
            for direction, unit_dy, unit_dx in normalized_directions():
                dy_m = unit_dy * shift_m
                dx_m = unit_dx * shift_m
                if not math.isclose(math.hypot(dy_m, dx_m), shift_m, abs_tol=1e-12):
                    raise RuntimeError("Direction normalization failed.")
                scenarios.append(
                    {
                        "scenario": f"shift_{shift_m:g}m_{direction}",
                        "scenario_group": group,
                        "shift_m": shift_m,
                        "direction": direction,
                        "shift_dy_m": dy_m,
                        "shift_dx_m": dx_m,
                        "dy_px": dy_m / pixel_size_m,
                        "dx_px": dx_m / pixel_size_m,
                        "primary_scenario": primary,
                    }
                )
    return scenarios


def required_scenarios(required_fraction: float, scenario_count: int) -> int:
    return math.ceil(required_fraction * scenario_count)


def endpoint_optical_class(value: float, threshold: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    return "vegetated" if value >= threshold else "bare"


def robust_scale(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        q25, q75 = np.percentile(values, [25, 75])
        scale = float((q75 - q25) / 1.349)
    return scale if np.isfinite(scale) and scale >= 1e-9 else 1.0


def contrast(object_median: np.ndarray, ring_values: np.ndarray) -> float:
    if ring_values.shape[1] == 0:
        return float("nan")
    ring_median = np.median(ring_values, axis=1)
    standardized = [
        (object_median[index] - ring_median[index])
        / robust_scale(ring_values[index])
        for index in range(3)
    ]
    return float(np.linalg.norm(standardized) / 2.0)


def read_single(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        return source.read(1), source.profile.copy()


def read_bands(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as source:
        return source.read()[:3].astype(np.float32), source.profile.copy()


def ensure_same_grid(reference: dict, candidate: dict, label: str) -> None:
    differences = [
        key
        for key in ("width", "height", "transform", "crs")
        if reference[key] != candidate[key]
    ]
    if differences:
        raise RuntimeError(f"{label} grid mismatch: {differences}")


def prepare_object_supports(
    labels: np.ndarray, footprint: np.ndarray, water: np.ndarray
) -> list[dict[str, object]]:
    supports: list[dict[str, object]] = []
    for object_id, object_slice in enumerate(ndimage.find_objects(labels), start=1):
        if object_slice is None:
            raise RuntimeError(f"Missing object {object_id}.")
        r0 = max(0, object_slice[0].start - R2_OUT - 1)
        r1 = min(labels.shape[0], object_slice[0].stop + R2_OUT + 1)
        c0 = max(0, object_slice[1].start - R2_OUT - 1)
        c1 = min(labels.shape[1], object_slice[1].stop + R2_OUT + 1)
        local_slice = (slice(r0, r1), slice(c0, c1))
        object_mask = labels[local_slice] == object_id
        local_water = water[local_slice]
        local_footprint = footprint[local_slice]
        exclusion = ndimage.binary_dilation(object_mask, D_EXCL)
        dilation_1 = ndimage.binary_dilation(object_mask, D1)
        dilation_2 = ndimage.binary_dilation(object_mask, D2)
        supports.append(
            {
                "object_id": object_id,
                "slice": local_slice,
                "object": object_mask,
                "ring1": dilation_1 & ~exclusion & local_water & ~local_footprint,
                "ring2": dilation_2 & ~dilation_1 & local_water & ~local_footprint,
            }
        )
    return supports


def scene_metrics(
    bands: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    supports: list[dict[str, object]],
) -> dict[int, dict[str, float]]:
    brightness = bands.mean(axis=0)
    output: dict[int, dict[str, float]] = {}
    for support in supports:
        object_id = int(support["object_id"])
        local_slice = support["slice"]
        object_mask = support["object"]
        local_valid = valid[local_slice]
        object_valid = object_mask & local_valid
        if int(object_valid.sum()) < 5:
            output[object_id] = {
                "C1": float("nan"),
                "C2": float("nan"),
                "TEX": float("nan"),
                "pveg": float("nan"),
                "n_pixels": int(object_valid.sum()),
            }
            continue
        local_bands = bands[:, local_slice[0], local_slice[1]]
        object_median = np.median(local_bands[:, object_valid], axis=1)
        ring1 = support["ring1"] & local_valid
        ring2 = support["ring2"] & local_valid
        object_brightness = brightness[local_slice][object_valid]
        pveg_values = probability[local_slice][object_valid]
        output[object_id] = {
            "C1": contrast(object_median, local_bands[:, ring1]),
            "C2": contrast(object_median, local_bands[:, ring2]),
            "TEX": float(
                np.std(object_brightness) / (np.mean(object_brightness) + 1e-9)
            ),
            "pveg": float(np.median(pveg_values[np.isfinite(pveg_values)])),
            "n_pixels": int(object_valid.sum()),
        }
    return output


def classify(
    baseline: dict[str, float],
    modern: dict[str, float],
    rules: dict[str, Any],
) -> tuple[str, int]:
    collapsed = []
    for signal in ("C1", "C2", "TEX"):
        old = baseline[signal]
        new = modern[signal]
        collapsed.append(
            bool(
                np.isfinite(old)
                and np.isfinite(new)
                and old > 1e-6
                and new < float(rules["collapse_ratio"]) * old
            )
        )
    collapse_score = int(sum(collapsed))
    homogenized = collapse_score >= int(rules["minimum_collapsed_signals"])
    p03 = baseline["pveg"]
    p23 = modern["pveg"]
    pveg_threshold = float(rules["endpoint_pveg_threshold"])
    if not np.isfinite(p03) or not np.isfinite(p23):
        return "unclassified", collapse_score
    if homogenized:
        return (
            "homogenized_vegetated"
            if p23 >= pveg_threshold
            else "homogenized_bare",
            collapse_score,
        )
    if p23 < pveg_threshold and (p03 - p23) >= float(
        rules["retained_bare_minimum_pveg_decline"]
    ):
        return "retained_bare_endpoint", collapse_score
    if (
        p23 >= pveg_threshold
        and np.isfinite(modern["C1"])
        and modern["C1"]
        >= float(rules["retained_vegetated_minimum_modern_c1"])
    ):
        return "retained_vegetated_endpoint", collapse_score
    return "unclassified", collapse_score


def shift_surface(
    bands: np.ndarray,
    probability: np.ndarray,
    valid: np.ndarray,
    dy_px: float,
    dx_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dy_px == 0 and dx_px == 0:
        return bands, probability, valid
    shifted_bands = np.stack(
        [
            ndimage.shift(
                band,
                shift=(dy_px, dx_px),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for band in bands
        ],
        axis=0,
    )
    clean_probability = np.where(np.isfinite(probability), probability, 0.0)
    shifted_probability = ndimage.shift(
        clean_probability,
        shift=(dy_px, dx_px),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    shifted_weight = ndimage.shift(
        valid.astype(np.float32),
        shift=(dy_px, dx_px),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    shifted_valid = shifted_weight >= 0.999
    shifted_probability[~shifted_valid] = np.nan
    return shifted_bands, shifted_probability, shifted_valid


def object_morphology(labels: np.ndarray) -> dict[int, dict[str, float | str]]:
    indices = np.arange(1, int(labels.max()) + 1)
    yy, xx = np.indices(labels.shape)
    area = np.array(ndimage.sum(labels > 0, labels, indices), dtype=float)
    sum_x = np.array(ndimage.sum(xx, labels, indices), dtype=float)
    sum_y = np.array(ndimage.sum(yy, labels, indices), dtype=float)
    sum_xx = np.array(ndimage.sum(xx * xx, labels, indices), dtype=float)
    sum_yy = np.array(ndimage.sum(yy * yy, labels, indices), dtype=float)
    sum_xy = np.array(ndimage.sum(xx * yy, labels, indices), dtype=float)
    mean_x = sum_x / area
    mean_y = sum_y / area
    var_x = sum_xx / area - mean_x * mean_x
    var_y = sum_yy / area - mean_y * mean_y
    covariance = sum_xy / area - mean_x * mean_y
    discriminant = np.sqrt((var_x - var_y) ** 2 + 4 * covariance**2)
    major = (var_x + var_y + discriminant) / 2
    minor = (var_x + var_y - discriminant) / 2
    elongation = np.sqrt((major + 0.25) / (np.maximum(minor, 0) + 0.25))
    return {
        int(object_id): {
            "elongation": float(elongation[object_id - 1]),
            "shape_type": (
                "compact" if elongation[object_id - 1] < 2.5 else "elongated"
            ),
        }
        for object_id in indices
    }


def write_raster(
    path: Path,
    values: np.ndarray,
    profile: dict,
    dtype: str,
    nodata: int,
    tags: dict[str, str],
) -> None:
    output_profile = profile.copy()
    output_profile.update(
        dtype=dtype,
        count=1,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(values.astype(dtype), 1)
        destination.update_tags(**tags)
        factors = [
            factor
            for factor in (2, 4, 8, 16)
            if min(values.shape) // factor >= 1
        ]
        destination.build_overviews(factors, rasterio.enums.Resampling.nearest)
        destination.update_tags(ns="rio_overview", resampling="nearest")


def pveg_classes(
    probability: np.ndarray, valid: np.ndarray, low: float, high: float
) -> np.ndarray:
    result = np.zeros(probability.shape, dtype=np.uint8)
    result[valid & (probability < low)] = 1
    result[valid & (probability >= low) & (probability < high)] = 2
    result[valid & (probability >= high)] = 3
    return result


def hectares(mask: np.ndarray) -> float:
    return round(float(mask.sum() * PIXEL_SIZE_M * PIXEL_SIZE_M / 10000.0), 4)


def main(config_path: Path) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    config_path = config_path.resolve()
    config = load_config(config_path)
    paths = input_paths(config)
    rules = config["classification"]
    gate = config["coregistration"]
    bins = config["diagnostic_bins"]
    config_hash = sha256(config_path)

    core_ids, reference_profile = read_single(paths["core_ids"])
    footprint_ids, footprint_profile = read_single(paths["footprint_ids"])
    water_raw, water_profile = read_single(paths["water"])
    bands_2003, bands_2003_profile = read_bands(paths["rboa_2003"])
    bands_2023, bands_2023_profile = read_bands(paths["rboa_2023"])
    pveg_2003, pveg_2003_profile = read_single(paths["pveg_2003"])
    pveg_2023, pveg_2023_profile = read_single(paths["pveg_2023"])
    for label, profile in [
        ("spectral footprints", footprint_profile),
        ("water", water_profile),
        ("RBOA 2003", bands_2003_profile),
        ("RBOA 2023", bands_2023_profile),
        ("PVEG 2003", pveg_2003_profile),
        ("PVEG 2023", pveg_2023_profile),
    ]:
        ensure_same_grid(reference_profile, profile, label)

    water = water_raw == 1
    core_ids = np.where(water, core_ids, 0).astype(np.int32)
    footprint_ids = np.where(water, footprint_ids, 0).astype(np.int32)
    object_count = int(core_ids.max())
    if object_count != 1011 or int(footprint_ids.max()) != object_count:
        raise RuntimeError("Core and footprint object IDs are not the expected 1,011.")
    core_mask = core_ids > 0
    supports = prepare_object_supports(core_ids, core_mask, water)

    valid_2003 = (
        water
        & np.all(np.isfinite(bands_2003), axis=0)
        & np.all(bands_2003 > 0, axis=0)
        & np.isfinite(pveg_2003)
        & (pveg_2003 >= 0)
        & (pveg_2003 <= 1)
    )
    valid_2023 = (
        water
        & np.all(np.isfinite(bands_2023), axis=0)
        & np.all(bands_2023 > 0, axis=0)
        & np.isfinite(pveg_2023)
        & (pveg_2023 >= 0)
        & (pveg_2023 <= 1)
    )
    baseline_metrics = scene_metrics(
        bands_2003, pveg_2003.astype(np.float32), valid_2003, supports
    )

    scenarios = build_shift_scenarios(gate)

    scenario_rows: list[dict[str, object]] = []
    classifications_by_object: dict[int, list[dict[str, object]]] = {
        object_id: [] for object_id in range(1, object_count + 1)
    }
    scenario_summaries: list[dict[str, object]] = []
    for scenario in scenarios:
        scenario_name = str(scenario["scenario"])
        shift_m = float(scenario["shift_m"])
        direction = str(scenario["direction"])
        primary = bool(scenario["primary_scenario"])
        shifted_bands, shifted_pveg, shifted_valid = shift_surface(
            bands_2023,
            pveg_2023.astype(np.float32),
            valid_2023,
            float(scenario["dy_px"]),
            float(scenario["dx_px"]),
        )
        modern_metrics = scene_metrics(
            shifted_bands, shifted_pveg, shifted_valid & water, supports
        )
        counts: Counter[str] = Counter()
        for object_id in range(1, object_count + 1):
            category, collapse_score = classify(
                baseline_metrics[object_id], modern_metrics[object_id], rules
            )
            retained = category in RETAINED_CLASSES
            baseline_optical = endpoint_optical_class(
                baseline_metrics[object_id]["pveg"],
                float(rules["endpoint_pveg_threshold"]),
            )
            endpoint_optical = endpoint_optical_class(
                modern_metrics[object_id]["pveg"],
                float(rules["endpoint_pveg_threshold"]),
            )
            row = {
                "scenario": scenario_name,
                "scenario_group": scenario["scenario_group"],
                "shift_m": shift_m,
                "direction": direction,
                "shift_dy_m": scenario["shift_dy_m"],
                "shift_dx_m": scenario["shift_dx_m"],
                "primary_scenario": int(primary),
                "object_id": object_id,
                "category": category,
                "retained": int(retained),
                "baseline_optical_class": baseline_optical,
                "endpoint_optical_class": endpoint_optical,
                "collapse_score": collapse_score,
                "C1_2023": modern_metrics[object_id]["C1"],
                "C2_2023": modern_metrics[object_id]["C2"],
                "TEX_2023": modern_metrics[object_id]["TEX"],
                "pveg_2023": modern_metrics[object_id]["pveg"],
            }
            scenario_rows.append(row)
            classifications_by_object[object_id].append(row)
            counts[category] += 1
        scenario_summaries.append(
            {
                "scenario": scenario_name,
                "scenario_group": scenario["scenario_group"],
                "shift_m": shift_m,
                "direction": direction,
                "shift_dy_m": scenario["shift_dy_m"],
                "shift_dx_m": scenario["shift_dx_m"],
                "verified_shift_magnitude_m": math.hypot(
                    float(scenario["shift_dy_m"]),
                    float(scenario["shift_dx_m"]),
                ),
                "primary_scenario": int(primary),
                **{f"n_{key}": counts[key] for key in CLASS_ORDER},
                "n_retained": sum(counts[key] for key in RETAINED_CLASSES),
            }
        )
        print(f"Classified {scenario_name}", flush=True)

    morphology = object_morphology(core_ids)
    core_counts = np.bincount(core_ids.ravel(), minlength=object_count + 1)
    footprint_counts = np.bincount(
        footprint_ids.ravel(), minlength=object_count + 1
    )
    object_rows: list[dict[str, object]] = []
    selected_ids: set[int] = set()
    primary_scenario_count = sum(
        int(bool(scenario["primary_scenario"])) for scenario in scenarios
    )
    required_fraction = float(gate["required_primary_fraction"])
    required_primary = required_scenarios(
        required_fraction, primary_scenario_count
    )
    for object_id in range(1, object_count + 1):
        object_scenarios = classifications_by_object[object_id]
        zero_shift_row = next(
            row for row in object_scenarios if row["scenario"] == "shift_0m"
        )
        primary_rows = [row for row in object_scenarios if row["primary_scenario"]]
        stress_rows = [
            row for row in object_scenarios if row["scenario_group"] == "stress"
        ]
        retained_primary = sum(int(row["retained"]) for row in primary_rows)
        retained_stress = sum(int(row["retained"]) for row in stress_rows)
        zero_shift_retained = bool(zero_shift_row["retained"])
        selected = zero_shift_retained and retained_primary >= required_primary
        if selected:
            selected_ids.add(object_id)
            final_status = (
                "selected_vegetated_endpoint"
                if zero_shift_row["category"] == "retained_vegetated_endpoint"
                else "selected_bare_endpoint"
            )
        elif zero_shift_retained:
            final_status = "excluded_shift_unstable"
        else:
            final_status = f"excluded_{zero_shift_row['category']}"
        core_area = float(core_counts[object_id] * PIXEL_SIZE_M**2)
        footprint_area = float(footprint_counts[object_id] * PIXEL_SIZE_M**2)
        object_rows.append(
            {
                "object_id": object_id,
                "core_class": zero_shift_row["category"],
                "final_status": final_status,
                "baseline_optical_class": zero_shift_row[
                    "baseline_optical_class"
                ],
                "endpoint_optical_class": zero_shift_row[
                    "endpoint_optical_class"
                ],
                "shape_type": morphology[object_id]["shape_type"],
                "elongation": morphology[object_id]["elongation"],
                "robust_frac": retained_primary / primary_scenario_count,
                "robust_n": retained_primary,
                "robust_required_n": required_primary,
                "stress_frac": (
                    retained_stress / len(stress_rows) if stress_rows else float("nan")
                ),
                "pveg_2003": baseline_metrics[object_id]["pveg"],
                "pveg_2023": zero_shift_row["pveg_2023"],
                "C1_2003": baseline_metrics[object_id]["C1"],
                "C1_2023": zero_shift_row["C1_2023"],
                "C2_2003": baseline_metrics[object_id]["C2"],
                "C2_2023": zero_shift_row["C2_2023"],
                "TEX_2003": baseline_metrics[object_id]["TEX"],
                "TEX_2023": zero_shift_row["TEX_2023"],
                "collapse_score": zero_shift_row["collapse_score"],
                "core_m2": core_area,
                "footprint_m2": footprint_area,
                "area_ratio": footprint_area / core_area,
            }
        )

    with CLASSIFICATION_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(object_rows[0]))
        writer.writeheader()
        writer.writerows(object_rows)
    with SCENARIO_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scenario_summaries[0]))
        writer.writeheader()
        writer.writerows(scenario_summaries)

    selected_mask = np.isin(footprint_ids, list(selected_ids)) & water
    selected_ids_array = np.where(selected_mask, footprint_ids, 0)
    write_raster(
        SELECTED_IDS_RASTER,
        selected_ids_array,
        reference_profile,
        "uint32",
        0,
        {
            "DESCRIPTION": "Selected 2023 structural footprint IDs; separate from vegetation likelihood.",
            "DIAGNOSTIC_SUPPORT": "clean supervised traces",
            "STRUCTURAL_SUPPORT": "spectral-growth footprints",
            "PRODUCT_ROLE": "paper-facing selected structural support",
            "PVEG_COMBINATION": "none",
            "ROBUST_RULE": (
                f"zero-shift retained and at least {required_primary}/"
                f"{primary_scenario_count} primary scenarios retained"
            ),
            "SHIFT_VECTOR_DEFINITION": "Euclidean-unit directions",
            "CONFIG_SHA256": config_hash,
        },
    )

    diagnostic_rows: list[dict[str, object]] = []
    for year, probability, structural_mask, output_path, support_description in [
        (
            2003,
            pveg_2003,
            (footprint_ids > 0) & water,
            DIAGNOSTIC_2003,
            "all 1,011 spectral-growth footprints",
        ),
        (
            2023,
            pveg_2023,
            selected_mask,
            DIAGNOSTIC_2023,
            f"{len(selected_ids)} selected structural footprints",
        ),
    ]:
        valid = (
            water
            & np.isfinite(probability)
            & (probability >= 0)
            & (probability <= 1)
        )
        diagnostic = pveg_classes(
            probability,
            valid,
            float(bins["low_upper_probability"]),
            float(bins["medium_upper_probability"]),
        )
        diagnostic[structural_mask & valid] = 4
        write_raster(
            output_path,
            diagnostic,
            reference_profile,
            "uint8",
            0,
            {
                "CLASS_1": "low vegetation influence",
                "CLASS_2": "medium vegetation influence",
                "CLASS_3": "high vegetation influence",
                "CLASS_4": "mapped structural support",
                "YEAR": str(year),
                "DIAGNOSTIC_SUPPORT": "clean supervised trace core",
                "STRUCTURAL_SUPPORT": support_description,
                "PRODUCT_ROLE": "diagnostic binned composite; not paper-facing",
                "PAPER_FACING": "false",
                "PHYSICAL_COEFFICIENT": "none",
            },
        )
        diagnostic_rows.append(
            {
                "year": year,
                "n_structures": object_count if year == 2003 else len(selected_ids),
                "structural_area_ha": hectares(structural_mask & valid),
                "class_1_area_ha": hectares(diagnostic == 1),
                "class_2_area_ha": hectares(diagnostic == 2),
                "class_3_area_ha": hectares(diagnostic == 3),
                "class_4_area_ha": hectares(diagnostic == 4),
            }
        )
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    zero_shift_counts = Counter(row["core_class"] for row in object_rows)
    final_counts = Counter(row["final_status"] for row in object_rows)
    retained_transitions = Counter(
        f"{row['baseline_optical_class']}_to_{row['endpoint_optical_class']}"
        for row in object_rows
        if row["core_class"] in RETAINED_CLASSES
    )
    selected_transitions = Counter(
        f"{row['baseline_optical_class']}_to_{row['endpoint_optical_class']}"
        for row in object_rows
        if str(row["final_status"]).startswith("selected_")
    )
    threshold_sensitivity = []
    for threshold_raw in gate["sensitivity_fractions"]:
        threshold = float(threshold_raw)
        required = required_scenarios(threshold, primary_scenario_count)
        ids = {
            int(row["object_id"])
            for row in object_rows
            if row["core_class"] in RETAINED_CLASSES
            and int(row["robust_n"]) >= required
        }
        threshold_sensitivity.append(
            {
                "threshold": threshold,
                "required_scenarios": required,
                "n_selected": len(ids),
                "expanded_area_ha": hectares(np.isin(footprint_ids, list(ids))),
            }
        )

    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "path": str(config_path),
            "sha256": config_hash,
            "schema_version": config["schema_version"],
        },
        "method": {
            "diagnostic_support": "clean supervised trace core",
            "structural_support": "spectral-growth footprint",
            "zero_shift_retention_rule": {
                **rules,
                "retained_vegetated_endpoint_semantics": (
                    "non-homogenized structure with a vegetated 2023 endpoint "
                    "and sufficient modern C1 contrast; no baseline optical-class "
                    "persistence is claimed"
                ),
                "retained_bare_endpoint_semantics": (
                    "non-homogenized structure with a bare 2023 endpoint and the "
                    "configured P(VEG) decline; no stable bare class is claimed"
                ),
            },
            "robust_selection": {
                "primary_shift_magnitudes_m": gate[
                    "primary_shift_magnitudes_m"
                ],
                "directions_per_nonzero_shift": 8,
                "direction_vector_normalization": gate[
                    "direction_vector_normalization"
                ],
                "primary_scenarios": primary_scenario_count,
                "required_retained_scenarios": required_primary,
                "required_fraction": required_fraction,
                "stress_shift_magnitudes_m": gate[
                    "stress_shift_magnitudes_m"
                ],
            },
            "paper_facing_output_rule": (
                "vegetation likelihood and structural support are separate layers"
            ),
        },
        "objects": object_count,
        "zero_shift_core_classes": dict(zero_shift_counts),
        "final_status": dict(final_counts),
        "retained_endpoint_transitions": dict(retained_transitions),
        "selected_endpoint_transitions": dict(selected_transitions),
        "selected_2023": {
            "objects": len(selected_ids),
            "expanded_area_ha": hectares(selected_mask),
            "ids": sorted(selected_ids),
        },
        "threshold_sensitivity": threshold_sensitivity,
        "diagnostic_composites": diagnostic_rows,
        "input_files": {
            key: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in paths.items()
        },
        "outputs": {
            "classification_csv": str(CLASSIFICATION_CSV),
            "scenario_csv": str(SCENARIO_CSV),
            "selected_structural_ids_raster": str(SELECTED_IDS_RASTER),
            "diagnostic_composite_2003": str(DIAGNOSTIC_2003),
            "diagnostic_composite_2023": str(DIAGNOSTIC_2023),
            "structural_manifest": str(STRUCTURAL_MANIFEST),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Hybrid core-footprint structural method",
        "",
        "## Decision rule",
        "",
        "- Clean supervised traces are the diagnostic core.",
        "- Spectral-growth polygons are used only as structural footprints.",
        f"- Zero-shift status uses a {float(rules['collapse_ratio']):g} collapse ratio and at least "
        f"{int(rules['minimum_collapsed_signals'])}/3 collapsed signals.",
        f"- A zero-shift retained object is selected only when retained in at least {required_primary}/"
        f"{primary_scenario_count} primary scenarios ({required_fraction:.0%}).",
        "- Every diagonal vector is Euclidean-normalized, so labelled 1, 2 and 4 m shifts have those actual magnitudes.",
        "- The configured 4 m displacement is a stress test and does not decide selection.",
        "- P(VEG) and structural support remain separate in every paper-facing map.",
        "",
        "## Zero-shift core classes",
        "",
        "| class | objects |",
        "|---|---:|",
    ]
    for key in CLASS_ORDER:
        report_lines.append(f"| {key} | {zero_shift_counts[key]} |")
    report_lines += [
        "",
        "The retained endpoint labels describe structural retention plus the 2023",
        "optical endpoint. They do not assert persistence of the optical class from 2003.",
        "",
        "## Final structural selection",
        "",
        f"- Selected objects: **{len(selected_ids)} / {object_count}**.",
        f"- Selected 2023 structural footprint: **{hectares(selected_mask):.2f} ha**.",
        "",
        "| required retained fraction | scenarios | selected objects | expanded area (ha) |",
        "|---:|---:|---:|---:|",
    ]
    for item in threshold_sensitivity:
        report_lines.append(
            f"| {item['threshold']:.0%} | {item['required_scenarios']}/{primary_scenario_count} | "
            f"{item['n_selected']} | {item['expanded_area_ha']:.2f} |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "The shift gate reduces sensitivity to small relative misregistration. It does",
        "not validate the spectral-growth boundary as the true physical footprint.",
        "The selected footprint raster is a structural-support layer, not a",
        "vegetation-probability or hydraulic-coefficient raster. Binned composites",
        "are diagnostic only and are excluded from the paper-facing structural map.",
    ]
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    manifest_outputs = (
        CLASSIFICATION_CSV,
        SCENARIO_CSV,
        SELECTED_IDS_RASTER,
        DIAGNOSTIC_2003,
        DIAGNOSTIC_2023,
        SUMMARY_JSON,
        SUMMARY_CSV,
        REPORT,
    )
    structural_manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "path": str(config_path),
            "sha256": config_hash,
        },
        "code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "claims": {
            "mapped_objects_2003": object_count,
            "mapped_area_2003_ha": hectares((footprint_ids > 0) & water),
            "selected_objects_2023": len(selected_ids),
            "selected_area_2023_ha": hectares(selected_mask),
            "primary_gate": f"{required_primary}/{primary_scenario_count}",
            "required_primary_fraction": required_fraction,
            "direction_vectors": "euclidean_unit",
        },
        "outputs": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in manifest_outputs
        },
    }
    STRUCTURAL_MANIFEST.write_text(
        json.dumps(structural_manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "selected_objects": len(selected_ids),
        "selected_area_ha": hectares(selected_mask),
        "zero_shift_counts": dict(zero_shift_counts),
        "final_counts": dict(final_counts),
        "required_primary": f"{required_primary}/{primary_scenario_count}",
    }, indent=2))
    print(RUN)


if __name__ == "__main__":
    main(parse_args().config)
