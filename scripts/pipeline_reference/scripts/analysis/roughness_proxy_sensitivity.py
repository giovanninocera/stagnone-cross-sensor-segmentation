"""Sensitivity analysis for the Stagnone continuous roughness proxy.

This analysis is deliberately separate from the manuscript. It tests:

1. the structural override applied to atoll/cordon footprints;
2. alternative monotonic transformations of calibrated P(VEG);
3. structural-selection thresholds from 70% to 100%;
4. aggregation at 2, 10 and 20 m.

Outputs are exploratory scenario products, not calibrated hydraulic
coefficients.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import Normalize
from rasterio.transform import Affine
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.stats import spearmanr


ROOT = Path(r"${PROJECT_ROOT}")
MINI = Path(r"${LICENSED_DATA_ROOT}")
RUN = ROOT / "out" / "runs" / "analysis" / "roughness_proxy_sensitivity"
EXPORT = MINI / "dati" / "roughness_sensitivity"
REPORT_PDF = MINI / "roughness_sensitivity_test.pdf"
REPORT_MD = MINI / "roughness_sensitivity_test.md"

GPKG = MINI / "dati/hybrid_core_footprint/hybrid_core_footprints.gpkg"
FOOTPRINT_IDS = (
    MINI
    / "dati/atolli_spettrali/expanded_spectral_2003"
    / "atolls_cordons_expanded_spectral_2003.tif"
)

SCENES = {
    2003: {
        "pveg": MINI / "dati/mean_pveg_20030709_qb.tif",
        "consensus": MINI / "dati/consensus_vegmask_20030709_qb.tif",
    },
    2023: {
        "pveg": MINI / "dati/mean_pveg_20230812_pl.tif",
        "consensus": MINI / "dati/consensus_vegmask_20230812_pl.tif",
    },
}

FINE_RESOLUTION = 2
COARSE_RESOLUTIONS = (10, 20)
NODATA = -9999.0


@dataclass
class Raster:
    values: np.ndarray
    profile: dict


def read_raster(path: Path, dtype: str | None = None) -> Raster:
    with rasterio.open(path) as src:
        values = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata
    if dtype is not None:
        values = values.astype(dtype)
    if np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32, copy=False)
        if nodata is not None:
            values[values == nodata] = np.nan
    return Raster(values, profile)


def same_grid(left: dict, right: dict) -> bool:
    return all(
        left[key] == right[key]
        for key in ("width", "height", "transform", "crs")
    )


def threshold_object_ids() -> dict[int, set[int]]:
    con = sqlite3.connect(GPKG)
    try:
        rows = list(
            con.execute(
                """
                SELECT object_id, robust_frac
                FROM core_diagnostic_classification
                WHERE core_class IN (
                    'retained_vegetated_endpoint',
                    'retained_bare_endpoint'
                )
                """
            )
        )
    finally:
        con.close()

    thresholds = {}
    for percent in (70, 80, 90, 100):
        threshold = percent / 100.0
        thresholds[percent] = {
            int(object_id)
            for object_id, robust_fraction in rows
            if float(robust_fraction) >= threshold
        }
    return thresholds


def pad_to_factor(
    values: np.ndarray,
    factor: int,
    fill: float | int,
) -> np.ndarray:
    rows = math.ceil(values.shape[0] / factor)
    cols = math.ceil(values.shape[1] / factor)
    padded = np.full((rows * factor, cols * factor), fill, dtype=values.dtype)
    padded[: values.shape[0], : values.shape[1]] = values
    return padded


def block_sum(values: np.ndarray, factor: int) -> np.ndarray:
    padded = pad_to_factor(values, factor, 0)
    rows = padded.shape[0] // factor
    cols = padded.shape[1] // factor
    return padded.reshape(rows, factor, cols, factor).sum(
        axis=(1, 3), dtype=np.float64
    )


def aggregate_mean(
    values: np.ndarray,
    valid: np.ndarray,
    factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = block_sum(valid.astype(np.uint8), factor).astype(np.float32)
    total = block_sum(
        np.where(valid, values, 0.0).astype(np.float32, copy=False),
        factor,
    )
    mean = np.divide(
        total,
        count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)
    return mean, count


def aggregate_fraction(
    mask: np.ndarray,
    valid: np.ndarray,
    factor: int,
) -> np.ndarray:
    count = block_sum(valid.astype(np.uint8), factor)
    positive = block_sum((mask & valid).astype(np.uint8), factor)
    return np.divide(
        positive,
        count,
        out=np.full(count.shape, np.nan, dtype=np.float64),
        where=count > 0,
    ).astype(np.float32)


def output_profile(profile: dict, resolution: int, shape: tuple[int, int]) -> dict:
    factor = resolution / abs(profile["transform"].a)
    transform = profile["transform"] * Affine.scale(factor, factor)
    out = profile.copy()
    out.update(
        width=shape[1],
        height=shape[0],
        transform=transform,
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    return out


def write_float(
    path: Path,
    values: np.ndarray,
    profile: dict,
    *,
    description: str,
    tags: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.where(np.isfinite(values), values, NODATA).astype(np.float32)
    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype="float32",
        nodata=NODATA,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, description)
        dst.update_tags(
            PRODUCT_STATUS="exploratory sensitivity test",
            PHYSICAL_COEFFICIENT="false",
            **tags,
        )


def copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def robust_mask(year: int, resolution: int, shape: tuple[int, int]) -> np.ndarray:
    path = MINI / f"dati/hydraulic_prior/qc_flag_{year}_{resolution}m.tif"
    raster = read_raster(path)
    if raster.values.shape != shape:
        raise RuntimeError(f"QC grid mismatch: {path}")
    return raster.values == 2


def rank_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    valid = mask & np.isfinite(candidate) & np.isfinite(reference)
    x = candidate[valid]
    y = reference[valid]
    difference = x - y
    rho = float(spearmanr(x, y).statistic)
    qx = float(np.quantile(x, 0.90))
    qy = float(np.quantile(y, 0.90))
    x_top = valid & (candidate >= qx)
    y_top = valid & (reference >= qy)
    union = int((x_top | y_top).sum())
    return {
        "n": int(valid.sum()),
        "spearman": rho,
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mean_difference": float(np.mean(difference)),
        "top10_jaccard": float((x_top & y_top).sum() / union) if union else 1.0,
    }


def summary_metrics(
    values: np.ndarray,
    mask: np.ndarray,
    structural_cells: np.ndarray,
) -> dict[str, float | int]:
    valid = mask & np.isfinite(values)
    x = values[valid]
    local = values[valid & structural_cells]
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "sd": float(np.std(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "fraction_ge_0p8": float(np.mean(x >= 0.8)),
        "structural_cell_mean": float(np.mean(local)) if local.size else float("nan"),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scenario_array(
    pveg: np.ndarray,
    valid: np.ndarray,
    structure: np.ndarray,
    *,
    gamma: float = 1.0,
    structural_floor: float | None = None,
) -> np.ndarray:
    out = np.full(pveg.shape, np.nan, dtype=np.float32)
    transformed = np.power(np.clip(pveg[valid], 0.0, 1.0), gamma)
    out[valid] = transformed.astype(np.float32)
    if structural_floor is not None:
        selected = structure & valid
        out[selected] = np.maximum(out[selected], structural_floor)
    return out


def analyse() -> dict:
    RUN.mkdir(parents=True, exist_ok=True)
    for folder in ("rasters_2m", "rasters_10m", "rasters_20m", "tables", "figures"):
        (RUN / folder).mkdir(parents=True, exist_ok=True)

    footprint = read_raster(FOOTPRINT_IDS)
    footprint_ids = footprint.values.astype(np.int32, copy=False)
    threshold_ids = threshold_object_ids()

    summary_rows: list[dict] = []
    comparison_rows: list[dict] = []
    threshold_rows: list[dict] = []
    scale_rows: list[dict] = []
    results: dict[str, dict] = {}
    map_cache: dict[tuple[int, str], np.ndarray] = {}

    for year, sources in SCENES.items():
        pveg_raster = read_raster(sources["pveg"], "float32")
        consensus_raster = read_raster(sources["consensus"])
        if not same_grid(pveg_raster.profile, footprint.profile):
            raise RuntimeError(f"{year}: footprint grid mismatch")
        if not same_grid(pveg_raster.profile, consensus_raster.profile):
            raise RuntimeError(f"{year}: consensus grid mismatch")

        pveg = pveg_raster.values
        consensus = consensus_raster.values
        valid = np.isfinite(pveg) & (pveg >= 0.0) & (pveg <= 1.0) & (consensus > 0)

        threshold_masks = {
            percent: valid & np.isin(footprint_ids, list(object_ids))
            for percent, object_ids in threshold_ids.items()
        }
        structure = valid & (
            (footprint_ids > 0) if year == 2003 else threshold_masks[80]
        )

        pveg_only = scenario_array(pveg, valid, structure)
        scenarios = {
            "pveg_only": pveg_only,
            "hybrid_rs075": scenario_array(
                pveg, valid, structure, structural_floor=0.75
            ),
            "hybrid_rs090": scenario_array(
                pveg, valid, structure, structural_floor=0.90
            ),
            "hybrid_rs100_current": scenario_array(
                pveg, valid, structure, structural_floor=1.00
            ),
            "hybrid_gamma05": scenario_array(
                pveg, valid, structure, gamma=0.5, structural_floor=1.00
            ),
            "hybrid_gamma20": scenario_array(
                pveg, valid, structure, gamma=2.0, structural_floor=1.00
            ),
        }
        binary = np.full(pveg.shape, np.nan, dtype=np.float32)
        binary[valid] = ((consensus[valid] == 2) | structure[valid]).astype(np.float32)
        scenarios["hybrid_binary_fraction"] = binary

        if year == 2023:
            for percent, mask in threshold_masks.items():
                scenarios[f"hybrid_threshold_{percent}"] = scenario_array(
                    pveg, valid, mask, structural_floor=1.00
                )

        uplift = scenarios["hybrid_rs100_current"] - pveg_only
        write_float(
            RUN / "rasters_2m" / f"roughness_structural_uplift_{year}_2m.tif",
            uplift,
            pveg_raster.profile,
            description=f"{year} structural uplift: current hybrid minus P(VEG)",
            tags={"SCENARIO": "hybrid_rs100_current minus pveg_only"},
        )
        for name in ("hybrid_rs075", "hybrid_rs090", "hybrid_rs100_current"):
            write_float(
                RUN / "rasters_2m" / f"roughness_{name}_{year}_2m.tif",
                scenarios[name],
                pveg_raster.profile,
                description=f"{year} exploratory relative roughness: {name}",
                tags={"SCENARIO": name, "YEAR": str(year), "RESOLUTION_M": "2"},
            )
        if year == 2023:
            for percent in (70, 90, 100):
                name = f"hybrid_threshold_{percent}"
                write_float(
                    RUN / "rasters_2m" / f"roughness_{name}_{year}_2m.tif",
                    scenarios[name],
                    pveg_raster.profile,
                    description=f"2023 threshold sensitivity: {percent} percent",
                    tags={
                        "SCENARIO": name,
                        "YEAR": str(year),
                        "RESOLUTION_M": "2",
                    },
                )

        year_results = {}
        coarse_values: dict[int, dict[str, np.ndarray]] = {}
        coarse_counts: dict[int, np.ndarray] = {}
        for resolution in COARSE_RESOLUTIONS:
            factor = resolution // FINE_RESOLUTION
            structure_fraction = aggregate_fraction(structure, valid, factor)
            pveg_outside = np.full(pveg.shape, np.nan, dtype=np.float32)
            outside = valid & ~structure
            pveg_outside[outside] = pveg[outside]
            outside_mean, outside_count = aggregate_mean(
                pveg_outside, outside, factor
            )
            coarse: dict[str, np.ndarray] = {}
            count = None
            for name, values in scenarios.items():
                aggregated, scenario_count = aggregate_mean(values, valid, factor)
                coarse[name] = aggregated
                count = scenario_count
            assert count is not None
            coarse_counts[resolution] = count

            mixture_identity = (
                (1.0 - structure_fraction)
                * np.where(np.isfinite(outside_mean), outside_mean, 0.0)
                + structure_fraction
            ).astype(np.float32)
            no_outside = (outside_count == 0) & (structure_fraction > 0)
            mixture_identity[no_outside] = 1.0
            mixture_identity[count == 0] = np.nan
            coarse["fractional_mixture_identity"] = mixture_identity
            any_structure = coarse["pveg_only"].copy()
            any_structure[structure_fraction > 0] = 1.0
            coarse["any_structure_cell_max"] = any_structure
            coarse["structural_uplift"] = (
                coarse["hybrid_rs100_current"] - coarse["pveg_only"]
            )
            coarse["structure_fraction"] = structure_fraction
            coarse_values[resolution] = coarse

            profile = output_profile(
                pveg_raster.profile,
                resolution,
                coarse["pveg_only"].shape,
            )
            qc = robust_mask(year, resolution, coarse["pveg_only"].shape)
            structural_cells = structure_fraction > 0

            for name, values in coarse.items():
                write_float(
                    RUN
                    / f"rasters_{resolution}m"
                    / f"roughness_{name}_{year}_{resolution}m.tif",
                    values,
                    profile,
                    description=f"{year} {resolution} m roughness sensitivity: {name}",
                    tags={
                        "SCENARIO": name,
                        "YEAR": str(year),
                        "RESOLUTION_M": str(resolution),
                    },
                )
                if name not in ("structure_fraction", "structural_uplift"):
                    metrics = summary_metrics(values, qc, structural_cells)
                    summary_rows.append(
                        {
                            "year": year,
                            "resolution_m": resolution,
                            "scenario": name,
                            **metrics,
                        }
                    )

            reference = coarse["hybrid_rs100_current"]
            for name, values in coarse.items():
                if name in (
                    "hybrid_rs100_current",
                    "structure_fraction",
                    "structural_uplift",
                ):
                    continue
                comparison_rows.append(
                    {
                        "year": year,
                        "resolution_m": resolution,
                        "scenario": name,
                        "reference": "hybrid_rs100_current",
                        **rank_metrics(values, reference, qc),
                    }
                )

            identity_mask = qc & np.isfinite(reference) & np.isfinite(mixture_identity)
            identity_error = np.abs(reference[identity_mask] - mixture_identity[identity_mask])
            year_results[str(resolution)] = {
                "robust_cells": int(qc.sum()),
                "pveg_fraction_spearman": float(
                    spearmanr(
                        coarse["pveg_only"][qc],
                        aggregate_fraction(consensus == 2, valid, factor)[qc],
                    ).statistic
                ),
                "current_fractional_identity_max_abs_error": float(
                    identity_error.max(initial=0.0)
                ),
                "mean_current": float(np.nanmean(reference[qc])),
                "mean_pveg": float(np.nanmean(coarse["pveg_only"][qc])),
                "mean_uplift_all": float(
                    np.nanmean(coarse["structural_uplift"][qc])
                ),
                "mean_uplift_structural_cells": float(
                    np.nanmean(coarse["structural_uplift"][qc & structural_cells])
                ),
                "structural_cells": int((qc & structural_cells).sum()),
            }
            if resolution == 10:
                map_cache[(year, "pveg")] = coarse["pveg_only"]
                map_cache[(year, "current")] = reference
                map_cache[(year, "uplift")] = coarse["structural_uplift"]
                map_cache[(year, "structure_fraction")] = structure_fraction

        for name in coarse_values[20]:
            if name in ("structure_fraction",):
                continue
            values10 = coarse_values[10][name]
            valid10 = np.isfinite(values10) & (coarse_counts[10] > 0)
            weighted = np.where(valid10, values10 * coarse_counts[10], 0.0)
            numerator = block_sum(weighted.astype(np.float32), 2)
            denominator = block_sum(
                np.where(valid10, coarse_counts[10], 0.0).astype(np.float32),
                2,
            )
            from10 = np.divide(
                numerator,
                denominator,
                out=np.full(numerator.shape, np.nan, dtype=np.float64),
                where=denominator > 0,
            ).astype(np.float32)
            direct20 = coarse_values[20][name]
            qc20 = robust_mask(year, 20, direct20.shape)
            metrics = rank_metrics(from10, direct20, qc20)
            scale_rows.append(
                {
                    "year": year,
                    "scenario": name,
                    "comparison": "10m area-weighted to 20m versus direct 20m",
                    **metrics,
                }
            )

        if year == 2023:
            for percent in (70, 80, 90, 100):
                mask = threshold_masks[percent]
                values = scenarios[f"hybrid_threshold_{percent}"]
                uplift_threshold = values - pveg_only
                threshold_rows.append(
                    {
                        "threshold_percent": percent,
                        "objects": len(threshold_ids[percent]),
                        "footprint_area_ha": float(mask.sum() * 4.0 / 10000.0),
                        "lagoon_mean_index": float(np.nanmean(values[valid])),
                        "lagoon_mean_uplift": float(
                            np.nanmean(uplift_threshold[valid])
                        ),
                        "mean_uplift_inside_selected_footprints": float(
                            np.nanmean(uplift_threshold[mask])
                        ),
                    }
                )

        results[str(year)] = year_results

    write_rows(RUN / "tables/scenario_summary.csv", summary_rows)
    write_rows(RUN / "tables/scenario_comparison_vs_current.csv", comparison_rows)
    write_rows(RUN / "tables/threshold_sensitivity_2023.csv", threshold_rows)
    write_rows(RUN / "tables/scale_consistency.csv", scale_rows)

    manifest = {
        "status": "exploratory sensitivity analysis; not hydraulic calibration",
        "paper_modified": False,
        "years": results,
        "threshold_sensitivity_2023": threshold_rows,
        "outputs": {
            "run": str(RUN),
            "export": str(EXPORT),
            "report_pdf": str(REPORT_PDF),
            "report_md": str(REPORT_MD),
        },
    }
    (RUN / "roughness_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    make_figures(summary_rows, comparison_rows, threshold_rows, map_cache)
    make_reports(manifest, summary_rows, comparison_rows, threshold_rows, scale_rows)

    copy_tree(RUN, EXPORT)
    shutil.copy2(RUN / "roughness_sensitivity_test.pdf", REPORT_PDF)
    shutil.copy2(RUN / "roughness_sensitivity_test.md", REPORT_MD)
    return manifest


def crop_map(values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    rows, cols = np.where(mask)
    if not rows.size:
        return values
    margin = 5
    r0 = max(0, rows.min() - margin)
    r1 = min(values.shape[0], rows.max() + margin + 1)
    c0 = max(0, cols.min() - margin)
    c1 = min(values.shape[1], cols.max() + margin + 1)
    return values[r0:r1, c0:c1]


def make_figures(
    summary_rows: list[dict],
    comparison_rows: list[dict],
    threshold_rows: list[dict],
    map_cache: dict[tuple[int, str], np.ndarray],
) -> None:
    figure_dir = RUN / "figures"

    fig, axes = plt.subplots(2, 4, figsize=(12, 10), constrained_layout=True)
    for row, year in enumerate((2003, 2023)):
        for col, (name, title, cmap, vmax) in enumerate(
            (
                ("pveg", "P(VEG) media", "viridis", 1.0),
                ("current", "Scenario corrente", "viridis", 1.0),
                ("uplift", "Incremento strutturale", "magma", None),
                ("structure_fraction", "Frazione strutturale", "cividis", 1.0),
            )
        ):
            values = crop_map(map_cache[(year, name)])
            ax = axes[row, col]
            finite = values[np.isfinite(values)]
            local_vmax = vmax
            if local_vmax is None:
                local_vmax = max(0.01, float(np.quantile(finite, 0.99)))
            image = ax.imshow(
                values,
                cmap=cmap,
                vmin=0.0,
                vmax=local_vmax,
                interpolation="nearest",
            )
            ax.set_title(f"{year} - {title}", fontsize=10)
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, fraction=0.036, pad=0.02)
    fig.suptitle(
        "Componenti dello scenario corrente sulla griglia da 10 m",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(figure_dir / "figure_1_component_maps.png", dpi=180)
    plt.close(fig)

    selected = [
        row
        for row in summary_rows
        if row["resolution_m"] == 10
        and row["scenario"]
        in (
            "pveg_only",
            "hybrid_rs075",
            "hybrid_rs090",
            "hybrid_rs100_current",
            "hybrid_gamma05",
            "hybrid_gamma20",
            "hybrid_binary_fraction",
            "any_structure_cell_max",
        )
    ]
    order = [
        "pveg_only",
        "hybrid_rs075",
        "hybrid_rs090",
        "hybrid_rs100_current",
        "hybrid_gamma05",
        "hybrid_gamma20",
        "hybrid_binary_fraction",
        "any_structure_cell_max",
    ]
    labels = {
        "pveg_only": "P(VEG)",
        "hybrid_rs075": "struttura >=0.75",
        "hybrid_rs090": "struttura >=0.90",
        "hybrid_rs100_current": "corrente (=1)",
        "hybrid_gamma05": "sqrt(P)",
        "hybrid_gamma20": "P^2",
        "hybrid_binary_fraction": "frazione binaria",
        "any_structure_cell_max": "cella intera=1",
    }
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    width = 0.38
    x = np.arange(len(order))
    for offset, year, color in ((-width / 2, 2003, "#277da1"), (width / 2, 2023, "#f3722c")):
        lookup = {
            row["scenario"]: row["mean"]
            for row in selected
            if row["year"] == year
        }
        ax.bar(
            x + offset,
            [lookup[name] for name in order],
            width,
            label=str(year),
            color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[name] for name in order], rotation=25, ha="right")
    ax.set_ylabel("Indice medio sulle celle robuste")
    ax.set_ylim(0, 1)
    ax.set_title("Sensibilita' alla formulazione dello scenario (10 m)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(figure_dir / "figure_2_scenario_means.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    thresholds = [row["threshold_percent"] for row in threshold_rows]
    objects = [row["objects"] for row in threshold_rows]
    areas = [row["footprint_area_ha"] for row in threshold_rows]
    uplift = [
        row["mean_uplift_inside_selected_footprints"] for row in threshold_rows
    ]
    axes[0].plot(thresholds, objects, marker="o", label="oggetti")
    second = axes[0].twinx()
    second.plot(thresholds, areas, marker="s", color="#f3722c", label="area")
    axes[0].set_xlabel("Soglia di scenari conservati (%)")
    axes[0].set_ylabel("Oggetti")
    second.set_ylabel("Area footprint (ha)")
    axes[0].set_title("Selezione strutturale 2023")
    axes[0].grid(alpha=0.25)
    axes[1].plot(thresholds, uplift, marker="o", color="#9c179e")
    axes[1].set_xlabel("Soglia di scenari conservati (%)")
    axes[1].set_ylabel("Incremento medio dentro le footprint")
    axes[1].set_title("Effetto locale dell'override")
    axes[1].grid(alpha=0.25)
    fig.savefig(figure_dir / "figure_3_threshold_sensitivity.png", dpi=180)
    plt.close(fig)

    selected_comparison = [
        row
        for row in comparison_rows
        if row["resolution_m"] == 10
        and row["scenario"]
        in (
            "pveg_only",
            "hybrid_rs075",
            "hybrid_rs090",
            "hybrid_gamma05",
            "hybrid_gamma20",
            "hybrid_binary_fraction",
            "any_structure_cell_max",
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "mae", "Differenza media assoluta"),
        (axes[1], "top10_jaccard", "Sovrapposizione del 10% piu' alto"),
    ):
        for year, color, marker in (
            (2003, "#277da1", "o"),
            (2023, "#f3722c", "s"),
        ):
            rows = [row for row in selected_comparison if row["year"] == year]
            names = [row["scenario"] for row in rows]
            values = [row[metric] for row in rows]
            positions = np.arange(len(names))
            ax.plot(
                positions,
                values,
                marker=marker,
                color=color,
                label=str(year),
            )
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [labels.get(name, name) for name in names],
                rotation=30,
                ha="right",
            )
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("MAE rispetto allo scenario corrente")
    axes[1].set_ylabel("Indice di Jaccard")
    axes[1].set_ylim(0, 1.02)
    axes[0].legend()
    axes[1].legend()
    fig.savefig(figure_dir / "figure_4_scenario_comparison.png", dpi=180)
    plt.close(fig)


def find_row(
    rows: list[dict],
    *,
    year: int,
    resolution: int,
    scenario: str,
) -> dict:
    return next(
        row
        for row in rows
        if row["year"] == year
        and row["resolution_m"] == resolution
        and row["scenario"] == scenario
    )


def make_reports(
    manifest: dict,
    summary_rows: list[dict],
    comparison_rows: list[dict],
    threshold_rows: list[dict],
    scale_rows: list[dict],
) -> None:
    r2003 = manifest["years"]["2003"]["10"]
    r2023 = manifest["years"]["2023"]["10"]
    gamma05_2003 = find_row(
        comparison_rows, year=2003, resolution=10, scenario="hybrid_gamma05"
    )
    gamma20_2023 = find_row(
        comparison_rows, year=2023, resolution=10, scenario="hybrid_gamma20"
    )
    current_2003 = find_row(
        summary_rows, year=2003, resolution=10, scenario="hybrid_rs100_current"
    )
    current_2023 = find_row(
        summary_rows, year=2023, resolution=10, scenario="hybrid_rs100_current"
    )

    md = f"""# Test di sensibilita' della roughness relativa

## Stato

Analisi esplorativa separata dal paper. Nessun coefficiente idraulico e'
stato calibrato. Il manoscritto non e' stato modificato.

## Risultati principali

1. Lo scenario corrente, quando viene mediato su celle da 10 o 20 m, e'
   matematicamente una miscela frazionaria:

   `R = (1 - f_struct) * P(VEG)_fuori + f_struct * 1`

   Errore numerico massimo a 10 m:
   - 2003: {r2003['current_fractional_identity_max_abs_error']:.2e}
   - 2023: {r2023['current_fractional_identity_max_abs_error']:.2e}

2. P(VEG) media e frazione di pixel classificati come vegetazione sono
   fortemente, ma non perfettamente, associate:
   - 2003: rho = {r2003['pveg_fraction_spearman']:.3f}
   - 2023: rho = {r2023['pveg_fraction_spearman']:.3f}

3. L'override strutturale ha un effetto medio limitato sull'intera laguna ma
   localmente importante:
   - incremento medio nelle celle strutturali 2003:
     {r2003['mean_uplift_structural_cells']:.3f}
   - incremento medio nelle celle strutturali 2023:
     {r2023['mean_uplift_structural_cells']:.3f}

4. Le trasformazioni monotone di P(VEG) conservano quasi interamente
   l'ordinamento spaziale:
   - sqrt(P), 2003: rho = {gamma05_2003['spearman']:.4f}
   - P^2, 2023: rho = {gamma20_2023['spearman']:.4f}

5. Scenario corrente medio sulle celle robuste da 10 m:
   - 2003: {current_2003['mean']:.3f}
   - 2023: {current_2023['mean']:.3f}

## Soglia strutturale 2023

| Regola | Oggetti | Area footprint (ha) | Incremento locale medio |
|---:|---:|---:|---:|
"""
    for row in threshold_rows:
        md += (
            f"| {row['threshold_percent']}% | {row['objects']} | "
            f"{row['footprint_area_ha']:.2f} | "
            f"{row['mean_uplift_inside_selected_footprints']:.3f} |\n"
        )
    md += """

## Interpretazione

- La trasformazione lineare di P(VEG) non e' il punto piu' fragile: il prodotto
  e' molto bimodale e le trasformazioni monotone cambiano soprattutto la
  magnitudine, non la geografia.
- La scelta piu' importante e' il valore assegnato alle strutture, soprattutto
  nel 2023, quando alcune footprint robuste hanno P(VEG) bassa.
- Sulla griglia idrodinamica non bisogna trasformare tutta la cella in valore
  massimo solo perche' contiene una piccola porzione di struttura. La media
  dello scenario corrente conserva gia' la frazione strutturale.
- Senza velocita', livelli o un modello idrodinamico, questi test misurano
  robustezza interna e sensibilita', non accuratezza idraulica.

## Proposta da discutere con il professore

Mantenere il raster a 2 m come scenario descrittivo e, per un eventuale modello
idrodinamico a 10-20 m, usare la formulazione frazionaria. Testare almeno tre
valori strutturali (0.75, 0.90, 1.00) come ensemble, senza sceglierne uno come
coefficiente fisico prima di una calibrazione.
"""
    (RUN / "roughness_sensitivity_test.md").write_text(md, encoding="utf-8")

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCentered",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionSimple",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.black,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    doc = SimpleDocTemplate(
        str(RUN / "roughness_sensitivity_test.pdf"),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Test di sensibilita' della roughness relativa",
    )

    story = [
        Paragraph("Test di sensibilita' della roughness relativa", styles["TitleCentered"]),
        Paragraph(
            "Analisi esplorativa separata dal paper. Sono stati confrontati "
            "l'override strutturale, diverse trasformazioni di P(VEG), le "
            "soglie di stabilita' e le griglie da 2, 10 e 20 m.",
            styles["SmallBody"],
        ),
        Spacer(1, 5),
        Paragraph("Risultati principali", styles["SectionSimple"]),
    ]
    bullets = [
        (
            "Lo scenario corrente aggregato e' gia' una miscela frazionaria: "
            "il contributo massimo delle strutture e' pesato dalla loro area "
            "nella cella, non applicato automaticamente all'intera cella."
        ),
        (
            f"P(VEG) media e frazione vegetata hanno rho={r2003['pveg_fraction_spearman']:.3f} "
            f"nel 2003 e rho={r2023['pveg_fraction_spearman']:.3f} nel 2023."
        ),
        (
            "Le trasformazioni sqrt(P) e P^2 cambiano la magnitudine ma quasi "
            "non modificano l'ordinamento spaziale."
        ),
        (
            f"L'incremento medio nelle celle strutturali e' {r2003['mean_uplift_structural_cells']:.3f} "
            f"nel 2003 e {r2023['mean_uplift_structural_cells']:.3f} nel 2023."
        ),
        (
            "La scelta piu' influente e' quindi il valore attribuito alle "
            "footprint, soprattutto nel 2023."
        ),
    ]
    for text in bullets:
        story.append(Paragraph("&#8226; " + text, styles["SmallBody"]))
    story.extend(
        [
            Spacer(1, 5),
            Image(
                str(RUN / "figures/figure_2_scenario_means.png"),
                width=17.5 * cm,
                height=8.75 * cm,
            ),
            Paragraph(
                "Figura 1. Valore medio degli scenari sulle celle robuste da "
                "10 m. Le trasformazioni di P(VEG) modificano soprattutto la "
                "scala numerica; l'ipotesi 'qualsiasi struttura rende massima "
                "l'intera cella' e' volutamente inclusa come caso aggressivo.",
                styles["SmallBody"],
            ),
            PageBreak(),
            Paragraph("Dove agisce l'override strutturale", styles["SectionSimple"]),
            Image(
                str(RUN / "figures/figure_1_component_maps.png"),
                width=17.5 * cm,
                height=14.6 * cm,
            ),
            Paragraph(
                "Figura 2. P(VEG), scenario corrente, incremento prodotto "
                "dalle strutture e frazione strutturale sulla griglia da 10 m.",
                styles["SmallBody"],
            ),
            PageBreak(),
            Paragraph("Sensibilita' della selezione 2023", styles["SectionSimple"]),
        ]
    )
    threshold_table = [
        ["Regola", "Oggetti", "Area (ha)", "Incremento locale"]
    ] + [
        [
            f"{row['threshold_percent']}%",
            str(row["objects"]),
            f"{row['footprint_area_ha']:.2f}",
            f"{row['mean_uplift_inside_selected_footprints']:.3f}",
        ]
        for row in threshold_rows
    ]
    table = Table(
        threshold_table,
        colWidths=[3.0 * cm, 3.0 * cm, 3.5 * cm, 4.2 * cm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#A0A0A0")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend(
        [
            table,
            Spacer(1, 8),
            Image(
                str(RUN / "figures/figure_3_threshold_sensitivity.png"),
                width=17.2 * cm,
                height=7.4 * cm,
            ),
            Paragraph(
                "Figura 3. Aumentando la soglia diminuiscono in modo regolare "
                "oggetti e area selezionata. Il 70% e' il valore operativo; "
                "le soglie superiori documentano la sensibilita' della selezione.",
                styles["SmallBody"],
            ),
            Spacer(1, 6),
            Image(
                str(RUN / "figures/figure_4_scenario_comparison.png"),
                width=17.2 * cm,
                height=7.5 * cm,
            ),
            Paragraph(
                "Figura 4. Differenza rispetto allo scenario corrente e "
                "stabilita' delle aree collocate nel 10% piu' alto.",
                styles["SmallBody"],
            ),
            PageBreak(),
            Paragraph("Conclusione operativa", styles["SectionSimple"]),
            Paragraph(
                "La formulazione lineare basata su P(VEG) non e' il principale "
                "punto debole. Il risultato e' molto piu' sensibile alla regola "
                "assegnata ad atolli e cordoni. Per un modello a 10-20 m e' "
                "preferibile mantenere la frazione occupata dalla struttura, "
                "invece di rendere massima un'intera cella quando contiene anche "
                "una sola porzione della footprint.",
                styles["SmallBody"],
            ),
            Paragraph("Proposta da inviare al professore", styles["SectionSimple"]),
            Paragraph(
                "Usare il prodotto corrente come scenario superiore e costruire "
                "un piccolo ensemble con valore strutturale minimo pari a 0.75, "
                "0.90 e 1.00. Le tre mappe mantengono la stessa evidenza "
                "spaziale e quantificano quanto i risultati dipendono "
                "dall'ipotesi strutturale. Senza osservazioni di velocita' o "
                "livello non si sceglie un valore come coefficiente fisico.",
                styles["SmallBody"],
            ),
            Paragraph("Cosa questi test non dimostrano", styles["SectionSimple"]),
            Paragraph(
                "Non validano Manning n, drag, velocita' o tempi di residenza. "
                "Misurano la robustezza interna della costruzione cartografica. "
                "La validazione idraulica richiede un modello e osservazioni "
                "indipendenti.",
                styles["SmallBody"],
            ),
            Spacer(1, 8),
            Paragraph(
                "File di supporto: raster GeoTIFF a 2, 10 e 20 m, tabelle CSV, "
                "figure e manifest disponibili nella cartella "
                "dati/roughness_sensitivity.",
                styles["SmallBody"],
            ),
        ]
    )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(A4[0] / 2, 0.75 * cm, str(document.page))
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    manifest = analyse()
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
