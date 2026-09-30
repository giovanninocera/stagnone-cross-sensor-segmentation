#!/usr/bin/env python3
"""Diagnostic controls for cross-scene comparability of calibrated P(VEG)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


ROOT = Path(r"${PROJECT_ROOT}")
MASKS = {
    "2003": ROOT / "out/training_mask/training_2003__FULL.tif",
    "2016": ROOT / "out/training_mask/training_2016__FULL.tif",
    "2023": ROOT / "out/training_mask/training_2023__FULL.tif",
}
PVEG = {
    "2003": ROOT / "out/runs/analysis/power_v3/prob_roughness_proxy/mean_pveg_20030709_qb.tif",
    "2016": ROOT / "out/runs/analysis/power_v3/prob_roughness_proxy/mean_pveg_20160825_wv.tif",
    "2023": ROOT / "out/runs/analysis/power_v3/prob_roughness_proxy/mean_pveg_20230812_pl.tif",
}
REEFS = (
    ROOT
    / "out/runs/analysis/atolli_spectral_typicality_supervised/"
    "atolli_degradation_2003_2023.csv"
)
OUT = ROOT / "out/runs/analysis/paper_v2_final/temporal_controls"
TEST_BLOCKS = (15, 18, 19, 22)
N_BLOCKS = 6
CELL_SIZE = 256


def block_bounds(width: int, height: int, block_id: int) -> tuple[int, int, int, int]:
    by, bx = divmod(block_id, N_BLOCKS)
    return (
        math.floor(bx * width / N_BLOCKS),
        math.floor((bx + 1) * width / N_BLOCKS),
        math.floor(by * height / N_BLOCKS),
        math.floor((by + 1) * height / N_BLOCKS),
    )


def test_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for block in TEST_BLOCKS:
        x0, x1, y0, y1 = block_bounds(shape[1], shape[0], block)
        mask[y0:y1, x0:x1] = True
    return mask


def bootstrap(values: np.ndarray, seed: int = 20260630) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(5000, dtype=np.float64)
    for i in range(draws.size):
        draws[i] = np.mean(rng.choice(values, size=values.size, replace=True))
    return (
        float(np.mean(values)),
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def cell_deltas(
    selector: np.ndarray, p03: np.ndarray, p23: np.ndarray, minimum: int = 512
) -> np.ndarray:
    values = []
    for block in TEST_BLOCKS:
        x0, x1, y0, y1 = block_bounds(selector.shape[1], selector.shape[0], block)
        for yy in range(y0, y1, CELL_SIZE):
            for xx in range(x0, x1, CELL_SIZE):
                y2 = min(y1, yy + CELL_SIZE)
                x2 = min(x1, xx + CELL_SIZE)
                local = selector[yy:y2, xx:x2]
                if int(local.sum()) < minimum:
                    continue
                delta = p23[yy:y2, xx:x2][local] - p03[yy:y2, xx:x2][local]
                values.append(float(np.mean(delta)))
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {}
    probabilities = {}
    profile = None
    for year in ("2003", "2016", "2023"):
        with rasterio.open(MASKS[year]) as source:
            labels[year] = source.read(1)
            profile = (source.height, source.width)
        with rasterio.open(PVEG[year]) as source:
            probabilities[year] = source.read(1).astype(np.float64)
    assert profile is not None
    held_out = test_mask(profile)
    finite = np.logical_and.reduce(
        [np.isfinite(probabilities[year]) & (probabilities[year] >= 0) for year in probabilities]
    )
    persistent_veg = finite & held_out & np.logical_and.reduce(
        [labels[year] == 2 for year in labels]
    )
    persistent_nonveg = finite & held_out & np.logical_and.reduce(
        [labels[year] == 1 for year in labels]
    )
    controls = {
        "persistent_reference_VEG": persistent_veg,
        "persistent_reference_NONVEG": persistent_nonveg,
    }

    rows = []
    for name, selector in controls.items():
        for year in ("2003", "2016", "2023"):
            values = probabilities[year][selector]
            rows.append(
                {
                    "control": name,
                    "year": year,
                    "n_pixels": int(values.size),
                    "mean_pveg": round(float(np.mean(values)), 6),
                    "median_pveg": round(float(np.median(values)), 6),
                    "p10": round(float(np.percentile(values, 10)), 6),
                    "p90": round(float(np.percentile(values, 90)), 6),
                }
            )
    delta_rows = []
    for index, (name, selector) in enumerate(controls.items()):
        values = cell_deltas(
            selector,
            probabilities["2003"],
            probabilities["2023"],
        )
        mean, low, high = bootstrap(values, seed=20260630 + index)
        delta_rows.append(
            {
                "control": name,
                "n_spatial_cells": int(values.size),
                "mean_cell_delta_2023_minus_2003": round(mean, 6),
                "ci95_low": round(low, 6),
                "ci95_high": round(high, 6),
            }
        )

    with (OUT / "persistent_control_probabilities.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "persistent_control_spatial_bootstrap.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)

    with REEFS.open(newline="", encoding="utf-8") as handle:
        reef_rows = list(csv.DictReader(handle))
    reef_medians = {
        year: float(np.median([float(row[f"pveg_{year}"]) for row in reef_rows]))
        for year in ("2003", "2016", "2023")
    }
    years = ("2003", "2016", "2023")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    styles = {
        "persistent_reference_VEG": ("#2b8a3e", "Persistent reference VEG"),
        "persistent_reference_NONVEG": ("#495057", "Persistent reference NONVEG"),
    }
    for name, (color, label) in styles.items():
        medians = [
            next(
                row["median_pveg"]
                for row in rows
                if row["control"] == name and row["year"] == year
            )
            for year in years
        ]
        ax.plot(years, medians, marker="o", linewidth=2.4, color=color, label=label)
    ax.plot(
        years,
        [reef_medians[year] for year in years],
        marker="o",
        linewidth=3.0,
        color="#c92a2a",
        label="Mapped reef objects",
    )
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Median calibrated P(VEG)")
    ax.set_title("Temporal controls and reef vegetation signal")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(OUT / "fig_temporal_controls.png", dpi=220, facecolor="white")
    plt.close(fig)

    report = [
        "# Temporal comparability controls",
        "",
        (
            "Controls use the same unique pixels in fixed test blocks that retain the "
            "same reference class in 2003, 2016, and 2023."
        ),
        "",
        "| control | P2003 median | P2016 median | P2023 median | spatial-cell delta 2003-2023 (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in controls:
        medians = {
            row["year"]: row["median_pveg"]
            for row in rows
            if row["control"] == name
        }
        delta = next(row for row in delta_rows if row["control"] == name)
        report.append(
            f"| {name} | {medians['2003']:.3f} | {medians['2016']:.3f} | "
            f"{medians['2023']:.3f} | "
            f"{delta['mean_cell_delta_2023_minus_2003']:+.3f} "
            f"[{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}] |"
        )
    report += [
        "",
        (
            f"Mapped reef median P(VEG): {reef_medians['2003']:.3f} -> "
            f"{reef_medians['2016']:.3f} -> {reef_medians['2023']:.3f}."
        ),
        "",
        "These are diagnostic controls based on photo-interpreted labels, not independent ecological validation.",
    ]
    (OUT / "TEMPORAL_COMPARABILITY_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (OUT / "temporal_control_manifest.json").write_text(
        json.dumps({"test_blocks": list(TEST_BLOCKS), "cell_size": CELL_SIZE}, indent=2),
        encoding="utf-8",
    )
    print(OUT)


if __name__ == "__main__":
    main()
