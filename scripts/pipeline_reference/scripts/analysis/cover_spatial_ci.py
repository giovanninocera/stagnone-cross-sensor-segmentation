from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.common.config import OUT


SEEDS = (123, 231, 312)
BRANCHES = ("control06", "replacement03")
YEARS = (2016, 2023)
MODES = ("checkpoint", "common_0p5", "soft")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def value_column(branch: str, year: int, mode: str) -> str:
    suffix = "soft_sum" if mode == "soft" else f"{mode}_positive"
    return f"{branch}_{year}_{suffix}"


def percentile(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.median(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def aggregate_seed_cover(
    rows: list[dict[str, str]], branch: str, year: int, mode: str
) -> float:
    column = value_column(branch, year, mode)
    numerator = sum(float(row[column]) for row in rows)
    denominator = sum(float(row["n_valid"]) for row in rows)
    if denominator <= 0:
        return float("nan")
    return 100.0 * numerator / denominator


def bootstrap_cover(
    cells: list[dict[str, str]],
    zone: str,
    branch: str,
    year: int,
    mode: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_seed = {
        item_seed: [
            row
            for row in cells
            if row["zone"] == zone and int(row["seed"]) == item_seed
        ]
        for item_seed in SEEDS
    }
    n_cells = len(by_seed[SEEDS[0]])
    if n_cells == 0 or any(len(rows) != n_cells for rows in by_seed.values()):
        raise ValueError(f"Inconsistent spatial cells for zone={zone}")

    observed_seed_values = [
        aggregate_seed_cover(by_seed[item_seed], branch, year, mode)
        for item_seed in SEEDS
    ]
    observed = float(np.mean(observed_seed_values))

    column = value_column(branch, year, mode)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        sampled_seed_indices = rng.integers(0, len(SEEDS), size=len(SEEDS))
        sampled_cell_indices = rng.integers(0, n_cells, size=n_cells)
        seed_covers = []
        for seed_index in sampled_seed_indices:
            sampled_rows = by_seed[SEEDS[int(seed_index)]]
            numerator = sum(
                float(sampled_rows[int(index)][column])
                for index in sampled_cell_indices
            )
            denominator = sum(
                float(sampled_rows[int(index)]["n_valid"])
                for index in sampled_cell_indices
            )
            seed_covers.append(100.0 * numerator / denominator)
        draws[iteration] = float(np.mean(seed_covers))

    median, lower, upper = percentile(draws)
    return {
        "zone": zone,
        "branch": branch,
        "year": year,
        "mode": mode,
        "observed_cover_percent": observed,
        "bootstrap_median_cover_percent": median,
        "ci_2p5_cover_percent": lower,
        "ci_97p5_cover_percent": upper,
        "ci_width_pp": upper - lower,
        "seed_cover_percent_min": float(np.min(observed_seed_values)),
        "seed_cover_percent_max": float(np.max(observed_seed_values)),
        "n_seeds": len(SEEDS),
        "n_cells_per_seed": n_cells,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "support": "common valid pixels for 2016 and 2023",
    }


def bootstrap_change(
    cells: list[dict[str, str]],
    zone: str,
    branch: str,
    mode: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_seed = {
        item_seed: [
            row
            for row in cells
            if row["zone"] == zone and int(row["seed"]) == item_seed
        ]
        for item_seed in SEEDS
    }
    n_cells = len(by_seed[SEEDS[0]])
    if n_cells == 0 or any(len(rows) != n_cells for rows in by_seed.values()):
        raise ValueError(f"Inconsistent spatial cells for zone={zone}")

    observed_seed_values = [
        aggregate_seed_cover(by_seed[item_seed], branch, 2023, mode)
        - aggregate_seed_cover(by_seed[item_seed], branch, 2016, mode)
        for item_seed in SEEDS
    ]
    observed = float(np.mean(observed_seed_values))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    columns = {
        year: value_column(branch, year, mode)
        for year in YEARS
    }
    for iteration in range(n_boot):
        sampled_seed_indices = rng.integers(0, len(SEEDS), size=len(SEEDS))
        sampled_cell_indices = rng.integers(0, n_cells, size=n_cells)
        seed_changes = []
        for seed_index in sampled_seed_indices:
            sampled_rows = by_seed[SEEDS[int(seed_index)]]
            denominator = sum(
                float(sampled_rows[int(index)]["n_valid"])
                for index in sampled_cell_indices
            )
            covers = {}
            for year in YEARS:
                numerator = sum(
                    float(sampled_rows[int(index)][columns[year]])
                    for index in sampled_cell_indices
                )
                covers[year] = 100.0 * numerator / denominator
            seed_changes.append(covers[2023] - covers[2016])
        draws[iteration] = float(np.mean(seed_changes))

    median, lower, upper = percentile(draws)
    return {
        "zone": zone,
        "branch": branch,
        "mode": mode,
        "observed_change_2016_2023_pp": observed,
        "bootstrap_median_change_pp": median,
        "ci_2p5_change_pp": lower,
        "ci_97p5_change_pp": upper,
        "ci_width_pp": upper - lower,
        "seed_change_pp_min": float(np.min(observed_seed_values)),
        "seed_change_pp_max": float(np.max(observed_seed_values)),
        "n_seeds": len(SEEDS),
        "n_cells_per_seed": n_cells,
        "n_boot": n_boot,
        "bootstrap_seed": seed,
        "support": "common valid pixels for 2016 and 2023",
    }


def write_report(
    path: Path,
    cover_rows: list[dict[str, Any]],
    change_rows: list[dict[str, Any]],
) -> None:
    lookup = {
        (row["zone"], row["branch"], int(row["year"]), row["mode"]): row
        for row in cover_rows
    }
    change_lookup = {
        (row["zone"], row["branch"], row["mode"]): row
        for row in change_rows
    }
    lines = [
        "# Spatial bootstrap cover CI on common 2016/2023 support",
        "",
        "These intervals quantify spatial uncertainty of the model-derived cover product.",
        "They are not Olofsson-style bias-adjusted area estimates because they do not use an independent probability reference sample.",
        "",
        "## Absolute cover CI",
        "",
        "| Zone | Branch | Mode | 2016 cover % [95% CI] | 2023 cover % [95% CI] | Change pp [95% CI] |",
        "|---|---|---|---:|---:|---:|",
    ]
    for zone in ("INT", "EXT"):
        for branch in BRANCHES:
            for mode in MODES:
                r2016 = lookup[(zone, branch, 2016, mode)]
                r2023 = lookup[(zone, branch, 2023, mode)]
                change = change_lookup[(zone, branch, mode)]
                lines.append(
                    f"| {zone} | {branch} | {mode} | "
                    f"{r2016['observed_cover_percent']:.2f} "
                    f"[{r2016['ci_2p5_cover_percent']:.2f}, {r2016['ci_97p5_cover_percent']:.2f}] | "
                    f"{r2023['observed_cover_percent']:.2f} "
                    f"[{r2023['ci_2p5_cover_percent']:.2f}, {r2023['ci_97p5_cover_percent']:.2f}] | "
                    f"{change['observed_change_2016_2023_pp']:+.2f} "
                    f"[{change['ci_2p5_change_pp']:+.2f}, {change['ci_97p5_change_pp']:+.2f}] |"
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the right CI for the mapped product under spatial autocorrelation: cells, not pixels, are resampled.",
            "- It supports statements about model-derived cover stability/change on the common 2016/2023 support.",
            "- It must not be described as an unbiased ecological area estimate.",
            "- Formal Olofsson-style area estimation would require an independent probability sample and reference labels.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Spatial bootstrap CI for cover on the common 2016/2023 support."
    )
    parser.add_argument(
        "--cells",
        type=Path,
        default=OUT
        / "runs"
        / "analysis"
        / "replacement_2003_paper_v2"
        / "matched"
        / "cover"
        / "common_change_cells.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT
        / "runs"
        / "analysis"
        / "replacement_2003_paper_v2"
        / "matched"
        / "cover",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260616)
    args = parser.parse_args()

    cells = read_csv(args.cells)
    zones = sorted({row["zone"] for row in cells})
    cover_rows = []
    change_rows = []
    for zone_index, zone in enumerate(zones):
        for branch_index, branch in enumerate(BRANCHES):
            for mode_index, mode in enumerate(MODES):
                base_seed = (
                    int(args.bootstrap_seed)
                    + zone_index * 10_000
                    + branch_index * 1_000
                    + mode_index * 100
                )
                for year_index, year in enumerate(YEARS):
                    cover_rows.append(
                        bootstrap_cover(
                            cells,
                            zone=zone,
                            branch=branch,
                            year=year,
                            mode=mode,
                            n_boot=int(args.bootstrap),
                            seed=base_seed + year_index,
                        )
                    )
                change_rows.append(
                    bootstrap_change(
                        cells,
                        zone=zone,
                        branch=branch,
                        mode=mode,
                        n_boot=int(args.bootstrap),
                        seed=base_seed + 50,
                    )
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cover_path = args.out_dir / "cover_spatial_ci_common_support.csv"
    change_path = args.out_dir / "cover_change_spatial_ci_common_support.csv"
    report_path = args.out_dir / "COVER_SPATIAL_CI_COMMON_SUPPORT.md"
    write_csv(cover_path, cover_rows)
    write_csv(change_path, change_rows)
    write_report(report_path, cover_rows, change_rows)
    manifest = {
        "cells": str(args.cells),
        "outputs": [str(cover_path), str(change_path), str(report_path)],
        "bootstrap": int(args.bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_unit": "non-overlapping spatial cells, resampled within zone and seed",
        "support": "common valid pixels for 2016 and 2023",
        "not_olofsson": (
            "No independent probability reference sample is used; these are product spatial CIs, "
            "not bias-adjusted area-estimation CIs."
        ),
    }
    (args.out_dir / "cover_spatial_ci_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
