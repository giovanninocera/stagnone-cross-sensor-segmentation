#!/usr/bin/env python3
"""Rebuild the consolidated numerical summary from aggregate bundle tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_OBJECTS = 598
AUTHORITATIVE_AREA_HA = 48.9456
AUTHORITATIVE_PVEG_SHA256 = {
    "2003": "3b3436249101dc83bc1f86e8fb18a65df8e98ae26fb2a683d2056323d755c82b",
    "2023": "cca69d4e6f0cc56376bd8417edcbd8752b8fed081f9b6a4cb58381701471c15b",
}


def read_csv(relative_path: str) -> list[dict[str, str]]:
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def as_int(value: str) -> int:
    return int(float(value))


def as_float(value: str) -> float:
    return float(value)


def scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if all(token not in lowered for token in (".", "e")):
            return int(text)
        return float(text)
    except ValueError:
        return text


def build_summary() -> dict[str, Any]:
    model_rows = [
        row
        for row in read_csv("tables/final_release/model_family_summary.csv")
        if row["scene_id"] == "ALL_SCENES"
    ]
    model_family = {}
    for row in sorted(model_rows, key=lambda item: item["model"]):
        model_family[row["model"]] = {
            "n_seeds": as_int(row["n_seeds"]),
            "n_unique_pixels": as_int(row["n_unique_pixels"]),
            "iou_veg_mean": as_float(row["iou_veg_mean"]),
            "iou_nonveg_mean": as_float(row["iou_nonveg_mean"]),
            "min_iou_mean": as_float(row["min_iou_mean"]),
            "min_iou_sd": as_float(row["min_iou_sd"]),
            "mcc_mean": as_float(row["mcc_mean"]),
            "ece_15_mean": as_float(row["ece_15_mean"]),
            "brier_mean": as_float(row["brier_mean"]),
        }

    cross_scene: dict[str, Any] = {}
    for row in read_csv("tables/final_release/cross_scene_domain_ablation.csv"):
        key = "m3" if row["design"] == "M3" else "scene_excluded"
        cross_scene.setdefault(row["year"], {})[key] = {
            "scene_id": row["scene_id"],
            "branch": row["branch"],
            "n_seeds": as_int(row["n_seeds"]),
            "min_iou_mean": as_float(row["min_iou_mean"]),
            "min_iou_sd": as_float(row["min_iou_sd"]),
            "mcc_mean": as_float(row["mcc_mean"]),
            "ece_mean": as_float(row["ece_mean"]),
        }

    cover = {}
    for row in read_csv("tables/final_release/hard_vs_soft_cover.csv"):
        cover[row["scene_id"]] = {
            "scene_label": row["scene_label"],
            "n_valid_pixels": as_int(row["n_valid_px"]),
            "hard_operational_mean_pct": as_float(row["hard_operational_mean_pct"]),
            "hard_operational_sd_pct": as_float(row["hard_operational_sd_pct"]),
            "soft_cover_pct": as_float(row["soft_cover_pct"]),
            "soft_minus_hard_percentage_points": as_float(
                row["soft_minus_operational_hard_pp"]
            ),
        }

    calibration = {}
    for row in read_csv("tables/current_alignment/calibration_ensemble_heldout.csv"):
        calibration[row["scene_id"]] = {
            "n_pixels": as_int(row["n_pixels"]),
            "prevalence": as_float(row["prevalence"]),
            "mean_probability": as_float(row["mean_probability"]),
            "ece_15": as_float(row["ece_15"]),
            "brier": as_float(row["brier"]),
            "nll": as_float(row["nll"]),
            "spatial_cells": as_int(row["spatial_cells"]),
            "ece_15_ci95": [
                as_float(row["ece_15_ci95_low"]),
                as_float(row["ece_15_ci95_high"]),
            ],
        }

    temporal_controls: dict[str, Any] = {}
    for row in read_csv("tables/controls/persistent_control_probabilities.csv"):
        temporal_controls.setdefault(row["control"], {})[row["year"]] = {
            "n_pixels": as_int(row["n_pixels"]),
            "mean_pveg": as_float(row["mean_pveg"]),
            "median_pveg": as_float(row["median_pveg"]),
            "p10": as_float(row["p10"]),
            "p90": as_float(row["p90"]),
        }

    support_rows = read_csv("tables/current_alignment/frozen_test_support.csv")
    support = {key: scalar(value) for key, value in support_rows[0].items()}

    current_hybrid = read_json("manifests/current/hybrid_summary_20260716.json")
    final_manifest = read_json("manifests/current/FINAL_RESULTS_MANIFEST.json")
    hybrid_selected = int(current_hybrid["selected_2023"]["objects"])
    hybrid_area = float(current_hybrid["selected_2023"]["expanded_area_ha"])
    final_structural = final_manifest["claims"]["structural_support"]
    final_selected = int(final_structural["selected_objects_2023"])
    final_area = float(final_structural["selected_area_2023_ha"])
    pveg_sha256 = {
        "2003": current_hybrid["input_files"]["pveg_2003"]["sha256"],
        "2023": current_hybrid["input_files"]["pveg_2023"]["sha256"],
    }

    if (hybrid_selected, hybrid_area) != (
        AUTHORITATIVE_OBJECTS,
        AUTHORITATIVE_AREA_HA,
    ):
        raise RuntimeError(
            "hybrid_summary_20260716.json does not match the authoritative freeze"
        )
    if (final_selected, final_area) != (
        AUTHORITATIVE_OBJECTS,
        AUTHORITATIVE_AREA_HA,
    ):
        raise RuntimeError(
            "FINAL_RESULTS_MANIFEST.json does not match the authoritative freeze"
        )
    if pveg_sha256 != AUTHORITATIVE_PVEG_SHA256:
        raise RuntimeError("P(VEG) input hashes do not match the frozen inputs")

    return {
        "schema_version": 1,
        "model_family_pooled": model_family,
        "cross_scene_ablation": cross_scene,
        "full_scene_cover": cover,
        "heldout_ensemble_calibration": calibration,
        "persistent_reference_controls": temporal_controls,
        "frozen_test_support": support,
        "authoritative_structural_freeze": {
            "status": "authoritative",
            "freeze_date": "2026-07-16",
            "selected_objects": AUTHORITATIVE_OBJECTS,
            "selected_area_ha": AUTHORITATIVE_AREA_HA,
            "hybrid_summary_created": current_hybrid.get("created"),
            "final_results_manifest_timestamp": final_manifest.get("timestamp"),
            "hybrid_summary_match": True,
            "final_results_manifest_match": True,
            "pveg_sha256": pveg_sha256,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON summary to this path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare the rebuilt summary with expected_summary.json",
    )
    args = parser.parse_args()

    summary = build_summary()
    payload = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    if args.output:
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    elif not args.check:
        sys.stdout.write(payload)

    if args.check:
        expected = read_json("expected_summary.json")
        if summary != expected:
            print("FAIL: rebuilt summary differs from expected_summary.json", file=sys.stderr)
            return 1
        print("PASS: consolidated numerical summary matches expected_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
