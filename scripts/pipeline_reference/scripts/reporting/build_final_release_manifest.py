"""Freeze the canonical numerical claims used by every final deliverable."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"${PROJECT_ROOT}")
OUT = ROOT / "out/deliverables"
ANALYSIS = ROOT / "out/runs/analysis"
MODEL_SUMMARY = ANALYSIS / "paper_v2_final/model_family/model_family_summary.csv"
MODEL_BOOTSTRAP = ANALYSIS / "paper_v2_final/model_family/model_family_bootstrap.csv"
COVER = (
    ANALYSIS
    / "final_scene_ablation_031623_v3/full_scene_products/"
    "hard_vs_soft_cover.csv"
)
CROSS = (
    ANALYSIS
    / "final_scene_ablation_031623_v3/cross_scene_2003/"
    "cross_scene_domain_ablation.csv"
)
CROSS_MANIFEST = (
    ANALYSIS
    / "final_scene_ablation_031623_v3/cross_scene_2003/"
    "cross_scene_2003_manifest.json"
)
FULL_SCENE_MANIFEST = (
    ANALYSIS
    / "final_scene_ablation_031623_v3/full_scene_products/"
    "full_scene_manifest.json"
)
HYBRID_SUMMARY = ANALYSIS / "paper_v3_hybrid/hybrid_summary.json"
STRUCTURAL_DIAGNOSTIC_MANIFEST = (
    ANALYSIS
    / "paper_v3_hybrid/review_products/structural_diagnostic_manifest.json"
)
SCENE_MATRIX_SUMMARY = (
    ANALYSIS
    / "final_scene_ablation_031623_v3/common_support/"
    "segformer_five_seed_summary.csv"
)
SCENE_MATRIX_BOOTSTRAP = (
    ANALYSIS
    / "final_scene_ablation_031623_v3/common_support/"
    "segformer_paired_spatial_bootstrap.csv"
)

SEEDS = (123, 231, 312, 423, 531)
SCENES = ("20030709_qb", "20160825_wv", "20230812_pl")
MODELS = ("segformer", "unet", "rf")
BRANCHES = (
    "M1_03",
    "M1_16",
    "M1_23",
    "M2_0316",
    "M2_0323",
    "M2_1623",
    "M3_031623",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_claims() -> dict[str, Any]:
    frame = pd.read_csv(MODEL_SUMMARY)
    pooled = frame[frame.scene_id == "ALL_SCENES"].set_index("model")
    if set(pooled.index.astype(str)) != set(MODELS):
        raise RuntimeError("The pooled model-family summary is incomplete.")
    result = {}
    for model in MODELS:
        row = pooled.loc[model]
        if int(row.n_seeds) != 5:
            raise RuntimeError(f"{model} does not have five seeds.")
        result[model] = {
            "n_seeds": int(row.n_seeds),
            "min_iou_mean": float(row.min_iou_mean),
            "min_iou_sd": float(row.min_iou_sd),
            "iou_veg_mean": float(row.iou_veg_mean),
            "iou_nonveg_mean": float(row.iou_nonveg_mean),
            "mcc_mean": float(row.mcc_mean),
            "ece_15_mean": float(row.ece_15_mean),
        }
    bootstrap = pd.read_csv(MODEL_BOOTSTRAP)
    pooled_boot = bootstrap[bootstrap.scene_id == "ALL_SCENES"].set_index(
        "competitor_model"
    )
    result["paired_bootstrap"] = {}
    for competitor in ("unet", "rf"):
        row = pooled_boot.loc[competitor]
        result["paired_bootstrap"][competitor] = {
            "delta_min_iou_mean": float(row.delta_min_iou_mean),
            "ci95_low": float(row.delta_min_iou_ci95_low),
            "ci95_high": float(row.delta_min_iou_ci95_high),
            "probability_positive": float(row.prob_delta_min_iou_gt_0),
        }
    return result


def cover_claims() -> dict[str, Any]:
    frame = pd.read_csv(COVER).set_index("scene_id")
    if set(frame.index.astype(str)) != set(SCENES):
        raise RuntimeError("Full-scene cover table has the wrong scenes.")
    result = {}
    for scene in SCENES:
        row = frame.loc[scene]
        thresholds = tuple(
            float(value) for value in str(row.seed_thresholds).split("|")
        )
        temperatures = tuple(
            float(value) for value in str(row.seed_temperatures).split("|")
        )
        if len(thresholds) != 5 or len(temperatures) != 5:
            raise RuntimeError(f"{scene} does not contain five calibrated seeds.")
        result[scene] = {
            "label": str(row.scene_label),
            "valid_pixels": int(row.n_valid_px),
            "hard_operational_cover_pct": float(
                row.hard_operational_mean_pct
            ),
            "hard_seed_sd_pct": float(row.hard_operational_sd_pct),
            "soft_cover_pct": float(row.soft_cover_pct),
            "thresholds": list(thresholds),
            "temperatures": list(temperatures),
        }
    return result


def cross_scene_claims() -> dict[str, Any]:
    frame = pd.read_csv(CROSS)
    if len(frame) != 6:
        raise RuntimeError("Expected six cross-scene rows.")
    result = {}
    for year in ("2003", "2016", "2023"):
        selected = frame[frame.year.astype(str) == year]
        excluded = selected[selected.design == "Scene excluded"].iloc[0]
        m3 = selected[selected.design == "M3"].iloc[0]
        if int(excluded.n_seeds) != 5 or int(m3.n_seeds) != 5:
            raise RuntimeError(f"Cross-scene result for {year} is not five-seed.")
        result[year] = {
            "excluded_branch": str(excluded.branch),
            "excluded_min_iou_mean": float(excluded.min_iou_mean),
            "excluded_min_iou_sd": float(excluded.min_iou_sd),
            "m3_min_iou_mean": float(m3.min_iou_mean),
            "m3_min_iou_sd": float(m3.min_iou_sd),
        }
    effect = json.loads(CROSS_MANIFEST.read_text(encoding="utf-8"))[
        "effect_2003"
    ]
    result["effect_2003"] = effect
    return result


def scene_matrix_claims() -> dict[str, Any]:
    frame = pd.read_csv(SCENE_MATRIX_SUMMARY)
    if set(frame.branch.astype(str)) != set(BRANCHES):
        raise RuntimeError("The M1/M2/M3 branch summary is incomplete.")
    if set(frame.scene_id.astype(str)) != set(SCENES) | {"ALL_SCENES"}:
        raise RuntimeError("The M1/M2/M3 scene summary is incomplete.")

    branches: dict[str, Any] = {}
    for branch in BRANCHES:
        selected = frame[frame.branch == branch].set_index("scene_id")
        if set(selected.index.astype(str)) != set(SCENES) | {"ALL_SCENES"}:
            raise RuntimeError(f"{branch} is missing one or more scenes.")
        if set(selected.n_seeds.astype(int)) != {5}:
            raise RuntimeError(f"{branch} does not have five seeds.")
        branches[branch] = {
            "pooled_min_iou_mean": float(
                selected.loc["ALL_SCENES", "min_iou_mean"]
            ),
            "pooled_min_iou_sd": float(
                selected.loc["ALL_SCENES", "min_iou_sd"]
            ),
            "per_scene": {
                scene: {
                    "min_iou_mean": float(
                        selected.loc[scene, "min_iou_mean"]
                    ),
                    "min_iou_sd": float(selected.loc[scene, "min_iou_sd"]),
                }
                for scene in SCENES
            },
        }

    bootstrap = pd.read_csv(SCENE_MATRIX_BOOTSTRAP)
    pooled = bootstrap[
        (bootstrap.scene_id == "ALL_SCENES")
        & (bootstrap.primary == "M3_031623")
    ].set_index("competitor")
    competitors = set(BRANCHES) - {"M3_031623"}
    if set(pooled.index.astype(str)) != competitors:
        raise RuntimeError("The pooled M3 pairwise bootstrap is incomplete.")
    paired = {
        competitor: {
            "delta_min_iou_mean": float(
                pooled.loc[competitor, "delta_min_iou_mean"]
            ),
            "ci95_low": float(
                pooled.loc[competitor, "delta_min_iou_ci95_low"]
            ),
            "ci95_high": float(
                pooled.loc[competitor, "delta_min_iou_ci95_high"]
            ),
            "probability_positive": float(
                pooled.loc[competitor, "prob_delta_min_iou_gt_0"]
            ),
        }
        for competitor in BRANCHES
        if competitor != "M3_031623"
    }
    return {
        "branches": branches,
        "m3_pairwise_pooled": paired,
        "best_m1": "M1_23",
        "best_m2": "M2_0323",
    }


def structural_support_claims() -> dict[str, Any]:
    payload = json.loads(HYBRID_SUMMARY.read_text(encoding="utf-8"))
    selected = payload["selected_2023"]
    status = payload["final_status"]
    selected_sum = int(status["selected_vegetated_endpoint"]) + int(
        status["selected_bare_endpoint"]
    )
    if selected_sum != int(selected["objects"]):
        raise RuntimeError("Selected hybrid status counts do not sum correctly.")
    return {
        "mapped_objects_2003": int(payload["objects"]),
        "mapped_area_2003_ha": float(
            next(
                row["structural_area_ha"]
                for row in payload["diagnostic_composites"]
                if int(row["year"]) == 2003
            )
        ),
        "selected_objects_2023": int(selected["objects"]),
        "selected_area_2023_ha": float(selected["expanded_area_ha"]),
        "selected_vegetated_endpoint": int(
            status["selected_vegetated_endpoint"]
        ),
        "selected_bare_endpoint": int(status["selected_bare_endpoint"]),
        "primary_gate": (
            f"{payload['method']['robust_selection']['required_retained_scenarios']}/"
            f"{payload['method']['robust_selection']['primary_scenarios']}"
        ),
        "direction_vector_normalization": payload["method"][
            "robust_selection"
        ]["direction_vector_normalization"],
        "paper_facing_output_rule": payload["method"][
            "paper_facing_output_rule"
        ],
    }


def structural_diagnostic_claims() -> dict[str, Any]:
    payload = json.loads(
        STRUCTURAL_DIAGNOSTIC_MANIFEST.read_text(encoding="utf-8")
    )
    summaries = {
        str(row["year"]): {
            "structural_area_ha": float(row["structural_area_ha"]),
            "mean_index": float(row["mean_index"]),
        }
        for row in payload["continuous_summary"]
    }
    return {
        "status": payload["status"],
        "paper_facing": bool(payload["paper_facing"]),
        "paper_facing_layers": payload["paper_facing_layers"],
        "continuous_rule": payload["continuous_rule"],
        "continuous_summary": summaries,
        "rings": payload["rings"],
    }


def main() -> int:
    sources = (
        MODEL_SUMMARY,
        MODEL_BOOTSTRAP,
        COVER,
        CROSS,
        CROSS_MANIFEST,
        FULL_SCENE_MANIFEST,
        HYBRID_SUMMARY,
        STRUCTURAL_DIAGNOSTIC_MANIFEST,
        SCENE_MATRIX_SUMMARY,
        SCENE_MATRIX_BOOTSTRAP,
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing freeze inputs: {missing}")
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "release": "final_scene_ablation_031623_v3_hybrid",
        "fixed_seeds": list(SEEDS),
        "claims": {
            "model_family": model_claims(),
            "full_scene_cover": cover_claims(),
            "cross_scene_ablation": cross_scene_claims(),
            "scene_matrix": scene_matrix_claims(),
            "structural_support": structural_support_claims(),
            "structural_support_diagnostics": structural_diagnostic_claims(),
        },
        "source_files": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sources
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "FINAL_RESULTS_MANIFEST.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
