"""Summarize the five-seed v3 model-family comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.analysis import summarize_final_ablation_common_support as summary_core


ROOT = Path(r"${PROJECT_ROOT}")
SOURCE = (
    ROOT
    / "out/runs/analysis/final_scene_ablation_031623_v3/model_family"
)
OUT = ROOT / "out/runs/analysis/paper_v2_final/model_family"
MODELS = ("segformer", "unet", "rf")
LABELS = {
    "segformer": "SegFormer MiT-B2",
    "unet": "U-Net ResNet-34",
    "rf": "Random Forest",
}
COLORS = {
    "segformer": "#167C80",
    "unet": "#D97731",
    "rf": "#6B7280",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20260704)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise ValueError("Use at least 100 bootstrap draws.")

    metrics = pd.read_csv(SOURCE / "all_model_family_metrics.csv")
    cells = pd.read_csv(SOURCE / "all_model_family_cells.csv")
    metrics_for_core = metrics.rename(columns={"model": "branch"})
    cells_for_core = cells.rename(columns={"model": "branch"})

    summary_core.BRANCHES = MODELS
    summary_core.PRIMARY_COMPARATOR = "segformer"
    summary_core.validate_metrics(metrics_for_core)
    summary = summary_core.summarize_seeds(metrics_for_core).rename(
        columns={"branch": "model"}
    )
    bootstrap = summary_core.paired_bootstrap(
        cells_for_core, args.bootstrap_draws, args.random_seed
    )
    bootstrap = bootstrap.rename(
        columns={"primary": "primary_model", "competitor": "competitor_model"}
    )

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "model_family_summary.csv"
    bootstrap_path = OUT / "model_family_bootstrap.csv"
    summary_core.write_atomic_csv(summary, summary_path)
    summary_core.write_atomic_csv(bootstrap, bootstrap_path)

    pooled = (
        summary[summary.scene_id == "ALL_SCENES"]
        .set_index("model")
        .loc[list(MODELS)]
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    values = pooled["min_iou_mean"].astype(float).to_numpy()
    errors = pooled["min_iou_sd"].astype(float).to_numpy()
    bars = axis.bar(
        [LABELS[model] for model in MODELS],
        values,
        yerr=errors,
        capsize=6,
        color=[COLORS[model] for model in MODELS],
        width=0.62,
    )
    axis.set_ylim(0, 1.0)
    axis.set_ylabel("Minimum class IoU")
    axis.set_title(
        "Model-family comparison on the same frozen spatial support"
    )
    axis.grid(axis="y", color="#D7DEE5", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value, error in zip(bars, values, errors):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.98, value + error + 0.035),
            f"{value:.3f} +/- {error:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    figure.tight_layout()
    figure_path = OUT / "fig_model_family_comparison.png"
    figure.savefig(figure_path, dpi=240, facecolor="white")
    plt.close(figure)

    pooled_bootstrap = bootstrap[bootstrap.scene_id == "ALL_SCENES"]
    report_lines = [
        "# Final five-seed model-family comparison",
        "",
        "All models use the corrected v3 M3 data allocation and the same frozen,",
        "deduplicated spatial support. Thresholds are selected only on validation.",
        "",
        "| Model | Mean minimum IoU | Seed SD |",
        "|---|---:|---:|",
    ]
    for model in MODELS:
        row = pooled.loc[model]
        report_lines.append(
            f"| {LABELS[model]} | {row.min_iou_mean:.4f} | "
            f"{row.min_iou_sd:.4f} |"
        )
    report_lines.extend(["", "## Paired spatial bootstrap", ""])
    for row in pooled_bootstrap.itertuples():
        report_lines.append(
            f"- SegFormer minus {LABELS[str(row.competitor_model)]}: "
            f"{row.delta_min_iou_mean:+.4f} "
            f"(95% CI {row.delta_min_iou_ci95_low:+.4f} to "
            f"{row.delta_min_iou_ci95_high:+.4f})."
        )
    report_lines.extend(
        [
            "",
            "These values quantify agreement with image-derived references, not",
            "independent ecological accuracy.",
            "",
        ]
    )
    report_path = OUT / "MODEL_FAMILY_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "experiment": "final_model_family_v3",
        "models": list(MODELS),
        "seeds": list(summary_core.SEEDS),
        "metrics_input": str(SOURCE / "all_model_family_metrics.csv"),
        "cells_input": str(SOURCE / "all_model_family_cells.csv"),
        "summary": str(summary_path),
        "bootstrap": str(bootstrap_path),
        "figure": str(figure_path),
        "report": str(report_path),
        "bootstrap_draws": args.bootstrap_draws,
        "random_seed": args.random_seed,
    }
    manifest_path = OUT / "model_family_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
