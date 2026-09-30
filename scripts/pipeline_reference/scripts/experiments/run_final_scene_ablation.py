"""Prepare and run the final M1/M2/M3 scene-ablation matrix.

Design:
- final summer scenes only: 2003 QuickBird, 2016 WorldView-2, 2023 Pleiades;
- seven scene combinations (three M1, three M2, one M3);
- constant total budget of 6,000 patches per configuration;
- identical fixed validation/test blocks;
- three deterministic training seeds;
- evaluation-only models: canonical full-scene products are not overwritten.

The command is resume-safe and writes only under the experiment-specific paths.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import rasterio

from scripts.common.config import OUT, RBOA, ROOT


EXPERIMENT = "final_scene_ablation_031623_v3"
HARDENING_CONFIG = (
    ROOT / "configs" / "train" / "hardening_newboa_rgb_01_02" / "m3_061623.json"
)
CONFIG_DIR = ROOT / "configs" / "train" / EXPERIMENT
CONFIG_PATH = CONFIG_DIR / "matrix_3seed.json"
ADDITIONAL_CONFIG_PATH = CONFIG_DIR / "matrix_additional_2seed.json"
SMOKE_CONFIG_PATH = CONFIG_DIR / "smoke_m1_03_seed123.json"
RUN_ROOT = OUT / "runs" / "train" / EXPERIMENT
ADDITIONAL_RUN_ROOT = OUT / "runs" / "train" / f"{EXPERIMENT}_seed_extension"
PREP_ROOT = OUT / "runs" / "hardening" / EXPERIMENT
MANIFEST_PATH = PREP_ROOT / "experiment_manifest.json"
PREP_LOG = PREP_ROOT / "prepare.log"

COMMON_SUFFIX = "_st3b_final_scene_ablation_031623_v3_common"
COMMON_STATS_CSV = (
    OUT / "patches" / "normalization_final_scene_ablation_031623_v3_train_blocks.csv"
)
FEATURE_PRESET = "CMP_BASE3"
TOTAL_PATCHES = 6000
PATCH_SEED = 123
TRAIN_SEEDS = (123, 231, 312)
ADDITIONAL_TRAIN_SEEDS = (423, 531)
VAL_BLOCKS = (6, 24, 28, 30)
TEST_BLOCKS = (15, 18, 19, 22)

BRANCHES: list[dict[str, Any]] = [
    {
        "key": "m1_03",
        "dataset": "FINALABL_M1_03",
        "scenes": ["20030709_qb"],
        "years": ["2003"],
    },
    {
        "key": "m1_16",
        "dataset": "FINALABL_M1_16",
        "scenes": ["20160825_wv"],
        "years": ["2016"],
    },
    {
        "key": "m1_23",
        "dataset": "FINALABL_M1_23",
        "scenes": ["20230812_pl"],
        "years": ["2023"],
    },
    {
        "key": "m2_0316",
        "dataset": "FINALABL_M2_0316",
        "scenes": ["20030709_qb", "20160825_wv"],
        "years": ["2003", "2016"],
    },
    {
        "key": "m2_0323",
        "dataset": "FINALABL_M2_0323",
        "scenes": ["20030709_qb", "20230812_pl"],
        "years": ["2003", "2023"],
    },
    {
        "key": "m2_1623",
        "dataset": "FINALABL_M2_1623",
        "scenes": ["20160825_wv", "20230812_pl"],
        "years": ["2016", "2023"],
    },
    {
        "key": "m3_031623",
        "dataset": "FINALABL_M3_031623",
        "scenes": ["20030709_qb", "20160825_wv", "20230812_pl"],
        "years": ["2003", "2016", "2023"],
    },
]


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(message, flush=True)
    PREP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PREP_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run(cmd: list[str], title: str) -> None:
    log("")
    log("=" * 96)
    log(f"[{now()}] {title}")
    log("CMD: " + subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{title} failed with return code {rc}")


def branch_by_key(key: str) -> dict[str, Any]:
    for branch in BRANCHES:
        if branch["key"] == key:
            return branch
    raise KeyError(key)


def own_patch_csv(branch: dict[str, Any]) -> Path:
    return OUT / "patches" / f"patches_{EXPERIMENT}_{branch['key']}.csv"


def own_splitplan(branch: dict[str, Any]) -> Path:
    return OUT / "patches" / f"patches_{EXPERIMENT}_{branch['key']}_splitplan.csv"


def own_h5(branch: dict[str, Any]) -> Path:
    return OUT / "patches_bin" / f"STORE_{EXPERIMENT}_{branch['key']}_st3b_t256.h5"


def dataset_paths(branch: dict[str, Any]) -> tuple[Path, Path, str]:
    return own_patch_csv(branch), own_h5(branch), COMMON_SUFFIX


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def fixed_block_args() -> list[str]:
    return [
        "--fixed_val_blocks",
        *[str(v) for v in VAL_BLOCKS],
        "--fixed_test_blocks",
        *[str(v) for v in TEST_BLOCKS],
    ]


def patch_command(
    branch: dict[str, Any],
    *,
    output_csv: Path,
    source_suffix: str,
    output_h5: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.patches.make_patches",
        "--out_csv",
        str(output_csv),
        "--source_suffix",
        source_suffix,
        "--suffix",
        "_st3b",
        "--feature_preset",
        FEATURE_PRESET,
        "--scene_id",
        *[str(v) for v in branch["scenes"]],
        "--mask_year",
        *[str(v) for v in branch["years"]],
        "--tile",
        "256",
        "--boundary_frac",
        "0.5",
        "--pos_frac_cap",
        "0.5",
        "--n_blocks_x",
        "6",
        "--n_blocks_y",
        "6",
        "--val_frac",
        "0.10",
        "--test_frac",
        "0.10",
        "--val_buffer_px",
        "256",
        "--test_buffer_px",
        "256",
        "--min_valid_frac",
        "0.95",
        "--seed",
        str(PATCH_SEED),
        "--n_patches_total",
        str(TOTAL_PATCHES),
        "--split_strategy",
        "balanced",
        "--split_seed_search",
        "1",
        "--split_balance_max_spread",
        "0.10",
        *fixed_block_args(),
    ]
    if output_h5 is not None:
        cmd.extend(["--out_h5", str(output_h5)])
    return cmd


def write_common_stats_csv() -> None:
    scene_dates = {
        "20030709_qb": "20030709",
        "20160825_wv": "20160825",
        "20230812_pl": "20230812",
    }
    train_blocks = sorted(set(range(36)) - set(VAL_BLOCKS) - set(TEST_BLOCKS))
    rows: list[dict[str, Any]] = []
    stats_tile = 64
    for scene, date in scene_dates.items():
        source = RBOA / f"RBOA_{date}.tif"
        if not source.exists():
            raise FileNotFoundError(source)
        with rasterio.open(source) as ds:
            width, height = ds.width, ds.height
        for block_id in train_blocks:
            by, bx = divmod(block_id, 6)
            xs = int(bx * width // 6)
            xe = int((bx + 1) * width // 6)
            ys = int(by * height // 6)
            ye = int((by + 1) * height // 6)
            for y0 in range(ys, ye - stats_tile + 1, stats_tile):
                for x0 in range(xs, xe - stats_tile + 1, stats_tile):
                    rows.append(
                        {
                            "scene_id": scene,
                            "split": "train",
                            "x0": x0,
                            "y0": y0,
                            "tile": stats_tile,
                            "block_id": block_id,
                        }
                    )
    COMMON_STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMMON_STATS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scene_id", "split", "x0", "y0", "tile", "block_id"],
        )
        writer.writeheader()
        writer.writerows(rows)
    log(
        f"[normalization] wrote {len(rows)} label-independent train-block windows "
        f"to {COMMON_STATS_CSV.name}"
    )


def prepare_common_features(resume: bool) -> None:
    expected = [
        ROOT / "data" / "features" / f"feat_{scene}{COMMON_SUFFIX}.tif"
        for scene in ("20030709_qb", "20160825_wv", "20230812_pl")
    ]
    write_common_stats_csv()
    if resume and all(path.exists() for path in expected):
        log("[SKIP] common label-independent normalized RGB stacks already exist")
        return
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.data_prep.prepare_features",
            "--scene_id",
            "20030709_qb",
            "20160825_wv",
            "20230812_pl",
            "--boa_source",
            "custom",
            "--boa_input_dir",
            str(RBOA),
            "--boa_input_pattern",
            "RBOA_{date}.tif",
            "--feature_preset",
            FEATURE_PRESET,
            "--suffix",
            COMMON_SUFFIX,
            "--stats_patch_csv",
            str(COMMON_STATS_CSV),
            "--stats_split",
            "train",
            "--overwrite",
        ],
        "Prepare common label-independent train-block-normalized RGB stacks",
    )


def prepare_branch(branch: dict[str, Any], resume: bool) -> None:
    patch_csv = own_patch_csv(branch)
    splitplan = own_splitplan(branch)
    h5_path = own_h5(branch)
    if (
        resume
        and count_rows(patch_csv) == TOTAL_PATCHES
        and h5_path.exists()
        and h5_path.stat().st_size > 0
    ):
        audit_branch(branch)
        log(f"[SKIP] {branch['key']}: dataset already prepared")
        return

    if h5_path.exists():
        h5_path.unlink()

    run(
        patch_command(
            branch,
            output_csv=splitplan,
            source_suffix=COMMON_SUFFIX,
            output_h5=None,
        ),
        f"{branch['key']}: frozen split plan",
    )
    audit_csv(branch, splitplan, "splitplan")
    run(
        patch_command(
            branch,
            output_csv=patch_csv,
            source_suffix=COMMON_SUFFIX,
            output_h5=h5_path,
        ),
        f"{branch['key']}: final CSV/HDF5",
    )
    audit_branch(branch)


def audit_branch(branch: dict[str, Any]) -> None:
    audit_csv(branch, own_patch_csv(branch), "final")


def audit_csv(branch: dict[str, Any], patch_csv: Path, stage: str) -> None:
    report_path = (
        PREP_ROOT / "overlap_audits" / f"{branch['key']}_{stage}.json"
    )
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.analysis.audit_patch_split_overlap",
            "--patch-csv",
            str(patch_csv),
            "--report-json",
            str(report_path),
            "--fail-on-overlap",
        ],
        f"{branch['key']}: {stage} zero-tolerance cross-split overlap audit",
    )


def make_training_config(training_seeds: tuple[int, ...]) -> dict[str, Any]:
    cfg = json.loads(HARDENING_CONFIG.read_text(encoding="utf-8"))
    cfg["expected_runs"] = len(BRANCHES) * len(training_seeds)
    cfg["continue_on_error"] = False
    cfg["datasets"] = {}
    for branch in BRANCHES:
        patch_csv, h5_path, suffix = dataset_paths(branch)
        cfg["datasets"][str(branch["dataset"])] = {
            "patch_csv": str(patch_csv.relative_to(ROOT)),
            "h5_path": str(h5_path.relative_to(ROOT)),
            "suffix": suffix,
            "in_ch": 3,
        }
    template_run = dict(cfg["runs"][0])
    cfg["runs"] = []
    for seed in training_seeds:
        run_cfg = dict(template_run)
        run_cfg["seed"] = seed
        cfg["runs"].append(run_cfg)
    cfg["_comment"] = (
        "Final constant-volume scene ablation. Seven combinations of the "
        "2003/2016/2023 summer scenes, 6,000 patches each, shared spatial holdouts."
    )
    return cfg


def write_configs() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = make_training_config(TRAIN_SEEDS)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    additional = make_training_config(ADDITIONAL_TRAIN_SEEDS)
    additional["_comment"] = (
        "Predeclared seed extension for the final scene ablation. These two "
        "seeds combine with 123/231/312 to give five seeds per branch."
    )
    ADDITIONAL_CONFIG_PATH.write_text(
        json.dumps(additional, indent=2), encoding="utf-8"
    )

    smoke = dict(cfg)
    smoke["expected_runs"] = 1
    first_name = str(BRANCHES[0]["dataset"])
    smoke["datasets"] = {first_name: cfg["datasets"][first_name]}
    smoke["runs"] = [dict(cfg["runs"][0])]
    smoke["stage2_epochs"] = 1
    smoke["stage2_patience"] = 1
    smoke["use_temp_scaling"] = False
    smoke["runs"][0]["use_temp_scaling"] = False
    smoke["_comment"] = "One-epoch runtime smoke test; not a scientific result."
    SMOKE_CONFIG_PATH.write_text(json.dumps(smoke, indent=2), encoding="utf-8")


def write_manifest() -> None:
    datasets = []
    for branch in BRANCHES:
        patch_csv, h5_path, suffix = dataset_paths(branch)
        datasets.append(
            {
                **branch,
                "patch_csv": str(patch_csv),
                "patch_rows": count_rows(patch_csv),
                "h5_path": str(h5_path),
                "h5_bytes": h5_path.stat().st_size if h5_path.exists() else 0,
                "suffix": suffix,
            }
        )
    manifest = {
        "experiment": EXPERIMENT,
        "created_or_updated": now(),
        "scientific_question": (
            "What is the marginal value of one, two, or three summer acquisitions "
            "under constant training volume and common spatial holdouts?"
        ),
        "status": "prepared",
        "design": {
            "scenes": ["20030709_qb", "20160825_wv", "20230812_pl"],
            "total_patches_per_configuration": TOTAL_PATCHES,
            "patch_seed": PATCH_SEED,
            "training_seeds": list(TRAIN_SEEDS),
            "additional_training_seeds": list(ADDITIONAL_TRAIN_SEEDS),
            "validation_blocks": list(VAL_BLOCKS),
            "test_blocks": list(TEST_BLOCKS),
            "split_fraction_semantics": (
                "The 0.10 CLI fractions guide block selection only. "
                "Realized row fractions are reported per dataset."
            ),
            "feature_preset": FEATURE_PRESET,
            "normalization": {
                "shared_across_configurations": True,
                "per_scene": True,
                "label_independent": True,
                "fit_support": "regular 64 px windows fully inside fixed train blocks",
                "stats_csv": str(COMMON_STATS_CSV),
            },
            "cross_split_overlap_audit": {
                "scope": "common analysis grid, including cross-scene windows",
                "tolerance_px2": 0,
                "required_to_pass_before_training": True,
            },
            "architecture": "SegFormer MiT-B2",
            "evaluation_only": True,
            "canonical_products_untouched": True,
        },
        "datasets": datasets,
        "config": str(CONFIG_PATH),
        "additional_seed_config": str(ADDITIONAL_CONFIG_PATH),
        "run_root": str(RUN_ROOT),
        "additional_seed_run_root": str(ADDITIONAL_RUN_ROOT),
    }
    PREP_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing CPU training")
    log(
        f"[runtime] python={sys.executable} torch={torch.__version__} "
        f"cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}"
    )


def run_suite(config: Path, out_root: Path, *, dry_run: bool = False) -> None:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "scripts.train.train_suite",
        "--config",
        str(config),
        "--mode",
        "stage2",
        "--out_root",
        str(out_root),
        "--amp",
    ]
    if dry_run:
        cmd.append("--dry_run")
    run(cmd, f"train suite: {config.name}")


def select_branches(value: str) -> list[dict[str, Any]]:
    if value == "all":
        return BRANCHES
    wanted = {part.strip() for part in value.split(",") if part.strip()}
    selected = [branch for branch in BRANCHES if branch["key"] in wanted]
    missing = wanted - {branch["key"] for branch in selected}
    if missing:
        raise SystemExit(f"Unknown branches: {sorted(missing)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser("Final 2003/2016/2023 scene ablation")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--additional-seeds", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--branches", default="all")
    args = parser.parse_args()

    if args.prepare_only and args.train_only:
        raise SystemExit("--prepare-only and --train-only are mutually exclusive")
    if args.smoke_test and args.additional_seeds:
        raise SystemExit("--smoke-test and --additional-seeds are mutually exclusive")
    selected = select_branches(args.branches)
    PREP_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.resume and not args.train_only:
        PREP_LOG.write_text("", encoding="utf-8")

    if not args.train_only:
        prepare_common_features(resume=args.resume)
        for branch in selected:
            prepare_branch(branch, resume=args.resume)
        write_configs()
        write_manifest()

    if args.prepare_only:
        log("[OK] preparation complete")
        return 0

    require_cuda()
    if args.smoke_test:
        run_suite(
            SMOKE_CONFIG_PATH,
            OUT / "runs" / "train" / f"{EXPERIMENT}_smoke",
            dry_run=args.dry_run,
        )
    elif args.additional_seeds:
        run_suite(
            ADDITIONAL_CONFIG_PATH,
            ADDITIONAL_RUN_ROOT,
            dry_run=args.dry_run,
        )
    else:
        run_suite(CONFIG_PATH, RUN_ROOT, dry_run=args.dry_run)
    log("[OK] requested scene-ablation stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
