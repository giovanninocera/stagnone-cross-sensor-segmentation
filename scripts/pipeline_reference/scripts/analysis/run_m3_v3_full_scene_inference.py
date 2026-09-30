"""Run five-seed M3 v3 full-scene inference with full Hann blending."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.analysis.evaluate_final_ablation_unique_pixels import find_run_dir


ROOT = Path(r"${PROJECT_ROOT}")
OUT_NAME = "final_scene_ablation_031623_v3_m3_full_hann"
OUT_ROOT = ROOT / "out/pred" / OUT_NAME
SCENES = ("20030709_qb", "20160825_wv", "20230812_pl")
SEEDS = (123, 231, 312, 423, 531)
SUFFIX = "_st3b_final_scene_ablation_031623_v3_common"


def command(seed: int, scene: str, overwrite: bool) -> list[str]:
    directory = find_run_dir("M3_031623", seed, require_complete=True)
    result = [
        sys.executable,
        "-u",
        "-m",
        "scripts.infer.infer_tiles",
        "--scene_id",
        scene,
        "--suffix",
        SUFFIX,
        "--source_suffix",
        SUFFIX,
        "--feature_preset",
        "CMP_BASE3",
        "--model_path",
        str(directory / "model_best.pt"),
        "--out_dir",
        f"{OUT_NAME}/seed{seed}",
        "--win",
        "512",
        "--pad",
        "128",
        "--blend_mode",
        "full_hann",
        "--batch",
        "8",
        "--amp",
        "--postprocess",
        "none",
        "--save_summary_json",
    ]
    if overwrite:
        result.append("--overwrite")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commands = [
        {
            "seed": seed,
            "scene": scene,
            "command": command(seed, scene, args.overwrite),
        }
        for seed in SEEDS
        for scene in SCENES
    ]
    if args.dry_run:
        print(json.dumps(commands, indent=2))
        return 0

    failures: list[dict[str, object]] = []
    for item in commands:
        seed = int(item["seed"])
        scene = str(item["scene"])
        print(f"\n[INFER] M3 v3 seed={seed} scene={scene}", flush=True)
        result = subprocess.run(
            list(item["command"]),
            cwd=ROOT,
            check=False,
        )
        if result.returncode:
            failures.append(
                {
                    "seed": seed,
                    "scene": scene,
                    "return_code": result.returncode,
                }
            )
            break
    if failures:
        raise RuntimeError(f"Full-scene inference failed: {failures}")
    print(f"[OK] Completed {len(commands)} full-scene predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
