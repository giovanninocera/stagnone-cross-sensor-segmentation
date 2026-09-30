#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/06_suite_train.py [v5.2 — _ST]
Suite di training per pipeline _ST operational con supporto esplicito a più dataset,
ciascuno con proprio CSV e proprio store HDF5/PT.

Correzioni chiave v5.1:
- supporto reviewer-grade a store separati per dataset:
    "datasets": {
      "A1_2006": {"patch_csv": "...csv", "h5_path": "...h5"},
      "A2_2016": {"patch_csv": "...csv", "h5_path": "...h5"},
      "A3_multi_fair": {"patch_csv": "...csv", "h5_path": "...h5"}
    }
- fix robusto di use_temp_scaling nei run espliciti (nessun default silenzioso a True)
- leaderboard arricchito con metriche raw/scaled quando disponibili

Compatibilità:
- mantiene compatibilità con il formato legacy:
    "datasets": {"A1_2006": "out/patches/patches_2006_st12s.csv"}
  In tal caso usa h5_path globale o pt_dir globale.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.common.config import OUT, expected_in_channels, p_pt_store


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _run(cmd: List[str], log_path: Optional[Path] = None) -> int:
    if log_path is None:
        return subprocess.call(cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"CMD: {' '.join(cmd)}\n\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def _tail_text(path: Path, n_chars: int = 6000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        return txt[-n_chars:].lower()
    except Exception:
        return ""


def _is_oom(log_tail: str) -> bool:
    return (("out of memory" in log_tail) or ("cuda out of memory" in log_tail) or ("cublas" in log_tail and "alloc" in log_tail))


@dataclass
class RunSpec:
    arch: str
    encoder: str
    lr: float
    wd: float
    loss_mix: str
    label_smooth: float
    batch: int
    seed: int
    grad_accum_steps: int = 1
    thr_metric: str = "iou"
    selection_metric_mode: str = "best"
    use_temp_scaling: bool = False
    augment: Optional[bool] = None
    boundary_weight: bool = True
    boundary_radius: int = 3
    boundary_factor: float = 3.0
    cutmix_prob: float = 0.0
    aug_profile: str = "default"
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7


@dataclass
class DatasetSpec:
    name: str
    patch_csv: Path
    h5_path: str = ""
    pt_dir: str = ""
    suffix: str = ""
    in_ch: int = 0


def spec_id(s: RunSpec) -> str:
    lm = s.loss_mix.replace(":", "").replace(",", "_").replace(".", "p")
    def _float_tag(prefix: str, value: float) -> str:
        mantissa, exponent = f"{value:.1e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
        exp_int = int(exponent)
        exp_tag = ("em" if exp_int < 0 else "ep") + str(abs(exp_int))
        return f"{prefix}{mantissa}{exp_tag}"
    lr_ = _float_tag("lr", s.lr)
    wd_ = _float_tag("wd", s.wd)
    ls_ = f"ls{str(s.label_smooth).replace('.', 'p')}"
    extra: List[str] = []
    if int(s.grad_accum_steps) > 1:
        extra.append(f"ga{int(s.grad_accum_steps)}")
    if "tversky" in s.loss_mix:
        extra.append(f"tv{str(s.tversky_alpha).replace('.', 'p')}_{str(s.tversky_beta).replace('.', 'p')}")
    if s.augment is not None:
        extra.append("augon" if s.augment else "augoff")
    if s.aug_profile != "default":
        extra.append(f"aug_{s.aug_profile}")
    extra_tag = ("_" + "_".join(extra)) if extra else ""
    return f"{s.arch}_{s.encoder}_{lr_}_{wd_}_{lm}_{ls_}_b{s.batch}_s{s.seed}{extra_tag}"


def _resolve_uts(r: Dict[str, Any], global_uts: bool) -> bool:
    if "use_temp_scaling" in r:
        return bool(r["use_temp_scaling"])
    return global_uts


def _cfg_list(block: Dict[str, Any], parent: Dict[str, Any], key: str, default: List[Any]) -> List[Any]:
    value = block.get(key, parent.get(key, default))
    if isinstance(value, list):
        return value
    return [value]


def _cfg_scalar(block: Dict[str, Any], parent: Dict[str, Any], key: str, default: Any) -> Any:
    return block.get(key, parent.get(key, default))


def _grid_from_block(block: Dict[str, Any], parent: Dict[str, Any], global_uts: bool) -> List[RunSpec]:
    arches = _cfg_list(block, parent, "arches", ["unetpp"])
    encoders = _cfg_list(block, parent, "encoders", ["resnet34"])
    lrs = _cfg_list(block, parent, "lrs", [2e-4])
    wds = _cfg_list(block, parent, "wds", [1e-4])
    loss_mixes = _cfg_list(block, parent, "loss_mixes", ["bce:0.5,dice:0.3,focal:0.2"])
    lss = _cfg_list(block, parent, "label_smooths", [0.0])
    batches = _cfg_list(block, parent, "batches", [32])
    seeds = _cfg_list(block, parent, "seeds", [123])
    grad_accums = _cfg_list(block, parent, "grad_accum_steps", [1])
    thr = str(_cfg_scalar(block, parent, "thr_metric", "iou"))
    selection_metric_mode = str(_cfg_scalar(block, parent, "selection_metric_mode", "best"))

    bw = bool(_cfg_scalar(block, parent, "boundary_weight", True))
    br = int(_cfg_scalar(block, parent, "boundary_radius", 3))
    bf = float(_cfg_scalar(block, parent, "boundary_factor", 3.0))
    cutmix = float(_cfg_scalar(block, parent, "cutmix_prob", 0.0))
    augment_flag = block.get("augment", parent.get("augment", None))
    aug_profile = str(_cfg_scalar(block, parent, "aug_profile", "default"))
    tversky_alpha = float(_cfg_scalar(block, parent, "tversky_alpha", 0.3))
    tversky_beta = float(_cfg_scalar(block, parent, "tversky_beta", 0.7))
    uts = _resolve_uts(block, global_uts)

    specs: List[RunSpec] = []
    for arch, enc, lr, wd, lm, ls, batch, ga, seed in itertools.product(
        arches, encoders, lrs, wds, loss_mixes, lss, batches, grad_accums, seeds
    ):
        specs.append(
            RunSpec(
                arch=str(arch),
                encoder=str(enc),
                lr=float(lr),
                wd=float(wd),
                loss_mix=str(lm),
                label_smooth=float(ls),
                batch=int(batch),
                seed=int(seed),
                grad_accum_steps=int(ga),
                thr_metric=thr,
                selection_metric_mode=selection_metric_mode,
                use_temp_scaling=uts,
                augment=(None if augment_flag is None else bool(augment_flag)),
                boundary_weight=bw,
                boundary_radius=br,
                boundary_factor=bf,
                cutmix_prob=cutmix,
                aug_profile=aug_profile,
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta,
            )
        )
    return specs


def make_grid(cfg: Dict[str, Any]) -> List[RunSpec]:
    global_uts = bool(cfg.get("use_temp_scaling", False))
    if cfg.get("runs"):
        specs: List[RunSpec] = []
        for r in cfg["runs"]:
            specs.append(
                RunSpec(
                    arch=r["arch"],
                    encoder=r["encoder"],
                    lr=float(r["lr"]),
                    wd=float(r.get("wd", 1e-4)),
                    loss_mix=r["loss_mix"],
                    label_smooth=float(r.get("label_smooth", 0.0)),
                    batch=int(r.get("batch", 32)),
                    seed=int(r.get("seed", 123)),
                    grad_accum_steps=int(r.get("grad_accum_steps", cfg.get("grad_accum_steps", 1))),
                    thr_metric=str(r.get("thr_metric", cfg.get("thr_metric", "iou"))),
                    selection_metric_mode=str(r.get("selection_metric_mode", cfg.get("selection_metric_mode", "best"))),
                    use_temp_scaling=_resolve_uts(r, global_uts),
                    augment=(None if "augment" not in r else bool(r.get("augment"))),
                    boundary_weight=bool(r.get("boundary_weight", cfg.get("boundary_weight", True))),
                    boundary_radius=int(r.get("boundary_radius", cfg.get("boundary_radius", 3))),
                    boundary_factor=float(r.get("boundary_factor", cfg.get("boundary_factor", 3.0))),
                    cutmix_prob=float(r.get("cutmix_prob", cfg.get("cutmix_prob", 0.0))),
                    aug_profile=str(r.get("aug_profile", cfg.get("aug_profile", "default"))),
                    tversky_alpha=float(r.get("tversky_alpha", cfg.get("tversky_alpha", 0.3))),
                    tversky_beta=float(r.get("tversky_beta", cfg.get("tversky_beta", 0.7))),
                )
            )
        return specs

    if cfg.get("model_groups"):
        specs: List[RunSpec] = []
        for group in cfg["model_groups"]:
            specs.extend(_grid_from_block(group, cfg, global_uts))
        return specs

    return _grid_from_block(cfg, cfg, global_uts)


def build_cmd(
    spec: RunSpec,
    patch_csv: Path,
    out_dir: Path,
    suffix: str,
    tile: int,
    in_ch: int,
    epochs: int,
    patience: int,
    warmup: int,
    scheduler: str,
    num_workers: int,
    eval_batch: int,
    grad_clip: float,
    encoder_weights: str,
    input_adapter: str,
    augment: bool,
    amp: bool,
    depth_aug_prob: float = 0.0,
    cutout_prob: float = 0.0,
    cache_ram: bool = False,
    perf_diag_epochs: int = 0,
    perf_diag_batches: int = 0,
    pt_dir: str = "",
    h5_path: str = "",
) -> List[str]:
    cmd = [
        sys.executable, "-m", "scripts.train.train_segformer",
        "--patch_csv", str(patch_csv),
        "--out_dir", str(out_dir),
        "--arch", spec.arch,
        "--encoder", spec.encoder,
        "--encoder_weights", encoder_weights,
        "--input_adapter", input_adapter,
        "--in_ch", str(in_ch),
        "--tile", str(tile),
        "--suffix", suffix,
        "--epochs", str(epochs),
        "--early_patience", str(patience),
        "--warmup_epochs", str(warmup),
        "--scheduler", scheduler,
        "--batch", str(spec.batch),
        "--grad_accum_steps", str(spec.grad_accum_steps),
        "--lr", str(spec.lr),
        "--wd", str(spec.wd),
        "--loss_mix", spec.loss_mix,
        "--label_smooth", str(spec.label_smooth),
        "--tversky_alpha", str(spec.tversky_alpha),
        "--tversky_beta", str(spec.tversky_beta),
        "--thr_metric", spec.thr_metric,
        "--selection_metric_mode", str(spec.selection_metric_mode),
        "--grad_clip", str(grad_clip),
        "--num_workers", str(num_workers),
        "--seed", str(spec.seed),
        "--aug_profile", str(spec.aug_profile),
    ]
    if eval_batch > 0:
        cmd += ["--eval_batch", str(eval_batch)]

    if h5_path:
        cmd += ["--h5_path", str(h5_path)]
    else:
        cmd += ["--pt_dir", str(pt_dir)]

    if amp:
        cmd.append("--amp")
    use_augment = spec.augment if spec.augment is not None else augment
    if use_augment:
        cmd.append("--augment")
    else:
        cmd.append("--no_augment")
    if spec.use_temp_scaling:
        cmd.append("--use_temp_scaling")
    # Boundary-weighted loss
    if spec.boundary_weight:
        cmd += ["--boundary_radius", str(spec.boundary_radius),
                "--boundary_factor", str(spec.boundary_factor)]
    else:
        cmd.append("--no_boundary_weight")
    # CutMix cross-patch
    if spec.cutmix_prob > 0.0:
        cmd += ["--cutmix_prob", str(spec.cutmix_prob)]
    if depth_aug_prob > 0.0:
        cmd += ["--depth_aug_prob", str(depth_aug_prob)]
    if cutout_prob > 0.0:
        cmd += ["--cutout_prob", str(cutout_prob)]
    if cache_ram:
        cmd.append("--cache_ram")
    if perf_diag_epochs > 0:
        cmd += ["--perf_diag_epochs", str(perf_diag_epochs)]
        cmd += ["--perf_diag_batches", str(max(0, perf_diag_batches))]
    return cmd


def _read_result(out_dir: Path) -> Optional[Dict[str, Any]]:
    mj = out_dir / "model_best.json"
    if not mj.exists():
        return None
    try:
        return json.loads(mj.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_complete_path(out_dir: Path) -> Path:
    return out_dir / "run_complete.json"


def _load_run_complete(out_dir: Path) -> Dict[str, Any]:
    marker = _run_complete_path(out_dir)
    if not marker.exists():
        return {}
    try:
        return json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_completed_ok(out_dir: Path) -> bool:
    payload = _load_run_complete(out_dir)
    if int(payload.get("return_code", 1)) != 0:
        return False
    meta = _read_result(out_dir)
    if meta is None:
        return False
    if bool(meta.get("calibration_enabled", False)) and not bool(meta.get("temp_scaling_done", False)):
        return False
    return True


def _reset_run_dir(out_dir: Path) -> None:
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return
    for child in out_dir.iterdir():
        if child.is_dir():
            import shutil
            shutil.rmtree(child)
        else:
            child.unlink(missing_ok=True)


def _write_run_complete(out_dir: Path, row: Dict[str, Any]) -> None:
    _run_complete_path(out_dir).write_text(json.dumps(row, indent=2), encoding="utf-8")


def update_leaderboard(lb_path: Path, ds_name: str, spec: RunSpec, out_dir: Path, elapsed: float, rc: int) -> None:
    res = _read_result(out_dir)
    row = {
        "dataset": ds_name,
        "run_id": spec_id(spec),
        "arch": spec.arch,
        "encoder": spec.encoder,
        "lr": spec.lr,
        "wd": spec.wd,
        "loss_mix": spec.loss_mix,
        "label_smooth": spec.label_smooth,
        "batch": spec.batch,
        "grad_accum_steps": spec.grad_accum_steps,
        "seed": spec.seed,
        "augment": ("" if spec.augment is None else spec.augment),
        "aug_profile": spec.aug_profile,
        "tversky_alpha": spec.tversky_alpha,
        "tversky_beta": spec.tversky_beta,
        "use_temp_scaling": spec.use_temp_scaling,
        "boundary_weight": spec.boundary_weight,
        "boundary_factor": spec.boundary_factor,
        "cutmix_prob": spec.cutmix_prob,
        "return_code": rc,
        "best_epoch":  res.get("best_epoch", "") if res else "",
        "best_metric": res.get("best_metric", "") if res else "",
        "best_metric_name": res.get("thr_metric", "") if res else "",
        "best_metric_raw": res.get("best_metric_raw", "") if res else "",
        "best_metric_scaled": res.get("best_metric_scaled", "") if res else "",
        "best_thr": res.get("best_thr", "") if res else "",
        "best_thr_raw": res.get("best_thr_raw", "") if res else "",
        "min_iou_at_05_raw": res.get("min_iou_at_05_raw", "") if res else "",
        "harmonic_iou_at_05_raw": res.get("harmonic_iou_at_05_raw", "") if res else "",
        "min_iou_best_raw": res.get("min_iou_best_raw", "") if res else "",
        "harmonic_iou_best_raw": res.get("harmonic_iou_best_raw", "") if res else "",
        "iou_at_05_raw": res.get("iou_at_05_raw", "") if res else "",
        "f1_at_05_raw": res.get("f1_at_05_raw", "") if res else "",
        "iou_veg_at_05_raw": res.get("iou_veg_at_05_raw", "") if res else "",
        "iou_sand_at_05_raw": res.get("iou_sand_at_05_raw", "") if res else "",
        "iou_veg_best_raw": res.get("iou_veg_best_raw", "") if res else "",
        "iou_sand_best_raw": res.get("iou_sand_best_raw", "") if res else "",
        "balanced_accuracy_raw": res.get("balanced_accuracy_raw", "") if res else "",
        "mcc_raw": res.get("mcc_raw", "") if res else "",
        "min_iou_at_05_scaled": res.get("min_iou_at_05_scaled", "") if res else "",
        "harmonic_iou_at_05_scaled": res.get("harmonic_iou_at_05_scaled", "") if res else "",
        "min_iou_best_scaled": res.get("min_iou_best_scaled", "") if res else "",
        "harmonic_iou_best_scaled": res.get("harmonic_iou_best_scaled", "") if res else "",
        "iou_at_05_scaled": res.get("iou_at_05_scaled", "") if res else "",
        "f1_at_05_scaled": res.get("f1_at_05_scaled", "") if res else "",
        "iou_veg_at_05_scaled": res.get("iou_veg_at_05_scaled", "") if res else "",
        "iou_sand_at_05_scaled": res.get("iou_sand_at_05_scaled", "") if res else "",
        "iou_veg_best_scaled": res.get("iou_veg_best_scaled", "") if res else "",
        "iou_sand_best_scaled": res.get("iou_sand_best_scaled", "") if res else "",
        "balanced_accuracy_scaled": res.get("balanced_accuracy_scaled", "") if res else "",
        "mcc_scaled": res.get("mcc_scaled", "") if res else "",
        "temperature": res.get("temperature", "") if res else "",
        "temp_scaling_done": res.get("temp_scaling_done", "") if res else "",
        "threshold_source": res.get("threshold_source", "") if res else "",
        "calibration_enabled": res.get("calibration_enabled", "") if res else "",
        "out_dir": str(out_dir),
        "elapsed_s": int(elapsed),
        "timestamp": _now(),
    }
    fields = list(row.keys())
    rows: List[Dict[str, Any]] = []
    if lb_path.exists():
        with open(lb_path, newline="", encoding="utf-8") as f:
            for old_row in csv.DictReader(f):
                if not (
                    old_row.get("dataset") == row["dataset"]
                    and old_row.get("run_id") == row["run_id"]
                ):
                    rows.append(old_row)
    rows.append(row)
    with open(lb_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_leaderboard(lb_path: Path, n: int = 10) -> None:
    if not lb_path.exists():
        return
    rows: List[Dict[str, Any]] = []
    with open(lb_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["best_metric"] = float(r["best_metric"])
            except Exception:
                r["best_metric"] = -1.0
            rows.append(r)
    rows.sort(key=lambda x: -x["best_metric"])
    print(f"\n{'='*70}")
    print(f" LEADERBOARD: {lb_path.name} (top {min(n, len(rows))}/{len(rows)})")
    print(f"{'='*70}")
    for r in rows[:n]:
        metric_name = r.get("best_metric_name", "metric")
        try:
            veg = float(r.get("iou_veg_best_raw", ""))
        except Exception:
            veg = -1.0
        try:
            sand = float(r.get("iou_sand_best_raw", ""))
        except Exception:
            sand = -1.0
        print(
            f" [{r.get('dataset','?'):14s}] {r['run_id']:<52} "
            f"{metric_name}={r['best_metric']:.4f}  veg={veg:.4f}  sand={sand:.4f}"
        )
    print(f"{'='*70}\n")


def parse_datasets(
    cfg: Dict[str, Any],
    args_h5: str,
    args_pt_dir: str,
    suffix: str,
    tile: int,
    in_ch_default: int,
) -> Dict[str, DatasetSpec]:
    ds_raw = cfg.get("datasets", {})
    if not ds_raw:
        raise SystemExit("Config JSON deve avere chiave 'datasets' con almeno un dataset.")

    global_h5 = args_h5 or str(cfg.get("h5_path", ""))
    global_pt = args_pt_dir or str(p_pt_store(suffix, tile))
    datasets: Dict[str, DatasetSpec] = {}

    for name, item in ds_raw.items():
        if isinstance(item, str):
            datasets[name] = DatasetSpec(
                name=name,
                patch_csv=Path(item),
                h5_path=global_h5,
                pt_dir=global_pt,
                suffix=suffix,
                in_ch=in_ch_default,
            )
        elif isinstance(item, dict):
            patch_csv = item.get("patch_csv") or item.get("csv")
            if not patch_csv:
                raise SystemExit(f"Dataset '{name}' deve definire 'patch_csv'.")
            datasets[name] = DatasetSpec(
                name=name,
                patch_csv=Path(patch_csv),
                h5_path=str(item.get("h5_path", global_h5) or ""),
                pt_dir=str(item.get("pt_dir", global_pt) or ""),
                suffix=str(item.get("suffix", suffix) or suffix),
                in_ch=int(item.get("in_ch", in_ch_default) or in_ch_default),
            )
        else:
            raise SystemExit(f"Formato dataset non valido per '{name}': {type(item)}")
    return datasets


def run_stage(
    stage: int,
    grid: List[RunSpec],
    datasets: Dict[str, DatasetSpec],
    out_root: Path,
    suffix: str,
    tile: int,
    in_ch: int,
    epochs: int,
    patience: int,
    warmup: int,
    scheduler: str,
    num_workers: int,
    eval_batch: int,
    grad_clip: float,
    encoder_weights: str,
    input_adapter: str,
    augment: bool,
    amp: bool,
    depth_aug_prob: float,
    cutout_prob: float,
    cache_ram: bool,
    perf_diag_epochs: int,
    perf_diag_batches: int,
    dry_run: bool,
    overwrite: bool,
    continue_on_error: bool,
    oom_batch_fallback: List[int],
) -> None:
    stage_tag = f"stage{stage}"
    print(f"\n{'='*70}")
    print(f" STAGE {stage}")
    print(f" {len(grid)} run - {len(datasets)} dataset = {len(grid)*len(datasets)} esecuzioni")
    print(f" epochs={epochs} patience={patience} augment={augment} amp={amp} num_workers={num_workers} eval_batch={eval_batch or 'train'} cache_ram={cache_ram}")
    print(f" perf_diag_epochs={perf_diag_epochs} perf_diag_batches={perf_diag_batches or 'all'}")
    print(f" continue_on_error={continue_on_error} oom_batch_fallback={oom_batch_fallback}")
    print(f"{'='*70}\n")

    for ds_name, ds in datasets.items():
        if not ds.patch_csv.exists():
            print(f" [SKIP] Dataset non trovato: {ds.patch_csv}")
            continue

        lb_path = out_root / f"leaderboard_{stage_tag}_{ds_name}.csv"
        log_path = out_root / "_logs" / f"{stage_tag}_{ds_name}.log"
        print(f"\n --- Dataset: {ds_name} ({ds.patch_csv.name}) ---")
        print(f"     suffix={ds.suffix} in_ch={ds.in_ch}")
        print(f"     store: {ds.h5_path if ds.h5_path else ds.pt_dir}")

        for i, spec in enumerate(grid, 1):
            rid = spec_id(spec)
            out_dir = out_root / stage_tag / ds_name / rid
            out_dir.mkdir(parents=True, exist_ok=True)

            if _run_completed_ok(out_dir) and not overwrite:
                bm = _load_run_complete(out_dir).get("best_metric", "?")
                print(f" [{i:03d}/{len(grid)}] SKIP {rid} (metric={bm})")
                continue

            # Resume-safe semantics:
            # - completed run: skip
            # - partial/interrupted run: wipe only that run dir and rerun it
            # - overwrite=True: force clean rerun
            if overwrite or any(out_dir.iterdir()):
                _reset_run_dir(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

            print(f" [{i:03d}/{len(grid)}] {ds_name}/{rid} [temp_scaling={spec.use_temp_scaling}]")

            cmd = build_cmd(
                spec=spec,
                patch_csv=ds.patch_csv,
                out_dir=out_dir,
                suffix=ds.suffix,
                tile=tile,
                in_ch=ds.in_ch,
                epochs=epochs,
                patience=patience,
                warmup=warmup,
                scheduler=scheduler,
                num_workers=num_workers,
                eval_batch=eval_batch,
                grad_clip=grad_clip,
                encoder_weights=encoder_weights,
                input_adapter=input_adapter,
                augment=augment,
                amp=amp,
                depth_aug_prob=depth_aug_prob,
                cutout_prob=cutout_prob,
                cache_ram=cache_ram,
                perf_diag_epochs=perf_diag_epochs,
                perf_diag_batches=perf_diag_batches,
                pt_dir=ds.pt_dir,
                h5_path=ds.h5_path,
            )
            if dry_run:
                print(f" [DRY] {' '.join(cmd[2:12])} ...")
                continue

            t0 = time.time()
            rc = _run(cmd, log_path=log_path)
            elapsed = time.time() - t0

            if rc != 0 and oom_batch_fallback:
                tail = _tail_text(log_path)
                if _is_oom(tail):
                    for b2 in oom_batch_fallback:
                        if b2 >= spec.batch:
                            continue
                        print(f"  [RETRY OOM] batch {spec.batch} -> {b2}")
                        # BUG FIX v5.2: copia TUTTI i campi dal spec originale
                        # In v5.1 mancavano boundary_weight/radius/factor/cutmix_prob
                        spec2 = RunSpec(
                            arch=spec.arch, encoder=spec.encoder, lr=spec.lr, wd=spec.wd,
                            loss_mix=spec.loss_mix, label_smooth=spec.label_smooth,
                            batch=b2, seed=spec.seed, thr_metric=spec.thr_metric,
                            use_temp_scaling=spec.use_temp_scaling,
                            boundary_weight=spec.boundary_weight,
                            boundary_radius=spec.boundary_radius,
                            boundary_factor=spec.boundary_factor,
                            cutmix_prob=spec.cutmix_prob,
                            aug_profile=spec.aug_profile,
                            tversky_alpha=spec.tversky_alpha,
                            tversky_beta=spec.tversky_beta,
                        )
                        rid2 = spec_id(spec2)
                        out_dir2 = out_root / stage_tag / ds_name / rid2
                        out_dir2.mkdir(parents=True, exist_ok=True)
                        cmd2 = build_cmd(
                            spec=spec2, patch_csv=ds.patch_csv, out_dir=out_dir2,
                            suffix=ds.suffix, tile=tile, in_ch=ds.in_ch, epochs=epochs,
                            patience=patience, warmup=warmup, scheduler=scheduler,
                            num_workers=num_workers, eval_batch=eval_batch, grad_clip=grad_clip,
                            encoder_weights=encoder_weights, input_adapter=input_adapter,
                            augment=augment, amp=amp, depth_aug_prob=depth_aug_prob,
                            cutout_prob=cutout_prob, cache_ram=cache_ram,
                            perf_diag_epochs=perf_diag_epochs,
                            perf_diag_batches=perf_diag_batches,
                            pt_dir=ds.pt_dir, h5_path=ds.h5_path,
                        )
                        t1 = time.time()
                        rc2 = _run(cmd2, log_path=log_path)
                        elapsed2 = time.time() - t1
                        res2 = _read_result(out_dir2)
                        bm2 = res2.get("best_metric", None) if res2 else None
                        update_leaderboard(lb_path, ds_name, spec2, out_dir2, elapsed2, rc2)
                        print(f"    rc={rc2} {elapsed2:.0f}s metric={bm2:.4f}" if bm2 is not None else f"    rc={rc2} {elapsed2:.0f}s (no metric)")
                        if rc2 == 0:
                            break
                    continue

            res = _read_result(out_dir)
            bm = res.get("best_metric", None) if res else None
            T = res.get("temperature", 1.0) if res else 1.0
            tsd = res.get("temp_scaling_done", False) if res else False
            print(f"  rc={rc} {elapsed:.0f}s metric={bm:.4f} T={T:.3f} temp_scaling_done={tsd}" if bm is not None else f"  rc={rc} {elapsed:.0f}s (no metric)")
            update_leaderboard(lb_path, ds_name, spec, out_dir, elapsed, rc)
            _write_run_complete(
                out_dir,
                {
                    "dataset": ds_name,
                    "run_id": rid,
                    "return_code": rc,
                    "best_metric": bm if bm is not None else "",
                    "temperature": T,
                    "temp_scaling_done": tsd,
                    "elapsed_s": int(elapsed),
                    "timestamp": _now(),
                    "out_dir": str(out_dir),
                },
            )
            # Stampa leaderboard solo all'ultimo run del dataset (meno rumoroso)
            if i == len(grid):
                print_leaderboard(lb_path)
            if rc != 0 and not continue_on_error:
                raise SystemExit(f"[FATAL] Run fallito (rc={rc}) e continue_on_error=False. Vedi log: {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Suite training - _ST v5.1")
    ap.add_argument("--config", required=True, help="Path al JSON di configurazione.")
    ap.add_argument("--mode", default="both", choices=["stage1", "stage2", "both"])
    ap.add_argument("--suffix", default="_st16s")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--in_ch", type=int, default=0,
                    help="Input channels. Default: inferred from JSON or suffix.")
    ap.add_argument("--out_root", default="", help="Override out_root.")
    ap.add_argument("--pt_dir", default="", help="PT store globale legacy. Supporta liste separate da virgole/; per component stores.")
    ap.add_argument("--h5_path", default="", help="H5 globale legacy. Supporta liste separate da virgole/; per component stores.")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--num_workers", type=int, default=-1,
                    help="Numero di worker DataLoader. -1 = legge dal JSON (default). "
                         "Passare un valore >= 0 sovrascrive il JSON.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--epochs_s1", type=int, default=-1)
    ap.add_argument("--epochs_s2", type=int, default=-1)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config non trovato: {cfg_path}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))

    root = Path(args.out_root).resolve() if args.out_root else (OUT / "suite_train")
    root.mkdir(parents=True, exist_ok=True)

    warmup = int(cfg.get("warmup_epochs", 3))
    sched = str(cfg.get("scheduler", "cosine"))
    nw_json = int(cfg.get("num_workers", 2))
    nw = nw_json if args.num_workers < 0 else args.num_workers
    if args.num_workers >= 0:
        print(f"  [CLI] num_workers={nw} (override JSON value={nw_json})")
    gc = float(cfg.get("grad_clip", 1.0))
    ew = str(cfg.get("encoder_weights", "imagenet"))
    input_adapter = str(cfg.get("input_adapter", "auto"))
    aug = bool(cfg.get("augment", True))
    amp_flag = args.amp or bool(cfg.get("amp", False))
    depth_aug_prob = float(cfg.get("depth_aug_prob", 0.0))
    cutout_prob = float(cfg.get("cutout_prob", 0.0))
    cache_ram = bool(cfg.get("cache_ram", False))
    eval_batch = int(cfg.get("eval_batch", 0))
    if eval_batch <= 0 and str(cfg.get("selection_metric_mode", "")).lower() == "at_05":
        eval_batch = 16
    perf_diag_epochs = max(0, int(cfg.get("perf_diag_epochs", 0)))
    perf_diag_batches = max(0, int(cfg.get("perf_diag_batches", 0)))
    continue_on_error = bool(cfg.get("continue_on_error", True))
    oom_fallback = cfg.get("oom_batch_fallback", [32, 24, 16])
    if not isinstance(oom_fallback, list) or not oom_fallback:
        oom_fallback = []
    oom_fallback = [int(x) for x in oom_fallback]

    # in_ch: CLI ha priorità, poi JSON, poi default da suffix
    in_ch_json = int(cfg.get("in_ch", 0))
    if args.in_ch <= 0:
        args.in_ch = in_ch_json if in_ch_json > 0 else expected_in_channels(args.suffix, fallback=16)
        source = "JSON" if in_ch_json > 0 else "suffix"
        print(f"  [{source}] in_ch={args.in_ch}")
    if False and args.in_ch == 12 and in_ch_json > 0 and in_ch_json != 12:
        # CLI non è stato esplicitamente settato (default=12) ma JSON dice diverso
        # Nota: non possiamo distinguere "CLI=12 esplicito" da "CLI default=12"
        # quindi se JSON ha in_ch != 12 lo usiamo (caso tipico: JSON ha in_ch=16)
        args.in_ch = in_ch_json
        print(f"  [JSON] in_ch={args.in_ch} (letto da config, CLI non esplicitato)")

    datasets = parse_datasets(cfg, args.h5_path, args.pt_dir, args.suffix, args.tile, args.in_ch)

    grid = make_grid(cfg)
    if not grid:
        raise SystemExit("Nessun run generato dalla griglia. Controlla config JSON.")

    n_uts_on = sum(s.use_temp_scaling for s in grid)
    print(f"\n[06_suite_train] AVVIO {_now()}")
    print(f"  config  : {cfg_path.name}")
    print(f"  mode    : {args.mode}")
    print(f"  out_root: {root}")
    print(f"  default suffix={args.suffix} tile={args.tile} in_ch={args.in_ch}")
    print(f"  encoder_weights={ew} input_adapter={input_adapter} augment={aug} amp={amp_flag}")
    print(f"  num_workers={nw} eval_batch={eval_batch or 'train'} scheduler={sched} warmup_epochs={warmup} cache_ram={cache_ram}")
    print(f"  perf_diag_epochs={perf_diag_epochs} perf_diag_batches={perf_diag_batches or 'all'}")
    print(f"  depth_aug_prob={depth_aug_prob} cutout_prob={cutout_prob}")
    print(f"  grid: {len(grid)} run - {len(datasets)} dataset = {len(grid)*len(datasets)} esecuzioni totali")
    # Stima tempo: ~600s per run Stage1 (20ep), ~3000s per run Stage2 (120ep)
    _est_s1 = int(cfg.get("stage1_epochs", cfg.get("epochs", 20))) * 30
    _est_s2 = int(cfg.get("stage2_epochs", cfg.get("epochs", 60))) * 30
    _n_runs = len(grid) * len(datasets)
    if args.mode == "stage1":
        print(f"  Tempo stimato: ~{_n_runs * _est_s1 // 3600}h {(_n_runs * _est_s1 % 3600) // 60}min (@ ~30s/ep)")
    elif args.mode == "stage2":
        print(f"  Tempo stimato: ~{_n_runs * _est_s2 // 3600}h {(_n_runs * _est_s2 % 3600) // 60}min (@ ~30s/ep)")
    print(f"  use_temp_scaling: ON in {n_uts_on}/{len(grid)} run")
    for name, ds in datasets.items():
        print(f"  dataset[{name}] csv={ds.patch_csv} suffix={ds.suffix} in_ch={ds.in_ch} store={ds.h5_path if ds.h5_path else ds.pt_dir}")
    print(f"  continue_on_error={continue_on_error} oom_batch_fallback={oom_fallback}")

    common_kw = dict(
        grid=grid, datasets=datasets, out_root=root, suffix=args.suffix,
        tile=args.tile, in_ch=args.in_ch, warmup=warmup, scheduler=sched,
        num_workers=nw, eval_batch=eval_batch, grad_clip=gc, encoder_weights=ew, input_adapter=input_adapter,
        augment=aug, amp=amp_flag, depth_aug_prob=depth_aug_prob,
        cutout_prob=cutout_prob, cache_ram=cache_ram,
        perf_diag_epochs=perf_diag_epochs, perf_diag_batches=perf_diag_batches,
        dry_run=args.dry_run, overwrite=args.overwrite,
        continue_on_error=continue_on_error, oom_batch_fallback=oom_fallback,
    )

    if args.mode in ("stage1", "both"):
        e1 = args.epochs_s1 if args.epochs_s1 > 0 else int(cfg.get("stage1_epochs", cfg.get("epochs", 15)))
        p1 = int(cfg.get("stage1_patience", cfg.get("patience", 5)))
        run_stage(stage=1, epochs=e1, patience=p1, **common_kw)

    if args.mode in ("stage2", "both"):
        e2 = args.epochs_s2 if args.epochs_s2 > 0 else int(cfg.get("stage2_epochs", cfg.get("epochs", 60)))
        p2 = int(cfg.get("stage2_patience", cfg.get("patience", 12)))
        run_stage(stage=2, epochs=e2, patience=p2, **common_kw)

    print(f"\n[06_suite_train] FINE {_now()}\n")


if __name__ == "__main__":
    main()




