#!/usr/bin/env python3
"""Tabular baseline for the paper-track branch: Random Forest."""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


_THR_SWEEP = np.round(np.arange(0.05, 0.86, 0.05), 4).tolist()
EPS = 1e-6


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_store_list(value: str) -> List[str]:
    parts: List[str] = []
    for chunk in str(value or "").replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def find_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def patch_keys_from_df(df: pd.DataFrame, tile: int) -> List[str]:
    scene_col = find_col(df, ["scene_id", "scene", "id"])
    col_col = find_col(df, ["x0", "x", "col", "col_off"])
    row_col = find_col(df, ["y0", "y", "row", "row_off"])
    if scene_col is None or col_col is None or row_col is None:
        raise ValueError(f"CSV mancano colonne scene_id/x0/y0. Colonne: {list(df.columns)}")
    return [
        f"{sid}__y{y0}_x{x0}_t{tile}"
        for sid, y0, x0 in zip(
            df[scene_col].astype(str),
            df[row_col].astype(int),
            df[col_col].astype(int),
        )
    ]


def split_train_val_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_col = find_col(df, ["split", "set", "subset", "phase"])
    if split_col is None:
        raise ValueError("Per il ramo tabulare serve una colonna split nel patch CSV.")
    s = df[split_col].astype(str).str.lower()
    tr = df[s.str.startswith("tr")].reset_index(drop=True)
    va = df[s.str.startswith("va")].reset_index(drop=True)
    te = df[s.str.startswith("te")].reset_index(drop=True)
    if len(tr) == 0 or len(va) == 0:
        raise ValueError("Split train/val non valido nel patch CSV.")
    return tr, va, te


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _binary_metrics(pred_pos: np.ndarray, target_pos: np.ndarray) -> Dict[str, float]:
    pred_pos = pred_pos.astype(bool)
    target_pos = target_pos.astype(bool)

    tp = float(np.logical_and(pred_pos, target_pos).sum())
    fp = float(np.logical_and(pred_pos, ~target_pos).sum())
    fn = float(np.logical_and(~pred_pos, target_pos).sum())
    tn = float(np.logical_and(~pred_pos, ~target_pos).sum())

    iou_veg = _safe_ratio(tp, tp + fp + fn)
    precision_veg = _safe_ratio(tp, tp + fp)
    recall_veg = _safe_ratio(tp, tp + fn)
    f1_veg = _safe_ratio(2 * precision_veg * recall_veg, precision_veg + recall_veg)

    iou_sand = _safe_ratio(tn, tn + fp + fn)
    precision_sand = _safe_ratio(tn, tn + fn)
    recall_sand = _safe_ratio(tn, tn + fp)
    f1_sand = _safe_ratio(2 * precision_sand * recall_sand, precision_sand + recall_sand)

    miou = 0.5 * (iou_veg + iou_sand)
    min_iou = min(iou_veg, iou_sand)
    harmonic_iou = _safe_ratio(2 * iou_veg * iou_sand, iou_veg + iou_sand)
    bal_acc = 0.5 * (recall_veg + recall_sand)
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = _safe_ratio(tp * tn - fp * fn, mcc_den)

    return {
        "iou_veg": float(iou_veg),
        "iou_sand": float(iou_sand),
        "f1_veg": float(f1_veg),
        "f1_sand": float(f1_sand),
        "precision_veg": float(precision_veg),
        "precision_sand": float(precision_sand),
        "recall_veg": float(recall_veg),
        "recall_sand": float(recall_sand),
        "miou": float(miou),
        "min_iou": float(min_iou),
        "harmonic_iou": float(harmonic_iou),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
    }


def sweep_threshold(y_true: np.ndarray, prob: np.ndarray, thr_metric: str) -> Dict[str, float]:
    target_pos = (y_true == 1)
    metrics_05 = _binary_metrics(prob >= 0.5, target_pos)

    best_thr = 0.5
    best_metrics = metrics_05
    best_score = float(metrics_05[thr_metric])
    for thr in _THR_SWEEP:
        metrics = _binary_metrics(prob >= thr, target_pos)
        score = float(metrics[thr_metric])
        if score > best_score + 1e-12:
            best_score = score
            best_thr = float(thr)
            best_metrics = metrics

    return {
        "best_thr": float(best_thr),
        "best_metric": float(best_score),
        "best_metric_name": str(thr_metric),
        "miou_at_05": float(metrics_05["miou"]),
        "iou_veg_at_05": float(metrics_05["iou_veg"]),
        "iou_sand_at_05": float(metrics_05["iou_sand"]),
        "miou_best": float(best_metrics["miou"]),
        "iou_veg_best": float(best_metrics["iou_veg"]),
        "iou_sand_best": float(best_metrics["iou_sand"]),
        "f1_veg_best": float(best_metrics["f1_veg"]),
        "f1_sand_best": float(best_metrics["f1_sand"]),
        "precision_veg_best": float(best_metrics["precision_veg"]),
        "precision_sand_best": float(best_metrics["precision_sand"]),
        "recall_veg_best": float(best_metrics["recall_veg"]),
        "recall_sand_best": float(best_metrics["recall_sand"]),
        "balanced_accuracy": float(best_metrics["balanced_accuracy"]),
        "mcc": float(best_metrics["mcc"]),
        "min_iou": float(best_metrics["min_iou"]),
        "harmonic_iou": float(best_metrics["harmonic_iou"]),
    }


class H5PatchReader:
    def __init__(self, h5_paths: Sequence[str]) -> None:
        import h5py

        self.paths = [str(Path(p).resolve()) for p in h5_paths]
        self.files = [h5py.File(path, "r", swmr=True) for path in self.paths]

    def close(self) -> None:
        for f in self.files:
            try:
                f.close()
            except Exception:
                pass

    def read_patch(self, key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs: List[np.ndarray] = []
        y = None
        w = None
        for i, handle in enumerate(self.files):
            grp = handle["patches"][key]
            xs.append(np.asarray(grp["x"][:], dtype=np.float32))
            if i == 0:
                y = np.asarray(grp["y"][:], dtype=np.uint8)
                w = np.asarray(grp["w"][:], dtype=np.float32)
        if y is None or w is None:
            raise RuntimeError(f"Missing y/w for key={key}")
        x = np.concatenate(xs, axis=0)
        return x, y, w


def sample_pixels(
    reader: H5PatchReader,
    keys: Sequence[str],
    seed: int,
    max_pixels_per_class: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    key_list = list(keys)
    rng.shuffle(key_list)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    ws: List[np.ndarray] = []
    taken = {0: 0, 1: 0}

    for key in key_list:
        if taken[0] >= max_pixels_per_class and taken[1] >= max_pixels_per_class:
            break
        x, y, w = reader.read_patch(key)
        flat_x = np.moveaxis(x, 0, -1).reshape(-1, x.shape[0])
        flat_y = y.reshape(-1)
        flat_w = w.reshape(-1)

        for raw_cls, bin_cls in ((1, 0), (2, 1)):
            need = max_pixels_per_class - taken[bin_cls]
            if need <= 0:
                continue
            idx = np.flatnonzero(flat_y == raw_cls)
            if idx.size == 0:
                continue
            if idx.size > need:
                idx = rng.choice(idx, size=need, replace=False)
            xs.append(flat_x[idx])
            ys.append(np.full(idx.shape[0], bin_cls, dtype=np.uint8))
            ws.append(flat_w[idx].astype(np.float32))
            taken[bin_cls] += int(idx.shape[0])

    if not xs:
        raise RuntimeError("Nessun pixel campionato.")
    X = np.concatenate(xs, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(ys, axis=0).astype(np.uint8, copy=False)
    w = np.concatenate(ws, axis=0).astype(np.float32, copy=False)
    return X, y, w


def infer_base_feature_names(n_features: int) -> List[str]:
    if n_features == 3:
        return ["B", "G", "R"]
    if n_features == 4:
        return ["B", "G", "R", "NIR"]
    return [f"f{i+1}" for i in range(n_features)]


def _derive_feature(name: str, values: Dict[str, np.ndarray]) -> np.ndarray:
    key = str(name).upper()
    if key in values:
        return values[key].astype(np.float32, copy=False)
    if key == "LOG_BG":
        return np.log(np.maximum(values["B"], EPS) / np.maximum(values["G"], EPS)).astype(np.float32, copy=False)
    if key == "LOG_GR":
        return np.log(np.maximum(values["G"], EPS) / np.maximum(values["R"], EPS)).astype(np.float32, copy=False)
    if key == "TURB_GBG":
        return (values["G"] / np.maximum(values["B"] + values["G"], EPS)).astype(np.float32, copy=False)
    raise KeyError(f"Unsupported tabular feature: {name}")


def build_feature_matrix(
    X: np.ndarray,
    requested_names: Sequence[str],
    stats: Dict[str, Dict[str, float]] | None = None,
) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, float]]]:
    base_names = infer_base_feature_names(int(X.shape[1]))
    values: Dict[str, np.ndarray] = {name: X[:, idx].astype(np.float32, copy=False) for idx, name in enumerate(base_names)}
    names = [str(name).strip() for name in requested_names if str(name).strip()] or base_names

    built: List[np.ndarray] = []
    out_stats: Dict[str, Dict[str, float]] = {} if stats is None else {k: dict(v) for k, v in stats.items()}
    for name in names:
        feat = _derive_feature(name, values)
        if name not in values:
            if stats is None:
                mu = float(np.nanmean(feat)) if feat.size else 0.0
                sd = float(np.nanstd(feat)) if feat.size else 1.0
                if not np.isfinite(sd) or sd < 1e-8:
                    sd = 1.0
                out_stats[name] = {"mean": mu, "std": sd}
            norm = out_stats[name]
            feat = np.clip((feat - float(norm["mean"])) / float(norm["std"]), -3.0, 3.0).astype(np.float32, copy=False)
        built.append(feat.reshape(-1, 1))
    return np.concatenate(built, axis=1).astype(np.float32, copy=False), names, out_stats


def train_model(args: argparse.Namespace, X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray):
    model_name = str(args.model).lower()
    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=int(args.rf_n_estimators),
            max_depth=(None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth)),
            min_samples_leaf=int(args.rf_min_samples_leaf),
            n_jobs=int(args.n_jobs),
            random_state=int(args.seed),
            class_weight="balanced_subsample",
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        return model

    raise ValueError(f"Unsupported model: {args.model}")


def save_feature_importance(model: Any, feature_names: Sequence[str], out_csv: Path) -> None:
    fi = getattr(model, "feature_importances_", None)
    if fi is None:
        return
    names = list(feature_names) if feature_names else [f"f{i+1}" for i in range(len(fi))]
    pd.DataFrame({"feature": names, "importance": fi}).sort_values(
        "importance", ascending=False
    ).to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser("07_train_tabular.py - RF baseline")
    ap.add_argument("--patch_csv", required=True)
    ap.add_argument("--h5_path", required=True, help="Lista store separata da virgole/;")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", required=True, choices=["rf"])
    ap.add_argument("--feature_names", default="")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--thr_metric", default="min_iou", choices=["miou", "min_iou", "harmonic_iou"])
    ap.add_argument("--train_pixels_per_class", type=int, default=250000)
    ap.add_argument("--val_pixels_per_class", type=int, default=150000)
    ap.add_argument("--n_jobs", type=int, default=-1)

    ap.add_argument("--rf_n_estimators", type=int, default=400)
    ap.add_argument("--rf_max_depth", type=int, default=20)
    ap.add_argument("--rf_min_samples_leaf", type=int, default=2)

    args = ap.parse_args()

    patch_csv = Path(args.patch_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_paths = parse_store_list(args.h5_path)
    if not h5_paths:
        raise SystemExit("Nessun HDF5 passato in --h5_path")
    feat_names = [s.strip() for s in str(args.feature_names).split(",") if s.strip()]

    df = pd.read_csv(patch_csv)
    df_tr, df_va, df_te = split_train_val_test(df)
    tr_keys = patch_keys_from_df(df_tr, tile=int(args.tile))
    va_keys = patch_keys_from_df(df_va, tile=int(args.tile))
    te_keys = patch_keys_from_df(df_te, tile=int(args.tile)) if len(df_te) else []

    print(f"[07_train_tabular] START {_now()}")
    print(f"  model={args.model}  thr_metric={args.thr_metric}  seed={args.seed}")
    print(f"  patch_csv={patch_csv}")
    print(f"  h5_paths={h5_paths}")
    print(f"  train patches={len(tr_keys)}  val patches={len(va_keys)}  test patches={len(te_keys)}")

    X_te = None
    y_te = None
    reader = H5PatchReader(h5_paths)
    try:
        X_tr, y_tr, w_tr = sample_pixels(reader, tr_keys, seed=int(args.seed), max_pixels_per_class=int(args.train_pixels_per_class))
        X_va, y_va, _w_va = sample_pixels(reader, va_keys, seed=int(args.seed) + 1, max_pixels_per_class=int(args.val_pixels_per_class))
        if te_keys:
            X_te, y_te, _w_te = sample_pixels(reader, te_keys, seed=int(args.seed) + 2, max_pixels_per_class=int(args.val_pixels_per_class))
    finally:
        reader.close()

    X_tr, feat_names_used, feat_stats = build_feature_matrix(X_tr, feat_names, stats=None)
    X_va, _feat_names_va, _ = build_feature_matrix(X_va, feat_names_used, stats=feat_stats)
    if X_te is not None:
        X_te, _feat_names_te, _ = build_feature_matrix(X_te, feat_names_used, stats=feat_stats)

    sampled_msg = f"  sampled train={X_tr.shape}  val={X_va.shape}"
    if X_te is not None:
        sampled_msg += f"  test={X_te.shape}"
    print(sampled_msg)

    t0 = time.time()
    model = train_model(args, X_tr, y_tr, w_tr)
    train_sec = time.time() - t0

    prob_va = model.predict_proba(X_va)[:, 1].astype(np.float32)
    metrics = sweep_threshold(y_va, prob_va, thr_metric=str(args.thr_metric))

    metric_key = {
        "miou": "miou",
        "min_iou": "min_iou",
        "harmonic_iou": "harmonic_iou",
    }[str(args.thr_metric)]
    test_summary: Dict[str, Any] = {}
    if X_te is not None and y_te is not None:
        prob_te = model.predict_proba(X_te)[:, 1].astype(np.float32)
        target_te = (y_te == 1)
        test_at_05 = _binary_metrics(prob_te >= 0.5, target_te)
        test_at_best_thr = _binary_metrics(prob_te >= float(metrics["best_thr"]), target_te)
        test_summary = {
            "test_shape": [int(X_te.shape[0]), int(X_te.shape[1])],
            "test_thr_from_val": float(metrics["best_thr"]),
            "test_metric_name": str(metrics["best_metric_name"]),
            "test_at_05": test_at_05,
            "test_at_best_thr_from_val": test_at_best_thr,
            "test_metric_at_05": float(test_at_05[metric_key]),
            "test_metric_at_best_thr_from_val": float(test_at_best_thr[metric_key]),
        }

    model_path = out_dir / "model.pkl"
    meta_path = out_dir / "model_best.json"
    fi_path = out_dir / "feature_importance.csv"

    with model_path.open("wb") as f:
        pickle.dump(model, f)
    save_feature_importance(model, feat_names_used, fi_path)

    meta = {
        "timestamp": _now(),
        "model_type": str(args.model),
        "patch_csv": str(patch_csv),
        "h5_paths": h5_paths,
        "feature_names": feat_names_used,
        "feature_stats": feat_stats,
        "tile": int(args.tile),
        "seed": int(args.seed),
        "thr_metric": str(args.thr_metric),
        "train_pixels_per_class": int(args.train_pixels_per_class),
        "val_pixels_per_class": int(args.val_pixels_per_class),
        "train_shape": [int(X_tr.shape[0]), int(X_tr.shape[1])],
        "val_shape": [int(X_va.shape[0]), int(X_va.shape[1])],
        "fit_seconds": float(train_sec),
        **metrics,
    }
    if test_summary:
        meta["test_summary"] = test_summary
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"  miou@.5={metrics['miou_at_05']:.4f}  "
        f"veg_iou={metrics['iou_veg_at_05']:.4f}  sand_iou={metrics['iou_sand_at_05']:.4f}  "
        f"thr*={metrics['best_thr']:.2f}  {metrics['best_metric_name']}*={metrics['best_metric']:.4f}"
    )
    if test_summary:
        print(
            f"  test@.5={test_summary['test_metric_at_05']:.4f}  "
            f"test@thr_val={test_summary['test_metric_at_best_thr_from_val']:.4f}  "
            f"thr={test_summary['test_thr_from_val']:.2f}"
        )
    print(f"  out={out_dir}")
    print(f"[07_train_tabular] END {_now()}")


if __name__ == "__main__":
    main()


