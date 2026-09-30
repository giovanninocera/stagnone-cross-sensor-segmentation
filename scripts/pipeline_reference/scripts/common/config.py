#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Central path configuration for the cleaned _ST project layout."""
from __future__ import annotations

import os
import re
from pathlib import Path


SCENE_SENSOR_BY_DATE = {
    "20030709": "qb",
    "20060408": "qb",
    "20160825": "wv",
    "20230812": "pl",
}
DATE_BY_SCENE_ID = {f"{date}_{sensor}": date for date, sensor in SCENE_SENSOR_BY_DATE.items()}


def project_root() -> Path:
    for key in ("ST_ROOT", "STAGNONE_ROOT"):
        env = os.environ.get(key, "").strip()
        if env:
            return Path(env)

    cwd = Path.cwd()
    if (cwd / "data").exists() and (cwd / "out").exists():
        return cwd

    here_root = Path(__file__).resolve().parents[2]
    if (here_root / "data").exists() and (here_root / "out").exists():
        return here_root

    return Path(r"${PROJECT_ROOT}")


ROOT = project_root()
DATA = ROOT / "data"
OUT = ROOT / "out"

GEOM = DATA / "geom"
RTOA = DATA / "rtoa"
RBOA = DATA / "rboa"
FEATURES = DATA / "features"

WM_ALL = GEOM / "wm_ALL_f.tif"
WM_INT = GEOM / "wm_ALL_f_INT.tif"
WM_OUT = GEOM / "wm_ALL_f_OUT.tif"


def scene_date(scene_id: str) -> str:
    scene_id = str(scene_id)
    if scene_id in DATE_BY_SCENE_ID:
        return DATE_BY_SCENE_ID[scene_id]
    match = re.match(r"^(\d{8})", scene_id)
    if match:
        return match.group(1)
    raise KeyError(f"Cannot infer acquisition date from scene_id={scene_id!r}")


def scene_sensor(scene_id: str) -> str:
    date = scene_date(scene_id)
    return SCENE_SENSOR_BY_DATE.get(date, scene_id.split("_", 1)[1] if "_" in scene_id else "")


def scene_id_from_date(date: str) -> str:
    sensor = SCENE_SENSOR_BY_DATE[str(date)]
    return f"{date}_{sensor}"


def scenes(source: str = "rboa") -> dict[str, Path]:
    """Return canonical scene_id -> raster path for the cleaned data layout."""
    source_l = str(source or "rboa").lower()
    if source_l == "rtoa":
        folder, prefix = RTOA, "RTOA"
    else:
        folder, prefix = RBOA, "RBOA"

    out: dict[str, Path] = {}
    for date, sensor in SCENE_SENSOR_BY_DATE.items():
        path = folder / f"{prefix}_{date}.tif"
        if path.exists():
            out[f"{date}_{sensor}"] = path
    return out


def expected_in_channels(suffix: str, fallback: int = 16) -> int:
    """Infer the canonical input channel count from a feature suffix."""
    match = re.search(r"st(\d+)", str(suffix or "").lower())
    return int(match.group(1)) if match else int(fallback)


def p_rtoa(scene_id: str) -> Path:
    return RTOA / f"RTOA_{scene_date(scene_id)}.tif"


def p_rboa(scene_id: str) -> Path:
    return RBOA / f"RBOA_{scene_date(scene_id)}.tif"


def p_al_boa(scene_id: str) -> Path:
    """Compatibility alias: aligned BOA is now the canonical RBOA raster."""
    return p_rboa(scene_id)


def p_boa_water(scene_id: str) -> Path:
    """Compatibility alias: water-masked BOA is now the canonical RBOA raster."""
    return p_rboa(scene_id)


def p_surface_artifact_mask(scene_id: str) -> Path:
    return DATA / "surface_artifacts" / f"artifact_{scene_id}.tif"


def p_surface_artifact_meta(scene_id: str) -> Path:
    return DATA / "surface_artifacts" / f"artifact_{scene_id}.json"


def p_boa_variant(scene_id: str, variant: str) -> Path:
    variant_l = str(variant).lower()
    if variant_l in {"rtoa", "toa"}:
        return p_rtoa(scene_id)
    return p_rboa(scene_id)


def p_boa_variant_meta(scene_id: str, variant: str) -> Path:
    return p_boa_variant(scene_id, variant).with_suffix(".tif.json")


def p_boa_final(scene_id: str) -> Path:
    """Canonical final BOA raster used before feature preparation."""
    return p_rboa(scene_id)


def p_boa_final_meta(scene_id: str) -> Path:
    return p_boa_final(scene_id).with_suffix(".tif.json")


def p_feat(scene_id: str, suffix: str) -> Path:
    return FEATURES / f"feat_{scene_id}{suffix}.tif"


def p_feat_meta(scene_id: str, suffix: str) -> Path:
    return FEATURES / f"feat_{scene_id}{suffix}.json"


def p_valid(scene_id: str, suffix: str) -> Path:
    return FEATURES / f"valid_{scene_id}{suffix}.tif"


def p_pt_store(suffix: str, tile: int) -> Path:
    suf = (suffix or "").lstrip("_") or "st"
    return OUT / "patches_bin" / f"STORE_{suf}_t{int(tile)}"


def p_patch_csv(year_tag: str, suffix: str) -> Path:
    return OUT / "patches" / f"patches_{year_tag}{suffix}.csv"


def p_training_mask(year: str) -> Path:
    return OUT / "training_mask" / f"training_{year}__FULL.tif"
