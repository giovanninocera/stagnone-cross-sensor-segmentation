#!/usr/bin/env python3
"""Canonical feature presets for the paper-oriented _ST operational pipeline."""
from __future__ import annotations

from typing import Dict, Iterable, List


FEATURE_PRESETS: Dict[str, Dict[str, object]] = {
    "CMP_BASE3": {
        "suffix": "_st3b",
        "channels": ["B", "G", "R"],
        "description": "Visible-only component store for deep models.",
    },
    "CMP_NIR1": {
        "suffix": "_st1n",
        "channels": ["NIR"],
        "description": "Single-channel NIR control component store.",
    },
    "CMP_RATIO3": {
        "suffix": "_st3r",
        "channels": ["LOG_BG", "LOG_GR", "TURB_GBG"],
        "description": "Compact auxiliary ratio/turbidity component store.",
    },
    "SEG3_MAIN": {
        "suffix": "_st3s",
        "channels": ["B", "G", "R"],
        "description": "Main SegFormer/CNN input: visible-only, radiometric and clean.",
    },
    "SEG4_CTRL": {
        "suffix": "_st4s",
        "channels": ["B", "G", "R", "NIR"],
        "description": "Visible baseline plus NIR control channel.",
    },
    "MASTER4_CTRL": {
        "suffix": "_st4m",
        "channels": ["B", "G", "R", "NIR"],
        "description": "Master scene-level stack for the SegFormer-first branch.",
    },
    "TAB6_MAIN": {
        "suffix": "_st6t",
        "channels": ["B", "G", "R", "LOG_BG", "LOG_GR", "TURB_GBG"],
        "description": "Tabular main set for Random Forest without NIR.",
    },
    "TAB7_CTRL": {
        "suffix": "_st7t",
        "channels": ["B", "G", "R", "NIR", "LOG_BG", "LOG_GR", "TURB_GBG"],
        "description": "Tabular control set with NIR retained as explicit control.",
    },
    "MASTER7_CTRL": {
        "suffix": "_st7m",
        "channels": ["B", "G", "R", "NIR", "LOG_BG", "LOG_GR", "TURB_GBG"],
        "description": "Master scene-level stack from which the current paper-track inputs are derived.",
    },
}


FEATURE_PRESET_ALIASES = {
    "cmp_base3": "CMP_BASE3",
    "st3b": "CMP_BASE3",
    "cmp_nir1": "CMP_NIR1",
    "st1n": "CMP_NIR1",
    "cmp_ratio3": "CMP_RATIO3",
    "st3r": "CMP_RATIO3",
    "seg3": "SEG3_MAIN",
    "seg3_main": "SEG3_MAIN",
    "st3s": "SEG3_MAIN",
    "vis3": "SEG3_MAIN",
    "vis3_base": "SEG3_MAIN",
    "seg4": "SEG4_CTRL",
    "seg4_ctrl": "SEG4_CTRL",
    "st4s": "SEG4_CTRL",
    "visn4": "SEG4_CTRL",
    "visn4_ctrl": "SEG4_CTRL",
    "master4": "MASTER4_CTRL",
    "master4_ctrl": "MASTER4_CTRL",
    "st4m": "MASTER4_CTRL",
    "tab6": "TAB6_MAIN",
    "tab6_main": "TAB6_MAIN",
    "st6t": "TAB6_MAIN",
    "ratio6": "TAB6_MAIN",
    "ratio6_main": "TAB6_MAIN",
    "tab7": "TAB7_CTRL",
    "tab7_ctrl": "TAB7_CTRL",
    "st7t": "TAB7_CTRL",
    "master": "MASTER7_CTRL",
    "master7": "MASTER7_CTRL",
    "st7m": "MASTER7_CTRL",
}


MASTER_FEATURE_PRESET = "MASTER7_CTRL"
MASTER_FEATURE_SUFFIX = "_st7m"


def resolve_feature_preset(name: str) -> str:
    key = str(name or "").strip()
    if not key:
        raise KeyError("Feature preset name is empty.")
    if key in FEATURE_PRESETS:
        return key
    alias = FEATURE_PRESET_ALIASES.get(key.lower())
    if alias:
        return alias
    raise KeyError(f"Unknown feature preset: {name}")


def feature_names_for_preset(name: str) -> List[str]:
    preset = resolve_feature_preset(name)
    return list(FEATURE_PRESETS[preset]["channels"])  # type: ignore[index]


def suffix_for_preset(name: str) -> str:
    preset = resolve_feature_preset(name)
    return str(FEATURE_PRESETS[preset]["suffix"])


def description_for_preset(name: str) -> str:
    preset = resolve_feature_preset(name)
    return str(FEATURE_PRESETS[preset]["description"])


def preset_for_suffix(suffix: str) -> str:
    suffix_l = str(suffix or "").lower()
    for preset_name, meta in FEATURE_PRESETS.items():
        if str(meta["suffix"]).lower() == suffix_l:
            return preset_name
    raise KeyError(f"No feature preset registered for suffix: {suffix}")


def feature_names_for_suffix(suffix: str) -> List[str]:
    return feature_names_for_preset(preset_for_suffix(suffix))


def resolve_feature_selection(feature_preset: str = "", suffix: str = "") -> str:
    if str(feature_preset or "").strip():
        return resolve_feature_preset(feature_preset)
    if str(suffix or "").strip():
        return preset_for_suffix(suffix)
    raise KeyError("A feature preset or a preset-registered suffix is required.")


def band_indexes_for_channels(
    available_channels: Iterable[str],
    requested_channels: Iterable[str],
) -> List[int]:
    available = [str(name) for name in available_channels]
    index_map = {name: idx + 1 for idx, name in enumerate(available)}
    missing = [name for name in requested_channels if name not in index_map]
    if missing:
        raise KeyError(
            f"Missing channels in source stack: {missing}. "
            f"Available: {available}"
        )
    return [index_map[name] for name in requested_channels]
