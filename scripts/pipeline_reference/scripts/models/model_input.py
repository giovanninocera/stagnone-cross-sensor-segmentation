#!/usr/bin/env python3
"""Shared input-adaptation helpers for training and inference."""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from scripts.common.feature_presets import feature_names_for_preset, resolve_feature_selection


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def infer_input_adapter(
    arch: str,
    encoder_weights: str | None,
    feature_preset: str = "",
    suffix: str = "",
    in_ch: int = 0,
) -> str:
    weights = str(encoder_weights or "none").lower()
    if weights == "none":
        return "identity"

    preset = ""
    try:
        preset = resolve_feature_selection(feature_preset=feature_preset, suffix=suffix)
    except Exception:
        preset = ""

    arch_l = str(arch or "").lower()
    if in_ch == 3 and (preset in {"SEG3_MAIN", "CMP_BASE3"} or arch_l in {"segformer", "segformer_hf"}):
        return "rgb_imagenet"
    if in_ch == 4 and preset in {"SEG4_CTRL"}:
        return "rgb_nir_centered"
    if in_ch == 6 and preset in {"TAB6_MAIN"}:
        return "rgb_aux_identity"
    if in_ch >= 7 and preset in {"TAB7_CTRL", "MASTER7_CTRL"}:
        return "rgb_nir_aux_centered"
    if in_ch == 3:
        return "rgb_imagenet"
    return "identity"


def feature_names_for_runtime(feature_preset: str = "", suffix: str = "", in_ch: int = 0) -> List[str]:
    try:
        preset = resolve_feature_selection(feature_preset=feature_preset, suffix=suffix)
        names = feature_names_for_preset(preset)
        if in_ch > 0:
            return names[: int(in_ch)]
        return names
    except Exception:
        return []


def _torch_imagenet_stats(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def adapt_batch_torch(x: torch.Tensor, input_adapter: str) -> torch.Tensor:
    if input_adapter == "identity":
        return x
    x = x.float()
    if x.ndim != 4:
        return x
    if input_adapter == "rgb_imagenet":
        mean, std = _torch_imagenet_stats(x.device, x.dtype)
        x[:, :3] = (x[:, :3] - mean) / std
        return x
    if input_adapter == "rgb_aux_identity":
        mean, std = _torch_imagenet_stats(x.device, x.dtype)
        x[:, :3] = (x[:, :3] - mean) / std
        return x
    if input_adapter == "rgb_nir_centered":
        mean, std = _torch_imagenet_stats(x.device, x.dtype)
        x[:, :3] = (x[:, :3] - mean) / std
        if x.shape[1] >= 4:
            x[:, 3:4] = (x[:, 3:4] - 0.5) / 0.25
        return x
    if input_adapter == "rgb_nir_aux_centered":
        mean, std = _torch_imagenet_stats(x.device, x.dtype)
        x[:, :3] = (x[:, :3] - mean) / std
        if x.shape[1] >= 4:
            x[:, 3:4] = (x[:, 3:4] - 0.5) / 0.25
        return x
    return x


def adapt_batch_numpy(x: np.ndarray, input_adapter: str) -> np.ndarray:
    if input_adapter == "identity":
        return x.astype(np.float32, copy=False)
    arr = x.astype(np.float32, copy=True)
    if arr.ndim != 4:
        return arr
    if input_adapter == "rgb_imagenet":
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, :, None, None]
        arr[:, :3] = (arr[:, :3] - mean) / std
        return arr
    if input_adapter == "rgb_aux_identity":
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, :, None, None]
        arr[:, :3] = (arr[:, :3] - mean) / std
        return arr
    if input_adapter == "rgb_nir_centered":
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, :, None, None]
        arr[:, :3] = (arr[:, :3] - mean) / std
        if arr.shape[1] >= 4:
            arr[:, 3:4] = (arr[:, 3:4] - 0.5) / 0.25
        return arr
    if input_adapter == "rgb_nir_aux_centered":
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
        std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, :, None, None]
        arr[:, :3] = (arr[:, :3] - mean) / std
        if arr.shape[1] >= 4:
            arr[:, 3:4] = (arr[:, 3:4] - 0.5) / 0.25
        return arr
    return arr
