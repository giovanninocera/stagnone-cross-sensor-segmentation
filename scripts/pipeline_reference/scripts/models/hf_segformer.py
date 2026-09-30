#!/usr/bin/env python3
"""Official Hugging Face SegFormer wrappers used by the paper-track branch."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.common.config import OUT

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    from transformers.utils import logging as hf_logging
except Exception as exc:  # pragma: no cover
    SegformerConfig = None
    SegformerForSemanticSegmentation = None
    hf_logging = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
    hf_logging.set_verbosity_error()


SEGFORMER_CHECKPOINTS: Dict[str, str] = {
    "mit_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "mit_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "mit_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "mit_b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "mit_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "mit_b5": "nvidia/segformer-b5-finetuned-ade-640-640",
}

LOCAL_SEGFORMER_CHECKPOINTS: Dict[str, Path] = {
    "nvidia/segformer-b2-finetuned-ade-512-512": (
        OUT / "hf_models" / "nvidia_segformer_b2_finetuned_ade_512_512"
    ),
}


class HFSegformerBinary(nn.Module):
    def __init__(
        self,
        encoder: str,
        in_ch: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if SegformerForSemanticSegmentation is None or SegformerConfig is None:
            raise ImportError(
                "transformers is not available for segformer_hf. "
                f"Original import error: {_IMPORT_ERROR}"
            )

        checkpoint = SEGFORMER_CHECKPOINTS.get(str(encoder).lower())
        if checkpoint is None:
            raise KeyError(
                f"Unsupported SegFormer encoder '{encoder}'. "
                f"Available: {sorted(SEGFORMER_CHECKPOINTS.keys())}"
            )
        load_checkpoint = self._resolve_checkpoint_source(checkpoint)

        if pretrained:
            # ignore_mismatched_sizes=True: loads all encoder weights that match the
            # pretrained checkpoint shape. For in_ch==3 this includes the first patch
            # embedding Conv2d (fully pretrained). For in_ch != 3 (e.g. 4-channel NIR
            # or 6-channel ratio ablation) the first SegformerOverlapPatchEmbeddings
            # layer has a different input shape and is therefore re-initialized with
            # random weights while all subsequent transformer layers remain pretrained.
            # This initialization asymmetry should be disclosed when comparing RGB vs
            # multi-channel results.
            base_kwargs = dict(
                num_labels=1,
                num_channels=int(in_ch),
                ignore_mismatched_sizes=True,
            )
            self.model = self._load_pretrained_checkpoint(load_checkpoint, base_kwargs)
            self.checkpoint = checkpoint
        else:
            config = SegformerConfig.from_pretrained(
                load_checkpoint,
                local_files_only=True,
                num_labels=1,
                num_channels=int(in_ch),
            )
            self.model = SegformerForSemanticSegmentation(config)
            self.checkpoint = None

    @staticmethod
    def _resolve_checkpoint_source(checkpoint: str) -> str:
        local_dir = LOCAL_SEGFORMER_CHECKPOINTS.get(str(checkpoint))
        if local_dir is not None and (local_dir / "config.json").exists():
            if (local_dir / "model.safetensors").exists() or (local_dir / "pytorch_model.bin").exists():
                return str(local_dir)
        return checkpoint

    @staticmethod
    def _load_pretrained_checkpoint(checkpoint: str, base_kwargs: Dict[str, object]) -> nn.Module:
        attempts = (
            {"use_safetensors": True},
            {"use_safetensors": False},
        )
        last_exc: Exception | None = None
        for fmt_kwargs in attempts:
            try:
                return SegformerForSemanticSegmentation.from_pretrained(
                    checkpoint,
                    local_files_only=True,
                    **base_kwargs,
                    **fmt_kwargs,
                )
            except Exception as exc:
                last_exc = exc

        if os.environ.get("ST_ALLOW_HF_DOWNLOAD", "").strip() == "1":
            for fmt_kwargs in attempts:
                try:
                    return SegformerForSemanticSegmentation.from_pretrained(
                        checkpoint,
                        **base_kwargs,
                        **fmt_kwargs,
                    )
                except Exception as exc:
                    last_exc = exc

        raise RuntimeError(
            f"SegFormer checkpoint '{checkpoint}' is not fully available in the local "
            "Hugging Face cache. Refusing network fallback during project training. "
            "Populate the cache first or set ST_ALLOW_HF_DOWNLOAD=1 for an explicit "
            "one-off download."
        ) from last_exc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        logits = out.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits
