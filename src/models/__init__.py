"""Siamese change-detection models: fc_siam_diff, siamese_segformer, dinov2_cd."""

from __future__ import annotations

from typing import Any

from src.models.fc_siam_diff import FCSiamDiff


def build_model(cfg: dict[str, Any]) -> Any:
    """Build a model from a resolved ``model`` config block."""
    name = str(cfg.get("name", "fc_siam_diff"))
    if name == "fc_siam_diff":
        return FCSiamDiff(
            in_ch=int(cfg.get("in_channels", 3)),
            base_channels=int(cfg.get("base_channels", 16)),
            out_channels=int(cfg.get("out_channels", 1)),
            fusion=str(cfg.get("fusion", "diff")),
        )
    raise ValueError(f"unknown model '{name}' (M1 supports: fc_siam_diff)")
