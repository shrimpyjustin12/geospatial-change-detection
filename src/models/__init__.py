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
    if name == "siamese_segformer":
        from src.models.siamese_segformer import SiameseSegFormer

        return SiameseSegFormer(
            encoder_name=str(cfg.get("encoder", "mit_b2")),
            in_ch=int(cfg.get("in_channels", 3)),
            out_channels=int(cfg.get("out_channels", 1)),
            fusion=str(cfg.get("fusion", "diff")),
            decoder_dim=int(cfg.get("decoder_dim", 256)),
            pretrained=bool(cfg.get("pretrained", True)),
            encoder_weights=str(cfg.get("encoder_weights", "imagenet")),
            dropout=float(cfg.get("dropout", 0.1)),
            freeze_encoder=bool(cfg.get("freeze_encoder", False)),
        )
    if name == "dinov2_cd":
        from src.models.dinov2_cd import DINOv2SiameseCD

        return DINOv2SiameseCD(
            model_name=str(cfg.get("encoder", "facebook/dinov2-base")),
            in_ch=int(cfg.get("in_channels", 3)),
            out_channels=int(cfg.get("out_channels", 1)),
            fusion=str(cfg.get("fusion", "diff")),
            image_size=int(cfg.get("image_size", 448)),
            out_indices=cfg.get("out_indices"),
            num_feature_layers=int(cfg.get("num_feature_layers", 4)),
            decoder_dim=int(cfg.get("decoder_dim", 256)),
            dropout=float(cfg.get("dropout", 0.1)),
            pretrained=bool(cfg.get("pretrained", True)),
            lora=bool(cfg.get("lora", True)),
            lora_r=int(cfg.get("lora_r", 16)),
            lora_alpha=int(cfg.get("lora_alpha", 32)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            lora_targets=cfg.get("lora_targets", ("query", "key", "value", "dense")),
            freeze_encoder=bool(cfg.get("freeze_encoder", True)),
            grad_checkpointing=bool(cfg.get("grad_checkpointing", False)),
            encoder_config=cfg.get("encoder_config"),
        )
    raise ValueError(
        f"unknown model '{name}' (supported: fc_siam_diff, siamese_segformer, dinov2_cd)"
    )
