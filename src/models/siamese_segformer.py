"""Siamese-SegFormer (PRD §6.1, "strong" tier): a pretrained MiT encoder in a weight-shared
Siamese configuration with difference fusion and a light all-MLP decoder.

Both dates are processed by a single **weight-shared** MiT (Mix Vision Transformer) encoder
from ``segmentation-models-pytorch`` (``mit_b0``..``mit_b5``, ImageNet-pretrained). At each of
the four hierarchical stages (strides 4/8/16/32) the two branches' features are fused
(``diff`` = |a-b|, the default, or ``concat`` for the fusion ablation, PRD §8). A SegFormer-style
all-MLP decoder projects every fused stage to a common width, upsamples to the stride-4 grid,
concatenates, fuses, and predicts single-channel change logits at full resolution.

Interface matches ``FCSiamDiff`` so ``train.py`` / ``evaluate.py`` are model-agnostic:
input ``(B, 2, C, H, W)`` -> logits ``(B, out_channels, H, W)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SiameseSegFormer(nn.Module):
    """Weight-shared MiT Siamese encoder + all-MLP change decoder."""

    def __init__(
        self,
        encoder_name: str = "mit_b2",
        in_ch: int = 3,
        out_channels: int = 1,
        fusion: str = "diff",
        decoder_dim: int = 256,
        pretrained: bool = True,
        encoder_weights: str = "imagenet",
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if fusion not in ("diff", "concat"):
            raise ValueError(f"fusion must be 'diff' or 'concat', got {fusion!r}")
        from segmentation_models_pytorch.encoders import get_encoder

        self.fusion = fusion
        # weights=None -> random init (offline-safe: no network); "imagenet" reads the staged
        # HF snapshot (smp-hub/<encoder>.imagenet) from the shared HF cache under HF_HUB_OFFLINE.
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_ch,
            depth=5,
            weights=encoder_weights if pretrained else None,
        )
        # MiT exposes no stride-2 feature, so the four real hierarchical stages are the last four
        # entries of out_channels (e.g. mit_b2 -> [64, 128, 320, 512] at strides 4/8/16/32).
        stage_ch = list(self.encoder.out_channels)[-4:]
        mult = 1 if fusion == "diff" else 2

        self.mlp = nn.ModuleList(
            [nn.Conv2d(mult * c, decoder_dim, kernel_size=1) for c in stage_ch]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(len(stage_ch) * decoder_dim, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def _fuse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.fusion == "diff":
            return torch.abs(a - b)
        return torch.cat([a, b], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x1, x2 = x[:, 0], x[:, 1]
        feats1 = self.encoder(x1)[-4:]  # four hierarchical stages, coarse-to-fine order preserved
        feats2 = self.encoder(x2)[-4:]

        fused = [self._fuse(a, b) for a, b in zip(feats1, feats2, strict=True)]
        target_size = fused[0].shape[-2:]  # stride-4 grid (finest of the four stages)
        projected = []
        for proj, feat in zip(self.mlp, fused, strict=True):
            y = proj(feat)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode="bilinear", align_corners=False)
            projected.append(y)

        y = self.fuse(torch.cat(projected, dim=1))
        logits = self.classifier(y)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
