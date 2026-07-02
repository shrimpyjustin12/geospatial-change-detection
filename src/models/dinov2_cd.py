"""DINOv2 foundation-model tier (PRD §6.1, "FM" tier): a frozen, ImageNet-scale self-supervised
ViT (``facebook/dinov2-*``) used as a **weight-shared** Siamese encoder, adapted with **LoRA**
(``peft``) instead of full fine-tuning, feeding a light multi-layer change decoder.

The comparison question this tier answers (README / PRD §8): *does foundation-model pretraining
beat an ImageNet backbone, and at what trainable-parameter cost?* LoRA keeps the trainable count in
the low millions (adapters + decoder) while the ~86M ViT-B backbone stays frozen, so the tier's
trainable params are directly comparable to the SegFormer strong model's 24.7M.

Design notes:
  * DINOv2 is a **single-scale** ViT (patch stride 14), unlike the hierarchical MiT. We recover a
    pseudo-multi-scale signal the standard ViT-dense-prediction way: read hidden states from four
    evenly-spaced transformer layers (ending at the last), all on the same patch grid, fuse the two
    dates per layer (``diff`` = |a-b| default, or ``concat``), project, concatenate, then decode.
  * Inputs are resized to ``image_size`` (a multiple of patch_size=14) for the encoder; the decoder
    upsamples logits back to the original tile resolution. ``interpolate_pos_encoding=True`` adapts
    the pretrained position grid to whatever ``image_size`` is chosen.
  * Three adaptation regimes via config: LoRA (default, ``lora: true``), frozen linear-probe
    (``lora: false``, ``freeze_encoder: true`` -> only the decoder trains), or full fine-tune
    (``lora: false``, ``freeze_encoder: false``).

Interface matches the other tiers so ``train.py`` / ``evaluate.py`` stay model-agnostic:
input ``(B, 2, C, H, W)`` -> logits ``(B, out_channels, H, W)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.utils import even_feature_layers


class DINOv2SiameseCD(nn.Module):
    """Weight-shared DINOv2 ViT Siamese encoder (frozen + LoRA) + multi-layer change decoder."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        in_ch: int = 3,
        out_channels: int = 1,
        fusion: str = "diff",
        image_size: int = 448,
        out_indices: Sequence[int] | None = None,
        num_feature_layers: int = 4,
        decoder_dim: int = 256,
        dropout: float = 0.1,
        pretrained: bool = True,
        lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: Sequence[str] = ("query", "key", "value", "dense"),
        freeze_encoder: bool = True,
        grad_checkpointing: bool = False,
        encoder_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if fusion not in ("diff", "concat"):
            raise ValueError(f"fusion must be 'diff' or 'concat', got {fusion!r}")
        if in_ch != 3:
            raise ValueError("DINOv2 expects 3-channel RGB input (in_channels=3)")
        from transformers import Dinov2Config, Dinov2Model

        self.fusion = fusion
        self.image_size = int(image_size)

        # weights: pretrained -> read the staged HF snapshot (offline, HF_HUB_OFFLINE=1); else a
        # fresh Dinov2Config (random init, offline-safe -- used by CPU unit tests).
        if pretrained:
            self.encoder = Dinov2Model.from_pretrained(model_name)
        else:
            base: dict[str, Any] = {
                "hidden_size": 192,
                "num_hidden_layers": 4,
                "num_attention_heads": 3,
                "mlp_ratio": 4,
                "patch_size": 14,
                "image_size": self.image_size,
            }
            if encoder_config:
                base.update(encoder_config)
            self.encoder = Dinov2Model(Dinov2Config(**base))

        self.patch_size = int(self.encoder.config.patch_size)
        hidden = int(self.encoder.config.hidden_size)
        n_layers = int(self.encoder.config.num_hidden_layers)
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}"
            )

        if out_indices is None:
            self.out_indices = even_feature_layers(n_layers, num_feature_layers)
        else:
            self.out_indices = [int(i) for i in out_indices]
        for i in self.out_indices:
            if not 1 <= i <= n_layers:
                raise ValueError(f"out_index {i} out of range 1..{n_layers}")

        # Freeze the backbone for LoRA (low-rank adapters carry the adaptation) or linear-probe.
        if freeze_encoder or lora:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.lora = bool(lora)
        if lora:
            from peft import LoraConfig, inject_adapter_in_model

            lora_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(lora_targets),
                bias="none",
            )
            self.encoder = inject_adapter_in_model(lora_cfg, self.encoder)
        if grad_checkpointing:
            # use_reentrant=False is required for DDP: reentrant checkpointing rebuilds the encoder
            # graph inside backward, so DDP's reducer never sees the LoRA params get grads and
            # aborts with "parameters that were not used in producing loss". Non-reentrant
            # checkpointing fires the param grad hooks within a single backward, so DDP works with
            # find_unused_parameters=False. (Single-GPU has no reducer, so both modes work there.)
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        mult = 1 if fusion == "diff" else 2
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(mult * hidden, decoder_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(decoder_dim),
                    nn.GELU(),
                )
                for _ in self.out_indices
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(len(self.out_indices) * decoder_dim, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
        # one learned 2x refinement before bilinear-to-full-res softens the 14x patch upsample.
        self.up = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(decoder_dim, out_channels, kernel_size=1)

    def _fuse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.fusion == "diff":
            return torch.abs(a - b)
        return torch.cat([a, b], dim=1)

    def _tokens_to_grid(self, hs: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        """(B, seq, C) -> (B, C, hp, wp), dropping the leading CLS/register prefix tokens."""
        b, seq, c = hs.shape
        prefix = seq - hp * wp  # robust to models with/without register tokens
        patches = hs[:, prefix:, :]
        return patches.transpose(1, 2).reshape(b, c, hp, wp)

    def _encode(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Encode one date at ``image_size`` -> per-layer patch-feature grids."""
        out = self.encoder(
            pixel_values=img,
            output_hidden_states=True,
            interpolate_pos_encoding=True,
        )
        hidden_states = out.hidden_states  # tuple length num_layers + 1
        hp = img.shape[-2] // self.patch_size
        wp = img.shape[-1] // self.patch_size
        return [self._tokens_to_grid(hidden_states[i], hp, wp) for i in self.out_indices]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x1, x2 = x[:, 0], x[:, 1]
        if (h, w) != (self.image_size, self.image_size):
            size = (self.image_size, self.image_size)
            x1 = F.interpolate(x1, size=size, mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=size, mode="bilinear", align_corners=False)

        feats1 = self._encode(x1)
        feats2 = self._encode(x2)
        fused = [self._fuse(a, b) for a, b in zip(feats1, feats2, strict=True)]
        projected = [proj(f) for proj, f in zip(self.proj, fused, strict=True)]

        y = self.fuse(torch.cat(projected, dim=1))
        y = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = self.up(y)
        logits = self.classifier(y)
        return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
