"""FC-Siam-diff (Daudt et al. 2018): fully-convolutional Siamese U-Net with difference skips.

Two dates are processed by a single **weight-shared** encoder; at each scale the two branches'
skip features are fused (``diff`` = |a-b|, the FC-Siam-diff default, or ``concat`` for ablation,
PRD §6/§8); a U-Net decoder produces single-channel change logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 conv-BN-ReLU layers."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """Shared 4-stage encoder; returns 4 skip features + bottleneck."""

    def __init__(self, in_ch: int, c: int) -> None:
        super().__init__()
        self.d1 = ConvBlock(in_ch, c)
        self.d2 = ConvBlock(c, 2 * c)
        self.d3 = ConvBlock(2 * c, 4 * c)
        self.d4 = ConvBlock(4 * c, 8 * c)
        self.bottleneck = ConvBlock(8 * c, 8 * c)
        self.pool = nn.MaxPool2d(2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.d1(x)
        s2 = self.d2(self.pool(s1))
        s3 = self.d3(self.pool(s2))
        s4 = self.d4(self.pool(s3))
        b = self.bottleneck(self.pool(s4))
        return s1, s2, s3, s4, b


class Up(nn.Module):
    """Transpose-conv upsample, concat the fused skip, then a ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:  # robustness for odd input sizes
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class FCSiamDiff(nn.Module):
    """FC-Siam-diff. Input ``(B, 2, C, H, W)`` -> change logits ``(B, out_channels, H, W)``."""

    def __init__(
        self,
        in_ch: int = 3,
        base_channels: int = 16,
        out_channels: int = 1,
        fusion: str = "diff",
    ) -> None:
        super().__init__()
        if fusion not in ("diff", "concat"):
            raise ValueError(f"fusion must be 'diff' or 'concat', got {fusion!r}")
        self.fusion = fusion
        c = base_channels
        m = 1 if fusion == "diff" else 2
        self.encoder = Encoder(in_ch, c)
        self.dec4 = Up(m * 8 * c, m * 8 * c, 4 * c)
        self.dec3 = Up(4 * c, m * 4 * c, 2 * c)
        self.dec2 = Up(2 * c, m * 2 * c, c)
        self.dec1 = Up(c, m * c, c)
        self.head = nn.Conv2d(c, out_channels, 1)

    def _fuse(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.fusion == "diff":
            return torch.abs(a - b)
        return torch.cat([a, b], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[:, 0], x[:, 1]
        s1a, s2a, s3a, s4a, ba = self.encoder(x1)
        s1b, s2b, s3b, s4b, bb = self.encoder(x2)
        y = self.dec4(self._fuse(ba, bb), self._fuse(s4a, s4b))
        y = self.dec3(y, self._fuse(s3a, s3b))
        y = self.dec2(y, self._fuse(s2a, s2b))
        y = self.dec1(y, self._fuse(s1a, s1b))
        return self.head(y)
