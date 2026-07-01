"""Losses for binary change detection: combined BCE + Dice (PRD §6.1)."""

from __future__ import annotations

import torch
from torch import nn


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss on the positive (change) class. Inputs shaped ``(B, 1, H, W)``."""
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = (0, 2, 3)
    num = 2.0 * (probs * target).sum(dim=dims) + eps
    den = probs.sum(dim=dims) + target.sum(dim=dims) + eps
    return 1.0 - (num / den).mean()


class BceDiceLoss(nn.Module):
    """``bce_weight * BCEWithLogits + dice_weight * Dice`` (weights exposed in config)."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        bce = self.bce(logits, target)
        dice = dice_loss(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice
