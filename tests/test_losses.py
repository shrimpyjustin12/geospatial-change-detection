"""BCE+Dice loss tests (need torch; skipped where torch is absent)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from src.losses import BceDiceLoss, dice_loss  # noqa: E402


def test_dice_loss_near_zero_for_perfect_prediction():
    logits = torch.full((1, 1, 8, 8), 10.0)  # sigmoid -> ~1
    target = torch.ones(1, 1, 8, 8)
    assert dice_loss(logits, target).item() < 0.01


def test_dice_loss_high_for_inverted_prediction():
    logits = torch.full((1, 1, 8, 8), -10.0)  # sigmoid -> ~0
    target = torch.ones(1, 1, 8, 8)
    assert dice_loss(logits, target).item() > 0.9


def test_bce_dice_weights_and_grad():
    loss = BceDiceLoss(bce_weight=1.0, dice_weight=1.0)
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 1, 16, 16)).float()
    val = loss(logits, target)
    assert val.item() > 0
    val.backward()
    assert logits.grad is not None
