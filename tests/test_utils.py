"""Tests for pure training helpers (torch-free — always runs in CI)."""

import pytest

from src.utils import scale_lr, warmup_factor


def test_scale_lr_linear_4x():
    # per-GPU batch 16, 4-GPU DDP -> effective 64 -> 4x the base LR
    assert scale_lr(0.001, effective_batch=64, reference_batch=16) == pytest.approx(0.004)


def test_scale_lr_identity():
    assert scale_lr(0.001, 16, 16) == pytest.approx(0.001)


def test_scale_lr_rejects_bad_reference():
    with pytest.raises(ValueError):
        scale_lr(0.001, 64, 0)


def test_warmup_factor_ramps_then_saturates():
    assert warmup_factor(0, 10) == pytest.approx(0.1)
    assert warmup_factor(9, 10) == pytest.approx(1.0)
    assert warmup_factor(100, 10) == 1.0
    assert warmup_factor(3, 0) == 1.0  # no warmup configured
