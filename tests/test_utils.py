"""Tests for pure training helpers (torch-free — always runs in CI)."""

import pytest

from src.utils import even_feature_layers, scale_lr, warmup_factor


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


def test_even_feature_layers_spans_and_ends_at_last():
    # 12-layer ViT, 4 taps -> evenly spaced, always includes the final layer
    assert even_feature_layers(12, 4) == [3, 6, 9, 12]
    assert even_feature_layers(24, 4) == [6, 12, 18, 24]


def test_even_feature_layers_edge_cases():
    assert even_feature_layers(4, 4) == [1, 2, 3, 4]
    assert even_feature_layers(2, 4) == [1, 2]  # k clamped to num_layers
    assert even_feature_layers(12, 1) == [12]  # single tap -> last layer
    assert even_feature_layers(12, 4)[-1] == 12  # last element is always the final layer
    assert all(1 <= i <= 12 for i in even_feature_layers(12, 4))
