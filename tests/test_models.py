"""FC-Siam-diff shape/behaviour tests (need torch; skipped where torch is absent)."""

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from src.models import build_model  # noqa: E402
from src.models.fc_siam_diff import FCSiamDiff  # noqa: E402


@pytest.mark.parametrize("fusion", ["diff", "concat"])
def test_forward_shape(fusion: str):
    model = FCSiamDiff(in_ch=3, base_channels=8, out_channels=1, fusion=fusion)
    x = torch.randn(2, 2, 3, 64, 64)  # (B, 2 dates, C, H, W)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)


def test_build_model_forward():
    model = build_model({"name": "fc_siam_diff", "base_channels": 8})
    y = model(torch.randn(1, 2, 3, 64, 64))
    assert y.shape == (1, 1, 64, 64)


def test_encoder_is_weight_shared():
    # A single encoder module is applied to both dates -> one set of encoder params.
    model = FCSiamDiff(base_channels=8)
    assert sum(p.numel() for p in model.encoder.parameters()) > 0


def test_rejects_bad_fusion():
    with pytest.raises(ValueError):
        FCSiamDiff(fusion="bogus")


def test_backward_produces_grads():
    model = FCSiamDiff(base_channels=8)
    y = model(torch.randn(1, 2, 3, 64, 64))
    y.sum().backward()
    assert any(p.grad is not None for p in model.parameters())
