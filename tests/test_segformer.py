"""Siamese-SegFormer shape/behaviour tests. Needs torch + segmentation-models-pytorch (smp);
skipped where either is absent (e.g. the CI runner, which installs only torch-cpu). Exercised on
Leonardo's .venv-train, which has smp/timm. Uses mit_b0 with random init (weights=None) so it runs
offline and fast on CPU.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

import torch  # noqa: E402

from src.models import build_model  # noqa: E402
from src.models.siamese_segformer import SiameseSegFormer  # noqa: E402


@pytest.mark.parametrize("fusion", ["diff", "concat"])
def test_forward_shape(fusion: str):
    model = SiameseSegFormer(encoder_name="mit_b0", pretrained=False, fusion=fusion, decoder_dim=64)
    x = torch.randn(2, 2, 3, 64, 64)  # (B, 2 dates, C, H, W)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)  # logits restored to full input resolution


def test_encoder_is_weight_shared():
    # a single encoder module is applied to both dates
    model = SiameseSegFormer(encoder_name="mit_b0", pretrained=False, decoder_dim=64)
    assert sum(1 for _ in model.encoder.parameters()) > 0
    n_encoders = sum(1 for name, _ in model.named_modules() if name == "encoder")
    assert n_encoders == 1


def test_build_model_forward():
    model = build_model(
        {"name": "siamese_segformer", "encoder": "mit_b0", "pretrained": False, "decoder_dim": 64}
    )
    y = model(torch.randn(1, 2, 3, 64, 64))
    assert y.shape == (1, 1, 64, 64)


def test_rejects_bad_fusion():
    with pytest.raises(ValueError):
        SiameseSegFormer(encoder_name="mit_b0", pretrained=False, fusion="bogus")


def test_backward_produces_grads():
    model = SiameseSegFormer(encoder_name="mit_b0", pretrained=False, decoder_dim=64)
    y = model(torch.randn(1, 2, 3, 64, 64))
    y.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_freeze_encoder_stops_encoder_grads():
    model = SiameseSegFormer(
        encoder_name="mit_b0", pretrained=False, decoder_dim=64, freeze_encoder=True
    )
    y = model(torch.randn(1, 2, 3, 64, 64))
    y.sum().backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.classifier.parameters())
