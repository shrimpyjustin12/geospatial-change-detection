"""DINOv2 FM-tier (DINOv2SiameseCD) shape/behaviour tests. Needs torch + transformers + peft;
skipped where any is absent (e.g. the CI runner, which installs only torch-cpu). Exercised on
Leonardo's .venv-train. Uses a tiny random-init ViT (pretrained=False, small encoder_config) so it
runs offline and fast on CPU — no network, no staged weights.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch  # noqa: E402

from src.models import build_model  # noqa: E402
from src.models.dinov2_cd import DINOv2SiameseCD  # noqa: E402

# tiny ViT so CPU forward/backward is fast (2 layers, 48-dim, patch 14, 28px -> 2x2 patch grid)
TINY = {"hidden_size": 48, "num_hidden_layers": 2, "num_attention_heads": 2}


def _tiny(**kw):
    opts = dict(
        pretrained=False,
        encoder_config=TINY,
        image_size=28,
        num_feature_layers=2,
        decoder_dim=32,
    )
    opts.update(kw)
    return DINOv2SiameseCD(**opts)


@pytest.mark.parametrize("fusion", ["diff", "concat"])
def test_forward_shape(fusion: str):
    model = _tiny(fusion=fusion, lora=False)
    x = torch.randn(2, 2, 3, 32, 32)  # (B, 2 dates, C, H, W); resized to image_size internally
    y = model(x)
    assert y.shape == (2, 1, 32, 32)  # logits restored to full input resolution


def test_build_model_forward():
    # build_model path uses the default random config (small but real) at a tiny image size
    model = build_model({"name": "dinov2_cd", "pretrained": False, "image_size": 28, "lora": False})
    y = model(torch.randn(1, 2, 3, 32, 32))
    assert y.shape == (1, 1, 32, 32)


def test_encoder_is_weight_shared():
    model = _tiny(lora=False)
    n_encoders = sum(1 for name, _ in model.named_modules() if name == "encoder")
    assert n_encoders == 1  # a single encoder applied to both dates


def test_out_indices_default_ends_at_last_layer():
    model = _tiny(lora=False)
    assert model.out_indices[-1] == model.encoder.config.num_hidden_layers


def test_rejects_bad_fusion():
    with pytest.raises(ValueError):
        _tiny(fusion="bogus")


def test_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        _tiny(in_ch=4)


def test_rejects_indivisible_image_size():
    with pytest.raises(ValueError):
        _tiny(image_size=30)  # not a multiple of patch_size=14


def test_linear_probe_freezes_encoder():
    # lora off + freeze on -> only the decoder trains
    model = _tiny(lora=False, freeze_encoder=True)
    y = model(torch.randn(1, 2, 3, 32, 32))
    y.sum().backward()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.classifier.parameters())


def test_lora_adapts_with_frozen_base():
    model = _tiny(lora=True, lora_r=4, lora_alpha=8)
    # base weights frozen; LoRA adapters trainable
    base = [p for n, p in model.encoder.named_parameters() if "lora_" not in n]
    adapters = [p for n, p in model.encoder.named_parameters() if "lora_" in n]
    assert len(adapters) > 0
    assert all(not p.requires_grad for p in base)
    assert all(p.requires_grad for p in adapters)

    y = model(torch.randn(1, 2, 3, 32, 32))
    y.sum().backward()
    assert any(p.grad is not None for p in adapters)  # LoRA receives gradients
    assert all(p.grad is None for p in base)  # frozen base does not

    # trainable = adapters + decoder is a small fraction of the total (the FM-tier headline)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < trainable < total
