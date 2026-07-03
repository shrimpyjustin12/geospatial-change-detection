"""ONNX export + PyTorch<->ONNXRuntime parity (src.export). Needs torch + onnx + onnxruntime;
the DINOv2 case additionally needs transformers + peft. Any missing dep skips (CI installs only a
subset). Everything uses tiny random-init models so it runs offline and fast on CPU.

The DINOv2 case is the important one (HANDOFF / PRD §9): it builds an encoder whose *native*
position grid (5x5) differs from the fixed export grid (3x3), so ``interpolate_pos_encoding`` truly
interpolates during the traced forward — the exact op that, if not baked into the graph, makes the
exported DINOv2 silently misbehave. Parity within tolerance proves the bake is correct.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from src.export import export_and_verify  # noqa: E402

DYN_AXES_HW = {
    "input": {0: "batch", 3: "height", 4: "width"},
    "logits": {0: "batch", 2: "height", 3: "width"},
}
BATCH_ONLY = {"input": {0: "batch"}, "logits": {0: "batch"}}


def _dyn_spec(size: int) -> dict:
    return {"size": size, "dynamic_hw": True, "tile": size, "dynamic_axes": DYN_AXES_HW}


def _static_spec(size: int, tile: int) -> dict:
    return {"size": size, "dynamic_hw": False, "tile": tile, "dynamic_axes": BATCH_ONLY}


def test_fc_siam_diff_parity(tmp_path):
    from src.models.fc_siam_diff import FCSiamDiff

    model = FCSiamDiff(in_ch=3, base_channels=8, out_channels=1, fusion="diff").eval()
    rec = export_and_verify(model, _dyn_spec(64), tmp_path / "fc.onnx")
    assert rec["passed"] and rec["primary"]["max_abs"] <= rec["tol"]
    assert rec["secondary"]["input_shape"][-1] != rec["primary"]["input_shape"][-1]  # H/W varied


def test_segformer_parity(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    from src.models.siamese_segformer import SiameseSegFormer

    # mit_b0 random-init (pretrained=False -> offline, no staged weights); dynamic H/W.
    model = SiameseSegFormer(
        encoder_name="mit_b0", pretrained=False, fusion="diff", decoder_dim=32
    ).eval()
    rec = export_and_verify(model, _dyn_spec(64), tmp_path / "sf.onnx")
    assert rec["passed"] and rec["primary"]["max_abs"] <= rec["tol"]


@pytest.mark.parametrize("lora", [True, False])
def test_dinov2_parity_forces_pos_encoding_interpolation(tmp_path, lora):
    pytest.importorskip("transformers")
    if lora:
        pytest.importorskip("peft")
    from src.models.dinov2_cd import DINOv2SiameseCD

    export_size = 42  # 3x3 patch grid at patch_size=14
    native_size = 70  # encoder trained/config'd for a 5x5 grid -> pos-embed must interpolate 5->3
    model = DINOv2SiameseCD(
        pretrained=False,
        encoder_config={
            "hidden_size": 24,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "image_size": native_size,
        },
        image_size=export_size,
        num_feature_layers=2,
        decoder_dim=32,
        lora=lora,
    ).eval()
    # sanity: the native grid really differs from the export grid, so interpolation is exercised.
    native_grid = native_size // model.patch_size
    export_grid = export_size // model.patch_size
    assert native_grid != export_grid

    rec = export_and_verify(model, _static_spec(export_size, tile=48), tmp_path / "dv2.onnx")
    assert rec["passed"], rec
    assert rec["primary"]["max_abs"] <= rec["tol"]
    assert not rec["dynamic_hw"]  # DINOv2 is exported at a fixed grid
    assert rec["secondary"]["input_shape"][0] == 2  # dynamic batch verified instead


def test_parity_failure_raises(tmp_path, monkeypatch):
    """A real parity gap must abort the export (PRD §9), not pass silently."""
    from src.models.fc_siam_diff import FCSiamDiff

    model = FCSiamDiff(in_ch=3, base_channels=8, out_channels=1, fusion="diff").eval()
    with pytest.raises(RuntimeError, match="PARITY FAILED"):
        export_and_verify(model, _dyn_spec(64), tmp_path / "fc.onnx", tol=-1.0)  # impossible tol
