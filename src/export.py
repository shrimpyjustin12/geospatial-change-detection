"""ONNX export + PyTorch<->ONNXRuntime parity check + artifact bundle (PRD §9).

Exports a trained Track-A change-detection model to ONNX, then **asserts numerical parity**
between the PyTorch forward and ONNXRuntime on a fixed sample (max abs difference below a
tolerance); a parity failure aborts the export and writes no bundle. On success it emits the
per-model *artifact bundle* — the contract the HF Space consumes (PRD §3):

    bundles/<run_id>/
      model.onnx           # the exported graph
      config.yaml          # the resolved training config (provenance)
      preprocessing.json   # normalization, input contract, tiling, threshold — how to feed it
      metrics_card.md      # headline metrics (from the eval harness summary if present)
      parity.json          # the recorded parity result (diffs, tol, opset, versions)

Usage:
    python -m src.export --config configs/levircd_segformer.yaml
    python -m src.export --config configs/levircd_dinov2.yaml --checkpoint /path/to/best.pt
    python -m src.export --config configs/levircd_segformer.yaml --random-init   # machinery check

**DINOv2 is the risky tier (HANDOFF / PRD §9).** ``dinov2_cd`` resizes each date to ``image_size``
(default 448, a 32x32 patch grid) and adapts the pretrained position grid with
``interpolate_pos_encoding=True``. If that resize + pos-embed interpolation are not baked into the
graph correctly the ONNX model *silently* misbehaves (wrong output, no error). We defuse this by
exporting DINOv2 at a **fixed input size equal to ``image_size``** (static H/W, only batch dynamic):
the model's internal date-resize branch is then never hit, and the pos-embed interpolation reduces
to a constant on the fixed 32x32 grid. The tile->448 resize is recorded in ``preprocessing.json``
so the demo does it before inference. The fully-convolutional CNN tiers (FC-Siam-diff, SegFormer)
export cleanly with dynamic H/W.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import expand_env, load_config
from src.data.levircd import IMAGENET_MEAN, IMAGENET_STD
from src.models import build_model

# Fixed opset: 17 covers LayerNorm / GELU / bicubic Resize used by the ViT + decoders, and is
# broadly supported by the onnxruntime CPU build the HF Space ships. Keep it pinned for repro.
DEFAULT_OPSET = 17
# Parity tolerance on raw logits. torch-CPU-fp32 vs ORT-CPU-fp32 differ only by op-ordering /
# Resize kernel rounding; 1e-3 on logits is comfortably tight (post-sigmoid diffs are far smaller).
DEFAULT_TOL = 1e-3


def _resolve_checkpoint(cfg: dict[str, Any], checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint)
    run_dir = Path(cfg["logging"]["log_dir"]) / cfg["run_id"]
    return run_dir / "checkpoints" / "best.pt"


def load_model(
    cfg: dict[str, Any],
    *,
    checkpoint: str | None = None,
    random_init: bool = False,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build the model on CPU in eval mode and load weights.

    ``random_init`` skips the checkpoint and forces ``pretrained: false`` so the *architecture*
    (real ViT-B/14 or MiT-b2) is exercised with random weights — used to verify the export/parity
    machinery offline, without the trained checkpoint or staged FM weights. It never yields a
    shippable bundle; the caller stamps the artifacts accordingly.
    """
    model_cfg = deepcopy(cfg["model"])
    meta: dict[str, Any] = {"random_init": bool(random_init)}
    if random_init:
        model_cfg["pretrained"] = False
        model = build_model(model_cfg)
        meta["checkpoint"] = None
        meta["epoch"] = -1
    else:
        ckpt_path = _resolve_checkpoint(cfg, checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {ckpt_path}. Pass --checkpoint, or --random-init to "
                f"verify the export machinery without trained weights."
            )
        state = torch.load(ckpt_path, map_location="cpu")
        model = build_model(model_cfg)
        model.load_state_dict(state["model"])
        meta["checkpoint"] = str(ckpt_path)
        meta["epoch"] = int(state.get("epoch", -1))
    model.eval()
    return model, meta


def input_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve the ONNX input contract for this model.

    DINOv2 -> fixed square input at ``image_size`` (static H/W). CNN tiers -> ``tile_size`` with
    dynamic H/W. Returns the export size plus the dynamic-axes map for ``torch.onnx.export``.
    """
    name = str(cfg["model"].get("name", "fc_siam_diff"))
    tile = int(cfg["data"].get("tile_size", 256))
    if name == "dinov2_cd":
        size = int(cfg["model"].get("image_size", 448))
        # input (B, 2, 3, H, W) / output (B, out, H, W): only batch is dynamic for DINOv2.
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
        return {"size": size, "dynamic_hw": False, "dynamic_axes": dynamic_axes, "tile": tile}
    dynamic_axes = {
        "input": {0: "batch", 3: "height", 4: "width"},
        "logits": {0: "batch", 2: "height", 3: "width"},
    }
    return {"size": tile, "dynamic_hw": True, "dynamic_axes": dynamic_axes, "tile": tile}


def _sample(batch: int, size: int, *, seed: int = 0) -> torch.Tensor:
    """Deterministic normalized-looking input ``(batch, 2, 3, size, size)`` for parity."""
    gen = torch.Generator().manual_seed(seed)
    # ImageNet-normalized pixels land roughly in [-2.5, 2.5]; a unit-ish normal exercises the graph.
    return torch.randn(batch, 2, 3, size, size, generator=gen)


@torch.no_grad()
def export_and_verify(
    model: torch.nn.Module,
    spec: dict[str, Any],
    onnx_path: Path,
    *,
    opset: int = DEFAULT_OPSET,
    tol: float = DEFAULT_TOL,
) -> dict[str, Any]:
    """Export ``model`` to ``onnx_path`` and assert PyTorch<->ONNXRuntime parity.

    Returns a parity record. Raises ``RuntimeError`` if the max abs logit difference exceeds
    ``tol`` (PRD §9: a parity failure must fail the export). Also re-runs ONNXRuntime on a second
    shape to prove the declared dynamic axes actually hold.
    """
    import onnxruntime as ort

    size = int(spec["size"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    example = _sample(1, size)

    torch.onnx.export(
        model,
        (example,),
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=spec["dynamic_axes"],
        opset_version=opset,
        do_constant_folding=True,
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def _parity_at(x: torch.Tensor) -> dict[str, float]:
        torch_out = model(x).cpu().numpy()
        ort_out = sess.run(["logits"], {"input": x.numpy()})[0]
        if torch_out.shape != ort_out.shape:
            raise RuntimeError(
                f"shape mismatch torch{torch_out.shape} vs onnx{ort_out.shape} "
                f"at input {tuple(x.shape)}"
            )
        max_abs = float(np.abs(torch_out - ort_out).max())
        mean_abs = float(np.abs(torch_out - ort_out).mean())
        # post-sigmoid is what the demo thresholds — report it as the operationally-relevant diff.
        prob_diff = float(np.abs(1 / (1 + np.exp(-torch_out)) - 1 / (1 + np.exp(-ort_out))).max())
        return {"max_abs": max_abs, "mean_abs": mean_abs, "max_prob_abs": prob_diff}

    primary = _parity_at(example)

    # Exercise a declared dynamic axis so a mis-declared graph fails loudly here, not in the demo.
    if spec["dynamic_hw"]:
        alt = _sample(1, size + 32, seed=1)  # different H/W (stays a multiple of 32 when size is)
        alt_desc = f"dynamic H/W at {alt.shape[-1]}px"
    else:
        alt = _sample(2, size, seed=1)  # different batch
        alt_desc = "dynamic batch=2"
    secondary = _parity_at(alt)

    passed = primary["max_abs"] <= tol and secondary["max_abs"] <= tol
    record = {
        "passed": bool(passed),
        "tol": tol,
        "opset": opset,
        "input_size": size,
        "dynamic_hw": bool(spec["dynamic_hw"]),
        "primary": {"input_shape": list(example.shape), **primary},
        "secondary": {"desc": alt_desc, "input_shape": list(alt.shape), **secondary},
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
    }
    if not passed:
        raise RuntimeError(
            f"PARITY FAILED: max|torch-onnx| primary={primary['max_abs']:.3e} "
            f"secondary={secondary['max_abs']:.3e} > tol={tol:.1e}. Export aborted."
        )
    return record


def _load_eval_summary(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort read of the eval harness summary for this run, if it was produced."""
    for base in (
        Path(cfg["logging"]["log_dir"]) / cfg["run_id"] / "eval_test",
        Path("results") / cfg["run_id"] / "eval_test",
    ):
        summary = base / "summary.json"
        if summary.exists():
            try:
                return json.loads(summary.read_text())
            except (OSError, json.JSONDecodeError):
                return None
    return None


def _recommended_threshold(summary: dict[str, Any] | None) -> tuple[float, str]:
    """Operating threshold for the demo: the val-selected one if we have it, else 0.5."""
    if summary and "operating_point" in summary:
        return float(summary["operating_point"]["threshold"]), "val-selected (max-F1)"
    return 0.5, "default (no eval summary found)"


def write_bundle(
    cfg: dict[str, Any],
    spec: dict[str, Any],
    meta: dict[str, Any],
    parity: dict[str, Any],
    bundle_dir: Path,
) -> None:
    """Write config.yaml, preprocessing.json, metrics_card.md, parity.json alongside model.onnx."""
    import yaml

    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    summary = _load_eval_summary(cfg)
    threshold, thr_source = _recommended_threshold(summary)
    name = str(cfg["model"].get("name"))
    size = int(spec["size"])
    preprocessing = {
        "input_name": "input",
        "output_name": "logits",
        "input_shape": ["batch", 2, 3, size, size],
        "input_layout": "(batch, 2 dates, 3 RGB channels, H, W)",
        "band_order": ["R", "G", "B"],
        "value_range": "float32; divide 8-bit RGB by 255 BEFORE normalization",
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "input_size": size,
        "dynamic_hw": bool(spec["dynamic_hw"]),
        "resize_to_input": (
            f"resize each {spec['tile']}px tile to {size}px (bilinear) before inference; "
            f"resize the {size}px output mask back to display resolution"
            if size != spec["tile"]
            else f"feed {size}px tiles directly (fully-convolutional; dynamic H/W also allowed)"
        ),
        "tiling": {"tile_size": int(cfg["data"].get("tile_size", 256)), "overlap": 0},
        "output": {
            "activation": "sigmoid",
            "meaning": "per-pixel change probability (channel 0)",
            "recommended_threshold": threshold,
            "threshold_source": thr_source,
        },
    }
    if name == "dinov2_cd":
        preprocessing["dinov2_note"] = (
            f"encoder runs on a FIXED {size}px / {size // 14}x{size // 14} patch grid with "
            "interpolate_pos_encoding baked in; do NOT feed a different size."
        )
    if meta.get("random_init"):
        preprocessing["weights"] = "RANDOM-INIT — parity-machinery verification only, not shippable"
    (bundle_dir / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))

    (bundle_dir / "parity.json").write_text(json.dumps(parity, indent=2))
    (bundle_dir / "metrics_card.md").write_text(_metrics_card(cfg, meta, parity, summary))


def _metrics_card(
    cfg: dict[str, Any],
    meta: dict[str, Any],
    parity: dict[str, Any],
    summary: dict[str, Any] | None,
) -> str:
    m = cfg["model"]
    ckpt = meta.get("checkpoint")
    ckpt_name = Path(ckpt).name if ckpt else "random-init"
    lines = [
        f"# Model card — {cfg['run_id']}",
        "",
        f"- **Architecture:** `{m.get('name')}` (encoder `{m.get('encoder')}`, fusion "
        f"`{m.get('fusion')}`)",
        "- **Dataset:** LEVIR-CD (binary building change, 0.5 m RGB aerial)",
        f"- **Checkpoint:** `{ckpt_name}` (epoch {meta.get('epoch')})",
        "- **Intended use:** portfolio/demo only; trained weights inherit LEVIR-CD "
        "research/non-commercial terms.",
        "",
        "## Metrics (LEVIR-CD test; threshold selected on val, applied to test)",
    ]
    if summary and "operating_point" in summary:
        op = summary["operating_point"]
        ps = summary.get("per_scene", {})
        lines += [
            "",
            "| F1 | IoU | Precision | Recall | AP | trainable params |",
            "|---|---|---|---|---|---|",
            f"| {op['f1']:.3f} | {op['iou']:.3f} | {op['precision']:.3f} | {op['recall']:.3f} | "
            f"{summary.get('average_precision', float('nan')):.3f} | "
            f"{summary.get('trainable_params', '—')} |",
            "",
            f"Per-scene F1 mean±std: {ps.get('f1_mean', float('nan')):.3f} ± "
            f"{ps.get('f1_std', float('nan')):.3f} (n={ps.get('n_scenes', '—')}). "
            "Overall pixel accuracy is intentionally NOT reported "
            "(change is a tiny pixel fraction).",
        ]
    else:
        lines += ["", "_No eval summary found for this run — metrics pending._"]
    verdict = "PASS" if parity["passed"] else "FAIL"
    lines += [
        "",
        "## Export parity (PyTorch ↔ ONNXRuntime)",
        f"- opset {parity['opset']}, tol {parity['tol']:.0e}, input {parity['input_size']}px, "
        f"dynamic_hw={parity['dynamic_hw']}",
        f"- max |logit diff| = {parity['primary']['max_abs']:.2e} "
        f"(post-sigmoid {parity['primary']['max_prob_abs']:.2e}) → **{verdict}**",
    ]
    if meta.get("random_init"):
        lines += [
            "",
            "> ⚠️ **Random-init export** — this bundle verifies the export/parity machinery only. "
            "Re-export from the trained checkpoint for a shippable bundle.",
        ]
    return "\n".join(lines) + "\n"


def export_bundle(
    config: str,
    *,
    checkpoint: str | None = None,
    random_init: bool = False,
    out_dir: str = "bundles",
    opset: int = DEFAULT_OPSET,
    tol: float = DEFAULT_TOL,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """End-to-end: build+load model, export to ONNX, verify parity, write the bundle."""
    cfg = expand_env(load_config(config, overrides or []))
    model, meta = load_model(cfg, checkpoint=checkpoint, random_init=random_init)
    spec = input_spec(cfg)
    bundle_dir = Path(out_dir) / str(cfg["run_id"])
    onnx_path = bundle_dir / "model.onnx"
    parity = export_and_verify(model, spec, onnx_path, opset=opset, tol=tol)
    write_bundle(cfg, spec, meta, parity, bundle_dir)
    return {"bundle_dir": str(bundle_dir), "parity": parity, "meta": meta}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a change-detection model to ONNX + verify parity."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="override best.pt path")
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="skip the checkpoint; verify export/parity machinery on the real architecture",
    )
    parser.add_argument("--out-dir", default="bundles")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--set", nargs="*", default=[], help="config overrides a.b=c")
    args = parser.parse_args()

    result = export_bundle(
        args.config,
        checkpoint=args.checkpoint,
        random_init=args.random_init,
        out_dir=args.out_dir,
        opset=args.opset,
        tol=args.tol,
        overrides=args.set,
    )
    p = result["parity"]
    print(
        f"[export] {result['bundle_dir']} | parity PASS "
        f"max|Δlogit|={p['primary']['max_abs']:.2e} "
        f"(post-sigmoid {p['primary']['max_prob_abs']:.2e}) "
        f"| opset {p['opset']} input {p['input_size']}px dynamic_hw={p['dynamic_hw']}"
    )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
