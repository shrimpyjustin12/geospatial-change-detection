"""ONNX bundle inference for the curated Space (Track A, CPU onnxruntime).

A *bundle* is exactly what ``src/export.py`` emits (PRD §3/§9): a directory with ``model.onnx``
and ``preprocessing.json`` (normalization, input size, band order, tiling, recommended threshold).
The demo consumes only the bundle — it never imports the training code. This module discovers
bundles under ``BUNDLES_DIR``, lazily builds an ``onnxruntime`` session per model, and runs a
before/after RGB pair through the documented preprocessing to produce a change-mask overlay + stats.

The preprocessing here mirrors ``src/data/levircd.py`` + the export contract: resize each date to
the bundle's ``input_size`` (DINOv2 needs the fixed 448 grid; the CNN tiers accept it too), scale to
[0, 1], standardize with the bundle's mean/std, stack the two dates to ``(1, 2, 3, S, S)``.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

OVERLAY_RGB = (220, 40, 40)  # change highlighted in red


@dataclass
class Bundle:
    """A loaded model bundle: its preprocessing contract + a lazily-created ORT session."""

    model_id: str
    root: Path
    preprocessing: dict[str, Any]
    metrics_card: str
    _session: ort.InferenceSession | None = field(default=None, repr=False)

    @property
    def input_size(self) -> int:
        return int(self.preprocessing.get("input_size", 256))

    @property
    def threshold(self) -> float:
        return float(self.preprocessing.get("output", {}).get("recommended_threshold", 0.5))

    @property
    def mean(self) -> np.ndarray:
        m = self.preprocessing.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
        return np.asarray(m, dtype=np.float32).reshape(3, 1, 1)

    @property
    def std(self) -> np.ndarray:
        s = self.preprocessing.get("normalization", {}).get("std", [0.229, 0.224, 0.225])
        return np.asarray(s, dtype=np.float32).reshape(3, 1, 1)

    @property
    def is_placeholder(self) -> bool:
        return "RANDOM-INIT" in str(self.preprocessing.get("weights", ""))

    def session(self) -> ort.InferenceSession:
        if self._session is None:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2  # HF free tier is ~2 vCPU; ORT clamps to the host anyway
            self._session = ort.InferenceSession(
                str(self.root / "model.onnx"),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def summary(self) -> dict[str, Any]:
        cfg_name = self.preprocessing.get("dinov2_note")
        return {
            "id": self.model_id,
            "input_size": self.input_size,
            "dynamic_hw": bool(self.preprocessing.get("dynamic_hw", False)),
            "threshold": self.threshold,
            "band_order": self.preprocessing.get("band_order", ["R", "G", "B"]),
            "is_placeholder": self.is_placeholder,
            "fixed_grid": cfg_name is not None,
        }


def _to_input(img: Image.Image, size: int, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """PIL RGB -> normalized ``(3, size, size)`` float32 (resize bilinear, /255, standardize)."""
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
    chw = arr.transpose(2, 0, 1) / 255.0
    return (chw - mean) / std


def _overlay_png(mask: np.ndarray, prob: np.ndarray, out_size: tuple[int, int]) -> str:
    """RGBA overlay: change pixels colored, alpha scaled by confidence; returned as a data URL."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = OVERLAY_RGB[0]
    rgba[..., 1] = OVERLAY_RGB[1]
    rgba[..., 2] = OVERLAY_RGB[2]
    # alpha only where predicted change; modulate by probability so faint calls look faint.
    alpha = np.where(mask, np.clip(120 + 135 * prob, 0, 255), 0).astype(np.uint8)
    rgba[..., 3] = alpha
    im = Image.fromarray(rgba, mode="RGBA").resize(out_size, Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class BundleRegistry:
    """Discovers and caches model bundles under ``bundles_dir``."""

    def __init__(self, bundles_dir: str | Path) -> None:
        self.bundles_dir = Path(bundles_dir)
        self._bundles: dict[str, Bundle] = {}
        self.reload()

    def reload(self) -> None:
        self._bundles.clear()
        if not self.bundles_dir.exists():
            return
        for child in sorted(self.bundles_dir.iterdir()):
            pre = child / "preprocessing.json"
            onnx = child / "model.onnx"
            if not (pre.exists() and onnx.exists()):
                continue
            card = child / "metrics_card.md"
            self._bundles[child.name] = Bundle(
                model_id=child.name,
                root=child,
                preprocessing=json.loads(pre.read_text()),
                metrics_card=card.read_text() if card.exists() else "",
            )

    def ids(self) -> list[str]:
        return list(self._bundles)

    def get(self, model_id: str) -> Bundle:
        if model_id not in self._bundles:
            raise KeyError(model_id)
        return self._bundles[model_id]

    def summaries(self) -> list[dict[str, Any]]:
        return [b.summary() for b in self._bundles.values()]

    def predict(self, model_id: str, before: Image.Image, after: Image.Image) -> dict[str, Any]:
        """Run the pair through the bundle -> overlay data URL + change stats."""
        bundle = self.get(model_id)
        size = bundle.input_size
        x = np.stack(
            [
                _to_input(before, size, bundle.mean, bundle.std),
                _to_input(after, size, bundle.mean, bundle.std),
            ],
            axis=0,
        )[None]  # (1, 2, 3, S, S)

        t0 = time.perf_counter()
        logits = bundle.session().run(["logits"], {"input": x.astype(np.float32)})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        prob = 1.0 / (1.0 + np.exp(-logits[0, 0]))  # (S, S)
        thr = bundle.threshold
        mask = prob >= thr

        out_size = before.size  # (W, H) for display
        overlay = _overlay_png(mask, prob, out_size)
        changed_frac = float(mask.mean())
        mean_conf_changed = float(prob[mask].mean()) if mask.any() else 0.0
        return {
            "overlay_png": overlay,
            "threshold": thr,
            "is_placeholder": bundle.is_placeholder,
            "stats": {
                "changed_fraction": changed_frac,
                "changed_percent": round(100.0 * changed_frac, 2),
                "mean_confidence_changed": round(mean_conf_changed, 4),
                "mean_confidence_overall": round(float(prob.mean()), 4),
                "changed_pixels": int(mask.sum()),
                "total_pixels": int(mask.size),
            },
            "elapsed_ms": round(elapsed_ms, 1),
            "input_size": size,
        }
