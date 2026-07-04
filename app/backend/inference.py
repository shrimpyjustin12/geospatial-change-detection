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
from PIL import Image, ImageFilter

# Amber "thermal" change signal, painted directly into the overlay PNG (the frontend no longer
# hue-rotates). Rendered as a translucent fill with a brighter, crisp 1-2px outline so the buildings
# underneath stay visible. The opacity slider (CSS) scales the whole overlay.
FILL_RGB = (255, 141, 52)
EDGE_RGB = (255, 201, 112)
FILL_ALPHA = 130  # ~51% in-PNG; slider default 0.75 -> ~0.38 effective fill
EDGE_ALPHA = 255


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
    def tile_size(self) -> int:
        """Native tile the model was trained on (0.5 m/px). The full scene is tiled into these,
        each inferred at ``input_size`` and stitched — the bundle's documented preprocessing."""
        return int(self.preprocessing.get("tiling", {}).get("tile_size", self.input_size))

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


def _overlay_png(mask: np.ndarray, out_size: tuple[int, int]) -> str:
    """Render the change mask as a translucent amber fill + a crisp outline, at display resolution.

    Display-only: the ``mask`` is the model's thresholded output (stats are computed from it
    upstream, unchanged). Here we only (a) lightly smooth the contour to drop stair-step edges,
    (b) fill each changed region at a low, uniform alpha so buildings stay visible, and (c) trace a
    brighter 1-2px outline around each region. Returned as a PNG data URL.
    """
    binary = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    # smooth the boundary (blur then re-binarize) — removes the stair-stepping of a hard mask
    smoothed = np.asarray(binary.filter(ImageFilter.GaussianBlur(1.2))) >= 128
    sm_img = Image.fromarray((smoothed.astype(np.uint8) * 255), mode="L")
    dil = np.asarray(sm_img.filter(ImageFilter.MaxFilter(3))) >= 128  # ~1px grow
    ero = np.asarray(sm_img.filter(ImageFilter.MinFilter(3))) >= 128  # ~1px shrink
    edge = dil & ~ero  # ~2px band straddling the boundary

    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for c in range(3):
        rgba[..., c] = np.where(smoothed, FILL_RGB[c], 0)
    rgba[..., 3] = np.where(smoothed, FILL_ALPHA, 0).astype(np.uint8)
    for c in range(3):
        rgba[edge, c] = EDGE_RGB[c]
    rgba[edge, 3] = EDGE_ALPHA

    im = Image.fromarray(rgba, mode="RGBA")
    if im.size != out_size:
        im = im.resize(out_size, Image.BILINEAR)
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

    def _infer_tile(
        self, bundle: Bundle, before_t: Image.Image, after_t: Image.Image
    ) -> np.ndarray:
        """One native tile through the model -> per-pixel change probability (model output grid)."""
        size = bundle.input_size
        x = np.stack(
            [
                _to_input(before_t, size, bundle.mean, bundle.std),
                _to_input(after_t, size, bundle.mean, bundle.std),
            ],
            axis=0,
        )[None].astype(np.float32)  # (1, 2, 3, S, S)
        logits = bundle.session().run(["logits"], {"input": x})[0]
        return 1.0 / (1.0 + np.exp(-logits[0, 0]))  # (S, S)

    def predict(self, model_id: str, before: Image.Image, after: Image.Image) -> dict[str, Any]:
        """Tile the full scene into native tiles, infer each, stitch the probability map, then
        threshold + render. This is the bundle's documented preprocessing (``tiling.tile_size``);
        the model, threshold and per-pixel metric are unchanged — they are just applied per tile,
        exactly as the evaluation harness does. Feeding the whole scene in one pass would break the
        model (a 0.5 m/px model at ~4x the trained field of view detects almost nothing)."""
        bundle = self.get(model_id)
        before = before.convert("RGB")
        after = after.convert("RGB")
        w, h = before.size
        tile = bundle.tile_size

        prob = np.zeros((h, w), dtype=np.float32)
        n_tiles = 0
        t0 = time.perf_counter()
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
                box = (x0, y0, x1, y1)
                p = self._infer_tile(bundle, before.crop(box), after.crop(box))  # (S, S)
                # resize this tile's probability back to its native footprint, then place it
                p_tile = np.asarray(
                    Image.fromarray(p.astype(np.float32), mode="F").resize(
                        (x1 - x0, y1 - y0), Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
                prob[y0:y1, x0:x1] = p_tile
                n_tiles += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        thr = bundle.threshold
        mask = prob >= thr  # stitched native-resolution mask; stats derive from this

        overlay = _overlay_png(mask, (w, h))
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
            "input_size": bundle.input_size,
            "n_tiles": n_tiles,
        }
