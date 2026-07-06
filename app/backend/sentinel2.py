"""Curated Sentinel-2 (Track-B) AOI registry for the ``/sentinel2`` demo tab (M5, PRD §6.3/§10).

A curated Sentinel-2 *AOI* is a real-world location with large, obvious change visible even at 10 m
(reclamation, a filling reservoir, an airport built from farmland, a solar park, a desert city). Its
before/after imagery and change prediction are **baked offline** by ``build_sentinel2.py`` and
served here straight from the cache — **no runtime inference, no runtime STAC, no GPU**, exactly
like the aerial curated mode. The runtime image stays STAC-free (no ``pystac``/``rasterio`` deps).

The registry reads ``<data_dir>/manifest.json`` (per-AOI metadata: title, MGRS tile, centre, the
acquisition dates + cloud cover) and the baked ``<data_dir>/_predictions.json`` (the same cache
schema ``inference.py.predict`` returns), and exposes both to the API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

_IMAGE_KINDS = ("before", "after", "overlay")


class Sentinel2Registry:
    """Reads the baked Sentinel-2 manifest + prediction cache and serves them (cache only)."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.aois: dict[str, dict[str, Any]] = {}
        self.predictions: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        self.aois.clear()
        self.predictions.clear()
        manifest = self.data_dir / "manifest.json"
        if manifest.exists():
            for entry in json.loads(manifest.read_text()).get("pairs", []):
                aid = str(entry["id"])
                if (self.data_dir / aid / "before.png").exists():
                    self.aois[aid] = entry
        cache = self.data_dir / "_predictions.json"
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
            except json.JSONDecodeError:
                data = {}
            for aid, pred in data.items():
                if aid in self.aois:
                    self.predictions[aid] = pred

    def list(self) -> list[dict[str, Any]]:
        """Per-AOI metadata + the baked prediction summary (stats/threshold/tiles), minus the heavy
        ``overlay_png`` data URL — the overlay is served as a PNG file via :meth:`image_path`."""
        out = []
        for aid, entry in self.aois.items():
            pred = self.predictions.get(aid, {})
            out.append(
                {
                    "id": aid,
                    "title": entry.get("title", aid),
                    "description": entry.get("description", ""),
                    "source": entry.get("source", "Sentinel-2 L2A · 10 m"),
                    "tile": entry.get("tile", ""),
                    "center": entry.get("center"),
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                    "date_before": entry.get("date_before"),
                    "date_after": entry.get("date_after"),
                    "cloud_before": entry.get("cloud_before"),
                    "cloud_after": entry.get("cloud_after"),
                    "model_id": pred.get("model_id", ""),
                    "threshold": pred.get("threshold"),
                    "is_placeholder": pred.get("is_placeholder", False),
                    "n_tiles": pred.get("n_tiles"),
                    "input_size": pred.get("input_size"),
                    "elapsed_ms": pred.get("elapsed_ms"),
                    "stats": pred.get("stats", {}),
                }
            )
        return out

    def image_path(self, aoi_id: str, which: str) -> Path:
        if aoi_id not in self.aois:
            raise KeyError(aoi_id)
        if which not in _IMAGE_KINDS:
            raise ValueError(which)
        return self.data_dir / aoi_id / f"{which}.png"

    def dimensions(self, aoi_id: str) -> tuple[int, int]:
        with Image.open(self.image_path(aoi_id, "before")) as im:
            return im.size
