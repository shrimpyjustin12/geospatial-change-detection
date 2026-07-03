"""Curated before/after pair registry for the ``/curated`` demo mode (PRD §10.1).

A curated *pair* is two co-registered RGB tiles of the same place at two times. On the real Space
these are LEVIR-CD test samples (plus a couple of well-known change events); the registry reads a
``manifest.json`` describing each pair and serves the images from disk. ``scripts/gen_sample_pairs``
can synthesize placeholder pairs so the pipeline is demonstrable before the real tiles are staged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image


class CuratedRegistry:
    """Reads ``<data_dir>/manifest.json`` -> pairs, each with ``before.png`` / ``after.png``."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self.pairs: dict[str, dict[str, Any]] = {}
        self.reload()

    def reload(self) -> None:
        self.pairs.clear()
        manifest = self.data_dir / "manifest.json"
        if not manifest.exists():
            return
        for entry in json.loads(manifest.read_text()).get("pairs", []):
            pid = str(entry["id"])
            if (self.data_dir / pid / "before.png").exists():
                self.pairs[pid] = entry

    def list(self) -> list[dict[str, Any]]:
        out = []
        for pid, entry in self.pairs.items():
            before = self.data_dir / pid / "before.png"
            with Image.open(before) as im:
                w, h = im.size
            out.append(
                {
                    "id": pid,
                    "title": entry.get("title", pid),
                    "description": entry.get("description", ""),
                    "source": entry.get("source", ""),
                    "width": w,
                    "height": h,
                }
            )
        return out

    def image_path(self, pair_id: str, which: str) -> Path:
        if pair_id not in self.pairs:
            raise KeyError(pair_id)
        if which not in ("before", "after"):
            raise ValueError(which)
        return self.data_dir / pair_id / f"{which}.png"

    def get_pair(self, pair_id: str) -> tuple[Image.Image, Image.Image]:
        before = Image.open(self.image_path(pair_id, "before")).convert("RGB")
        after = Image.open(self.image_path(pair_id, "after")).convert("RGB")
        return before, after
