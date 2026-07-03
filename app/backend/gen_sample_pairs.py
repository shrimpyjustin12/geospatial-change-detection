"""Generate synthetic before/after curated pairs so the Space is demonstrable offline.

These are **placeholders**, not LEVIR-CD: textured pseudo-aerial tiles where the "after" image
adds/removes a few bright rectangles ("buildings"). They exercise the full curated pipeline (slider,
overlay, stats) end to end. Replace with real LEVIR-CD test tiles for the shipped Space.

    python app/backend/gen_sample_pairs.py --out app/backend/data/curated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256


def _texture(rng: np.random.Generator, base: tuple[int, int, int]) -> np.ndarray:
    """Low-frequency muted tonal ground texture around a base RGB colour (pseudo-aerial)."""
    # a single smooth grayscale field (upsampled low-res noise) drives tonal variation on all
    # channels so the result reads like ground/vegetation, not colour noise.
    small = rng.normal(0, 1, (12, 12)).astype(np.float32)
    norm = (small - small.min()) / (float(small.max() - small.min()) + 1e-6)
    field_img = Image.fromarray((norm * 255).astype("uint8")).resize((SIZE, SIZE), Image.BICUBIC)
    field = (np.asarray(field_img, dtype=np.float32) - 128.0) * 0.42  # signed, ~[-54, 54]
    img = np.asarray(base, dtype=np.float32)[None, None, :] + field[:, :, None]
    img += rng.normal(0, 4, (SIZE, SIZE, 3))  # fine grain
    return np.clip(img, 0, 255)


def _add_building(img: np.ndarray, rng: np.random.Generator) -> None:
    """Paint one bright rectangular 'building' with a slight roof-colour jitter (in place)."""
    h = int(rng.integers(14, 40))
    w = int(rng.integers(14, 40))
    y = int(rng.integers(0, SIZE - h))
    x = int(rng.integers(0, SIZE - w))
    roof = np.asarray([210, 205, 195], dtype=np.float32) + rng.normal(0, 12, 3)
    img[y : y + h, x : x + w] = np.clip(roof, 60, 255)
    img[y : y + 2, x : x + w] = np.clip(roof - 45, 0, 255)  # a little shadow/eave


def make_pair(
    seed: int, n_common: int, n_added: int, n_removed: int
) -> tuple[Image.Image, Image.Image]:
    rng = np.random.default_rng(seed)
    base = (95 + int(rng.integers(-10, 10)), 110 + int(rng.integers(-10, 10)), 80)
    ground = _texture(rng, base)
    before = ground.copy()
    after = ground.copy()

    for _ in range(n_common):  # unchanged buildings — present in both dates
        r2 = np.random.default_rng(int(rng.integers(0, 1 << 31)))
        h = int(r2.integers(14, 40))
        w = int(r2.integers(14, 40))
        y = int(r2.integers(0, SIZE - h))
        x = int(r2.integers(0, SIZE - w))
        roof = np.clip(np.asarray([205, 200, 190], np.float32) + r2.normal(0, 10, 3), 60, 255)
        before[y : y + h, x : x + w] = roof
        after[y : y + h, x : x + w] = roof

    for _ in range(n_removed):  # in before only (demolished)
        _add_building(before, rng)
    for _ in range(n_added):  # in after only (constructed) — the real "change"
        _add_building(after, rng)

    return (
        Image.fromarray(before.astype("uint8"), "RGB"),
        Image.fromarray(after.astype("uint8"), "RGB"),
    )


SPECS = [
    (
        "suburban_growth",
        "Suburban growth",
        "Several new houses appear on former open ground.",
        5,
        4,
        0,
    ),
    (
        "infill_and_demo",
        "Infill + demolition",
        "Two buildings removed, three added between dates.",
        6,
        3,
        2,
    ),
    (
        "dense_block",
        "Dense urban block",
        "Mostly stable; one new structure on the block edge.",
        9,
        1,
        0,
    ),
    (
        "no_change",
        "Stable scene (control)",
        "No construction — tests the false-positive rate.",
        7,
        0,
        0,
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="app/backend/data/curated")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for i, (pid, title, desc, common, added, removed) in enumerate(SPECS):
        before, after = make_pair(seed=100 + i, n_common=common, n_added=added, n_removed=removed)
        (out / pid).mkdir(exist_ok=True)
        before.save(out / pid / "before.png")
        after.save(out / pid / "after.png")
        pairs.append(
            {"id": pid, "title": title, "description": desc, "source": "synthetic placeholder"}
        )
    (out / "manifest.json").write_text(json.dumps({"pairs": pairs}, indent=2))
    print(f"wrote {len(pairs)} synthetic pairs to {out}")


if __name__ == "__main__":
    main()
