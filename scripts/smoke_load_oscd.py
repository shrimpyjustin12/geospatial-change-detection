#!/usr/bin/env python
"""M5 acceptance smoke: confirm staged OSCD is loadable via TiledOSCD (torchgeo's pinned version
has no OSCD class) AND compute the per-band normalization stats needed for D2.

Prints per-split tile counts + one item's 4-band shapes / value range, then per-band mean/std over
the train split (paste into ``src/data/oscd.py`` ``OSCD_MEAN`` / ``OSCD_STD``). Because the loader's
placeholder stats are mean 0 / std 1, ``ds[i]["image"]`` is exactly the ÷10000 reflectance, so the
stats computed here are the true reflectance stats to standardize by.

Run on Leonardo in a venv with torch + rasterio (``.venv-train``), via a serial job — NOT the login
node (leonardo.md: login nodes SIGKILL compute):
    OSCD_ROOT=$WORK/sat-change-detection/data/oscd python scripts/smoke_load_oscd.py
"""

from __future__ import annotations

import os
import sys

# Make ``src`` importable when run as a standalone script (sys.path[0] is scripts/, not repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _train_band_stats(root: str, max_tiles: int = 500) -> dict[str, list[float]]:
    """Per-band (R,G,B,NIR) mean/std over up to ``max_tiles`` train tiles of ÷10000 reflectance."""
    import torch

    from src.data.oscd import TiledOSCD

    ds = TiledOSCD(root=root, split="train", tile_size=256)  # placeholder norm -> raw reflectance
    n = min(len(ds), max_tiles)
    csum = torch.zeros(4)
    csq = torch.zeros(4)
    cnt = 0
    for i in range(n):
        x = ds[i]["image"].permute(1, 0, 2, 3).reshape(4, -1)  # (4, 2*t*t): both dates pooled
        csum += x.sum(dim=1)
        csq += (x * x).sum(dim=1)
        cnt += x.shape[1]
    mean = csum / cnt
    std = (csq / cnt - mean**2).clamp_min(1e-12).sqrt()
    return {
        "mean": [round(v, 5) for v in mean.tolist()],
        "std": [round(v, 5) for v in std.tolist()],
    }


def main() -> None:
    root = os.environ.get("OSCD_ROOT")
    if root is None:
        work = os.environ.get("WORK")
        root = os.path.join(work, "sat-change-detection", "data", "oscd") if work else "data/oscd"
    print(f"[smoke] OSCD root: {root}")

    from src.data.oscd import BAND_ORDER, TiledOSCD

    total = 0
    for split in ("train", "val", "test"):
        try:
            ds = TiledOSCD(root=root, split=split, tile_size=256)
        except Exception as exc:  # report per-split; keep going
            print(f"[smoke] split={split:5s} SKIPPED ({type(exc).__name__}: {exc})")
            continue
        n = len(ds)
        total += n
        print(f"[smoke] split={split:5s} cities={len(ds.cities)} tiles={n}")
        if n:
            item = ds[0]
            img, mask = item["image"], item["mask"]
            print(
                f"[smoke]   image shape={tuple(img.shape)} dtype={img.dtype} "
                f"min={img.min():.3f} max={img.max():.3f} bands={BAND_ORDER}"
            )
            print(f"[smoke]   mask  shape={tuple(mask.shape)} changed_frac={mask.mean():.4f}")

    if total == 0:
        sys.exit("[smoke][FAIL] 0 tiles loaded — check staging/layout (D1)")

    try:
        stats = _train_band_stats(root)
        print("[smoke] OSCD train per-band stats (reflectance) — paste into src/data/oscd.py:")
        print(f"[smoke]   OSCD_MEAN = {tuple(stats['mean'])}")
        print(f"[smoke]   OSCD_STD  = {tuple(stats['std'])}")
    except Exception as exc:
        print(f"[smoke] stats SKIPPED ({type(exc).__name__}: {exc})")

    print(f"[smoke] OK — total tiles across splits: {total}")
    sys.stdout.flush()
    os._exit(0)  # dodge the login-node torch teardown hang (leonardo.md)


if __name__ == "__main__":
    main()
