#!/usr/bin/env python
"""M0 acceptance smoke: confirm staged LEVIR-CD is loadable via torchgeo.

Run on Leonardo (login node or a serial job) inside the staging venv:
    LEVIRCD_ROOT=$WORK/sat-change-detection/data/levircd python scripts/smoke_load_levircd.py
"""

from __future__ import annotations

import os
import sys

from torchgeo.datasets import LEVIRCD


def main() -> None:
    root = os.environ.get("LEVIRCD_ROOT")
    if root is None:
        work = os.environ.get("WORK")
        root = (
            os.path.join(work, "sat-change-detection", "data", "levircd")
            if work
            else "data/levircd"
        )
    print(f"[smoke] LEVIR-CD root: {root}")

    total = 0
    for split in ("train", "val", "test"):
        try:
            ds = LEVIRCD(root=root, split=split, download=False, checksum=False)
        except Exception as exc:  # report per-split; keep going
            print(f"[smoke] split={split:5s} SKIPPED ({type(exc).__name__}: {exc})")
            continue
        n = len(ds)
        total += n
        print(f"[smoke] split={split:5s} samples={n}")
        if n:
            sample = ds[0]
            print(f"[smoke]   keys={list(sample.keys())}")
            for key, value in sample.items():
                shape = getattr(value, "shape", None)
                dtype = getattr(value, "dtype", None)
                shp = tuple(shape) if shape is not None else None
                print(f"[smoke]     {key}: shape={shp} dtype={dtype}")

    if total == 0:
        sys.exit("[smoke][FAIL] 0 samples loaded — check staging/layout")
    print(f"[smoke] OK — total samples across splits: {total}")
    sys.stdout.flush()
    os._exit(0)  # dodge the login-node torch teardown hang (leonardo.md)


if __name__ == "__main__":
    main()
