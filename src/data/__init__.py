"""Dataset wrappers, tiling, transforms, and a name-dispatched dataset factory."""

from __future__ import annotations

from typing import Any


def build_dataset(dcfg: dict[str, Any], split: str, augment: bool) -> Any:
    """Construct a tiled change-detection dataset from the ``data`` config block.

    Dispatches on ``data.name`` (``levircd`` = 3-band aerial, ``oscd`` = 4-band Sentinel-2) so
    train/eval/export stay dataset-agnostic. Loader modules are imported lazily to keep importing
    this package light (each pulls torch + its IO stack only when actually used). Returns ``Any``
    because the concrete tiled dataset type is chosen at runtime.
    """
    name = str(dcfg.get("name", "levircd"))
    tile_size = int(dcfg.get("tile_size", 256))
    if name == "levircd":
        from src.data.levircd import TiledLEVIRCD

        return TiledLEVIRCD(root=dcfg["root"], split=split, tile_size=tile_size, augment=augment)
    if name == "oscd":
        from src.data.oscd import TiledOSCD

        return TiledOSCD(root=dcfg["root"], split=split, tile_size=tile_size, augment=augment)
    raise ValueError(f"unknown dataset {name!r} (supported: levircd, oscd)")
