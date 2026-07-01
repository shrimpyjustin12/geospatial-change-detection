"""Pure tiling geometry (torch-free — unit-tested in CI)."""

from __future__ import annotations


def num_tiles(image_size: int, tile_size: int) -> int:
    """Number of non-overlapping tiles per image (must divide evenly)."""
    if tile_size <= 0 or image_size % tile_size != 0:
        raise ValueError(
            f"image_size {image_size} must be a positive multiple of tile_size {tile_size}"
        )
    return (image_size // tile_size) ** 2


def tile_location(idx: int, per_side: int, tile_size: int) -> tuple[int, int, int]:
    """Map a flat tile index to ``(image_index, y0, x0)`` top-left crop coordinates.

    Tiles are laid out row-major within each image; ``idx`` runs over all images' tiles.
    """
    per_image = per_side * per_side
    img_idx, within = divmod(idx, per_image)
    row, col = divmod(within, per_side)
    return img_idx, row * tile_size, col * tile_size
