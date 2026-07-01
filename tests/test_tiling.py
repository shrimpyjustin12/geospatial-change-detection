"""Tests for tiling geometry (torch-free — always runs in CI)."""

import pytest

from src.data.tiling import num_tiles, tile_location


def test_num_tiles():
    assert num_tiles(1024, 256) == 16
    assert num_tiles(256, 256) == 1


def test_num_tiles_rejects_non_multiple():
    with pytest.raises(ValueError):
        num_tiles(1000, 256)


def test_tile_location_row_major_within_and_across_images():
    assert tile_location(0, 4, 256) == (0, 0, 0)  # image 0, top-left
    assert tile_location(1, 4, 256) == (0, 0, 256)  # next column
    assert tile_location(4, 4, 256) == (0, 256, 0)  # next row
    assert tile_location(15, 4, 256) == (0, 768, 768)  # last tile of image 0
    assert tile_location(16, 4, 256) == (1, 0, 0)  # first tile of image 1
