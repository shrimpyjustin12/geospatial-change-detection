"""TiledOSCD loader (src.data.oscd). Needs torch + rasterio + numpy (writing tiny GeoTIFF
fixtures); missing deps skip. Exercises the OSCD-specific tricky bits vs LEVIR-CD: 4-band
(RGB+NIR) output, variable per-scene size with edge-aligned tiling, sub-tile reflect/replicate
padding, robust {0,255}/{1,2}/{0,1} mask binarization, and the deterministic val holdout.
"""

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("rasterio")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.oscd import _BAND_IDS, TiledOSCD  # noqa: E402


def _write_tif(path: Path, arr: np.ndarray) -> None:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=arr.dtype,
    ) as dst:
        dst.write(arr, 1)


def _write_city(root: Path, city: str, size: tuple[int, int], mask_vals: tuple[int, int]) -> None:
    """One OSCD city: 4 band tifs per date under imgs_{1,2}_rect + a cm/cm.tif change map."""
    h, w = size
    rng = np.random.default_rng(abs(hash(city)) % (2**32))
    for which in (1, 2):
        for band_id in _BAND_IDS:
            arr = rng.integers(0, 10000, size=(h, w), dtype=np.uint16)
            _write_tif(root / city / f"imgs_{which}_rect" / f"{band_id}.tif", arr)
    lo, hi = mask_vals
    cm = np.full((h, w), lo, dtype=np.uint8)
    cm[: h // 2, : w // 2] = hi  # a changed quadrant
    _write_tif(root / city / "cm" / "cm.tif", cm)


def write_oscd_fixture(
    root: Path,
    train: list[str],
    test: list[str],
    size: tuple[int, int] = (64, 64),
    mask_vals: tuple[int, int] = (0, 255),
) -> None:
    for city in train:
        _write_city(root, city, size, mask_vals)
    for city in test:
        _write_city(root, city, size, mask_vals)
    (root / "train.txt").write_text(",".join(train))
    (root / "test.txt").write_text(",".join(test))


def test_shapes_dtype_and_binarization(tmp_path: Path) -> None:
    write_oscd_fixture(tmp_path, train=[], test=["alpha"], size=(64, 64), mask_vals=(0, 255))
    ds = TiledOSCD(root=tmp_path, split="test", tile_size=32)
    assert len(ds) == 4  # 64/32 -> 2x2 tiles
    item = ds[0]
    assert item["image"].shape == (2, 4, 32, 32)  # two dates, 4 bands (RGB+NIR)
    assert item["mask"].shape == (1, 32, 32)
    assert item["image"].dtype == torch.float32
    # {0,255} change map must binarize to {0,1}, and the changed quadrant must survive.
    vals = set(torch.unique(item["mask"]).tolist())
    assert vals <= {0.0, 1.0}
    assert item["mask"].max().item() == 1.0


def test_variable_size_edge_aligned_tiling(tmp_path: Path) -> None:
    # 80 is not a multiple of 32 -> starts [0,32,48] per side (last tile edge-aligned) -> 3x3 = 9.
    write_oscd_fixture(tmp_path, train=[], test=["big"], size=(80, 80))
    ds = TiledOSCD(root=tmp_path, split="test", tile_size=32)
    assert len(ds) == 9
    for i in range(len(ds)):  # every tile, including edge tiles, is a full (t,t) crop
        assert ds[i]["image"].shape == (2, 4, 32, 32)


def test_sub_tile_scene_is_padded(tmp_path: Path) -> None:
    write_oscd_fixture(tmp_path, train=[], test=["tiny"], size=(20, 20))
    ds = TiledOSCD(root=tmp_path, split="test", tile_size=32)
    assert len(ds) == 1
    assert ds[0]["image"].shape == (2, 4, 32, 32)
    assert ds[0]["mask"].shape == (1, 32, 32)


def test_val_holdout_is_deterministic(tmp_path: Path) -> None:
    train = ["c0", "c1", "c2", "c3"]  # _VAL_HOLDOUT=2 -> train=c0,c1 ; val=c2,c3
    write_oscd_fixture(tmp_path, train=train, test=["t0"], size=(32, 32))
    assert TiledOSCD(root=tmp_path, split="train", tile_size=32).cities == ["c0", "c1"]
    assert TiledOSCD(root=tmp_path, split="val", tile_size=32).cities == ["c2", "c3"]
    assert TiledOSCD(root=tmp_path, split="test", tile_size=32).cities == ["t0"]


def test_scene_id_maps_tiles_to_cities(tmp_path: Path) -> None:
    # Eval harness groups tiles into scenes via scene_id; OSCD has variable tiles per city.
    write_oscd_fixture(tmp_path, train=[], test=["c0", "c1"], size=(64, 64))  # 2x2=4 tiles each
    ds = TiledOSCD(root=tmp_path, split="test", tile_size=32)
    assert len(ds) == 8
    assert [ds.scene_id(i) for i in range(len(ds))] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_mask_encoding_1_2(tmp_path: Path) -> None:
    # OSCD's original {1,2} encoding (1=no-change, 2=change) must map to {0,1}, not all-ones.
    write_oscd_fixture(tmp_path, train=[], test=["enc"], size=(32, 32), mask_vals=(1, 2))
    ds = TiledOSCD(root=tmp_path, split="test", tile_size=32)
    m = ds[0]["mask"]
    assert set(torch.unique(m).tolist()) == {0.0, 1.0}
    assert 0.0 < m.mean().item() < 1.0  # both classes present, not collapsed
