"""Tiled OSCD dataset: Sentinel-2 RGB+NIR change detection (torchgeo-free).

OSCD (Onera Satellite Change Detection) ships, per city, Sentinel-2 band GeoTIFFs for two dates
plus a binary change map. Two things differ from LEVIR-CD and drive this loader:

1. **4 bands, not 3.** We read B04, B03, B02, B08 = R, G, B, NIR, scale reflectance to ~[0, 1]
   (Sentinel-2 L2A is uint16 ×10000), and standardize by OSCD-train stats (NOT ImageNet).
2. **Variable scene size.** OSCD cities differ in H×W, so the fixed-grid ``tiling`` helpers do not
   apply; we build a per-scene tile index with edge-aligned coverage (the last row/col tile is
   flush with the far edge, overlapping slightly, so every pixel is covered with no padding —
   except scenes smaller than a tile, which are reflect-padded up to one tile).

Item: ``{"image": (2, 4, t, t) float, "mask": (1, t, t) float}`` — two dates, 4 bands.

torchgeo's pinned version has no OSCD class, so file IO is done here directly (``rasterio``,
imported lazily like ``levircd`` does with torchgeo, to keep ``src.data`` imports light).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Output channel order and the Sentinel-2 band ids that fill it.
BAND_ORDER: tuple[str, ...] = ("R", "G", "B", "NIR")
_BAND_IDS: tuple[str, ...] = ("B04", "B03", "B02", "B08")  # R, G, B, NIR

# Sentinel-2 L2A reflectance is uint16 scaled by 10000 — divide before standardizing.
_S2_SCALE = 10000.0

# D2 (plan Task 9): replace with per-band mean/std computed over the OSCD *train* split, on the
# post-``/_S2_SCALE`` reflectance values. Until staged, standardization is a no-op (mean 0 / std 1)
# so scaling alone drives the input distribution; refined + kept in sync with export's
# ``preprocessing.json`` once the real stats are known.
OSCD_MEAN: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
OSCD_STD: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

# Cities held out from the train list to form a scene-disjoint val split (threshold selection
# stays off test — standing decision). Deterministic: the last ``_VAL_HOLDOUT`` train cities.
_VAL_HOLDOUT = 2


def _augment(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Random hflip/vflip/rot90 applied identically to both dates and the mask."""
    if torch.rand(()) < 0.5:
        image, mask = image.flip(-1), mask.flip(-1)
    if torch.rand(()) < 0.5:
        image, mask = image.flip(-2), mask.flip(-2)
    k = int(torch.randint(0, 4, ()))
    if k:
        image = torch.rot90(image, k, dims=(-2, -1))
        mask = torch.rot90(mask, k, dims=(-2, -1))
    return image, mask


def _tile_starts(size: int, tile: int) -> list[int]:
    """Top-left crop starts covering ``[0, size)`` with ``tile``-sized non-overlapping steps and a
    final edge-aligned tile (overlaps the previous one) so no pixels are dropped. Scenes smaller
    than a tile yield ``[0]`` (the caller reflect-pads them up to one tile)."""
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, tile))
    if starts[-1] != size - tile:
        starts.append(size - tile)
    return starts


def _read_split_cities(root: Path, split: str) -> list[str]:
    """City names for ``split`` in {train, val, test}.

    Reads ``train.txt`` / ``test.txt`` at the dataset root (comma- or newline-separated, as OSCD
    ships them). ``val`` is a deterministic holdout of the last ``_VAL_HOLDOUT`` train cities;
    ``train`` is the remainder.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r} (expected train|val|test)")
    list_name = "test.txt" if split == "test" else "train.txt"
    list_path = root / list_name
    if not list_path.exists():
        raise FileNotFoundError(f"OSCD split list not found: {list_path}")
    raw = list_path.read_text().replace(",", "\n")
    cities = [c.strip() for c in raw.splitlines() if c.strip()]
    if split == "test":
        return cities
    if len(cities) <= _VAL_HOLDOUT:
        raise ValueError(
            f"train list has {len(cities)} cities; need > {_VAL_HOLDOUT} to hold out a val split"
        )
    if split == "val":
        return cities[-_VAL_HOLDOUT:]
    return cities[:-_VAL_HOLDOUT]


class TiledOSCD(Dataset):
    """OSCD tiled to ``tile_size`` crops. Item: ``{"image": (2,4,t,t), "mask": (1,t,t)}``."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile_size: int = 256,
        mean: tuple[float, float, float, float] = OSCD_MEAN,
        std: tuple[float, float, float, float] = OSCD_STD,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.tile_size = tile_size
        self.augment = augment
        self.mean = torch.tensor(mean).view(4, 1, 1)
        self.std = torch.tensor(std).view(4, 1, 1)
        self.cities = _read_split_cities(self.root, split)
        # Build the flat per-scene tile index: one (city_idx, y0, x0) entry per tile.
        self.index: list[tuple[int, int, int]] = []
        for ci, city in enumerate(self.cities):
            h, w = self._scene_size(city)
            for y0 in _tile_starts(h, tile_size):
                for x0 in _tile_starts(w, tile_size):
                    self.index.append((ci, y0, x0))
        self._cache_idx = -1
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.index)

    # ---- file resolution -------------------------------------------------------------------
    def _date_dir(self, city: str, which: int) -> Path:
        """Directory of per-band tifs for date ``which`` (1 or 2). Prefer the co-registered
        ``imgs_{which}_rect`` variant used for change detection; fall back to ``imgs_{which}``."""
        city_dir = self.root / city
        for name in (f"imgs_{which}_rect", f"imgs_{which}"):
            if (city_dir / name).is_dir():
                return city_dir / name
        raise FileNotFoundError(
            f"no imgs_{which}[_rect] dir for OSCD city {city!r} under {city_dir}"
        )

    def _band_path(self, date_dir: Path, band_id: str) -> Path:
        """Resolve a single band tif by id (e.g. 'B04') via glob, tolerating city-prefixed names."""
        hits = sorted(date_dir.glob(f"*{band_id}*.tif")) + sorted(
            date_dir.glob(f"*{band_id}*.tiff")
        )
        if not hits:
            raise FileNotFoundError(f"band {band_id} not found in {date_dir}")
        return hits[0]

    def _mask_path(self, city: str) -> Path:
        """Resolve the change map (OSCD ships ``cm/cm.png``; tolerate tif/other names via glob)."""
        cm_dir = self.root / city / "cm"
        for pat in ("*cm*.png", "*cm*.tif", "*.png", "*.tif"):
            hits = sorted(cm_dir.glob(pat))
            if hits:
                return hits[0]
        raise FileNotFoundError(f"no change map under {cm_dir}")

    def _scene_size(self, city: str) -> tuple[int, int]:
        """(H, W) of a city from a single band's metadata (no full raster read)."""
        import rasterio

        with rasterio.open(self._band_path(self._date_dir(city, 1), _BAND_IDS[0])) as src:
            return int(src.height), int(src.width)

    # ---- raster reads ----------------------------------------------------------------------
    def _read_date(self, city: str, which: int) -> np.ndarray:
        """Stack the 4 bands for one date into ``(4, H, W)`` float reflectance (÷ _S2_SCALE)."""
        import rasterio

        date_dir = self._date_dir(city, which)
        bands = []
        for band_id in _BAND_IDS:
            with rasterio.open(self._band_path(date_dir, band_id)) as src:
                bands.append(src.read(1).astype(np.float32))
        return np.stack(bands, axis=0) / _S2_SCALE

    def _read_mask(self, city: str) -> np.ndarray:
        """Binary change map ``(H, W)`` in {0,1}. Robust to OSCD's {0,255}, {0,1}, {1,2} encodings:
        the max label is 'change' when the map spans >1 value, else a plain ``>0``."""
        import rasterio

        with rasterio.open(self._mask_path(city)) as src:
            arr = src.read(1).astype(np.float32)
        thresh = 1.0 if float(arr.max()) > 1.0 else 0.0
        return (arr > thresh).astype(np.float32)

    def _load_scene(self, city_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardized image ``(2,4,H',W')`` + mask ``(1,H',W')`` for a city, reflect-padded up to
        one tile if the scene is smaller. Caches the most recent scene (tiles reuse it)."""
        if city_idx == self._cache_idx and self._cache is not None:
            return self._cache
        city = self.cities[city_idx]
        d1 = torch.from_numpy(self._read_date(city, 1))  # (4,H,W)
        d2 = torch.from_numpy(self._read_date(city, 2))
        image = torch.stack([d1, d2], dim=0)  # (2,4,H,W)
        image = (image - self.mean) / self.std
        mask = torch.from_numpy(self._read_mask(city)).unsqueeze(0)  # (1,H,W)
        t = self.tile_size
        h, w = image.shape[-2:]
        pad_h, pad_w = max(0, t - h), max(0, t - w)
        if pad_h or pad_w:
            # pad bottom/right so a sub-tile scene still yields a full (t,t) crop. ``replicate``
            # (vs reflect) has no pad<size constraint, so it is safe for very small scenes.
            image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
            mask = torch.nn.functional.pad(
                mask.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate"
            ).squeeze(0)
        self._cache_idx, self._cache = city_idx, (image, mask)
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        city_idx, y0, x0 = self.index[idx]
        image, mask = self._load_scene(city_idx)
        t = self.tile_size
        image_tile = image[:, :, y0 : y0 + t, x0 : x0 + t].clone()
        mask_tile = mask[:, y0 : y0 + t, x0 : x0 + t].clone()
        if self.augment:
            image_tile, mask_tile = _augment(image_tile, mask_tile)
        return {"image": image_tile, "mask": mask_tile}
