"""Tiled LEVIR-CD dataset: torchgeo LEVIRCD (1024²) -> normalized 256² tiles.

torchgeo returns ``{"image": (2,3,H,W) float, "mask": (1,H,W)}`` per scene (two dates). We tile
each scene into non-overlapping crops, scale to [0,1], standardize (ImageNet stats by default),
and optionally apply synced geometric augmentation. A one-scene cache avoids re-decoding a PNG
for each of its tiles.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.tiling import num_tiles, tile_location

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class TiledLEVIRCD(Dataset):
    """LEVIR-CD tiled to ``tile_size`` crops. Item: ``{"image": (2,3,t,t), "mask": (1,t,t)}``."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        tile_size: int = 256,
        image_size: int = 1024,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        augment: bool = False,
    ) -> None:
        from torchgeo.datasets import LEVIRCD

        self.base = LEVIRCD(root=str(root), split=split, download=False)
        self.tile_size = tile_size
        self.per_side = image_size // tile_size
        self.per_image = num_tiles(image_size, tile_size)
        self.augment = augment
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self._cache_idx = -1
        self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.base) * self.per_image

    def scene_id(self, idx: int) -> int:
        """Source-scene index for a flat tile index (LEVIR: fixed ``per_image`` tiles per scene).
        Used by the eval harness to group tiles into scenes for the per-scene breakdown."""
        return idx // self.per_image

    def _load_scene(self, img_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if img_idx == self._cache_idx and self._cache is not None:
            return self._cache
        sample = self.base[img_idx]
        image = sample["image"].float() / 255.0  # (2,3,H,W)
        image = (image - self.mean) / self.std
        mask = (sample["mask"] > 0).float()  # binarize: handles {0,1} and {0,255} encodings
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # (1,H,W)
        self._cache_idx, self._cache = img_idx, (image, mask)
        return image, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_idx, y0, x0 = tile_location(idx, self.per_side, self.tile_size)
        image, mask = self._load_scene(img_idx)
        t = self.tile_size
        image_tile = image[:, :, y0 : y0 + t, x0 : x0 + t].clone()
        mask_tile = mask[:, y0 : y0 + t, x0 : x0 + t].clone()
        if self.augment:
            image_tile, mask_tile = _augment(image_tile, mask_tile)
        return {"image": image_tile, "mask": mask_tile}
