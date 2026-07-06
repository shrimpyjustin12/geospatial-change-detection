"""Build-time Sentinel-2 AOI bake for the curated Sentinel-2 (Track-B) tab — M5 Phase 4.

**BUILD-TIME ONLY.** This is the *only* place Planetary Computer STAC / rasterio run. Its deps
(``pystac-client``, ``planetary-computer``, ``rasterio``) are **build-time only** and are never
added to ``app/backend/requirements.txt`` — the runtime Space image stays STAC-free and serves the
baked cache exactly like the aerial curated mode (no runtime inference, no runtime STAC, no GPU).

For each confirmed AOI (a real-world location with large, obvious change visible even at 10 m) it:

1. Resolves the two confirmed low-cloud S2 **L2A** acquisitions from PC STAC (exact day + tile).
2. Reads a fixed window of bands **B04,B03,B02,B08** (R,G,B,NIR) around the AOI centre.
   Co-registration is trivial: each AOI's before/after share one MGRS tile → identical UTM grid, so
   the same pixel window is aligned by construction (asserted).
3. Harmonises reflectance across processing baselines (baseline ≥ 04.00 carries ``BOA_ADD_OFFSET``
   ``-1000``; OSCD predates it) so both dates and the model input match training's radiometry.
4. Renders 8-bit RGB display PNGs (joint percentile stretch so before/after are radiometrically
   comparable).
5. Runs the OSCD 4-band ONNX bundle **offline** with its own tile-stitch — deliberately *not* the
   ``inference.py`` path, which is hard-coded RGB (3-band). ``inference.py._overlay_png`` IS
   band-agnostic and is reused for the amber-fill + outline overlay.
6. Bakes ``before.png`` / ``after.png`` / ``overlay.png`` + ``manifest.json`` + a predictions cache
   under the gitignored ``app/backend/data/sentinel2/`` — the cache schema the app already serves.

Usage (from the repo root, with the CPU ``.venv`` that has the STAC/rasterio deps)::

    python app/backend/build_sentinel2.py                 # bake all AOIs
    python app/backend/build_sentinel2.py --only gerd_reservoir --window 1536
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import planetary_computer as pc
import pystac_client
import rasterio
from PIL import Image
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

# reuse the band-agnostic overlay renderer from inference.py (flat import, as in app.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import _overlay_png  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
BUNDLE_DIR = _REPO / "bundles" / "oscd_s2_baseline"
OUT_DIR = _HERE / "data" / "sentinel2"
MODEL_ID = "oscd_s2_baseline"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Sentinel-2 bands in the OSCD band order [R, G, B, NIR].
BANDS = ("B04", "B03", "B02", "B08")


@dataclass
class AOI:
    """A curated Sentinel-2 change site: two confirmed low-cloud acquisitions on one MGRS tile."""

    id: str
    title: str
    description: str
    lat: float
    lon: float
    tile: str  # MGRS tile, e.g. "T40RCP" — before/after share it → aligned by construction
    date_before: str  # "YYYY-MM-DD"
    date_after: str
    window: int = 1024  # crop side in 10 m pixels (10.24 km); tuned per feature scale


# The 5 confirmed AOIs (arid/engineered, crispest imagery — reclamation / reservoir / airport /
# solar / desert city). Dates + tiles verified low-cloud S2 L2A with all four bands present.
AOIS: list[AOI] = [
    AOI(
        "dubai_deira",
        "Dubai — Deira Islands reclamation",
        "Land reclaimed from the Persian Gulf: new coastline and island fill off Deira.",
        25.30,
        55.34,
        "T40RCP",
        "2016-03-03",
        "2023-04-28",
        window=1024,
    ),
    AOI(
        "gerd_reservoir",
        "Grand Ethiopian Renaissance Dam — reservoir filling",
        "The Blue Nile reservoir behind the GERD fills — open water where there was valley.",
        11.21,
        35.09,
        "T36PYT",
        "2020-02-14",
        "2023-12-25",
        window=1536,
    ),
    AOI(
        "beijing_daxing",
        "Beijing Daxing International Airport",
        "A greenfield mega-airport built from farmland — terminal, aprons and runways appear.",
        39.51,
        116.41,
        "T50SMJ",
        "2016-10-10",
        "2019-11-19",
        window=1024,
    ),
    AOI(
        "bhadla_solar",
        "Bhadla Solar Park, Rajasthan",
        "One of the world's largest solar parks spreads across the Thar desert.",
        27.53,
        71.91,
        "T42RYR",
        "2017-12-17",
        "2021-12-16",
        window=1280,
    ),
    AOI(
        "egypt_newcapital",
        "New Administrative Capital, Egypt",
        "A new capital city rises from open desert east of Cairo.",
        30.01,
        31.75,
        "T36RUU",
        "2016-08-27",
        "2023-08-26",
        window=1280,
    ),
]


# -------------------------------------------------------------------------------------------------
# OSCD 4-band ONNX inference (standalone tile-stitch; the eval-harness / bundle preprocessing)
# -------------------------------------------------------------------------------------------------
@dataclass
class OscdModel:
    """The OSCD 4-band ONNX bundle + its documented preprocessing (band order, scaling, norm)."""

    pre: dict[str, Any]
    _session: ort.InferenceSession = field(repr=False)

    @classmethod
    def load(cls, bundle_dir: Path) -> OscdModel:
        pre = json.loads((bundle_dir / "preprocessing.json").read_text())
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        sess = ort.InferenceSession(
            str(bundle_dir / "model.onnx"), sess_options=so, providers=["CPUExecutionProvider"]
        )
        return cls(pre=pre, _session=sess)

    @property
    def mean(self) -> np.ndarray:
        return np.asarray(self.pre["normalization"]["mean"], dtype=np.float32).reshape(-1, 1, 1)

    @property
    def std(self) -> np.ndarray:
        return np.asarray(self.pre["normalization"]["std"], dtype=np.float32).reshape(-1, 1, 1)

    @property
    def threshold(self) -> float:
        return float(self.pre["output"]["recommended_threshold"])

    @property
    def tile_size(self) -> int:
        return int(self.pre.get("tiling", {}).get("tile_size", 256))

    @property
    def input_size(self) -> int:
        return int(self.pre.get("input_size", 256))

    def _standardize(self, refl: np.ndarray) -> np.ndarray:
        """(4, H, W) float32 reflectance -> standardized model input (÷scale already applied)."""
        return (refl - self.mean) / self.std

    def predict(
        self, before: np.ndarray, after: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Tile the scene into ``tile_size`` crops, run each 4-band pair through the ONNX model, and
        stitch the probability map — the same per-tile procedure as ``src/evaluate.py`` and the
        bundle's documented preprocessing. Returns ``(mask, prob, n_tiles, elapsed_ms)``.

        ``before``/``after`` are ``(4, H, W)`` standardized inputs. The model is fully convolutional
        with dynamic H/W and ``input_size == tile_size``, so each tile is fed at its native size."""
        _, h, w = before.shape
        tile = self.tile_size
        prob = np.zeros((h, w), dtype=np.float32)
        n_tiles = 0
        t0 = time.perf_counter()
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                b = before[:, y0:y1, x0:x1]
                a = after[:, y0:y1, x0:x1]
                x = np.stack([b, a], axis=0)[None].astype(np.float32)  # (1, 2, 4, t, t)
                logits = self._session.run(["logits"], {"input": x})[0]
                prob[y0:y1, x0:x1] = 1.0 / (1.0 + np.exp(-logits[0, 0]))
                n_tiles += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        mask = prob >= self.threshold
        return mask, prob, n_tiles, elapsed_ms


# -------------------------------------------------------------------------------------------------
# STAC fetch + windowed read + radiometric harmonisation
# -------------------------------------------------------------------------------------------------
COVER_MIN = 0.995  # a scene must fill the AOI window; same MGRS tile ≠ same orbit swath coverage
CLOUD_MAX = 25.0  # generous ceiling for a fallback; confirmed scenes are all < 1%


def open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def _day(item: Any) -> str:
    return item.datetime.strftime("%Y-%m-%d")


def _coverage(item: Any, aoi: AOI) -> float:
    """Fraction of the AOI window with valid (non-nodata) data. A given acquisition only fills the
    part of its MGRS tile its orbit swath covers, so ``eo:cloud_cover`` alone is not enough."""
    with rasterio.open(item.assets["B04"].href) as ds:
        dn = ds.read(1, window=_window_from_center(ds, aoi.lon, aoi.lat, aoi.window))
    return float((dn > 0).mean())


def find_item(cat: pystac_client.Client, aoi: AOI, date: str) -> tuple[Any, str, bool]:
    """Resolve an S2 L2A item that actually *covers* the AOI window. Prefers the confirmed ``date``
    (lowest cloud if reprocessed twice); if that acquisition's swath misses the window, falls back
    to the lowest-cloud covering scene that month on the same tile. Returns ``(item, day, fb)``."""
    y, m, _ = date.split("-")
    search = cat.search(
        collections=["sentinel-2-l2a"],
        intersects={"type": "Point", "coordinates": [aoi.lon, aoi.lat]},
        datetime=f"{y}-{m}-01/{y}-{m}-28",
    )
    want_tile = aoi.tile.lstrip("T")
    same_tile = [
        it
        for it in search.items()
        if it.properties.get("s2:mgrs_tile") == want_tile and it.datetime is not None
    ]
    if not same_tile:
        raise RuntimeError(f"{aoi.id}: no S2 L2A item on tile {aoi.tile} in {y}-{m}")

    cloud = lambda it: it.properties.get("eo:cloud_cover", 100.0)  # noqa: E731

    # 1) the confirmed day, if its swath covers the window
    for it in sorted((x for x in same_tile if _day(x) == date), key=cloud):
        if _coverage(it, aoi) >= COVER_MIN:
            return it, date, False

    # 2) fallback — any covering low-cloud scene that month; lowest cloud, then nearest target day
    target = int(date[8:10])
    covering = [
        it for it in same_tile if cloud(it) <= CLOUD_MAX and _coverage(it, aoi) >= COVER_MIN
    ]
    if not covering:
        raise RuntimeError(
            f"{aoi.id}: no covering low-cloud S2 scene near {date} on tile {aoi.tile}"
        )
    covering.sort(key=lambda it: (cloud(it), abs(int(_day(it)[8:10]) - target)))
    chosen = covering[0]
    return chosen, _day(chosen), True


def _boa_offset(item: Any) -> float:
    """DN offset to subtract before ÷10000. Baseline ≥ 04.00 (post 2022-01-25) carries
    ``BOA_ADD_OFFSET = -1000``; OSCD predates it, so we harmonise every scene to that convention."""
    baseline = item.properties.get("s2:processing_baseline")
    try:
        if baseline is not None and float(baseline) >= 4.0:
            return 1000.0
    except (TypeError, ValueError):
        pass
    # fallback on acquisition date if the baseline field is absent
    dt = item.datetime
    if dt is not None and dt.strftime("%Y-%m-%d") >= "2022-01-25":
        return 1000.0
    return 0.0


def _window_from_center(ds: rasterio.io.DatasetReader, lon: float, lat: float, win: int) -> Window:
    """A ``win×win`` pixel window centred on (lon,lat), clamped inside the raster."""
    xs, ys = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
    row, col = ds.index(xs[0], ys[0])
    col_off = int(round(col - win / 2))
    row_off = int(round(row - win / 2))
    col_off = max(0, min(col_off, ds.width - win))
    row_off = max(0, min(row_off, ds.height - win))
    return Window(col_off, row_off, win, win)


def read_reflectance(item: Any, aoi: AOI) -> tuple[np.ndarray, tuple[float, float]]:
    """Read the AOI window for all four bands and return harmonised reflectance ``(4, win, win)``
    plus the geographic ``(transform_a, origin)`` fingerprint used to assert co-registration."""
    offset = _boa_offset(item)
    bands: list[np.ndarray] = []
    grid_fp: tuple[float, float] | None = None
    win: Window | None = None
    for band in BANDS:
        href = item.assets[band].href  # already pc-signed via sign_inplace modifier
        with rasterio.open(href) as ds:
            if win is None:
                win = _window_from_center(ds, aoi.lon, aoi.lat, aoi.window)
                grid_fp = (ds.transform.a, ds.transform.c + ds.transform.f)
            dn = ds.read(1, window=win).astype(np.float32)
        refl = np.clip((dn - offset) / 10000.0, 0.0, None)
        bands.append(refl)
    assert grid_fp is not None
    return np.stack(bands, axis=0), grid_fp


def stretch_rgb(
    before_rgb: np.ndarray, after_rgb: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0
) -> tuple[np.ndarray, np.ndarray]:
    """Joint per-channel percentile stretch of two ``(3, H, W)`` reflectance stacks → uint8 HxWx3.

    Shared bounds (computed over both dates) keep before/after radiometrically comparable, so the
    swipe slider shows real change, not a stretch artefact."""

    def to_uint8(rgb: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        out = np.empty(rgb.shape, dtype=np.float32)
        for c in range(3):
            span = max(float(hi[c] - lo[c]), 1e-6)
            out[c] = np.clip((rgb[c] - lo[c]) / span, 0.0, 1.0)
        return (out * 255.0).round().astype(np.uint8).transpose(1, 2, 0)

    both = np.concatenate([before_rgb, after_rgb], axis=1)  # (3, 2H, W)
    lo = np.array([np.percentile(both[c], lo_pct) for c in range(3)], dtype=np.float32)
    hi = np.array([np.percentile(both[c], hi_pct) for c in range(3)], dtype=np.float32)
    return to_uint8(before_rgb, lo, hi), to_uint8(after_rgb, lo, hi)


# -------------------------------------------------------------------------------------------------
# Bake
# -------------------------------------------------------------------------------------------------
def _overlay_data_url_to_png(data_url: str, path: Path) -> None:
    """Write the base64 PNG produced by ``_overlay_png`` to ``path``."""
    _, _, b64 = data_url.partition(",")
    path.write_bytes(base64.b64decode(b64))


def bake_aoi(cat: pystac_client.Client, model: OscdModel, aoi: AOI) -> tuple[dict, dict]:
    """Fetch, run the model offline, write the cache artefacts. Returns ``(manifest, pred)``."""
    item_b, day_b, fb_b = find_item(cat, aoi, aoi.date_before)
    item_a, day_a, fb_a = find_item(cat, aoi, aoi.date_after)
    cloud_b = float(item_b.properties.get("eo:cloud_cover", 0.0))
    cloud_a = float(item_a.properties.get("eo:cloud_cover", 0.0))
    fb = lambda flag, want, got: (  # noqa: E731
        f" (fallback from {want}: confirmed swath misses AOI)" if flag else ""
    )
    print(
        f"[{aoi.id}] before={day_b} {item_b.id} ({cloud_b:.2f}%)"
        f"{fb(fb_b, aoi.date_before, day_b)}\n"
        f"          after={day_a} {item_a.id} ({cloud_a:.2f}%){fb(fb_a, aoi.date_after, day_a)}"
    )

    refl_b, grid_b = read_reflectance(item_b, aoi)
    refl_a, grid_a = read_reflectance(item_a, aoi)
    # co-registration check: same MGRS tile => identical grid (pixel size + origin) within tolerance
    if not (abs(grid_b[0] - grid_a[0]) < 1e-6 and abs(grid_b[1] - grid_a[1]) < 1.0):
        raise RuntimeError(f"{aoi.id}: before/after grids differ ({grid_b} vs {grid_a})")

    before_rgb_u8, after_rgb_u8 = stretch_rgb(refl_b[:3], refl_a[:3])

    inp_b = model._standardize(refl_b)
    inp_a = model._standardize(refl_a)
    mask, prob, n_tiles, elapsed_ms = model.predict(inp_b, inp_a)

    h, w = mask.shape
    overlay_url = _overlay_png(mask, (w, h))

    out = OUT_DIR / aoi.id
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(before_rgb_u8, mode="RGB").save(out / "before.png")
    Image.fromarray(after_rgb_u8, mode="RGB").save(out / "after.png")
    _overlay_data_url_to_png(overlay_url, out / "overlay.png")

    changed_frac = float(mask.mean())
    mean_conf_changed = float(prob[mask].mean()) if mask.any() else 0.0
    stats = {
        "changed_fraction": changed_frac,
        "changed_percent": round(100.0 * changed_frac, 2),
        "mean_confidence_changed": round(mean_conf_changed, 4),
        "mean_confidence_overall": round(float(prob.mean()), 4),
        "changed_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
    }
    manifest = {
        "id": aoi.id,
        "title": aoi.title,
        "description": aoi.description,
        "source": "Sentinel-2 L2A · 10 m",
        "tile": aoi.tile,
        "center": [aoi.lon, aoi.lat],
        "width": w,
        "height": h,
        "date_before": day_b,
        "date_after": day_a,
        "cloud_before": round(cloud_b, 2),
        "cloud_after": round(cloud_a, 2),
        "scene_before": item_b.id,
        "scene_after": item_a.id,
    }
    prediction = {
        "overlay_png": overlay_url,
        "threshold": model.threshold,
        "is_placeholder": False,
        "stats": stats,
        "elapsed_ms": round(elapsed_ms, 1),
        "input_size": model.input_size,
        "n_tiles": n_tiles,
        "pair_id": aoi.id,
        "model_id": MODEL_ID,
    }
    print(
        f"    baked {w}×{h}px · {n_tiles} tiles · changed {stats['changed_percent']}% · "
        f"{elapsed_ms:.0f} ms"
    )
    return manifest, prediction


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake curated Sentinel-2 AOIs (build-time).")
    ap.add_argument("--only", default=None, help="bake a single AOI id (default: all)")
    ap.add_argument("--window", type=int, default=None, help="override crop side in px for --only")
    ap.add_argument("--bundle", default=str(BUNDLE_DIR), help="OSCD ONNX bundle dir")
    args = ap.parse_args()

    aois = AOIS if args.only is None else [a for a in AOIS if a.id == args.only]
    if not aois:
        raise SystemExit(f"no AOI matches --only {args.only!r}")
    if args.window is not None:
        for a in aois:
            a.window = args.window

    model = OscdModel.load(Path(args.bundle))
    cat = open_catalog()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # merge with any existing manifest/cache so `--only` bakes are incremental
    manifest_path = OUT_DIR / "manifest.json"
    cache_path = OUT_DIR / "_predictions.json"
    manifest_by_id: dict[str, dict] = {}
    if manifest_path.exists():
        for e in json.loads(manifest_path.read_text()).get("pairs", []):
            manifest_by_id[e["id"]] = e
    cache: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    for aoi in aois:
        # STAC/COG network reads are occasionally flaky (RemoteDisconnected) — retry the whole AOI.
        last: Exception | None = None
        for attempt in range(1, 5):
            try:
                manifest, prediction = bake_aoi(cat, model, aoi)
                manifest_by_id[aoi.id] = manifest
                cache[aoi.id] = prediction
                break
            except Exception as exc:  # noqa: BLE001 — build-time resilience over flaky network reads
                if isinstance(exc, RuntimeError) and "no covering" in str(exc):
                    raise  # a real coverage failure, not a transient network blip — surface it
                last = exc
                print(f"    [{aoi.id}] attempt {attempt}/4 failed ({exc}); retrying…")
                time.sleep(3 * attempt)
        else:
            raise RuntimeError(f"{aoi.id}: baking failed after retries") from last

    ordered = [manifest_by_id[a.id] for a in AOIS if a.id in manifest_by_id]
    manifest_path.write_text(json.dumps({"pairs": ordered}, indent=2))
    cache_path.write_text(json.dumps(cache))
    print(f"\nwrote {manifest_path.relative_to(_REPO)} ({len(ordered)} AOIs) + {cache_path.name}")


if __name__ == "__main__":
    main()
