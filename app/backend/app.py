"""FastAPI app for the curated change-detection Space (PRD §10).

Single container (HF Spaces Docker SDK): this process serves the built React/MapLibre frontend as
static files AND the inference API on port 7860. CPU-only ``onnxruntime``.

Endpoints
    GET  /api/health            liveness + what's loaded
    GET  /api/models            available model bundles (id, input size, threshold, ...)
    GET  /api/models/{id}/card  the bundle's metrics_card.md (markdown)
    GET  /api/curated           curated before/after pairs (id, title, size, ...)
    GET  /api/curated/{id}/{which}.png   the before/after image
    POST /api/predict           {pair_id, model_id} -> change overlay (PNG data URL) + stats
    GET  /api/sentinel2         curated Sentinel-2 AOIs (baked results only; no runtime inference)
    GET  /api/sentinel2/{id}/{which}.png   the before / after / overlay image

Config via env:
    BUNDLES_DIR   dir of exported model bundles     (default: app/backend/models)
    CURATED_DIR   dir of curated pairs + manifest    (default: app/backend/data/curated)
    SENTINEL2_DIR dir of baked Sentinel-2 AOIs       (default: app/backend/data/sentinel2)
    FRONTEND_DIST built React app to serve at /      (default: app/frontend/dist)
    HF_BUNDLE_REPO  optional HF model repo to pull bundles from at startup (id or url)
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from curated import CuratedRegistry
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from inference import BundleRegistry
from pydantic import BaseModel
from sentinel2 import Sentinel2Registry

_HERE = Path(__file__).resolve().parent
BUNDLES_DIR = Path(os.environ.get("BUNDLES_DIR", _HERE / "models"))
CURATED_DIR = Path(os.environ.get("CURATED_DIR", _HERE / "data" / "curated"))
SENTINEL2_DIR = Path(os.environ.get("SENTINEL2_DIR", _HERE / "data" / "sentinel2"))
FRONTEND_DIST = Path(os.environ.get("FRONTEND_DIST", _HERE.parent / "frontend" / "dist"))


def _maybe_pull_bundles() -> None:
    """If HF_BUNDLE_REPO is set and BUNDLES_DIR is empty, pull the bundles from the HF Model repo.

    This is the HF-Space networking path (PRD §3/§9): the Space pulls weights from a companion
    Model repo at startup. It is a no-op locally (repo unset) and never runs on Leonardo.
    """
    repo = os.environ.get("HF_BUNDLE_REPO", "").strip()
    if not repo or (BUNDLES_DIR.exists() and any(BUNDLES_DIR.iterdir())):
        return
    try:
        from huggingface_hub import snapshot_download

        BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, repo_type="model", local_dir=str(BUNDLES_DIR))
    except Exception as exc:  # noqa: BLE001 — startup pull is best-effort; app still serves UI
        print(f"[startup] bundle pull from {repo!r} failed: {exc}")


_maybe_pull_bundles()
models = BundleRegistry(BUNDLES_DIR)
curated = CuratedRegistry(CURATED_DIR)
sentinel2 = Sentinel2Registry(SENTINEL2_DIR)
_predict_cache: dict[tuple[str, str], dict[str, Any]] = {}
# Curated pairs + models are fixed, so predictions are deterministic. tile+stitch inference is a few
# seconds of CPU per scene (DINOv2), so we bake predictions to disk (the deploy plan): the file is
# loaded instantly at startup and any gaps are filled by a background prewarm.
_CACHE_FILE = CURATED_DIR / "_predictions.json"


def _load_prediction_cache() -> None:
    if not _CACHE_FILE.exists():
        return
    try:
        data = json.loads(_CACHE_FILE.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"[cache] could not read {_CACHE_FILE}: {exc}")
        return
    valid = set(curated.pairs)
    for key, result in data.items():
        pair_id, _, model_id = key.partition("||")
        if pair_id in valid and model_id in models.ids():
            _predict_cache[(pair_id, model_id)] = result


def _save_prediction_cache() -> None:
    try:
        data = {f"{p}||{m}": r for (p, m), r in _predict_cache.items()}
        _CACHE_FILE.write_text(json.dumps(data))
    except Exception as exc:  # noqa: BLE001
        print(f"[cache] could not write {_CACHE_FILE}: {exc}")


def _compute_prediction(pair_id: str, model_id: str) -> dict[str, Any]:
    """Cached tile+stitch prediction for a (pair, model) — populated on demand and by prewarm."""
    key = (pair_id, model_id)
    cached = _predict_cache.get(key)
    if cached is not None:
        return cached
    before, after = curated.get_pair(pair_id)
    result = models.predict(model_id, before, after)
    result.update({"pair_id": pair_id, "model_id": model_id})
    _predict_cache[key] = result
    return result


def _prewarm() -> None:
    """Fill any (pair, model) predictions missing from the on-disk cache, then persist them so the
    next startup is instant. On-demand requests still work while this runs."""
    changed = False
    for pair_id in list(curated.pairs):
        for model_id in models.ids():
            if (pair_id, model_id) in _predict_cache:
                continue
            try:
                _compute_prediction(pair_id, model_id)
                changed = True
            except Exception as exc:  # noqa: BLE001 — best-effort warm; on-demand path still serves
                print(f"[prewarm] {pair_id}/{model_id} failed: {exc}")
    if changed:
        _save_prediction_cache()


_load_prediction_cache()
threading.Thread(target=_prewarm, daemon=True).start()

app = FastAPI(title="Satellite Change Detection — curated demo", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # single-origin in prod; permissive so `vite dev` can call the API too
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    pair_id: str
    model_id: str


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models": models.ids(),
        "n_curated": len(curated.pairs),
        "n_sentinel2": len(sentinel2.aois),
        "bundles_dir": str(BUNDLES_DIR),
    }


@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    return models.summaries()


@app.get("/api/models/{model_id}/card", response_class=PlainTextResponse)
def model_card(model_id: str) -> str:
    try:
        return models.get(model_id).metrics_card or "_No metrics card in bundle._"
    except KeyError as exc:
        raise HTTPException(404, f"unknown model {model_id!r}") from exc


@app.get("/api/curated")
def list_curated() -> list[dict[str, Any]]:
    return curated.list()


@app.get("/api/curated/{pair_id}/{which}.png")
def curated_image(pair_id: str, which: str) -> FileResponse:
    try:
        path = curated.image_path(pair_id, which)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, f"no {which} image for pair {pair_id!r}") from exc
    return FileResponse(path, media_type="image/png")


@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    if req.model_id not in models.ids():
        raise HTTPException(404, f"unknown model {req.model_id!r}")
    if req.pair_id not in curated.pairs:
        raise HTTPException(404, f"unknown pair {req.pair_id!r}")
    return _compute_prediction(req.pair_id, req.model_id)


# --- Sentinel-2 (Track B): curated AOIs served ENTIRELY from the baked cache --------------------
# No runtime inference, no STAC, no GPU — the OSCD 4-band predictions were computed offline by
# build_sentinel2.py (aerial models do NOT transfer to 10 m, so this tab uses the Sentinel-2-native
# OSCD model only). The runtime never imports pystac/rasterio; it just serves PNGs + baked stats.
@app.get("/api/sentinel2")
def list_sentinel2() -> list[dict[str, Any]]:
    return sentinel2.list()


@app.get("/api/sentinel2/{aoi_id}/{which}.png")
def sentinel2_image(aoi_id: str, which: str) -> FileResponse:
    try:
        path = sentinel2.image_path(aoi_id, which)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, f"no {which} image for Sentinel-2 AOI {aoi_id!r}") from exc
    if not path.exists():
        raise HTTPException(404, f"no {which} image for Sentinel-2 AOI {aoi_id!r}")
    return FileResponse(path, media_type="image/png")


# --- static frontend (mounted LAST so the /api/* routes above take precedence) ---------------
# html=True serves index.html for "/" and correctly serves hashed JS/CSS + binary assets.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @app.get("/", response_class=HTMLResponse)
    def index_missing() -> str:
        return (
            "<h1>Frontend not built</h1><p>Run <code>npm ci &amp;&amp; npm run build</code> in "
            "<code>app/frontend</code>, or use the multi-stage Dockerfile.</p>"
        )
