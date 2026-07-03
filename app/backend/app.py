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

Config via env:
    BUNDLES_DIR   dir of exported model bundles     (default: app/backend/models)
    CURATED_DIR   dir of curated pairs + manifest    (default: app/backend/data/curated)
    FRONTEND_DIST built React app to serve at /      (default: app/frontend/dist)
    HF_BUNDLE_REPO  optional HF model repo to pull bundles from at startup (id or url)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from curated import CuratedRegistry
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from inference import BundleRegistry
from pydantic import BaseModel

_HERE = Path(__file__).resolve().parent
BUNDLES_DIR = Path(os.environ.get("BUNDLES_DIR", _HERE / "models"))
CURATED_DIR = Path(os.environ.get("CURATED_DIR", _HERE / "data" / "curated"))
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
# curated pairs + models are fixed, so predictions are deterministic — cache them so switching back
# to a scene is instant (real DINOv2-base is ~seconds per tile on CPU).
_predict_cache: dict[tuple[str, str], dict[str, Any]] = {}

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
    key = (req.pair_id, req.model_id)
    if key in _predict_cache:
        return _predict_cache[key]
    try:
        before, after = curated.get_pair(req.pair_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown pair {req.pair_id!r}") from exc
    result = models.predict(req.model_id, before, after)
    result.update({"pair_id": req.pair_id, "model_id": req.model_id})
    _predict_cache[key] = result
    return result


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
