---
title: Satellite Change Detection (Curated)
emoji: 🛰️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Satellite Change Detection — curated demo (Track A)

A single-container [Hugging Face Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker):
**FastAPI** (uvicorn, `:7860`) serves a **React + MapLibre GL** build and a CPU **onnxruntime**
inference API. This is the **curated** before/after mode (PRD §10.1) — the always-fast demo path;
the live Sentinel-2 AOI mode is a later milestone.

## What it does

- Pick a curated before/after scene pair and a model bundle.
- The backend runs the pair through the exported **ONNX** change-detection model (CPU) and returns a
  change-mask overlay + stats (% area changed, mean confidence, inference time).
- A MapLibre **swipe slider** compares before vs after; the detected change is a colored overlay with
  an **opacity** control. A **model-card** page carries the real LEVIR-CD results and limitations.

## Model bundles (the contract)

The app consumes only **artifact bundles** produced by `src/export.py` (PRD §3/§9), never the
training code. Each bundle is a directory with:

```
model.onnx           # exported graph (parity-checked against PyTorch)
preprocessing.json   # normalization, input size, band order, tiling, recommended threshold
config.yaml          # provenance
metrics_card.md      # headline metrics
parity.json          # recorded PyTorch↔ONNXRuntime parity
```

Bundles are either baked into `./models` at build time or pulled at startup from a companion HF
**Model repo** by setting the `HF_BUNDLE_REPO` env var — this keeps the Space lean and separates
weights from the app.

## Run locally

```bash
# 1. export at least one bundle (from the repo root, with the train env):
python -m src.export --config configs/levircd_segformer.yaml   # or --random-init to smoke it
cp -r bundles/* app/backend/models/

# 2. synthesize curated pairs (or drop real LEVIR-CD tiles into app/backend/data/curated/):
python app/backend/gen_sample_pairs.py --out app/backend/data/curated

# 3. build the frontend + run the API (serves the build at http://localhost:7860):
cd app/frontend && npm ci && npm run build && cd ..
BUNDLES_DIR=backend/models CURATED_DIR=backend/data/curated FRONTEND_DIST=frontend/dist \
  uvicorn backend.app:app --host 0.0.0.0 --port 7860
```

Or build the whole thing with Docker: `docker build -t sat-cd app/ && docker run -p 7860:7860 sat-cd`.

## Notes / honesty

- Change-class metrics only — overall pixel accuracy is meaningless when change is a tiny pixel
  fraction (see the model card).
- Trained weights inherit LEVIR-CD research/non-commercial terms — showcase use only.
