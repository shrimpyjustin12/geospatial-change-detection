# Curated change-detection Space — single container, FastAPI serves the React/MapLibre build +
# the inference API on :7860 (PRD §10). Multi-stage: node builds the static frontend, then a slim
# python runtime serves it alongside onnxruntime (CPU) inference.

# ---- stage 1: build the frontend ------------------------------------------------------------
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

# ---- stage 2: python runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    BUNDLES_DIR=/app/models \
    CURATED_DIR=/app/data/curated \
    SENTINEL2_DIR=/app/data/sentinel2 \
    FRONTEND_DIST=/app/frontend_dist

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./
COPY --from=frontend /frontend/dist ./frontend_dist

# Model bundles: either baked into ./models at build time, or pulled at startup from a companion
# HF Model repo by setting HF_BUNDLE_REPO (PRD §3/§9 — the Space pulls weights, never trains).
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
