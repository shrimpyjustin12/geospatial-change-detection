#!/usr/bin/env bash
# Pre-download pretrained / foundation-model weights on a LOGIN NODE into a shared HF cache.
# Compute nodes have NO egress (leonardo.md); training reads only from this cache with
# HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1. Exercised in M2 (MiT/SegFormer) and M3 (DINOv2).
# PRD §5.2 / §5.3: note any *gated* weights (license click + token) in DECISIONS.md.
set -euo pipefail

export HF_HOME="${HF_HOME:-${WORK:?set WORK or HF_HOME}/sat-change-detection/.cache/huggingface}"
mkdir -p "$HF_HOME"
if [ -f "$HOME/.hf_token" ]; then
  HF_TOKEN="$(cat "$HOME/.hf_token")"; export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

log() { printf '[stage_weights] %s\n' "$*"; }
log "HF_HOME=$HF_HOME"

# Weight list grows per milestone. Example (M3), once transformers/huggingface_hub is installed:
#   python - <<'PY'
#   from huggingface_hub import snapshot_download
#   snapshot_download("facebook/dinov2-base")   # ungated at time of writing — confirm
#   PY
log "M0: baseline FC-Siam-diff uses no pretrained weights. Populated in M2 (MiT) / M3 (DINOv2)."
