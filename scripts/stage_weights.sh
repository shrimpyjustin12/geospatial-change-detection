#!/usr/bin/env bash
# Pre-download pretrained / foundation-model weights on a LOGIN NODE into a shared HF cache.
# Compute nodes have NO egress (leonardo.md); training reads only from this cache with
# HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1. M2 stages the SegFormer MiT encoders; M3 adds DINOv2.
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

# smp's MiT encoders load ImageNet weights from HF repos (smp-hub/<encoder>.imagenet) AT A PINNED
# REVISION. We read repo_id + revision from the installed smp so the cached commit is exactly the
# one get_encoder(weights="imagenet") requests — otherwise offline load misses the cache and smp
# silently falls back to a network URL (which fails on no-egress compute nodes). mit_b0 -> smoke,
# mit_b2 -> full strong model. Both ungated at time of writing.
python - <<'PY'
import os
from huggingface_hub import snapshot_download
from segmentation_models_pytorch.encoders import encoders

TARGETS = ["mit_b0", "mit_b2"]
for name in TARGETS:
    ps = encoders[name]["pretrained_settings"]["imagenet"]
    repo_id, revision = ps["repo_id"], ps.get("revision")
    path = snapshot_download(repo_id=repo_id, revision=revision)
    print(f"[stage_weights] cached {repo_id}@{revision} -> {path}")
print(f"[stage_weights] HF_HOME={os.environ.get('HF_HOME')}")
PY

# DINOv2 foundation-model encoders (M3, PRD §6.1). facebook/dinov2-{small,base} are UNGATED (no
# license click) at time of writing. Unlike smp, transformers' from_pretrained HONORS
# HF_HUB_OFFLINE and ERRORS on a cache miss (no silent network fallback) -- so the offline risk is
# a missing/incomplete snapshot, not the smp silent-fallback trap. We stage the exact 'main' commit
# (recorded below) and then verify an offline load in a fresh process so a cache miss fails HERE
# (login node, egress) rather than on a no-egress GPU node.
DINOV2_MODELS="${DINOV2_MODELS:-facebook/dinov2-small facebook/dinov2-base}"
log "staging DINOv2: $DINOV2_MODELS"
python - "$DINOV2_MODELS" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

for repo in sys.argv[1].split():
    path = snapshot_download(repo_id=repo, revision="main")
    commit = Path(path).name  # .../snapshots/<commit_sha>
    print(f"[stage_weights] cached {repo}@{commit} -> {path}")
PY

# offline-load verification: no network, cache-only. Success == the snapshot is complete.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - "$DINOV2_MODELS" <<'PY'
import sys
from transformers import Dinov2Model

for repo in sys.argv[1].split():
    m = Dinov2Model.from_pretrained(repo)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    c = m.config
    print(
        f"[stage_weights] OFFLINE load OK {repo}: {n:.1f}M params, "
        f"hidden={c.hidden_size} layers={c.num_hidden_layers} patch={c.patch_size}"
    )
print("[stage_weights] DINOv2 offline verification passed (no network fallback)")
PY

log "done. Record the staged DINOv2 commits in DECISIONS.md (re-run if transformers is upgraded)."
