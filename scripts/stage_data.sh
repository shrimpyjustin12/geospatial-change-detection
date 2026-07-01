#!/usr/bin/env bash
# Stage datasets on a LOGIN NODE (has egress). Compute nodes have NO internet (leonardo.md),
# so all downloads happen here and are read from shared storage by training jobs.
#
# Downloads -> verifies md5 (vs the torchgeo-pinned upstream) -> verifies/records sha256
# (committed manifest, PRD §5.2) -> extracts into torchgeo's expected layout. Idempotent.
#
# Usage:
#   scripts/stage_data.sh                 # stage LEVIR-CD into $WORK/sat-change-detection/data
#   DATA_ROOT=/path/to/data scripts/stage_data.sh levircd
set -euo pipefail

DATASET="${1:-levircd}"
DATA_ROOT="${DATA_ROOT:-${WORK:?WORK not set — export WORK or pass DATA_ROOT}/sat-change-detection/data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKSUM_DIR="${SCRIPT_DIR}/checksums"
mkdir -p "$CHECKSUM_DIR"

log() { printf '[stage_data] %s\n' "$*"; }
die() { printf '[stage_data][ERROR] %s\n' "$*" >&2; exit 1; }

command -v curl  >/dev/null || die "curl not found"
command -v unzip >/dev/null || die "unzip not found"
command -v md5sum    >/dev/null || die "md5sum not found"
command -v sha256sum >/dev/null || die "sha256sum not found"

verify_md5() {  # <file> <expected_md5>
  local got; got="$(md5sum "$1" | awk '{print $1}')"
  [ "$got" = "$2" ] || die "md5 mismatch for $(basename "$1"): got=$got want=$2"
  log "md5 OK: $(basename "$1")"
}

record_or_verify_sha256() {  # <file> <manifest>
  local base got want line
  base="$(basename "$1")"
  got="$(sha256sum "$1" | awk '{print $1}')"
  if line="$(grep -E "  ${base}\$" "$2" 2>/dev/null)"; then
    want="$(echo "$line" | awk '{print $1}')"
    [ "$got" = "$want" ] || die "sha256 mismatch for $base (vs manifest): got=$got want=$want"
    log "sha256 OK (manifest): $base"
  else
    printf '%s  %s\n' "$got" "$base" >> "$2"
    log "sha256 recorded: $base -> $got"
  fi
}

download() {  # <url> <dest>
  if [ -f "$2" ]; then log "exists, skip download: $(basename "$2")"; return; fi
  log "downloading $(basename "$2") ..."
  curl -fL --retry 3 --retry-delay 5 -o "${2}.part" "$1"
  mv "${2}.part" "$2"
}

stage_levircd() {
  local root="${DATA_ROOT}/levircd"
  local arc="${root}/_archives"
  local manifest="${CHECKSUM_DIR}/levircd.sha256"
  mkdir -p "$arc"
  # torchgeo-pinned upstream: HuggingFace satellite-image-deep-learning/LEVIR-CD @ 6a6bb0a5
  local base="https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/6a6bb0a5b389403d81c05e33bf08bc0b9e5f13a6"
  local zips=(train.zip val.zip test.zip)
  local md5s=(a638e71f480628652dea78d8544307e4 f7b857978524f9aa8c3bf7f94e3047a4 07d5dd89e46f5c1359e2eca746989ed9)

  for i in "${!zips[@]}"; do
    download "${base}/${zips[$i]}" "${arc}/${zips[$i]}"
    verify_md5 "${arc}/${zips[$i]}" "${md5s[$i]}"
    record_or_verify_sha256 "${arc}/${zips[$i]}" "$manifest"
  done

  # Extract all splits into $root; torchgeo LEVIRCD globs {root}/{A,B,label}/{split}*.png .
  for z in "${zips[@]}"; do
    log "extracting ${z} ..."
    unzip -q -o "${arc}/${z}" -d "$root"
  done

  log "LEVIR-CD staged at: $root"
  log "top-level dirs:"; find "$root" -maxdepth 1 -mindepth 1 -type d | sort | sed 's/^/    /'
  for d in A B label; do
    if [ -d "${root}/${d}" ]; then
      log "  ${d}/: $(find "${root}/${d}" -name '*.png' | wc -l | tr -d ' ') png"
    fi
  done
}

case "$DATASET" in
  levircd) stage_levircd ;;
  *) die "unknown dataset '$DATASET' (supported: levircd)" ;;
esac
log "done: ${DATASET}"
