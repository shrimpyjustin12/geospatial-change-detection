#!/usr/bin/env bash
# Stage datasets on a LOGIN NODE (has egress). Compute nodes have NO internet (leonardo.md),
# so all downloads happen here and are read from shared storage by training jobs.
#
# Downloads -> verifies md5 (vs the torchgeo-pinned upstream) -> verifies/records sha256
# (committed manifest, PRD §5.2) -> extracts into torchgeo's expected layout. Idempotent.
#
# Prefers the HuggingFace CLI with hf_transfer (fast, parallel, uses ~/.hf_token) and falls
# back to curl. Run inside the staging venv so `hf` is on PATH:
#   source .venv-stage/bin/activate
#   DATA_ROOT=$WORK/sat-change-detection/data scripts/stage_data.sh levircd
set -euo pipefail

DATASET="${1:-levircd}"
DATA_ROOT="${DATA_ROOT:-${WORK:?WORK not set — export WORK or pass DATA_ROOT}/sat-change-detection/data}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKSUM_DIR="${SCRIPT_DIR}/checksums"
mkdir -p "$CHECKSUM_DIR"

log() { printf '[stage_data] %s\n' "$*"; }
die() { printf '[stage_data][ERROR] %s\n' "$*" >&2; exit 1; }

command -v unzip     >/dev/null || die "unzip not found"
command -v md5sum    >/dev/null || die "md5sum not found"
command -v sha256sum >/dev/null || die "sha256sum not found"

HF_CLI="$(command -v hf || command -v huggingface-cli || true)"
[ -n "$HF_CLI" ] || command -v curl >/dev/null || die "need either the 'hf' CLI or curl"
if [ -f "$HOME/.hf_token" ]; then
  HF_TOKEN="$(cat "$HOME/.hf_token")"; export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

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

fetch_all() {  # <repo> <rev> <dir> <file...>
  local repo="$1" rev="$2" dir="$3"; shift 3
  local files=("$@") missing=() f
  mkdir -p "$dir"
  for f in "${files[@]}"; do [ -f "${dir}/${f}" ] || missing+=("$f"); done
  if [ ${#missing[@]} -eq 0 ]; then log "all archives present; skip download"; return; fi
  if [ -n "$HF_CLI" ]; then
    log "downloading via hf CLI (Xet high-performance transfer): ${missing[*]}"
    "$HF_CLI" download "$repo" "${missing[@]}" \
      --repo-type dataset --revision "$rev" --local-dir "$dir"
  else
    for f in "${missing[@]}"; do
      log "downloading $f via curl ..."
      curl -fL --retry 3 --retry-delay 5 -o "${dir}/${f}.part" \
        "https://huggingface.co/datasets/${repo}/resolve/${rev}/${f}"
      mv "${dir}/${f}.part" "${dir}/${f}"
    done
  fi
}

stage_levircd() {
  local root="${DATA_ROOT}/levircd"
  local arc="${root}/_archives"
  local manifest="${CHECKSUM_DIR}/levircd.sha256"
  # torchgeo-pinned upstream: HuggingFace satellite-image-deep-learning/LEVIR-CD @ 6a6bb0a5
  local repo="satellite-image-deep-learning/LEVIR-CD"
  local rev="6a6bb0a5b389403d81c05e33bf08bc0b9e5f13a6"
  local zips=(train.zip val.zip test.zip)
  local md5s=(a638e71f480628652dea78d8544307e4 f7b857978524f9aa8c3bf7f94e3047a4 07d5dd89e46f5c1359e2eca746989ed9)

  fetch_all "$repo" "$rev" "$arc" "${zips[@]}"
  for i in "${!zips[@]}"; do
    verify_md5 "${arc}/${zips[$i]}" "${md5s[$i]}"
    record_or_verify_sha256 "${arc}/${zips[$i]}" "$manifest"
  done

  # Extract all splits into $root; torchgeo LEVIRCD globs {root}/{A,B,label}/{split}*.png .
  for z in "${zips[@]}"; do
    log "extracting ${z} ..."
    unzip -q -o "${arc}/${z}" -d "$root"
  done
  rm -rf "${arc}/.cache"  # hf CLI metadata

  log "LEVIR-CD staged at: $root"
  log "top-level dirs:"; find "$root" -maxdepth 1 -mindepth 1 -type d | sort | sed 's/^/    /'
  for d in A B label; do
    if [ -d "${root}/${d}" ]; then
      log "  ${d}/: $(find "${root}/${d}" -name '*.png' | wc -l | tr -d ' ') png"
    fi
  done
}

# Canonical OSCD split (Daudt et al. / torchgeo): 14 train cities, 10 test cities. Used as a
# fallback if the downloaded archive does not ship train.txt/test.txt at the dataset root.
OSCD_CANON_TRAIN="aguasclaras,bercy,bordeaux,nantes,paris,rennes,saclay_e,abudhabi,cupertino,pisa,beihai,hongkong,beirut,mumbai"
OSCD_CANON_TEST="brasilia,chongqing,dubai,lasvegas,milano,montpellier,norcia,rio,saclay_w,valencia"

# OSCD source: taken from torchgeo 0.8.1's OSCD dataset (the Onera partage.imt.fr distribution).
# Three Nextcloud archives (not on HuggingFace) — we curl them directly and verify md5.
# NOTE: torchgeo lists the Train Labels URL on partage.mines-telecom.fr, whose TLS cert does not
# match its hostname; the same share token is served (valid cert) from partage.imt.fr, so we use
# that host for all three and keep TLS verification on.
OSCD_IMAGES_URL="https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download"
OSCD_TRAIN_URL="https://partage.imt.fr/index.php/s/2D6n03k58ygBSpu/download"
OSCD_TEST_URL="https://partage.imt.fr/index.php/s/gpStKn4Mpgfnr63/download"
OSCD_IMAGES_MD5="c50d4a2941da64e03a47ac4dec63d915"
OSCD_TRAIN_MD5="4d2965af8170c705ebad3d6ee71b6990"
OSCD_TEST_MD5="8177d437793c522653c442aa4e66c617"
OSCD_IMAGES_WRAP="Onera Satellite Change Detection dataset - Images"
OSCD_TRAIN_WRAP="Onera Satellite Change Detection dataset - Train Labels"
OSCD_TEST_WRAP="Onera Satellite Change Detection dataset - Test Labels"

curl_fetch() {  # <url> <dest>
  [ -f "$2" ] && { log "present: $(basename "$2")"; return; }
  log "downloading $(basename "$2") ..."
  # -sS silences the progress meter but keeps errors; connect-timeout avoids indefinite hangs.
  curl -fL -sS --connect-timeout 30 --retry 3 --retry-delay 5 -o "${2}.part" "$1"
  mv "${2}.part" "$2"
}

stage_oscd() {
  # OSCD (Onera Satellite Change Detection) — Sentinel-2, 24 pairs (14 train / 10 test), 13 bands.
  # curl the 3 archives -> md5 verify -> sha256 record -> extract -> normalize the wrapper layout
  # into <root>/<city>/{imgs_*_rect, dates.txt, cm/cm.png} (what TiledOSCD reads) -> derive
  # train.txt/test.txt from the Train/Test Labels folders (authoritative split; canonical fallback).
  command -v curl >/dev/null || die "curl not found (needed for OSCD Nextcloud archives)"
  local root="${DATA_ROOT}/oscd"
  local arc="${root}/_archives"
  local raw="${root}/_raw"
  local manifest="${CHECKSUM_DIR}/oscd.sha256"
  mkdir -p "$arc"

  local names=("Images.zip" "Train Labels.zip" "Test Labels.zip")
  local urls=("$OSCD_IMAGES_URL" "$OSCD_TRAIN_URL" "$OSCD_TEST_URL")
  local md5s=("$OSCD_IMAGES_MD5" "$OSCD_TRAIN_MD5" "$OSCD_TEST_MD5")
  local i
  for i in "${!names[@]}"; do
    curl_fetch "${urls[$i]}" "${arc}/${names[$i]}"
    verify_md5 "${arc}/${names[$i]}" "${md5s[$i]}"
    record_or_verify_sha256 "${arc}/${names[$i]}" "$manifest"
  done

  rm -rf "$raw"; mkdir -p "$raw"
  for i in "${!names[@]}"; do
    log "extracting ${names[$i]} ..."
    unzip -q -o "${arc}/${names[$i]}" -d "$raw"
  done

  # Authoritative split = the region folders under each Labels wrapper (canonical fallback).
  local train_csv test_csv
  train_csv=$(find "$raw/$OSCD_TRAIN_WRAP" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; 2>/dev/null | sort | paste -sd, -)
  test_csv=$(find "$raw/$OSCD_TEST_WRAP" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; 2>/dev/null | sort | paste -sd, -)
  [ -n "$train_csv" ] || train_csv="$OSCD_CANON_TRAIN"
  [ -n "$test_csv" ]  || test_csv="$OSCD_CANON_TEST"

  # Normalize: images for all regions, then merge each region's cm/ from the Labels wrappers.
  local regdir reg wrap
  for regdir in "$raw/$OSCD_IMAGES_WRAP"/*/; do
    [ -d "$regdir" ] || continue
    reg=$(basename "$regdir")
    mkdir -p "$root/$reg"
    mv "$regdir"/imgs_* "$regdir"/dates.txt "$root/$reg/" 2>/dev/null || true
  done
  for wrap in "$OSCD_TRAIN_WRAP" "$OSCD_TEST_WRAP"; do
    for regdir in "$raw/$wrap"/*/; do
      [ -d "$regdir" ] || continue
      reg=$(basename "$regdir")
      mkdir -p "$root/$reg"
      mv "$regdir"/cm "$root/$reg/" 2>/dev/null || true
    done
  done
  rm -rf "$raw"

  printf '%s\n' "$train_csv" > "${root}/train.txt"
  printf '%s\n' "$test_csv"  > "${root}/test.txt"
  log "split: $(echo "$train_csv" | tr ',' '\n' | grep -c .) train / $(echo "$test_csv" | tr ',' '\n' | grep -c .) test cities"

  log "OSCD staged at: $root"
  local ncities; ncities=$(find "$root" -maxdepth 2 -type d -name 'imgs_*' 2>/dev/null | sed 's#/imgs_[^/]*$##' | sort -u | wc -l | tr -d ' ')
  log "cities with imagery (imgs_*): ${ncities}"
  local sample; sample=$(find "$root" -maxdepth 2 -type d -name 'imgs_1_rect' 2>/dev/null | head -1)
  if [ -n "$sample" ]; then
    log "sample city '$(basename "$(dirname "$sample")")': $(find "$sample" -name '*.tif' | wc -l | tr -d ' ') band tifs; cm=[$(ls "$(dirname "$sample")/cm" 2>/dev/null | tr '\n' ' ')]"
  fi
}

case "$DATASET" in
  levircd) stage_levircd ;;
  oscd)    stage_oscd ;;
  *) die "unknown dataset '$DATASET' (supported: levircd, oscd)" ;;
esac
log "done: ${DATASET}"
