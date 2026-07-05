#!/bin/bash
# Single-GPU SMOKE on the debug QoS (fast to schedule, 30-min cap). For Track-B / OSCD, which trains
# SINGLE-GPU (ddp:false) — the 4-GPU DDP default hurts convergence on such a tiny dataset. Validates
# the real GPU path (bf16 AMP, eval, checkpoint) end-to-end before a full single-GPU run. The node is
# billed whole on Leonardo regardless, but correctness > utilisation for OSCD (24 pairs).
#   sbatch --account=<ALLOCATION> slurm/smoke_1gpu.sbatch [configs/oscd_s2_smoke.yaml]
#SBATCH --job-name=cd-smoke1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # single process, single GPU (no DDP)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

CONFIG="${1:-configs/oscd_s2_smoke.yaml}"
cd "$WORK/sat-change-detection"
mkdir -p logs

module load python/3.11.7
source .venv-train/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="$WORK/sat-change-detection/.cache/huggingface"
export GIT_SHA="$(cat REVISION 2>/dev/null || echo unknown)"

# Single process on the one allocated GPU (train.py detects world=1 -> no DDP). Extra args forwarded.
python -m src.train --config "$CONFIG" "${@:2}"
