#!/bin/bash
# Budget-free CPU logic-smoke on the serial partition (no GPU-hours). Validates the training
# loop / offline weight load / checkpoint / eval plumbing cheaply. Login nodes SIGKILL compute,
# so run it here. Optional config arg (default: baseline smoke).
#   sbatch --account=<ALLOCATION> slurm/smoke_cpu.sbatch [configs/levircd_segformer_smoke.yaml]
#SBATCH --job-name=cd-cpusmoke
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

CONFIG="${1:-configs/levircd_baseline_smoke.yaml}"
cd "$WORK/sat-change-detection"
mkdir -p logs

module load python/3.11.7
source .venv-train/bin/activate   # cu121 torch runs on CPU here; also provides smp for SegFormer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="$WORK/sat-change-detection/.cache/huggingface"   # staged MiT/FM weights (offline)
export GIT_SHA="$(cat REVISION 2>/dev/null || echo unknown)"

python -m src.train --config "$CONFIG"
