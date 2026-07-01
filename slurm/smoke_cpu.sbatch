#!/bin/bash
# Budget-free CPU logic-smoke on the serial partition (no GPU-hours). Validates the training
# loop / checkpoint / eval plumbing cheaply. Login nodes SIGKILL compute, so run it here.
#   sbatch --account=<ALLOCATION> slurm/smoke_cpu.sbatch
#SBATCH --job-name=cd-cpusmoke
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "$WORK/sat-change-detection"
mkdir -p logs

module load python/3.11.7
source .venv-stage/bin/activate  # CPU torch is fine here
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export GIT_SHA="$(cat REVISION 2>/dev/null || echo unknown)"

python -m src.train --config configs/levircd_baseline_smoke.yaml
