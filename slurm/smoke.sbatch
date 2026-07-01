#!/bin/bash
# 4-GPU DDP SMOKE on the debug QoS (fast to schedule, 30-min cap). Validates the real DDP path
# (rank-correct ckpt, DistributedSampler, all-reduce, bf16 AMP) end-to-end before a full run.
#   sbatch --account=<ALLOCATION> slurm/smoke.sbatch
#SBATCH --job-name=cd-smoke
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

cd "$WORK/sat-change-detection"
mkdir -p logs

module load python/3.11.7
source .venv-train/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export GIT_SHA="$(cat REVISION 2>/dev/null || echo unknown)"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

srun python -m src.train --config configs/levircd_baseline_smoke.yaml
