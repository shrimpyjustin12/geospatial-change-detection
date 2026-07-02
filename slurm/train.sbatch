#!/bin/bash
# Full 4-GPU DDP training on Leonardo Booster (one rank per GPU; billed per node, so use all 4).
# The allocation is supplied at submit time so it never lives in the repo. Extra args after the
# config are forwarded to src.train (e.g. --set for the fusion ablation):
#   sbatch --account=<ALLOCATION> slurm/train.sbatch [configs/levircd_baseline.yaml]
#   sbatch --account=<ALLOCATION> slurm/train.sbatch configs/levircd_segformer.yaml \
#          --set run_id=levircd_segformer_concat model.fusion=concat
#SBATCH --job-name=cd-train
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task == one rank == one GPU
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120          # USR1 to batch script 120s before walltime
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

CONFIG="${1:-configs/levircd_baseline.yaml}"
cd "$WORK/sat-change-detection"
mkdir -p logs

module load python/3.11.7
source .venv-train/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="$WORK/sat-change-detection/.cache/huggingface"   # staged MiT/FM weights (offline)
export GIT_SHA="$(cat REVISION 2>/dev/null || echo unknown)"

# DDP rendezvous (single node)
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))

# On USR1 (near walltime): requeue. Python's SIGTERM handler + periodic checkpoints (every
# ckpt_every_min) mean --resume-if-exists picks up where we left off.
_on_usr1() { echo "[sbatch] USR1 received; requeuing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap _on_usr1 USR1

srun python -m src.train --config "$CONFIG" --resume-if-exists "${@:2}" &
wait $!
