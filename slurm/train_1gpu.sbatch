#!/bin/bash
# Full SINGLE-GPU training on Leonardo Booster for Track-B / OSCD. Single-GPU is intentional (the
# revised M5 scope): OSCD is tiny (24 pairs) and the 4-GPU DDP default's large effective batch hurts
# convergence. The node is billed whole regardless, but correctness wins here. Allocation is supplied
# at submit time so it never lives in the repo. Extra args after the config are forwarded to src.train.
#   sbatch --account=<ALLOCATION> slurm/train_1gpu.sbatch [configs/oscd_s2.yaml]
#SBATCH --job-name=cd-train1
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # single process, single GPU (no DDP)
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120          # USR1 to batch script 120s before walltime
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

CONFIG="${1:-configs/oscd_s2.yaml}"
cd "$WORK/sat-change-detection"
mkdir -p logs

module load python/3.11.7
source .venv-train/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME="$WORK/sat-change-detection/.cache/huggingface"
export GIT_SHA="$(cat REVISION 2>/dev/null || echo unknown)"

# On USR1 (near walltime): requeue. train.py's SIGTERM handler + periodic checkpoints (ckpt_every_min)
# mean --resume-if-exists picks up where it left off. OSCD is small and normally finishes in one window.
_on_usr1() { echo "[sbatch] USR1 received; requeuing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"; }
trap _on_usr1 USR1

python -m src.train --config "$CONFIG" --resume-if-exists "${@:2}" &
wait $!
