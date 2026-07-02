# HANDOFF

Continuation notes for a fresh session with zero prior conversation context. Read this first;
then **[PRD.md](PRD.md) is the authoritative build spec** and `leonardo.md` is authoritative for
all Leonardo/HPC specifics.

- **Project:** a satellite **change-detection** system (high-res aerial + Sentinel-2) with a
  first-class evaluation harness and a deployed web demo. Full spec: **[PRD.md](PRD.md)**.
- **GitHub:** https://github.com/shrimpyjustin12/geospatial-change-detection (branch `main`).
- **Local, gitignored notes:** `DECISIONS.md` (resolved cluster placeholders + every decision) and
  `experiments/LOG.md` (run trail). Keep both current. Injected memory also carries key facts.

## Milestone status
- **M0 — done.** Repo skeleton, CI, LEVIR-CD staged (637 pairs), container definition drafted.
- **M1 — done.** FC-Siam-diff baseline, 4-GPU DDP on Leonardo. **LEVIR-CD test change-class
  F1 0.884 / IoU 0.793** (val best F1 0.887). CI green.
- **M2 — next.** Strong model + full eval harness (scope below).

## Environment facts — do NOT rediscover the hard way
- **Do NOT build a Singularity/Apptainer container on Leonardo.** It fails twice over: login-node
  SIGKILL during `mksquashfs`, then `mksquashfs` errors on Lustre xattrs (`Unrecognised xattr
  prefix lustre.lov`) even with `$SCRATCH` as tmpdir. **Use the venv method** (documented in
  `leonardo.md`): `.venv-train` under `$WORK/sat-change-detection` (torch **2.5.1+cu121**, torchgeo,
  tensorboard), launched via `srun`. `.venv-stage` (CPU torch) exists for staging + budget-free CPU
  smokes. `container/changedet.def` is retained only as a portable artifact — do not retry building it.
- **Cluster:** partition `boost_usr_prod` (4×A100-64GB/node → DDP = 4 tasks / 4 GPUs); smokes on
  `boost_qos_dbg` (30-min cap, fast schedule); budget-free CPU on `lrd_all_serial`. Modules
  `python/3.11.7`, `cuda/12.2` (default) → torch **cu121** wheels. SSH host alias: `leonardo`.
- **Allocation account + absolute `$WORK`/`$SCRATCH` paths live in the local (gitignored)
  `DECISIONS.md` and in injected memory — they are NEVER committed.** Submit jobs with
  `sbatch --account=<allocation> slurm/<file>`. Never commit account IDs, absolute paths,
  usernames, or the user's other project names.
- **Login nodes SIGKILL compute** — even a CPU smoke must run on a compute node (`lrd_all_serial`)
  or via SLURM, never on the login node.
- **Data** is staged at `$WORK/sat-change-detection/data/levircd`. Compute nodes have **no internet
  egress** — stage datasets/weights on the login node (HF token already at `~/.hf_token`).

## Decisions that contradict a naive PRD read — do NOT revert
- **4-GPU DDP is the intentional default** (user override of PRD §6.4 "single-GPU is sufficient"),
  because Leonardo bills the whole node regardless of GPUs used. `train.py` scales the base LR by
  `effective_batch / reference_batch` (×4 on one node) and adds a short warmup. Keep DDP-default.
- **Flat `{A,B,label}/{split}*.png` data layout is authoritative over PRD §5.4's nested sketch** —
  it is what torchgeo's `LEVIRCD` expects, and M0 acceptance was "loadable via torchgeo."
- **Commits use the user's git identity (already set in this repo's `git config`; see `git log`)
  with NO Claude attribution** — never add a `Co-Authored-By: Claude` trailer, and don't add Claude
  as a repo collaborator. Push over SSH (remote + key already configured).

## File map (entry points)
- `src/models/fc_siam_diff.py` — FC-Siam-diff (weight-shared Siamese U-Net, `diff|concat` fusion).
  `src/models/__init__.py` exposes `build_model(cfg["model"])`.
- `src/losses.py` — `BceDiceLoss`, `dice_loss`. `src/metrics.py` — `ChangeMetrics` + `prf1_iou`
  (change-class P/R/F1/IoU; count→metric math is torch-free and unit-tested in CI).
- `src/data/levircd.py` — `TiledLEVIRCD` (torchgeo LEVIR-CD → 256² tiles, ImageNet-normalize,
  synced geometric aug; binarizes mask `{0,255}→{0,1}`). `src/data/tiling.py` — pure tile geometry.
- `src/dist.py` — DDP helpers (NCCL init from SLURM/torchrun env; single-process fallback).
- `src/train.py` — **`python -m src.train --config <cfg> [--resume-if-exists] [--set k=v ...]`**.
  DDP auto-engages when world>1; bf16 AMP; rank-correct checkpoint (save rank0 / load all ranks);
  `DistributedSampler.set_epoch`; LR scaling + warmup; `run_manifest.json` (git SHA, eff-batch, LR);
  optional TensorBoard; SIGUSR1/SIGTERM checkpoint. Smoke controls: `max_steps`, `limit_batches`.
- `src/evaluate.py` — **`python -m src.evaluate --config <cfg> --split test`**. Change-class
  P/R/F1/IoU at a threshold; writes `results/<run_id>/eval_<split>.json`. Minimal — full harness is M2.
- `src/config.py` — YAML load + dotted `key=value` overrides + `${VAR}` expansion.
- `configs/` — `levircd_baseline.yaml` (+ `levircd_baseline_smoke.yaml`). One yaml per model + a smoke.
- `slurm/` — `train.sbatch` (full 4-GPU DDP, USR1→requeue, `--resume-if-exists`),
  `smoke.sbatch` (debug-QoS 4-GPU DDP), `smoke_cpu.sbatch` (serial CPU). Submit with
  `sbatch --account=<allocation> slurm/<file>`.
- `scripts/` — `stage_data.sh` (LEVIR-CD download + md5/sha256), `stage_weights.sh` (shared HF
  cache), `smoke_load_levircd.py`. CI: `.github/workflows/ci.yml` (ruff + mypy + pytest, torch-cpu).

## M2 — scope + first step
- **Strong model:** Siamese-SegFormer — pretrained MiT (SegFormer) encoder from `smp`/`timm`,
  weight-shared, difference fusion, light MLP/U-Net decoder. Target LEVIR-CD change-F1 ≈ 0.90.
- **Full eval harness** (grow `src/evaluate.py`): PR-curve **threshold selection** (justify the
  operating point, not a fixed 0.5), **per-scene breakdown** (mean±std), auto-generated
  **failure-case gallery** (before/after/pred/GT, cause-bucketed), **≥1 ablation** (e.g. diff vs
  concat fusion), and the 3-tier comparison table on the identical LEVIR-CD test split.
- **First action:** stage the pretrained **MiT / timm** weights on the **login node** into the
  shared HF cache via `scripts/stage_weights.sh` (HF token present; compute nodes have no egress).
  Then smoke the SegFormer config on `boost_qos_dbg` before any full run.

## Working conventions
- **Smoke before full** — run the smoke config (CPU serial or `boost_qos_dbg`) before any full
  submission. **PAUSE and ask the user before the first full multi-GPU submission.**
- Checkpoint every ~30 min + `--resume-if-exists` so a walltime cut never loses progress.
- Commit granularly (user identity, no Claude attribution); keep `DECISIONS.md` +
  `experiments/LOG.md` current; confirm CI is green after each push.
- **When blocked, or when an assumption is load-bearing, ask the user** — they relay questions to a
  senior reviewer and return answers. Don't guess on irreversible or costly choices.
