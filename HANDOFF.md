# HANDOFF

Continuation notes for a fresh session with zero prior conversation context. Read this first;
then **[PRD.md](PRD.md) is the authoritative build spec** and `leonardo.md` (in the repo's parent
dir) is authoritative for all Leonardo/HPC specifics.

- **Project:** a satellite **change-detection** system (high-res aerial + Sentinel-2) with a
  first-class evaluation harness and a deployed web demo. Full spec: **[PRD.md](PRD.md)**.
- **GitHub:** https://github.com/shrimpyjustin12/geospatial-change-detection (branch `main`).
- **Local, gitignored notes:** `DECISIONS.md` (resolved cluster placeholders + every decision) and
  `experiments/LOG.md` (run trail). Keep both current. Injected memory also carries key facts.

## Milestone status
- **M0 — done.** Repo skeleton, CI, LEVIR-CD staged (637 pairs), container def drafted.
- **M1 — done.** FC-Siam-diff baseline, 4-GPU DDP. LEVIR-CD test **F1 0.884** (fixed thr 0.5).
- **M2 — done.** Siamese-SegFormer (smp **MiT-b2**, ImageNet) + full eval harness. Tier comparison
  on LEVIR-CD **test** (threshold selected on **val**, applied to test — never tuned on test):

  | Model | Trainable params | F1 | IoU | AP |
  |---|---|---|---|---|
  | FC-Siam-diff (baseline) | 0.83M | 0.886 | 0.796 | — |
  | **Siamese-SegFormer MiT-b2 (diff)** — headline strong model | 24.72M | **0.911** | 0.836 | 0.943 |
  | Siamese-SegFormer MiT-b2 (concat) | 24.98M | 0.907 | 0.829 | 0.939 |

  **Fusion ablation:** difference > concat confirmed (0.9106 vs 0.9066), same encoder/LR/epochs/seed.
  **Per-scene (n=128):** strong model lifts the mean (0.734 → 0.761) but std stays ~0.31 and the
  hardest scenes (tiny/subtle changes) are still F1≈0 — the aggregate gain exceeds the per-scene-mean
  gain because the aggregate is pixel-weighted. PR curve + failure gallery committed in `docs/results/`.
- **M3 — next.** DINOv2 foundation-model tier (scope below).

## Environment facts — do NOT rediscover the hard way
- **Do NOT build a Singularity/Apptainer container on Leonardo.** It fails twice (login-node SIGKILL
  during `mksquashfs`, then Lustre-xattr errors). **Use the venv method** (per `leonardo.md`):
  `.venv-train` under `$WORK/sat-change-detection` (torch **2.5.1+cu121**, torchgeo, **smp 0.5.0**,
  timm, matplotlib, numpy, pytest), launched via `srun`. `.venv-stage` (CPU torch) is for staging.
  `container/changedet.def` is a portable artifact only — do not retry building it.
- **⚠ `transformers` is NOT installed in `.venv-train`.** M2 used smp/timm (no transformers). **M3
  (DINOv2) needs it** → the **first M3 step** is `pip install transformers peft` on the **LOGIN node**
  (compute nodes have no egress). The HF token is at `~/.hf_token`.
- **Pretrained-weight offline trap (learned in M2 — apply to DINOv2):** smp/HF loaders pin an *exact
  HF revision* per model. A plain `snapshot_download(repo)` grabs a *different* commit, so under
  `HF_HUB_OFFLINE=1` the loader misses the cache and **silently falls back to a network URL — which
  FAILS on no-egress GPU nodes** (model trains from scratch). `scripts/stage_weights.sh` reads
  repo_id+revision from the installed library and stages that exact commit. **Do the same for
  `facebook/dinov2-*`** (pin the exact revision) and verify offline load `fell_back=False`.
- **Cluster:** partition `boost_usr_prod` (4×A100-64GB/node → DDP = 4 tasks / 4 GPUs); smokes on
  `boost_qos_dbg` (30-min cap); budget-free CPU on `lrd_all_serial`. Modules `python/3.11.7`,
  `cuda/12.2` (default) → torch **cu121** wheels. SSH host alias `leonardo`. **Login nodes SIGKILL
  compute** — even a CPU smoke runs via SLURM (`lrd_all_serial`), never on the login node.
- **Allocation account + absolute `$WORK`/`$SCRATCH` paths + username live ONLY in the local
  (gitignored) `DECISIONS.md` and injected memory — NEVER committed to this repo.** Submit jobs with
  `sbatch --account=<allocation> slurm/<file>`. Data is staged at `$WORK/sat-change-detection/data/levircd`;
  the shared offline HF cache is `$WORK/sat-change-detection/.cache/huggingface` (set `HF_HOME` to it).

## Standing decisions — do NOT revert
- **4-GPU DDP is the intentional default** (Leonardo bills the whole node regardless of GPUs used).
  `train.py` scales the base LR by `effective_batch/reference_batch` (×4 on one node) + a short
  warmup. SegFormer full run used base lr 4e-5 → scaled **1.6e-4**, AdamW wd 0.01, cosine, bf16,
  per-GPU batch 8, 200 epochs (~2 h).
- **Eval threshold is ALWAYS selected on val (max-F1) and applied to test — never tuned on test.**
  The harness also reports metrics at 0.5, average precision, per-scene mean±std, and a
  cause-bucketed failure gallery.
- **Flat `{A,B,label}/{split}*.png` layout is authoritative** over PRD §5.4's nested sketch (it is
  what torchgeo's `LEVIRCD` expects).
- **Commits use the user's identity (umaraslam66, no Claude attribution)** — never a
  `Co-Authored-By: Claude` trailer, never name Claude in commit messages / PR bodies. **Standing OK
  to push to `main`** (solo portfolio repo; CI gates every push) — do not pause to ask.

## File map (entry points; M2 additions in **bold**)
- `src/models/fc_siam_diff.py` — FC-Siam-diff (M1). **`src/models/siamese_segformer.py`** — weight-
  shared smp MiT encoder + `diff|concat` fusion + all-MLP decoder. `src/models/__init__.py`
  `build_model(cfg["model"])` dispatches on `model.name` (`fc_siam_diff` | `siamese_segformer`).
  All models share the `(B,2,C,H,W) → (B,out,H,W)` interface, so train/eval are model-agnostic.
- `src/train.py` — `python -m src.train --config <cfg> [--resume-if-exists] [--set k=v ...]`. DDP
  auto-engages when world>1; bf16 AMP; rank-correct checkpoint; `run_manifest.json`; SIGUSR1/TERM
  checkpoint. Smoke controls: `max_steps`, `limit_batches`.
- **`src/eval_harness.py`** — pure, torch-free, unit-tested: PR curve, best-F1 operating point, AP,
  per-scene mean±std, `classify_failure` cause buckets, comparison-table renderer.
- **`src/evaluate.py`** — full harness (torch passes → histograms + per-scene + gallery).
  `python -m src.evaluate --config <cfg> --split test [--select-split val] [--threshold t]`. Writes
  `results/<run_id>/eval_<split>/{summary.json,pr_curve.png,gallery.png,pr_curve.json}`.
- **`src/compare.py`** — `python -m src.compare --manifest <yaml>` → tier/ablation markdown+JSON tables.
- `src/{losses,metrics,config,dist,utils}.py`, `src/data/{levircd,tiling}.py` — as M1.
- `configs/` — `levircd_baseline{,_smoke}.yaml`, **`levircd_segformer{,_smoke}.yaml`**,
  **`compare_levircd.yaml`** (baseline+diff+concat), **`ablation_fusion.yaml`** (diff vs concat).
- `slurm/` — `train.sbatch` (4-GPU DDP; forwards args after the config, e.g. `--set run_id=… model.fusion=…`),
  `smoke.sbatch` (debug-QoS 4-GPU; takes a config arg), `smoke_cpu.sbatch` (serial CPU; uses `.venv-train`).
  `scripts/stage_weights.sh` stages smp MiT weights at pinned revisions (extend for DINOv2).

## M3 — scope + first steps
DINOv2 foundation-model tier: **frozen or LoRA-adapted** (via `peft`) DINOv2 ViT encoder, shared-
weight Siamese, a custom change decoder, same `(B,2,C,H,W)→logits` interface as M1/M2. **Deliverable:**
extend the 3-way comparison (baseline vs SegFormer-diff vs DINOv2) on the identical LEVIR-CD test
split via the identical harness, answering **"does foundation-model pretraining beat an ImageNet
backbone, and at what trainable-parameter cost?"** Add the DINOv2 row to `configs/compare_levircd.yaml`.
**First steps:** (1) `pip install transformers peft` on the login node; (2) stage `facebook/dinov2-*`
at a pinned revision via `stage_weights.sh` and verify offline load; (3) smoke on `boost_qos_dbg`
before the full run.

## Working conventions
- **Smoke before full** (CPU serial or `boost_qos_dbg`). **PAUSE and ask the user before the first
  full multi-GPU submission.**
- Checkpoint every ~30 min + `--resume-if-exists` so a walltime cut never loses progress.
- Commit granularly (user identity, no Claude attribution); keep `DECISIONS.md` + `experiments/LOG.md`
  current; confirm CI is green after each push.
- **When blocked, or when an assumption is load-bearing, ask the user** — they relay questions to a
  senior reviewer and return answers. Don't guess on irreversible or costly choices.
