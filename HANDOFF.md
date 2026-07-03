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
- **M3 — done.** DINOv2 FM tier — **frozen `facebook/dinov2-base` (ViT-B/14) + LoRA**, weight-shared
  Siamese, multi-layer change decoder. LEVIR-CD **test** (threshold-on-val→test), full 4-tier table:

  | Model | Trainable | F1 | IoU | AP |
  |---|---|---|---|---|
  | FC-Siam-diff (baseline) | 0.83M | 0.886 | 0.796 | 0.932 |
  | Siamese-SegFormer MiT-b2 (diff) | 24.72M | 0.911 | 0.836 | 0.943 |
  | DINOv2-base frozen linear-probe | 1.64M | 0.889 | 0.800 | 0.924 |
  | **DINOv2-base + LoRA** (headline FM tier) | **2.82M** | **0.913** | **0.839** | **0.946** |

  **Headline = parameter efficiency (the defensible claim), NOT an accuracy win.** DINOv2+LoRA
  matches-to-slightly-beats the specialist SegFormer (F1 0.913 vs 0.911 is **within noise**; IoU 0.839
  vs 0.836; AP 0.946 vs 0.943) at **~9× fewer trainable params** (2.82M vs 24.72M). Frame it as
  parity-plus-efficiency; do not oversell the fractional F1 lead. The causal story (frozen-probe
  ablation): frozen features alone (1.64M, decoder-only) ≈ the baseline (0.889 vs 0.886) → the
  self-supervised representation already carries most of the change signal; the frozen→LoRA delta
  (+0.024 F1) is the adaptation lift — the win comes from *both*, not either alone.
  **Honest limitation (keep prominent):** per-scene LoRA mean **0.767**, std **~0.31**, **min 0.00** —
  the FM lifts the mean but does NOT tighten variance or fix the hardest small/subtle tiles (10/12
  worst tiles `small_subtle`; 11/12 for frozen). Artifacts in `docs/results/` (dinov2_lora_*).
- **M4 — export + parity + local Space are DONE and verified on real weights; only the HF push/deploy
  remains, gated on the user's HF org name.** `src/export.py` (ONNX export + parity + artifact bundle)
  done; parity **verified on the real trained checkpoints** — SegFormer `mit_b2` max |Δlogit|
  **2.29e-5**, DINOv2+LoRA (fixed 448) **8.01e-5** (tol 1e-3) — see "M4 status" below. Curated HF
  Space **built and verified locally, end-to-end**, on the real bundles + real LEVIR-CD test crops:
  DINOv2 detects real building change at the val-selected threshold 0.508 (31% predicted vs ~30% GT
  on a test crop); no-change control 0%. **Only open (deploy-side, DO NOT do until the user hands over
  the org name):** HF org/username → push bundles to a Model repo + create the Space; local
  `docker build` (Docker not installed on the dev Mac).

## Environment facts — do NOT rediscover the hard way
- **Do NOT build a Singularity/Apptainer container on Leonardo.** It fails twice (login-node SIGKILL
  during `mksquashfs`, then Lustre-xattr errors). **Use the venv method** (per `leonardo.md`):
  `.venv-train` under `$WORK/sat-change-detection` (torch **2.5.1+cu121**, torchgeo, **smp 0.5.0**,
  timm, matplotlib, numpy, pytest), launched via `srun`. `.venv-stage` (CPU torch) is for staging.
  `container/changedet.def` is a portable artifact only — do not retry building it.
- **`transformers` 4.57.6 + `peft` 0.19.1 (+ `accelerate`) ARE now installed in `.venv-train`** (M3,
  login-node pip). torch untouched (2.5.1+cu121); the install pulled `numpy` 2.4.4→2.0.2 (now matches
  the pyproject pin `<2.1`) and `huggingface_hub` 1.21.0→0.36.2 (transformers 4.57 ceiling) — both
  benign, re-verified. **DINOv2 weights staged + offline-verified:** `facebook/dinov2-small`
  @`ed25f3a3`, `facebook/dinov2-base`@`f9e44c81`, both **ungated**. HF token at `~/.hf_token`. Unlike
  smp, `transformers` from_pretrained HONORS `HF_HUB_OFFLINE` (errors on cache miss, no silent URL
  fallback). Re-run `stage_weights.sh` if transformers is upgraded (revisions are library-tied).
- **DDP + gradient checkpointing trap (fixed in M3):** reentrant activation checkpointing rebuilds the
  encoder graph in backward, so DDP's reducer flags all LoRA params as "unused" and aborts. Fix =
  `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` in the model
  (already applied in `dinov2_cd.py`). Single-GPU has no reducer so it hides the bug — always confirm a
  capped 4-GPU DDP run of the full config before a full submission.
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
- `src/models/fc_siam_diff.py` — FC-Siam-diff (M1). `src/models/siamese_segformer.py` (M2) — weight-
  shared smp MiT encoder + `diff|concat` fusion + all-MLP decoder. **`src/models/dinov2_cd.py`** (M3) —
  frozen `facebook/dinov2-*` ViT + LoRA (peft `inject_adapter_in_model`) + multi-layer decoder; regimes
  via config (`lora` / frozen linear-probe / full finetune); resizes to `image_size` (mult. of 14),
  `use_reentrant=False` checkpointing. `src/models/__init__.py` `build_model(cfg["model"])` dispatches on
  `model.name` (`fc_siam_diff` | `siamese_segformer` | `dinov2_cd`). All models share the
  `(B,2,C,H,W) → (B,out,H,W)` interface, so train/eval/compare are model-agnostic.
- **`src/train.py` (M3 tweaks):** optimizer takes only `requires_grad` params (excludes the frozen ViT);
  config knob `train.ddp_find_unused_parameters` (default False). `src/utils.even_feature_layers` (M3,
  CI-tested) picks the ViT hidden-state taps (ends at the last layer → no DDP-unused params).
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
- `configs/` — `levircd_baseline{,_smoke}.yaml`, `levircd_segformer{,_smoke}.yaml`,
  **`levircd_dinov2{,_smoke}.yaml`** (M3 FM tier; smoke uses dinov2-small@224),
  `compare_levircd.yaml` (now 5 rows: baseline + SF diff/concat + DINOv2 frozen/LoRA),
  `ablation_fusion.yaml` (diff vs concat).
- `slurm/` — `train.sbatch` (4-GPU DDP; forwards args after the config, e.g. `--set run_id=… model.fusion=…`),
  `smoke.sbatch` (debug-QoS 4-GPU; takes a config arg), `smoke_cpu.sbatch` (serial CPU; uses `.venv-train`).
  `scripts/stage_weights.sh` stages smp MiT weights at pinned revisions (extend for DINOv2).

## M4 — scope + first steps
ONNX export + PyTorch↔ONNX parity check + **curated** HF Space. **Export (`src/export.py`, PRD §9):**
export each Track-A model to ONNX (fixed opset; dynamic batch/H/W where feasible); **assert
PyTorch↔ONNXRuntime parity** (max abs diff below tol, else FAIL the export); emit the per-model
artifact bundle — `model.onnx`, `config.yaml`, `preprocessing.json` (norm stats, input size, band
order, tiling), `metrics_card.md`. Push the bundles to an **HF Model repo**; the Space pulls from
there at build/startup (keeps the Space lean, separates weights from app). **Space (PRD §10):** Docker
HF Space, FastAPI (uvicorn :7860) serving a React+MapLibre static build; **/curated** before/after mode
— swipe slider, change-mask overlay + opacity, stats panel — on LEVIR-CD test pairs, CPU `onnxruntime`.
Defer live-AOI (Track-B/OSCD) to M5.

- **\*\*\* M4 NEEDS THE USER'S HF ORG/USERNAME: `[FILL IN ONCE USER PROVIDES]`** — the Model repo (weight
  bundles) and the Space both live there. Ask before pushing bundles or creating the Space; it is a
  NEEDS-CONFIRMATION item in DECISIONS "Open items". Do not invent a name.
- **KNOWN CAVEAT — DINOv2 ONNX (verify parity SPECIFICALLY on the DINOv2 model, not just the CNN
  tiers):** `dinov2_cd` resizes inputs to `image_size=448` internally and uses
  `interpolate_pos_encoding=True`. If the 448 resize + pos-embed interpolation are not baked into the
  traced graph correctly, the exported model **silently misbehaves** (wrong output, no error). Export
  at a **fixed 448 grid** (bake the resize into `preprocessing.json`) and assert PyTorch↔ONNX parity on
  a real LEVIR-CD pair for the DINOv2 checkpoint. FC-Siam-diff and SegFormer are fully-convolutional and
  export cleanly with dynamic H/W — DINOv2 is the risky one.
- **Surface note:** M0–M3 train on **Leonardo** (compute nodes have **no egress** → weights pre-staged).
  M4's Space **deploys on Hugging Face**, which **does** have egress (it pulls the bundle from the HF
  Model repo at build/startup) — a different surface with different networking rules. The offline guards
  are a Leonardo concern, not an HF-Space one.

### M4 status (what's done)
- **`src/export.py`** — CLI `python -m src.export --config <cfg> [--checkpoint <best.pt>] [--random-init]`.
  Builds the model on CPU, exports to ONNX (opset 17), **asserts PyTorch↔ONNXRuntime parity** on a
  fixed sample (max abs logit diff ≤ `--tol` 1e-3, else RuntimeError → no bundle), and writes the
  bundle: `model.onnx`, `config.yaml`, `preprocessing.json`, `metrics_card.md`, `parity.json`.
  **DINOv2 export = fixed `image_size` grid (static H/W, only batch dynamic)** so the 448 resize +
  `interpolate_pos_encoding` bake in as constants; **CNN tiers = dynamic batch + H/W**. The parity
  pass also re-runs ORT on a second shape to prove the declared dynamic axes actually hold.
- **`tests/test_export.py`** — parity for all three tiers on tiny random-init models (CI-safe:
  `importorskip` onnx/onnxruntime/transformers/peft). The DINOv2 case forces a native pos-grid ≠
  export grid so `interpolate_pos_encoding` truly interpolates — the exact silent-misbehavior trap.
- **Verified locally** (Python 3.12 CPU venv pinned to the training stack: torch 2.5.1, transformers
  4.57.6, peft 0.19.1, smp 0.5.0): SegFormer `mit_b2` max |Δlogit| **6.3e-7** (dynamic H/W); DINOv2
  native-518→448 max |Δlogit| **8.1e-8** (fixed 448). Parity is a graph property (independent of the
  trained weight values), so this proves the export machinery; **re-run on the real `best.pt` for the
  shippable bundle** (see blockers).
- **`build_model` gained an `encoder_config` passthrough** (dinov2 only) so a random-init encoder can
  be given a custom native grid — purely additive.
- **Curated Space (`app/`)** — multi-stage `Dockerfile` (node build → py3.11 runtime, uvicorn :7860),
  FastAPI (`backend/app.py`: `/api/health|models|models/{id}/card|curated|curated/{id}/{which}.png|predict`,
  serves the built SPA), CPU-onnxruntime inference (`backend/inference.py`, consumes a bundle dir via
  `BUNDLES_DIR`; startup pull from `HF_BUNDLE_REPO` for the Space), React+Vite+MapLibre frontend
  (`frontend/`: two view-synced maps + draggable swipe divider, change overlay, opacity, stats,
  model-card page with the real 4-tier results). **The change overlay is an aligned HTML `<img>`
  over the maps, NOT a MapLibre raster layer** — a second stacked image-source raster proved
  unreliable to paint (add-after-idle never repaints); an HTML element positioned via `map.project()`
  always renders and gives smooth CSS opacity. Base imagery IS a MapLibre raster image layer.
- **Verified in-browser** (chrome-devtools): frontend `npm run build` compiles, uvicorn serves the
  build + API, both bundles predict (DINOv2 448 ~200 ms, SegFormer 256 ~750 ms on CPU), swipe +
  overlay + opacity + pair/model switching + model-card all work. Demo runs on **synthetic placeholder
  pairs** (`backend/gen_sample_pairs.py`) + **random-init bundles** (a banner says so); both are
  gitignored (regenerate the pairs with the script; bundles come from HF / re-export).

### M4 done locally — reproduce the real bundles + demo data
The real artifacts live **only on the dev machine** (gitignored — bundles pull from HF at deploy,
LEVIR imagery is not redistributed per PRD §5.3). To regenerate after `scp`-ing the checkpoints
(under `$WORK/.../results/<run_id>/checkpoints/best.pt`) to a local path:
```
python -m src.export --config configs/levircd_segformer.yaml --checkpoint <seg best.pt> --out-dir bundles
python -m src.export --config configs/levircd_dinov2.yaml   --checkpoint <dv2 best.pt> --out-dir bundles
cp -r bundles/* app/backend/models/                       # the Space's BUNDLES_DIR
# curated pairs: drop real LEVIR test A/B tiles into app/backend/data/curated (256px before/after +
# a manifest.json), or use gen_sample_pairs.py for synthetic placeholders.
```
(Locally: Python 3.12 CPU venv pinned to the training stack; `Dinov2Model.from_pretrained` /smp
download the *architecture* over the internet, then the checkpoint overwrites the weights.)

### Remaining M4 steps (deploy-side — need the user)
1. **HF org/username** (still a placeholder) → create the Model repo (push bundles; set
   `HF_BUNDLE_REPO` on the Space to pull them) + create the Docker Space.
2. **Docker** is not installed on the dev Mac, so `docker build app/` wasn't run — the Dockerfile is
   authored and the frontend build + backend serve were verified outside Docker. Install Docker for a
   full local container build, or let HF build it at deploy.
3. **Do NOT push bundles or deploy the Space until the user hands over the HF org name** (standing
   convention + explicit hold — the user is waiting on the client and on approval to showcase publicly
   under their name). Stay paused here.
4. **HF free-tier performance plan (do at deploy time):** real DINOv2-base on CPU is ~4 s/tile
   (SegFormer ~1 s). The curated pairs + models are fixed, so **precompute every (pair, model)
   prediction at build time and bake the overlay-PNG + stats into the bundle/cache** (e.g. a
   `curated_predictions.json` the backend loads at startup, or warm `_predict_cache` in a startup
   hook) → the public Space serves curated results **instantly** with no CPU inference. Keep live
   inference for the M5 live-AOI path. Optionally add ONNX INT8 quantization for the Track-B model.

## Working conventions
- **Smoke before full** (CPU serial or `boost_qos_dbg`). **PAUSE and ask the user before the first
  full multi-GPU submission — and (M4+) before deploying/making public the HF Space or pushing weight
  bundles to a public HF Model repo** (outward-facing, hard to reverse).
- Checkpoint every ~30 min + `--resume-if-exists` so a walltime cut never loses progress.
- Commit granularly (user identity, no Claude attribution); keep `DECISIONS.md` + `experiments/LOG.md`
  current; confirm CI is green after each push.
- **When blocked, or when an assumption is load-bearing, ask the user** — they relay questions to a
  senior reviewer and return answers. Don't guess on irreversible or costly choices.
