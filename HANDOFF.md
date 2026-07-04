# HANDOFF

Continuation notes for a fresh session with zero prior conversation context. Read this first;
then **[PRD.md](PRD.md) is the authoritative build spec** and `leonardo.md` (in the repo's parent
dir) is authoritative for all Leonardo/HPC specifics.

- **Project:** a satellite **change-detection** system (high-res aerial + Sentinel-2) with a
  first-class evaluation harness and a deployed web demo. Full spec: **[PRD.md](PRD.md)**.
- **Repo:** GitHub, branch `main` (exact URL + owner in the local, gitignored `DECISIONS.md`).
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
- **M4 — export + parity + curated Space are DONE and verified on real weights; only the HF push/deploy
  remains, and it is now UNBLOCKED (an HF org is available — see "Deploy" below).** `src/export.py`
  (ONNX export + parity + artifact bundle) done; parity **verified on the real trained checkpoints** —
  SegFormer `mit_b2` max |Δlogit| **2.29e-5**, DINOv2+LoRA (fixed 448) **8.01e-5** (tol 1e-3). The
  curated Space is **built and verified locally, end-to-end**, on the real bundles + real LEVIR-CD test
  tiles, with a polished dark **"Observatory Console"** theme: **native-1024 before/after imagery** with
  **tile-stitch masks** (the scene is split into 256px native tiles, each inferred, the mask stitched —
  the bundle's documented `tiling.tile_size`; same model/threshold/metric as the eval harness, since
  feeding the whole 1024 tile at once collapses to ~0% change), a **full-bleed cover-fit** map layout,
  and the **change mask stays pixel-aligned to the imagery through zoom/pan/resize**. Full-scene
  predictions match the GT change fraction (e.g. test_45 25.8% vs 25% annotated; stable control 1.3% vs
  1.5%). See "M4 status" and "Deploy" below.

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
- **Commits use the maintainer's own git identity, with no AI/tool attribution** — never a
  `Co-Authored-By` AI trailer, never name any AI assistant in commit messages / PR bodies, never write
  a personal name/handle/username into tracked files (those stay in local, gitignored notes / env).
  **Standing OK to push to `main`** (solo portfolio repo; CI gates every push) — do not pause to ask.

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

## M4 — scope
ONNX export + PyTorch↔ONNX parity check + **curated** HF Space. **Export (`src/export.py`, PRD §9):**
export each Track-A model to ONNX (fixed opset; dynamic batch/H/W where feasible); **assert
PyTorch↔ONNXRuntime parity** (max abs diff below tol, else FAIL the export); emit the per-model
artifact bundle — `model.onnx`, `config.yaml`, `preprocessing.json` (norm stats, input size, band
order, tiling), `metrics_card.md`. Push the bundles to an **HF Model repo**; the Space pulls from
there at build/startup (keeps the Space lean, separates weights from app). **Space (PRD §10):** Docker
HF Space, FastAPI (uvicorn :7860) serving a React+MapLibre static build; **/curated** before/after mode
— swipe slider, change-mask overlay + opacity, stats panel — on LEVIR-CD test pairs, CPU `onnxruntime`.
Defer live-AOI (Track-B/OSCD) to M5.

- **KNOWN CAVEAT — DINOv2 ONNX (verify parity SPECIFICALLY on the DINOv2 model, not just the CNN
  tiers):** `dinov2_cd` resizes inputs to `image_size=448` internally and uses
  `interpolate_pos_encoding=True`. If the 448 resize + pos-embed interpolation are not baked into the
  traced graph correctly, the exported model **silently misbehaves** (wrong output, no error). Export
  at a **fixed 448 grid** (bake the resize into `preprocessing.json`) and assert PyTorch↔ONNX parity on
  a real LEVIR-CD pair for the DINOv2 checkpoint. FC-Siam-diff and SegFormer are fully-convolutional and
  export cleanly with dynamic H/W — DINOv2 is the risky one. (All verified — see status.)
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
  `interpolate_pos_encoding` bake in as constants; **CNN tiers = dynamic batch + H/W**. Parity verified
  on the **real trained checkpoints** (SegFormer 2.29e-5, DINOv2+LoRA 8.01e-5).
- **`tests/test_export.py`** — parity for all three tiers on tiny random-init models (CI-safe:
  `importorskip` onnx/onnxruntime/transformers/peft). The DINOv2 case forces a native pos-grid ≠
  export grid so `interpolate_pos_encoding` truly interpolates — the exact silent-misbehavior trap.
- **Curated Space (`app/`)** — multi-stage `Dockerfile` (node build → py3.11 runtime, uvicorn :7860),
  FastAPI (`backend/app.py`: `/api/health|models|models/{id}/card|curated|curated/{id}/{which}.png|predict`,
  serves the built SPA), CPU-onnxruntime inference (`backend/inference.py`, consumes a bundle dir via
  `BUNDLES_DIR`; startup pull from `HF_BUNDLE_REPO` for the Space), React+Vite+MapLibre frontend
  (`frontend/`: two view-synced maps + draggable swipe divider, change overlay, opacity, stats,
  model-card page with the real 4-tier results). Dark **"Observatory Console"** theme; fonts are
  **self-hosted** (`@fontsource` IBM Plex, vendored into the build — no CDN at runtime).
- **`inference.py` runs tile+stitch (`predict`):** the full scene is split into `tiling.tile_size`
  (256px) native tiles, each run through the ONNX model at `input_size`, and the probability map is
  stitched back to native resolution before thresholding. This is the bundle's documented preprocessing
  and matches the eval harness — the single-pass shortcut is gone because a 0.5 m/px model fed the whole
  1024 tile (≈4× its trained field of view) detects almost nothing. Stats derive from the stitched
  mask; the overlay is rendered at native res as a **translucent amber fill + crisp 1–2px outline**.
- **The change overlay is an aligned HTML `<img>` over the maps, NOT a MapLibre raster layer** — a
  second stacked image-source raster proved unreliable to paint (add-after-idle never repaints); an HTML
  element positioned via `map.project()` always renders, gives smooth CSS opacity, and **stays
  pixel-aligned through zoom/pan** (re-aligned on every map `move`). Base imagery IS a MapLibre raster
  image layer (`raster-resampling: linear`), cover-fit to fill the stage.
- **Verified in-browser** (chrome-devtools + DOM measurement): frontend `npm run build` compiles, uvicorn
  serves the build + API, both bundles predict, swipe + overlay + opacity + pair/model switching +
  model-card all work; cover-fit is full-bleed at 1280/1440/ultrawide and the mask tracks the imagery
  through zoom (954→1908 px lockstep) and pan. Demo runs on **real LEVIR-CD test tiles** (the 7 sharpest
  by variance-of-Laplacian, change spread 1.5%→25%) + **real trained bundles** — all gitignored (LEVIR
  imagery is not redistributed; bundles come from HF / re-export). `gen_sample_pairs.py` still makes
  synthetic placeholder pairs if the real tiles aren't staged (a banner flags placeholder weights).

### M4 done locally — reproduce the real bundles + demo data
The real artifacts live **only on the dev machine** (gitignored — bundles pull from HF at deploy,
LEVIR imagery is not redistributed per PRD §5.3). To regenerate after `scp`-ing the checkpoints
(under `$WORK/.../results/<run_id>/checkpoints/best.pt`) to a local path:
```
python -m src.export --config configs/levircd_segformer.yaml --checkpoint <seg best.pt> --out-dir bundles
python -m src.export --config configs/levircd_dinov2.yaml   --checkpoint <dv2 best.pt> --out-dir bundles
cp -r bundles/* app/backend/models/                       # the Space's BUNDLES_DIR
# curated pairs: drop real LEVIR test A/B tiles into app/backend/data/curated as native-1024
#   before.png/after.png (A=before, B=after) + a manifest.json, or use gen_sample_pairs.py for
#   synthetic placeholders. Then bake predictions to data/curated/_predictions.json (below).
```
(Locally: Python 3.12 CPU venv pinned to the training stack; `Dinov2Model.from_pretrained` /smp
download the *architecture* over the internet, then the checkpoint overwrites the weights.)

**Baked prediction cache (`app/backend/data/curated/_predictions.json`, gitignored):** curated pairs +
models are fixed, so every (pair, model) tile-stitch prediction is precomputed and written there. The
backend loads it at startup (`_load_prediction_cache`) so the Space serves curated results **instantly**;
a background `_prewarm` fills any gaps and re-saves. Regenerate by deleting the file and hitting each
`/api/predict`, or by running `_prewarm` once.

### Deploy — now UNBLOCKED (still gated on the maintainer's explicit go-ahead)
1. **HF org name + identity/visibility decision** come from the maintainer **at deploy time** — do NOT
   hard-code them in tracked files (local `DECISIONS.md` / env only). Create the Model repo (push
   bundles; set `HF_BUNDLE_REPO` on the Space to pull them) + the Docker Space under that org.
2. **Docker** is not installed on the dev Mac, so `docker build app/` wasn't run — the Dockerfile is
   authored and the frontend build + backend serve were verified outside Docker. Let HF build it at
   deploy, or install Docker for a full local container build first.
3. **PAUSE for the explicit go-ahead before pushing bundles or making the Space public** (outward-facing,
   hard to reverse) — even though the org is now available.
4. **HF free-tier performance is handled by the baked cache** — the public Space serves curated
   predictions with **no live CPU inference**. For reference, full-scene tile-stitch on the ~2-vCPU free
   tier is **DINOv2 ~51 s / SegFormer ~3 s per scene**, which is exactly why the cache exists; keep live
   inference deferred to the M5 live-AOI path (optionally ONNX INT8 for the Track-B model there).

### Known display note (accepted)
- **Ultrawide cover-fit upscales to fill:** the native tile is 1024 px, so on a very wide stage cover-fit
  scales it up (~2.18× on a 2560-wide window) to leave no letterbox voids — smooth (linear resampling),
  and zoom/pan recover native pixels. At ≤1440-wide it fills at ≤~1.09× (native-sharp); ~1280 fills at
  0.93× (downscaled/crisp). Accepted tradeoff (fill-first). Opt-out if ever wanted: cap the cover bump in
  `CompareView.fitCover` — e.g. `zoom = cam.zoom + Math.min(coverBump, Math.log2(MAX_SCALE))` — to trade
  a little letterbox back for a guaranteed ≤MAX_SCALE display scale.

## Roadmap after M4
- **M4 deploy** — gated on org name + identity/visibility decision (above), then push bundles + create
  the HF Model repo & Space, and (optionally) a full Docker build.
- **M5 — live Sentinel-2 AOI** (Track-B/OSCD): STAC → Planetary Computer → tile → Track-B ONNX → overlay;
  AOI draw + date pickers + latency caps. This is where live CPU inference belongs.
- **M6 — xBD disaster track** (multi-class building damage, xView2 weighted-F1 metric). The hardest and
  latest milestone.

## Working conventions
- **Smoke before full** (CPU serial or `boost_qos_dbg`). **PAUSE and ask the maintainer before the first
  full multi-GPU submission — and (M4+) before deploying/making public the HF Space or pushing weight
  bundles to a public HF Model repo** (outward-facing, hard to reverse).
- Checkpoint every ~30 min + `--resume-if-exists` so a walltime cut never loses progress.
- Commit granularly (maintainer identity, no AI/tool attribution); keep `DECISIONS.md` +
  `experiments/LOG.md` current; confirm CI is green after each push.
- **When blocked, or when an assumption is load-bearing, ask the maintainer** — they relay questions to a
  senior reviewer and return answers. Don't guess on irreversible or costly choices.
