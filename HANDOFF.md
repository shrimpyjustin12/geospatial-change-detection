# HANDOFF

Continuation notes for a fresh session with zero prior conversation context. Read this first;
then **[PRD.md](PRD.md) is the authoritative build spec** and `leonardo.md` (in the repo's parent
dir) is authoritative for all Leonardo/HPC specifics.

- **Project:** a satellite **change-detection** system (high-res aerial + Sentinel-2) with a
  first-class evaluation harness and a deployed web demo. Full spec: **[PRD.md](PRD.md)**.
- **Repo:** GitHub, branch `main` (exact URL + owner in the local, gitignored `DECISIONS.md`).
- **Notes:** `DECISIONS.md` (resolved cluster placeholders + every decision) is **local & gitignored**
  — never commit it. `experiments/LOG.md` (run trail) is **TRACKED/committed**, so it must stay
  identity-clean (job IDs, partition names, configs, git SHAs, outcomes only — NO allocation account /
  absolute `$WORK`/`$SCRATCH` paths / username). Keep both current. Injected memory also carries key facts.

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
- **M4 — COMPLETE (deployed to Hugging Face 2026-07-05; public Space live and verified logged-out).**
  Live URLs — Space (public): `huggingface.co/spaces/GeospatiaProject/geospatial1`; direct app:
  `geospatiaproject-geospatial1.hf.space`; Model repo (public): `GeospatiaProject/geospatial-1`. The
  curated aerial mode serves **precomputed predictions from the baked cache** (no runtime inference) and
  is **verified working for anonymous visitors**. Build/deploy detail under "Deploy — DONE" below.
  `src/export.py`
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

### Deploy — DONE (2026-07-05)
- **Live.** Public Docker Space `GeospatiaProject/geospatial1` (app URL `geospatiaproject-geospatial1.hf.space`)
  pulls the two Track-A bundles at startup from the public Model repo `GeospatiaProject/geospatial-1` via the
  `HF_BUNDLE_REPO` Space variable. Bundles uploaded with `HfApi.upload_folder`; `app/` uploaded EXCLUDING
  `backend/models/` (kept lean) but INCLUDING the baked cache (`_predictions.json`, 14 entries) + curated
  pairs/images. HF built the image — no local Docker build needed.
- **Verified logged-out** (anonymous, isolated browser context): health ok, real weights
  (`is_placeholder=False`), model cards clean, `/api/predict` served instantly from the baked cache
  (~0.3–0.5 s round-trip, NOT the ~51 s DINOv2 tile-stitch), and swipe / overlay / opacity / pair-switch /
  model-card all work.
- **Identity:** repo content + served artifacts are free of personal/tool identifiers. A leak was caught
  pre-push — both `metrics_card.md` embedded the absolute local checkpoint path; sanitized to `best.pt`
  and `src/export.py` patched to emit the basename only (on `main`, CI green). The org/visibility choice
  and exact deploy specifics live in the local gitignored `DECISIONS.md` ("M4 DEPLOY" section).
- **Free-tier perf** is handled by the baked cache — the public Space serves curated predictions with **no
  live CPU inference**. For reference, full-scene tile-stitch on the ~2-vCPU free tier is **DINOv2 ~51 s /
  SegFormer ~3 s per scene**, which is exactly why the cache exists. The same cache-first pattern is the
  basis for M5 (below).

### Known display note (accepted)
- **Ultrawide cover-fit upscales to fill:** the native tile is 1024 px, so on a very wide stage cover-fit
  scales it up (~2.18× on a 2560-wide window) to leave no letterbox voids — smooth (linear resampling),
  and zoom/pan recover native pixels. At ≤1440-wide it fills at ≤~1.09× (native-sharp); ~1280 fills at
  0.93× (downscaled/crisp). Accepted tradeoff (fill-first). Opt-out if ever wanted: cap the cover bump in
  `CompareView.fitCover` — e.g. `zoom = cam.zoom + Math.min(coverBump, Math.log2(MAX_SCALE))` — to trade
  a little letterbox back for a guaranteed ≤MAX_SCALE display scale.

## Roadmap after M4
- **M5 — curated Sentinel-2 (Track-B) mode — IN PROGRESS: Phases 1–3 DONE, Phase 4–5 remaining (both
  before Gate 2). Full status + exact next steps in the "## M5 — status" section below.** REVISED SCOPE
  (decided with the reviewer; simpler than the
  PRD §10.2 fully-live plan). Add a CURATED Sentinel-2 mode to the SAME Space: a handful of real-world
  AOIs whose predictions are PRECOMPUTED and served instantly from cache, exactly like the curated aerial
  mode. Free-tier, CPU-only — **no GPU, no runtime/live inference, no runtime STAC, no latency caps.**
  - **Core work (STILL REQUIRED):** train a Sentinel-2-native model on **OSCD** on Leonardo (Track-B, PRD
    §6.3). Aerial models do NOT transfer to 10 m Sentinel-2. Compact Siamese CD model; input = Sentinel-2
    **RGB+NIR** (full 13-band optional via config). Export to ONNX like the aerial tiers.
  - **STAC / Planetary Computer = BUILD-TIME ONLY:** fetch ~4–6 low-cloud Sentinel-2 L2A pairs for chosen
    AOIs, co-register once, precompute predictions, bake into the cache/bundle. No STAC or auth at runtime.
  - **AOI selection:** real-world locations with LARGE, OBVIOUS change visible even at 10 m (major urban
    expansion, large new construction, reservoir/dam filling, airport builds, deforestation). Prioritize
    clearest-change + lowest-cloud for visual quality.
  - **Frontend:** new Sentinel-2 tab in the existing Space. Show AOIs as pins on a MapLibre basemap;
    clicking a pin loads the before/after S2 pair + change overlay + stats, reusing the existing
    swipe/overlay/stats + dark theme. UI copy must be honest: label **Sentinel-2 10 m**, note it's a
    coarser domain than the aerial track, and that these are curated real-world examples.
  - **Training:** OSCD is tiny (24 pairs) → **single-GPU is sufficient and RECOMMENDED over the 4-GPU
    default** (avoids effective-batch / convergence issues on a small dataset); if DDP is used, scale LR +
    warmup. Stage OSCD on the Leonardo **login node** (no egress on compute nodes), same pattern as LEVIR-CD.
  - **Honesty:** OSCD is small → expect modest F1; present directionally-correct with caveats. Aerial stays
    the crisp/high-res showcase; Sentinel-2 is the any-real-location showcase.
  - **OUT OF SCOPE for now (optional later):** a live "run on a fresh STAC fetch" button — deferred; do NOT
    build unless explicitly asked.
- **M6 — xBD disaster track** (multi-class building damage, xView2 weighted-F1 metric). The hardest and
  latest milestone.

## M5 — status (Phases 1–3 DONE 2026-07-06; Phase 4–5 remaining, both BEFORE Gate 2)

**Gate 1 (before first full training) — PASSED, maintainer approved.** Gate 2 unchanged (see bottom).

### Phases 1–3 — DONE (OSCD Track-B trained, evaluated, exported)
- **New code (all TDD, CI-green, on `main`):**
  - `src/data/oscd.py` — `TiledOSCD`: 4-band **RGB+NIR = S2 B04/B03/B02/B08**, torchgeo-free (torchgeo
    0.8.1 IS on the cluster but its import hangs on the login node — we read files directly). Variable
    per-scene **edge-aligned tiling** (OSCD cities differ in size), robust `{0,255}/{1,2}/{0,1}` mask
    binarize, `scene_id(idx)` for the eval per-scene breakdown, per-band norm stats baked in
    (`OSCD_MEAN`/`OSCD_STD`, computed on the staged train split; NIR 0.20 > RGB ~0.13).
  - `src/data/__init__.py` — `build_dataset(dcfg, split, augment)` dispatch on `data.name`
    (`levircd`|`oscd`); wired into `train.py` + `evaluate.py` (both previously hard-coded `TiledLEVIRCD`).
  - `configs/oscd_s2{,_segformer}{,_smoke}.yaml` — Track-B. **SINGLE-GPU (`ddp:false`) is intentional —
    do NOT revert to the 4-GPU DDP default** (24 pairs → large effective batch hurts convergence).
  - `slurm/{smoke,train}_1gpu.sbatch` — single-GPU (the 4-GPU `train.sbatch`/`smoke.sbatch` are Track-A).
  - `src/export.py` — 4-band bundle (band order `[R,G,B,NIR]`, ÷10000 S2 scaling, OSCD norm stats, and an
    **honest-framing** model card). `src/evaluate.py` — `scene_id`-based per-scene grouping + RGB-slice gallery.
- **OSCD staging (login node):** `scripts/stage_data.sh oscd` — curl torchgeo 0.8.1's Onera archive URLs,
  **md5-verified**. **TLS gotcha:** the Train Labels host `partage.mines-telecom.fr` serves a cert that
  doesn't match its hostname → we use `partage.imt.fr` (same Nextcloud share token, valid cert; TLS stays
  ON). Normalized to `<root>/<city>/{imgs_*_rect, cm/cm.png}` + `train.txt`/`test.txt`. **24 cities: 14
  train / 10 test; val = the last 2 train cities (held out, scene-disjoint).** `scripts/smoke_load_oscd.py`
  = load smoke + recomputes the D2 norm stats.
- **Results (single-GPU, 100 ep):** baseline **`fc_siam_diff` (0.83M) test F1 0.453 / IoU 0.293** (thr
  selected on val→test); `siamese_segformer` mit_b0 4-band (ImageNet-pretrained) **F1 0.413** →
  **ImageNet transfers poorly to 10 m multispectral; the from-scratch baseline is the demo model.** Jobs:
  train 48688859 (baseline) / 48689122 (segformer); eval 48690012. Bundle **exported to
  `bundles/oscd_s2_baseline/`** (4-band ONNX, parity **1.4e-6** on the real checkpoint; card carries the
  honest 10 m framing). Trained checkpoint: cluster `results/oscd_s2_baseline/checkpoints/best.pt`; a local
  copy is `ckpts_local/oscd_s2_baseline_best.pt` (gitignored). Reproduce the bundle:
  `WORK=/tmp/w python -m src.export --config configs/oscd_s2.yaml --checkpoint <best.pt> --out-dir bundles`.
- **Env:** the repo's own `.venv` was rebuilt as **Python 3.12** with the CPU stack (torch 2.5.1, rasterio,
  onnx/onnxruntime, ruff/mypy/pytest) — it had been an empty py3.14. Build-time STAC deps installed locally
  (`pystac-client`, `planetary-computer`, `rioxarray`). CI installs only `.[dev]` + CPU torch, so the
  rasterio/onnx-gated OSCD tests SKIP in CI (they run locally).

### Phase 4 — build_sentinel2.py offline bake (NOT built yet — do this first in the fresh session)
Build-time ONLY (no runtime STAC/torch/GPU). Fetch the 5 AOIs from Planetary Computer STAC, co-register
(**trivial — each AOI's before/after are on the SAME MGRS tile**, so identical UTM grid), make RGB display
PNGs (percentile-stretch B04/B03/B02), run the OSCD ONNX **offline** on 4-band standardized input, threshold
(bundle's 0.469), compute stats, render the overlay.
- **KEY FINDING — the S2 bake MUST be a SEPARATE offline pipeline:** `app/backend/inference.py` is
  **hard-coded RGB (3-band)** — `Bundle.mean/std` do `reshape(3,1,1)` and `_to_input` does `.convert("RGB")`
  — so the OSCD 4-band model **cannot** run through its `predict`. Write your own 4-band tile-stitch (tile at
  the bundle's `tiling.tile_size`=256, same as the eval harness). **`inference.py._overlay_png(mask, out_size)`
  IS band-agnostic and reusable — import it** for the amber-fill+outline overlay.
- Emit per-AOI `before.png`/`after.png`/`overlay.png` + `manifest.json` + `_predictions.json` under
  gitignored `app/backend/data/sentinel2/`. **Cache-entry shape MUST match `inference.py.predict`'s return:**
  `{overlay_png, threshold, is_placeholder:false, stats{changed_fraction, changed_percent,
  mean_confidence_changed, mean_confidence_overall, changed_pixels, total_pixels}, elapsed_ms, input_size,
  n_tiles, pair_id, model_id}`.
- **The 5 confirmed AOIs (all verified low-cloud S2 L2A; exact acquisitions; same tile → aligned):**

  | id | title | before | after | tile | center lat,lon |
  |---|---|---|---|---|---|
  | dubai_deira | Dubai Deira Islands reclamation | 2016-03-03 (0.2%) | 2023-04-28 (0.0%) | T40RCP | 25.30, 55.34 |
  | gerd_reservoir | Grand Ethiopian Renaissance Dam | 2020-02-14 (0.0%) | 2023-12-25 (0.0%) | T36PYT | 11.21, 35.09 |
  | beijing_daxing | Beijing Daxing airport | 2016-10-10 (0.8%) | 2019-11-19 (0.8%) | T50SMJ | 39.51, 116.41 |
  | bhadla_solar | Bhadla Solar Park | 2017-12-17 (0.0%) | 2021-12-16 (0.0%) | T43RBL | 27.54, 72.02 |
  | egypt_newcapital | New Administrative Capital, Egypt | 2016-08-27 (0.7%) | 2023-08-26 (0.0%) | T36RUU | 30.01, 31.75 |

  Direction chosen with the maintainer: **arid/engineered, crispest imagery** (reclamation / reservoir /
  airport / solar / desert city) — no cloudy tropical scenes. Feasibility probe: `scratchpad/stac_probe.py`.

### Phase 5 — S2 backend routes + Sentinel-2 tab (NOT built yet — after Phase 4)
- Backend `app/backend/sentinel2.py` registry + `/api/sentinel2` routes serving the **baked results only**
  (NO runtime inference/STAC). **Keep STAC deps OUT of `app/backend/requirements.txt`** — the runtime image
  stays STAC-free.
- Frontend `Sentinel2View.tsx` — MapLibre pins at the AOI centers → clicking loads before/after + the baked
  overlay + stats into **`CompareView` (fully generic — reuse as-is)**. New "Sentinel-2" tab in `App.tsx`.
  Single OSCD model → **no model dropdown** on this tab.
- **HONESTY (maintainer directive — state PLAINLY, no dressing up):** label **"Sentinel-2 · 10 m"**, note
  it's a **coarser domain** than the aerial track, that these are **curated real-world examples**, and
  surface each pair's **acquisition dates + cloud %**. The aerial LEVIR-CD track stays the high-accuracy
  showcase. (The OSCD model card already states this framing; the tab must too.)
- Basemap: lean self-hosted/minimal to keep the app's no-CDN-at-runtime ethos (revisit at Gate 2).

**Plan doc (local, uncommitted):** `docs/superpowers/plans/2026-07-05-m5-curated-sentinel2.md`.

## Working conventions
- **Smoke before full** (CPU serial or `boost_qos_dbg`). **PAUSE and ask the maintainer before the first
  full training submission — and (M4+) before modifying/redeploying/making public the HF Space or pushing
  weight bundles to a public HF Model repo** (outward-facing, hard to reverse). **M5 Gate 2 specifically:
  the maintainer eyeballs the new Sentinel-2 tab on a PRIVATE preview before it goes live.**
- Checkpoint every ~30 min + `--resume-if-exists` so a walltime cut never loses progress.
- Commit granularly (maintainer identity, no AI/tool attribution); keep `DECISIONS.md` +
  `experiments/LOG.md` current; confirm CI is green after each push.
- **When blocked, or when an assumption is load-bearing, ask the maintainer** — they relay questions to a
  senior reviewer and return answers. Don't guess on irreversible or costly choices.
