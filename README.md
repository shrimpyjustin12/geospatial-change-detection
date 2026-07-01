# Satellite Change Detection

Given two images of the same place at two times, produce a map of **what changed** — and
serve it through an interactive web demo. This is a portfolio project: the **evaluation
harness and reproducibility are first-class deliverables**, not afterthoughts.

> **Status:** M0 (setup) complete — CI green, LEVIR-CD staged. M1 (baseline on HPC) next.
> See [milestones](#milestones). This README is the
> front door and will grow with results tables, a failure gallery, and honest limitations
> as milestones land (final polish in M7).

## Two imagery tracks (they must not be mixed)

- **Track A — high-res aerial (0.5 m RGB):** LEVIR-CD (binary building change) + xBD (disaster
  damage, multi-class). Powers the curated demo.
- **Track B — Sentinel-2 (10 m multispectral):** OSCD. Powers the **live AOI** demo, which pulls
  fresh Sentinel-2 scenes at inference.

**Why the split matters (domain gap):** a model trained on 0.5 m aerial imagery does **not**
transfer to 10 m Sentinel-2 — resolution and spectral characteristics differ by more than an
order of magnitude. Running the LEVIR-CD model on live Sentinel-2 would produce meaningless
output. The live mode therefore uses a Sentinel-2-native model trained on OSCD. This is a
documented design decision, not a footnote.

## Architecture

```
 LEONARDO HPC (trains)                          HUGGING FACE (serves)
 login: stage data + FM weights (egress)        FastAPI + onnxruntime (CPU) :7860
   -> $SCRATCH/$WORK data                          |- /curated : Track-A ONNX, before/after
   -> SLURM (Singularity, GPU) -> checkpoints      |- /live-aoi: STAC -> Sentinel-2 ->
   -> evaluate.py -> results/ (metrics, PR,        |             Track-B ONNX -> overlay
      failure gallery)                           React + MapLibre (swipe slider, AOI draw)
   -> export.py -> artifact bundle  --push-->  HF Model repo --pull--> Space
```

The **artifact bundle** is the contract between the two surfaces: per model, a directory with
`model.onnx`, `config.yaml`, `preprocessing.json`, and `metrics_card.md`. The demo consumes
only the bundle — never the training code.

## Repository layout

```
configs/     one yaml per model + a smoke config each
src/         data/ · models/ · train.py · evaluate.py · export.py  (config-driven)
scripts/     stage_data.sh · stage_weights.sh   (login-node downloads + checksums)
slurm/       train.sbatch   (templated for Leonardo)
container/   changedet.def  (Singularity/Apptainer)
experiments/ LOG.md         (job id · config · git sha · outcome)
results/     metrics tables, PR curves, failure images (large binaries gitignored)
app/         HF Space: Dockerfile · backend (FastAPI) · frontend (React+MapLibre)
```

## Development

Local dev needs only the light tooling (heavy ML/geo deps run on Leonardo / the HF Space):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # ruff, mypy, pytest
ruff check . && ruff format --check . && mypy && pytest -q
```

CI (GitHub Actions, Python 3.11) runs exactly this on every push.

## Training on Leonardo (HPC)

Cluster-specific settings (allocation, partition, container image, torch build) are resolved
from `leonardo.md` and kept in local project notes (not committed). Data and pretrained weights
are pre-staged on the login node (compute nodes have **no internet egress**); training reads
only local storage. Always run a **smoke config** before any full submission.

## Milestones

| | Milestone | State |
|---|---|---|
| M0 | Setup: skeleton, CI, data staging, container draft | ✅ done |
| M1 | Baseline (FC-Siam-diff) end-to-end on HPC | todo |
| M2 | Strong model (Siamese-SegFormer) + full eval harness | todo |
| M3 | Foundation-model tier (DINOv2) + 3-model comparison | todo |
| M4 | ONNX export + curated HF Space | todo |
| M5 | Live Sentinel-2 AOI mode | todo |
| M6 | Disaster xBD multi-class track | todo |
| M7 | Polish: README, model cards, demo GIF | todo |

## License & data hygiene

Code: [MIT](LICENSE). **Trained weights** inherit the research/non-commercial terms of their
datasets (LEVIR-CD, xBD, OSCD) → showcase/demo use only. Datasets are **never committed** — see
`scripts/` for download scripts only.
