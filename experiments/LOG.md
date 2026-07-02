# Experiments Log

Submission + run trail (PRD §7.3). One row per notable action — smoke/full submissions, job
IDs, config, git SHA, outcome. Keep it human-readable; never silently re-run.

| Date (UTC) | Milestone | Action | Job/Run ID | Config | Git SHA | Outcome |
|---|---|---|---|---|---|---|
| 2026-07-01 | M0 | Repo skeleton + tooling; pushed to GitHub | Actions 28545571645 | — | 96abb23 | CI success |
| 2026-07-01 | M0 | Stage LEVIR-CD (login node, Xet) + md5/sha256 verify | — | stage_data.sh | 96abb23 | 637 pairs; md5 OK |
| 2026-07-01 | M0 | torchgeo LEVIRCD load smoke | — | smoke_load_levircd.py | 96abb23 | train=445 val=64 test=128 OK |
| 2026-07-01 | M1 | Unit tests (model/loss/metric) on CPU torch | — | tests/ | ffc3395 | 22 pass |
| 2026-07-01 | M1 | CPU logic-smoke (serial partition) | srun | levircd_baseline_smoke | ffc3395 | loop/eval/ckpt OK |
| 2026-07-01 | M1 | 4-GPU DDP smoke (boost_qos_dbg) | 48233882 | levircd_baseline_smoke | 5a80faf | COMPLETED 2:03; world=4 eff_batch=8 lr=4e-3; val+ckpt+TB OK |
| 2026-07-01 | M1 | CI green (ruff+mypy+25 pytest); fixed .gitignore hiding src/data | Actions | — | d745ffa | success |
| 2026-07-02 | M1 | Submit FULL baseline (4-GPU DDP, boost_usr_prod, 100 ep) | 48236696 | levircd_baseline.yaml | d745ffa | submitted |
| 2026-07-02 | M1 | FULL baseline COMPLETED (55 min, no requeue) | 48236696 | levircd_baseline.yaml | d745ffa | val F1=0.877, best=0.887 |
| 2026-07-02 | M1 | Test eval (change class, thr=0.5) | srun serial | eval --split test | af46744 | P=0.920 R=0.851 **F1=0.884** IoU=0.793 |
| 2026-07-02 | M2 | Stage smp MiT-b0/b2 ImageNet weights (login node, pinned revs) | — | stage_weights.sh | 489c647 | offline load OK, fell_back=False |
| 2026-07-02 | M2 | Validation job: CPU segformer smoke + baseline harness eval | 48269993 | serial | 489c647 | COMPLETED; smoke OK |
| 2026-07-02 | M2 | Full eval harness on M1 baseline (val-selected thr) | 48269993 | evaluate levircd_baseline | 489c647 | thr=0.148 **F1=0.886** IoU=0.796 AP=0.932; per-scene 0.734±0.314 |
| 2026-07-02 | M2 | Real-stack pytest (torch+smp) | — | tests/ | 489c647 | 28 passed |
| 2026-07-02 | M2 | 4-GPU DDP SegFormer smoke (boost_qos_dbg) | 48271942 | levircd_segformer_smoke | 489c647 | COMPLETED 2:40; world=4 lr=1.6e-4; val+ckpt OK |
