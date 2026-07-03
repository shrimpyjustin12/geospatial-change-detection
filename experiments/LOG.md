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
| 2026-07-02 | M2 | FULL SegFormer diff (4-GPU DDP, 200 ep) COMPLETED 2:01 | 48276349 | levircd_segformer.yaml | dd0971d | val best F1=0.9142 (ep 181) |
| 2026-07-02 | M2 | FULL SegFormer concat (4-GPU DDP, 200 ep) COMPLETED 2:00 | 48276351 | levircd_segformer.yaml --set fusion=concat | dd0971d | val best F1=0.9084 |
| 2026-07-02 | M2 | Harness eval both strong models + tier/ablation tables | 48293204 | evaluate + compare | 9bbedf4 | diff **F1=0.9106** IoU=0.836 AP=0.943; concat F1=0.9066 IoU=0.829 |
| 2026-07-02 | M2 | Tier comparison (test, thr-on-val): baseline vs diff vs concat | 48293204 | compare_levircd.yaml | 9bbedf4 | 0.886 / **0.911** / 0.907 F1 — M2 target met |
| 2026-07-02 | M3 | Install transformers 4.57.6 + peft 0.19.1 in .venv-train (login node) | — | pip | bdadabc | torch untouched; numpy→2.0.2, hub→0.36.2 |
| 2026-07-02 | M3 | Stage DINOv2 small/base (pinned commits) + offline-load verify | — | stage_weights.sh | bdadabc | ungated; 22.1M/86.6M; no net fallback |
| 2026-07-02 | M3 | Real-stack pytest (torch+transformers+peft+smp) | 48315172 | tests/ | bdadabc | 54 passed |
| 2026-07-02 | M3 | CPU logic smoke (dinov2-small, offline FM load) | 48315458 | levircd_dinov2_smoke | bdadabc | loop/eval/ckpt OK |
| 2026-07-02 | M3 | 4-GPU DDP smoke (dinov2-small, LoRA) | 48315460 | levircd_dinov2_smoke | bdadabc | COMPLETED; world=4 lr=4e-4 |
| 2026-07-02 | M3 | Single-GPU mem/speed probe (dinov2-base@448) | srun | levircd_dinov2.yaml | bdadabc | 89.4M/**2.82M** trainable; 1.0GB/64; 74ms/step |
| 2026-07-02 | M3 | Capped 4-GPU DDP full-config run — DDP+reentrant-ckpt unused-param error | 48316121 | levircd_dinov2 max_steps=15 | bdadabc | FAILED → fix use_reentrant=False |
| 2026-07-02 | M3 | Capped 4-GPU DDP full-config run (use_reentrant=False) | 48316662 | levircd_dinov2 max_steps=15 | bdadabc | COMPLETED 2:40; val F1=0.587@15steps |
| 2026-07-02 | M3 | Go-ahead received: launch LoRA + frozen linear-probe ablation (reviewer) | — | — | bdadabc | approved |
| 2026-07-02 | M3 | Frozen-probe capped 4-GPU DDP smoke (lora=false + grad-ckpt) | 48318779 | levircd_dinov2 --set model.lora=false max_steps=15 | bdadabc | COMPLETED 2:36; DDP OK |
| 2026-07-02 | M3 | Submit FULL DINOv2-base + LoRA (diff, 4-GPU DDP, 200 ep) | 48318927 | levircd_dinov2.yaml | bdadabc | submitted |
| 2026-07-02 | M3 | Submit FULL DINOv2-base FROZEN linear-probe (ablation, 4-GPU DDP, 200 ep) | 48318928 | levircd_dinov2 --set model.lora=false run_id=levircd_dinov2_frozen | bdadabc | submitted |
| 2026-07-03 | M3 | FULL DINOv2 + LoRA COMPLETED (2:50, no requeue, 200 ep) | 48318927 | levircd_dinov2.yaml | bdadabc | val best F1=0.9166 |
| 2026-07-03 | M3 | FULL DINOv2 frozen linear-probe COMPLETED (1:56, 200 ep) | 48318928 | levircd_dinov2_frozen | bdadabc | val best F1=0.8961 |
| 2026-07-03 | M3 | 4-tier comparison eval (test, thr-on-val→test) | 48333943 | compare_levircd.yaml | bdadabc | LoRA **F1=0.9125** IoU=0.839 AP=0.946; frozen 0.889/0.800/0.924 |
| 2026-07-03 | M3 | Tier table: baseline / SF-diff / DINOv2-frozen / DINOv2-LoRA | 48333943 | compare_levircd.yaml | bdadabc | 0.886 / 0.911 / 0.889 / **0.913** F1 — FM wins at **2.82M vs 24.72M** trainable |
