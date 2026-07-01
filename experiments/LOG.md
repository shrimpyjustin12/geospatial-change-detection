# Experiments Log

Submission + run trail (PRD §7.3). One row per notable action — smoke/full submissions, job
IDs, config, git SHA, outcome. Keep it human-readable; never silently re-run.

| Date (UTC) | Milestone | Action | Job/Run ID | Config | Git SHA | Outcome |
|---|---|---|---|---|---|---|
| 2026-07-01 | M0 | Repo skeleton + tooling; pushed to GitHub | Actions 28545571645 | — | 96abb23 | CI success |
| 2026-07-01 | M0 | Stage LEVIR-CD (login node, Xet) + md5/sha256 verify | — | stage_data.sh | 96abb23 | 637 pairs; md5 OK |
| 2026-07-01 | M0 | torchgeo LEVIRCD load smoke | — | smoke_load_levircd.py | 96abb23 | train=445 val=64 test=128 OK |
