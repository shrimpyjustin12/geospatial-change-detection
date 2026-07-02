"""Tabulate several trained models on the *identical* split via the identical harness (PRD §8).

Drives both headline artifacts:
  * the Track-A tier comparison (baseline vs strong vs FM), and
  * ablations (e.g. difference vs concatenation fusion),
which are just different manifests. Each model is evaluated with ``evaluate.evaluate_model`` (so
every model's threshold is selected the same way, on val), then the operating-point metrics are
rendered into a markdown + JSON table.

    python -m src.compare --manifest configs/compare_levircd.yaml

Manifest (YAML)::

    split: test
    select_split: val
    models:
      - name: FC-Siam-diff (baseline)
        config: configs/levircd_baseline.yaml
      - name: Siamese-SegFormer (diff)
        config: configs/levircd_segformer.yaml
      - name: Siamese-SegFormer (concat)
        config: configs/levircd_segformer.yaml
        set: ["run_id=levircd_segformer_concat", "model.fusion=concat"]
        checkpoint: /path/to/best.pt   # optional override
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.config import expand_env, load_config
from src.eval_harness import render_comparison_markdown
from src.evaluate import evaluate_model


def run_comparison(manifest_path: str, out_dir: Path | None = None) -> dict[str, Any]:
    """Evaluate every model in the manifest and emit a comparison table."""
    manifest = expand_env(load_config(manifest_path))
    split = str(manifest.get("split", "test"))
    select_split = str(manifest.get("select_split", "val"))
    title = str(manifest.get("title", "LEVIR-CD comparison"))

    rows: list[dict[str, Any]] = []
    for entry in manifest["models"]:
        cfg = expand_env(load_config(entry["config"], entry.get("set", [])))
        summary = evaluate_model(
            cfg,
            split=split,
            select_split=select_split,
            checkpoint=entry.get("checkpoint"),
        )
        op = summary["operating_point"]
        rows.append(
            {
                "name": entry["name"],
                "run_id": summary["run_id"],
                "trainable_params": summary["trainable_params"],
                "threshold": op["threshold"],
                "precision": op["precision"],
                "recall": op["recall"],
                "f1": op["f1"],
                "iou": op["iou"],
                "average_precision": summary["average_precision"],
            }
        )
        print(
            f"[compare] {entry['name']}: F1={op['f1']:.4f} IoU={op['iou']:.4f} "
            f"thr={op['threshold']:.3f} params={summary['trainable_params'] / 1e6:.2f}M"
        )

    table = render_comparison_markdown(rows)
    out_dir = out_dir or Path(manifest.get("out_dir", "results/comparison"))
    out_dir.mkdir(parents=True, exist_ok=True)
    subtitle = f"Split: `{split}` · threshold selected on `{select_split}` (max-F1)."
    md = f"# {title}\n\n{subtitle}\n\n{table}\n"
    (out_dir / f"{split}.md").write_text(md)
    (out_dir / f"{split}.json").write_text(
        json.dumps({"title": title, "split": split, "rows": rows}, indent=2)
    )
    print(f"\n{table}\n")
    print(f"[compare] wrote {out_dir / (split + '.md')}")
    return {"title": title, "split": split, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_comparison(args.manifest, Path(args.out_dir) if args.out_dir else None)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # dodge the GPU-less interpreter-teardown hang on Leonardo (leonardo.md)


if __name__ == "__main__":
    main()
