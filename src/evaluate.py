"""Evaluate a trained checkpoint: change-class P/R/F1/IoU on a split (M1 acceptance).

The full harness (PR-curve threshold selection, per-scene breakdown, failure gallery,
ablations — PRD §8) lands in M2. Runs single-process on one GPU/CPU.

    python -m src.evaluate --config configs/levircd_baseline.yaml --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import expand_env, load_config
from src.data.levircd import TiledLEVIRCD
from src.metrics import ChangeMetrics
from src.models import build_model


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    """Change-class metrics over a split at a fixed probability threshold."""
    model.eval()
    metrics = ChangeMetrics()
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        pred = torch.sigmoid(model(images).float()) >= threshold
        metrics.update(pred, masks.bool())
    return metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--checkpoint", default=None, help="default: <log_dir>/<run_id>/checkpoints/best.pt"
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = expand_env(load_config(args.config, args.set))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    threshold = (
        args.threshold if args.threshold is not None else float(cfg["eval"].get("threshold", 0.5))
    )

    run_dir = Path(cfg["logging"]["log_dir"]) / cfg["run_id"]
    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    state = torch.load(ckpt_path, map_location=device)

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(state["model"])

    dcfg = cfg["data"]
    dataset = TiledLEVIRCD(
        root=dcfg["root"], split=args.split, tile_size=int(dcfg.get("tile_size", 256))
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    metrics = evaluate_split(model, loader, device, threshold)
    result = {
        "run_id": cfg["run_id"],
        "split": args.split,
        "checkpoint": str(ckpt_path),
        "threshold": threshold,
        "epoch": int(state.get("epoch", -1)),
        "metrics": metrics,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"eval_{args.split}.json"
    out.write_text(json.dumps(result, indent=2))
    print(
        f"[eval] {cfg['run_id']} split={args.split} thr={threshold} "
        f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} IoU={metrics['iou']:.4f}"
    )
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
