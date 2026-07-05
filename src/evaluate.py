"""Full evaluation harness (PRD §8): change-class metrics with a justified operating point,
per-scene variance, and an auto-generated failure-case gallery.

Each split is reduced to change/no-change probability histograms + per-scene TP/FP/FN, then all
arithmetic is deferred to ``src.eval_harness`` (torch-free, unit-tested). The operating threshold
is selected on ``--select-split`` (val by default) by maximising F1, then applied to ``--split``
(test) so the reported number is not tuned on the test set. Outputs land in
``results/<run_id>/eval_<split>/``.

    python -m src.evaluate --config configs/levircd_segformer.yaml --split test

Legacy fixed-threshold behaviour is preserved via ``--threshold`` (skips val selection).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import expand_env, load_config
from src.data import build_dataset
from src.eval_harness import (
    average_precision,
    best_f1_operating_point,
    classify_failure,
    pr_curve_from_hist,
    summarize_per_scene,
)
from src.metrics import prf1_iou
from src.models import build_model

NBINS = 256
GALLERY_K = 12
_BOUNDARY_KERNEL = 5  # ~2px band around the GT boundary for the registration heuristic


def _denormalize(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet standardization -> RGB in [0, 1]. ``img`` shaped ``(3, H, W)``."""
    return (img * std + mean).clamp(0.0, 1.0)


def _boundary_error_ratio(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Share of error pixels (pred XOR gt) lying within a few px of the GT boundary.

    A high ratio means the model is essentially right but mislocated at object edges — the classic
    signature of registration misalignment (or annotation-edge noise). ``pred``/``gt``: ``(1,H,W)``.
    """
    err = pred ^ gt
    total = int(err.sum())
    if total == 0:
        return 0.0
    gt_f = gt.float().unsqueeze(0)  # (1,1,H,W)
    k, p = _BOUNDARY_KERNEL, _BOUNDARY_KERNEL // 2
    dil = F.max_pool2d(gt_f, k, stride=1, padding=p)
    ero = -F.max_pool2d(-gt_f, k, stride=1, padding=p)
    band = ((dil - ero) > 0).squeeze(0)  # (1,H,W)
    return int((err & band).sum()) / total


def _tile_stats(
    a: torch.Tensor, b: torch.Tensor, pred: torch.Tensor, gt: torch.Tensor
) -> dict[str, float]:
    """Cause-bucket features for one tile: a/b denormalized (3,H,W); masks (1,H,W)."""
    lum_a = a.mean(dim=0)
    lum_b = b.mean(dim=0)
    bright = ((lum_a > 0.9) | (lum_b > 0.9)).float().mean()
    dark = ((lum_a < 0.1) | (lum_b < 0.1)).float().mean()
    return {
        "gt_pos_frac": float(gt.float().mean()),
        "pred_pos_frac": float(pred.float().mean()),
        "img_absdiff_mean": float((a - b).abs().mean()),
        "bright_frac": float(bright),
        "dark_frac": float(dark),
        "boundary_error_ratio": _boundary_error_ratio(pred, gt),
    }


@torch.no_grad()
def _run_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    *,
    threshold: float | None,
    per_image: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    gallery_k: int,
) -> dict[str, Any]:
    """One pass over a split. Always builds the PR histograms; when ``threshold`` is given also
    accumulates per-scene counts at that threshold and collects the worst tiles for the gallery.
    """
    model.eval()
    pos_hist = torch.zeros(NBINS, dtype=torch.int64, device=device)
    neg_hist = torch.zeros(NBINS, dtype=torch.int64, device=device)
    scene_counts: dict[int, list[int]] = {}
    gallery: list[dict[str, Any]] = []
    worst_kept_f1 = float("inf")
    gidx = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = model(images)
        else:
            logits = model(images)
        probs = torch.sigmoid(logits.float())  # (B,1,H,W)
        target = masks.bool()

        bins = torch.clamp((probs * NBINS).long(), 0, NBINS - 1)
        pos_hist += torch.bincount(bins[target].flatten(), minlength=NBINS)
        neg_hist += torch.bincount(bins[~target].flatten(), minlength=NBINS)

        if threshold is not None:
            pred = probs >= threshold
            tp = (pred & target).sum(dim=(1, 2, 3))
            fp = (pred & ~target).sum(dim=(1, 2, 3))
            fn = (~pred & target).sum(dim=(1, 2, 3))
            gt_pos = target.sum(dim=(1, 2, 3))
            for i in range(images.shape[0]):
                scene = (gidx + i) // per_image
                acc = scene_counts.setdefault(scene, [0, 0, 0])
                acc[0] += int(tp[i])
                acc[1] += int(fp[i])
                acc[2] += int(fn[i])
                if gallery_k > 0 and int(gt_pos[i]) > 0:
                    m = prf1_iou(int(tp[i]), int(fp[i]), int(fn[i]))
                    if len(gallery) < gallery_k or m["f1"] < worst_kept_f1:
                        a = _denormalize(images[i, 0].cpu(), mean, std)
                        b = _denormalize(images[i, 1].cpu(), mean, std)
                        p1 = pred[i].cpu()
                        g1 = target[i].cpu()
                        stats = _tile_stats(a, b, p1, g1)
                        gallery.append(
                            {
                                "f1": m["f1"],
                                "a": a,
                                "b": b,
                                "pred": p1,
                                "gt": g1,
                                "bucket": classify_failure(stats),
                                "stats": stats,
                            }
                        )
                        gallery.sort(key=lambda e: e["f1"])
                        del gallery[gallery_k:]
                        worst_kept_f1 = gallery[-1]["f1"]
        gidx += images.shape[0]

    return {
        "pos_hist": pos_hist.cpu().tolist(),
        "neg_hist": neg_hist.cpu().tolist(),
        "scene_counts": [tuple(v) for v in scene_counts.values()],
        "gallery": gallery,
    }


def _save_pr_curve(curve: list[dict[str, float]], op: dict[str, float], path: Path) -> bool:
    """Precision-recall plot with the selected operating point marked. Returns False if no mpl."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    recall = [e["recall"] for e in curve]
    precision = [e["precision"] for e in curve]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, "-", color="#1f77b4", lw=1.5, label="PR curve")
    ax.plot(
        op["recall"],
        op["precision"],
        "o",
        color="#d62728",
        label=f"op@{op['threshold']:.3f} (F1={op['f1']:.3f})",
    )
    ax.set_xlabel("Recall (change class)")
    ax.set_ylabel("Precision (change class)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("LEVIR-CD change-class PR curve")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def _save_gallery(gallery: list[dict[str, Any]], path: Path) -> bool:
    """Grid of worst tiles: before / after / prediction / ground-truth, tagged by cause bucket."""
    if not gallery:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = len(gallery)
    fig, axes = plt.subplots(rows, 4, figsize=(9, 2.3 * rows))
    if rows == 1:
        axes = axes.reshape(1, 4)
    col_titles = ["Before (A)", "After (B)", "Prediction", "Ground truth"]
    for r, item in enumerate(gallery):
        panels = [
            item["a"].permute(1, 2, 0).numpy(),
            item["b"].permute(1, 2, 0).numpy(),
            item["pred"].squeeze(0).numpy(),
            item["gt"].squeeze(0).numpy(),
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap=None if c < 2 else "gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=9)
        axes[r, 0].set_ylabel(f"F1={item['f1']:.2f}\n{item['bucket']}", fontsize=8)
    fig.suptitle("Failure-case gallery (worst change tiles)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def _make_loader(cfg: dict[str, Any], split: str) -> tuple[DataLoader, Any]:
    dcfg = cfg["data"]
    dataset = build_dataset(dcfg, split=split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("eval", {}).get("batch_size", cfg["train"]["batch_size"])),
        shuffle=False,  # SequentialSampler -> stable tile order for per-scene grouping
        num_workers=int(dcfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset


def evaluate_model(
    cfg: dict[str, Any],
    *,
    split: str = "test",
    select_split: str = "val",
    checkpoint: str | None = None,
    threshold_override: float | None = None,
    gallery_k: int = GALLERY_K,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Full harness for one model; returns the summary dict (also written to ``out_dir``)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    autocast_dtype = torch.bfloat16 if use_cuda else None

    run_dir = Path(cfg["logging"]["log_dir"]) / cfg["run_id"]
    ckpt_path = Path(checkpoint) if checkpoint else run_dir / "checkpoints" / "best.pt"
    state = torch.load(ckpt_path, map_location=device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(state["model"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- select the operating threshold on select_split (unless explicitly overridden)
    if threshold_override is not None:
        op_threshold = threshold_override
        selection: dict[str, Any] = {"method": "fixed", "threshold": op_threshold}
    else:
        sel_loader, sel_ds = _make_loader(cfg, select_split)
        sel = _run_pass(
            model,
            sel_loader,
            device,
            autocast_dtype,
            threshold=None,
            per_image=sel_ds.per_image,
            mean=sel_ds.mean,
            std=sel_ds.std,
            gallery_k=0,
        )
        sel_curve = pr_curve_from_hist(sel["pos_hist"], sel["neg_hist"])
        sel_op = best_f1_operating_point(sel_curve)
        op_threshold = sel_op["threshold"]
        selection = {
            "method": "max_f1",
            "selected_on": select_split,
            "threshold": op_threshold,
            "val_f1": sel_op["f1"],
            "val_precision": sel_op["precision"],
            "val_recall": sel_op["recall"],
        }

    # --- report on split at the selected threshold; build test PR curve + per-scene + gallery
    loader, dataset = _make_loader(cfg, split)
    res = _run_pass(
        model,
        loader,
        device,
        autocast_dtype,
        threshold=op_threshold,
        per_image=dataset.per_image,
        mean=dataset.mean,
        std=dataset.std,
        gallery_k=gallery_k,
    )
    curve = pr_curve_from_hist(res["pos_hist"], res["neg_hist"])
    op_on_curve = min(curve, key=lambda e: abs(e["threshold"] - op_threshold))
    ref_05 = min(curve, key=lambda e: abs(e["threshold"] - 0.5))
    per_scene = summarize_per_scene(res["scene_counts"])
    buckets: dict[str, int] = {}
    for item in res["gallery"]:
        buckets[item["bucket"]] = buckets.get(item["bucket"], 0) + 1

    out_dir = out_dir or (run_dir / f"eval_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pr_png = out_dir / "pr_curve.png"
    gallery_png = out_dir / "gallery.png"
    has_pr = _save_pr_curve(curve, op_on_curve, pr_png)
    has_gallery = _save_gallery(res["gallery"], gallery_png)
    (out_dir / "pr_curve.json").write_text(json.dumps(curve, indent=2))

    summary = {
        "run_id": cfg["run_id"],
        "model": cfg["model"].get("name"),
        "encoder": cfg["model"].get("encoder"),
        "fusion": cfg["model"].get("fusion"),
        "split": split,
        "checkpoint": str(ckpt_path),
        "epoch": int(state.get("epoch", -1)),
        "trainable_params": int(trainable),
        "threshold_selection": selection,
        "operating_point": {
            "threshold": op_on_curve["threshold"],
            "precision": op_on_curve["precision"],
            "recall": op_on_curve["recall"],
            "f1": op_on_curve["f1"],
            "iou": op_on_curve["iou"],
        },
        "reference_0p5": {
            "precision": ref_05["precision"],
            "recall": ref_05["recall"],
            "f1": ref_05["f1"],
            "iou": ref_05["iou"],
        },
        "average_precision": average_precision(curve),
        "per_scene": {k: v for k, v in per_scene.items() if k != "per_scene"},
        "failure_buckets": buckets,
        "artifacts": {
            "pr_curve": pr_png.name if has_pr else None,
            "gallery": gallery_png.name if has_gallery else None,
            "pr_curve_json": "pr_curve.json",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--select-split", default="val", help="split used to pick the threshold")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--threshold", type=float, default=None, help="fixed threshold; skips selection"
    )
    parser.add_argument("--gallery-k", type=int, default=GALLERY_K)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()

    cfg = expand_env(load_config(args.config, args.set))
    summary = evaluate_model(
        cfg,
        split=args.split,
        select_split=args.select_split,
        checkpoint=args.checkpoint,
        threshold_override=args.threshold,
        gallery_k=args.gallery_k,
    )
    op = summary["operating_point"]
    ps = summary["per_scene"]
    print(
        f"[eval] {summary['run_id']} split={summary['split']} "
        f"thr={op['threshold']:.3f} P={op['precision']:.4f} R={op['recall']:.4f} "
        f"F1={op['f1']:.4f} IoU={op['iou']:.4f} | AP={summary['average_precision']:.4f} | "
        f"per-scene F1={ps['f1_mean']:.4f}+-{ps['f1_std']:.4f} (n={ps['n_scenes']})"
    )
    print(
        f"[eval] wrote {Path(cfg['logging']['log_dir']) / cfg['run_id'] / ('eval_' + args.split)}"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # dodge the GPU-less interpreter-teardown hang on Leonardo (leonardo.md)


if __name__ == "__main__":
    main()
