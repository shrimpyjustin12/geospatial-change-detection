"""Reusable evaluation-harness logic (PRD §8), kept torch-free so it is unit-tested in CI.

The pixel-level work in ``evaluate.py`` reduces each split to two probability histograms (one for
change pixels, one for no-change pixels) plus per-scene TP/FP/FN counts. Everything downstream —
the precision/recall curve, the justified operating point, the per-scene mean+-std breakdown, the
failure-cause bucketing, and the comparison table — is pure arithmetic on those reductions and
lives here.
"""

from __future__ import annotations

import math
from typing import Any

from src.metrics import prf1_iou

# --- failure-cause heuristic thresholds (PRD §8 gallery buckets). Deliberately simple and
# documented; they *tag* worst-case tiles by a likely cause, they do not claim to be a classifier.
_SMALL_GT_FRAC = 0.01  # change occupies <1% of the tile -> small / subtle
_EXTREME_FRAC = 0.30  # >=30% near-saturated or near-black pixels -> cloud / shadow
_ABSDIFF_HIGH = 0.30  # large global before/after intensity gap -> seasonal / illumination
_MODERATE_GT_FRAC = 0.05
_BOUNDARY_HIGH = 0.55  # >=55% of error pixels hug the GT boundary -> registration misalignment
_MIN_PRED_FRAC = 0.001  # a "misalignment" needs a real (mislocated) prediction, not a blank tile

CAUSE_BUCKETS = (
    "registration_misalignment",
    "cloud_shadow",
    "seasonal_illumination",
    "small_subtle",
    "other",
)


def _suffix_sums(hist: list[int]) -> list[int]:
    """``out[k] = sum(hist[k:])`` for k in ``0..len(hist)`` (out has len+1 entries)."""
    out = [0] * (len(hist) + 1)
    running = 0
    for i in range(len(hist) - 1, -1, -1):
        running += hist[i]
        out[i] = running
    return out


def pr_curve_from_hist(pos_hist: list[int], neg_hist: list[int]) -> list[dict[str, float]]:
    """Precision/recall/F1/IoU at every bin-edge threshold from change/no-change histograms.

    ``pos_hist[j]`` / ``neg_hist[j]`` count change / no-change pixels whose predicted probability
    fell in bin ``j`` of ``nbins`` equal-width bins over [0, 1]. A pixel is predicted *change* at
    threshold ``t = k / nbins`` when its bin index is ``>= k``. Returns one entry per threshold
    ``k/nbins`` for ``k`` in ``0..nbins`` (ascending threshold).
    """
    if len(pos_hist) != len(neg_hist):
        raise ValueError("pos_hist and neg_hist must have equal length")
    nbins = len(pos_hist)
    pos_suffix = _suffix_sums(pos_hist)
    neg_suffix = _suffix_sums(neg_hist)
    total_pos = pos_suffix[0]
    curve: list[dict[str, float]] = []
    for k in range(nbins + 1):
        tp = pos_suffix[k] if k < len(pos_suffix) else 0
        fp = neg_suffix[k] if k < len(neg_suffix) else 0
        fn = total_pos - tp
        entry = {"threshold": k / nbins, **prf1_iou(tp, fp, fn)}
        curve.append(entry)
    return curve


def best_f1_operating_point(curve: list[dict[str, float]]) -> dict[str, float]:
    """The curve entry maximising F1 (tie-break: higher recall, then lower threshold).

    Justifies the reported operating point as an explicit precision/recall trade-off rather than a
    default 0.5 (PRD §8).
    """
    if not curve:
        raise ValueError("empty PR curve")
    return max(curve, key=lambda e: (e["f1"], e["recall"], -e["threshold"]))


def average_precision(curve: list[dict[str, float]]) -> float:
    """Average precision = sum of (delta recall) * precision as the threshold descends.

    ``curve`` is ascending in threshold, so iterating it in reverse walks the threshold down and
    recall up monotonically (a lower threshold predicts more positives). This is the standard AP
    (sklearn's ``average_precision_score`` convention), robust to the repeated-recall plateaus a
    near-perfectly-separated classifier produces.
    """
    ap = 0.0
    prev_recall = 0.0
    for e in reversed(curve):
        ap += (e["recall"] - prev_recall) * e["precision"]
        prev_recall = e["recall"]
    return ap


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Population mean and standard deviation (numpy-free)."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def summarize_per_scene(scene_counts: list[tuple[int, int, int]]) -> dict[str, Any]:
    """Per-scene change-class metrics reduced to mean+-std (and min/max) across scenes.

    ``scene_counts`` is one ``(tp, fp, fn)`` triple per test scene. Reporting variance across
    scenes (PRD §8) exposes whether a headline F1 is uniform or driven by a few easy scenes.
    """
    per_scene = [prf1_iou(tp, fp, fn) for tp, fp, fn in scene_counts]
    f1s = [m["f1"] for m in per_scene]
    ious = [m["iou"] for m in per_scene]
    f1_mean, f1_std = _mean_std(f1s)
    iou_mean, iou_std = _mean_std(ious)
    return {
        "n_scenes": len(scene_counts),
        "f1_mean": f1_mean,
        "f1_std": f1_std,
        "f1_min": min(f1s) if f1s else 0.0,
        "f1_max": max(f1s) if f1s else 0.0,
        "iou_mean": iou_mean,
        "iou_std": iou_std,
        "per_scene": per_scene,
    }


def classify_failure(stats: dict[str, float]) -> str:
    """Tag a worst-case tile with a likely failure cause (PRD §8 gallery buckets).

    ``stats`` keys (all in [0, 1] over the tile):
      ``gt_pos_frac`` / ``pred_pos_frac`` fraction of GT / predicted change pixels;
      ``img_absdiff_mean`` mean |A-B| over RGB; ``bright_frac`` / ``dark_frac`` near-saturated /
      near-black pixel fraction; ``boundary_error_ratio`` share of error pixels within a few px of
      the GT boundary. Checked most-specific first; falls through to ``other``.
    """
    gt = stats.get("gt_pos_frac", 0.0)
    # registration = a prediction exists but is spatially offset (errors hug the GT boundary). A
    # (near-)blank prediction is a *miss*, not a misalignment, so it must not claim this bucket.
    if (
        stats.get("pred_pos_frac", 0.0) >= _MIN_PRED_FRAC
        and stats.get("boundary_error_ratio", 0.0) >= _BOUNDARY_HIGH
    ):
        return "registration_misalignment"
    if (
        stats.get("bright_frac", 0.0) >= _EXTREME_FRAC
        or stats.get("dark_frac", 0.0) >= _EXTREME_FRAC
    ):
        return "cloud_shadow"
    if stats.get("img_absdiff_mean", 0.0) >= _ABSDIFF_HIGH and gt < _MODERATE_GT_FRAC:
        return "seasonal_illumination"
    if 0.0 < gt <= _SMALL_GT_FRAC:
        return "small_subtle"
    return "other"


def render_comparison_markdown(rows: list[dict[str, Any]]) -> str:
    """Markdown comparison table over models evaluated on the identical split (PRD §8).

    Each row needs: ``name``, ``trainable_params``, ``threshold``, ``precision``, ``recall``,
    ``f1``, ``iou``. Trainable params are rendered in millions.
    """
    header = (
        "| Model | Trainable params | Threshold | Precision | Recall | F1 | IoU |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in rows:
        params_m = f"{r['trainable_params'] / 1e6:.2f}M"
        lines.append(
            f"| {r['name']} | {params_m} | {r['threshold']:.3f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | "
            f"**{r['f1']:.4f}** | {r['iou']:.4f} |"
        )
    return "\n".join(lines)
