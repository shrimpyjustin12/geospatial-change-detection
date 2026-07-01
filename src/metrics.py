"""Change-class metrics (PRD §8).

We report precision / recall / F1 / IoU **on the change class only** — never overall pixel
accuracy, which is ~99% for a trivial "no change" predictor and therefore useless.

The pure count->metric math (`prf1_iou`) has no torch dependency so it is unit-tested in CI
without the ML stack; `ChangeMetrics` accumulates TP/FP/FN from boolean tensors at eval time.
"""

from __future__ import annotations

from typing import Any


def prf1_iou(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall, F1 and IoU for the change class from TP/FP/FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


class ChangeMetrics:
    """Accumulate change-class TP/FP/FN over batches, then compute P/R/F1/IoU."""

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred: Any, target: Any) -> None:
        """``pred``/``target``: boolean tensors (change == True), any matching shape."""
        p = pred.bool()
        t = target.bool()
        self.tp += int((p & t).sum())
        self.fp += int((p & ~t).sum())
        self.fn += int((~p & t).sum())

    def compute(self) -> dict[str, float]:
        return prf1_iou(self.tp, self.fp, self.fn)

    def reset(self) -> None:
        self.tp = self.fp = self.fn = 0
