"""Tests for change-class metric math (torch-free — always runs in CI)."""

from src.metrics import prf1_iou


def test_prf1_iou_perfect():
    assert prf1_iou(tp=10, fp=0, fn=0) == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "iou": 1.0,
    }


def test_prf1_iou_empty_counts_are_zero_not_nan():
    assert prf1_iou(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}


def test_prf1_iou_mixed():
    m = prf1_iou(tp=5, fp=5, fn=10)
    assert m["precision"] == 0.5
    assert abs(m["recall"] - 5 / 15) < 1e-9
    assert abs(m["iou"] - 5 / 20) < 1e-9
    expected_f1 = 2 * 0.5 * (5 / 15) / (0.5 + 5 / 15)
    assert abs(m["f1"] - expected_f1) < 1e-9
