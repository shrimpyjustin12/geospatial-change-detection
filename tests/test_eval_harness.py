"""Unit tests for the torch-free evaluation-harness logic (PRD §8). No torch/numpy needed."""

from src.eval_harness import (
    average_precision,
    best_f1_operating_point,
    classify_failure,
    pr_curve_from_hist,
    render_comparison_markdown,
    summarize_per_scene,
)


def test_pr_curve_perfect_separation():
    # all change pixels in the top bin, all no-change in the bottom bin -> F1=1 achievable
    curve = pr_curve_from_hist(pos_hist=[0, 0, 0, 4], neg_hist=[4, 0, 0, 0])
    assert len(curve) == 5  # nbins + 1
    op = best_f1_operating_point(curve)
    assert op["f1"] == 1.0
    assert op["precision"] == 1.0
    assert op["recall"] == 1.0
    # tie among thresholds 0.25/0.50/0.75 -> tie-break picks the lowest threshold
    assert op["threshold"] == 0.25


def test_pr_curve_endpoints():
    curve = pr_curve_from_hist(pos_hist=[1, 1], neg_hist=[1, 1])
    lo, hi = curve[0], curve[-1]
    assert lo["threshold"] == 0.0 and lo["recall"] == 1.0  # predict everything positive
    assert hi["threshold"] == 1.0 and hi["recall"] == 0.0  # predict nothing positive


def test_pr_curve_length_mismatch_raises():
    try:
        pr_curve_from_hist([1, 2, 3], [1, 2])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched histogram lengths")


def test_average_precision_perfect_is_one():
    curve = pr_curve_from_hist(pos_hist=[0, 0, 0, 4], neg_hist=[4, 0, 0, 0])
    assert abs(average_precision(curve) - 1.0) < 1e-9


def test_average_precision_bounds():
    curve = pr_curve_from_hist(pos_hist=[2, 1, 1, 2], neg_hist=[3, 2, 2, 1])
    ap = average_precision(curve)
    assert 0.0 <= ap <= 1.0


def test_summarize_per_scene_mean_std():
    # two scenes: one perfect (F1=1), one all-miss (F1=0) -> mean 0.5, std 0.5
    summary = summarize_per_scene([(10, 0, 0), (0, 0, 10)])
    assert summary["n_scenes"] == 2
    assert abs(summary["f1_mean"] - 0.5) < 1e-9
    assert abs(summary["f1_std"] - 0.5) < 1e-9
    assert summary["f1_min"] == 0.0
    assert summary["f1_max"] == 1.0


def test_classify_failure_buckets():
    # registration needs a real (mislocated) prediction, not a blank tile
    assert (
        classify_failure({"boundary_error_ratio": 0.8, "pred_pos_frac": 0.05})
        == "registration_misalignment"
    )
    assert classify_failure({"bright_frac": 0.5}) == "cloud_shadow"
    assert classify_failure({"dark_frac": 0.5}) == "cloud_shadow"
    assert (
        classify_failure({"img_absdiff_mean": 0.5, "gt_pos_frac": 0.01}) == "seasonal_illumination"
    )
    assert classify_failure({"gt_pos_frac": 0.002}) == "small_subtle"
    assert classify_failure({"gt_pos_frac": 0.4}) == "other"


def test_classify_failure_blank_prediction_is_not_registration():
    # a small GT change the model missed entirely (blank prediction) -> small_subtle, NOT
    # registration, even though the tiny GT blob is mostly its own boundary
    stats = {"boundary_error_ratio": 0.95, "pred_pos_frac": 0.0, "gt_pos_frac": 0.003}
    assert classify_failure(stats) == "small_subtle"


def test_classify_failure_priority_boundary_wins():
    # boundary is checked first even when brightness would also trigger (given a real prediction)
    stats = {"boundary_error_ratio": 0.9, "bright_frac": 0.9, "pred_pos_frac": 0.05}
    assert classify_failure(stats) == "registration_misalignment"


def test_render_comparison_markdown():
    rows = [
        {
            "name": "baseline",
            "trainable_params": 1_100_000,
            "threshold": 0.5,
            "precision": 0.92,
            "recall": 0.85,
            "f1": 0.884,
            "iou": 0.793,
        }
    ]
    md = render_comparison_markdown(rows)
    assert md.startswith("| Model |")
    assert "baseline" in md
    assert "1.10M" in md
    assert "**0.8840**" in md
