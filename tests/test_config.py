"""Tests for config loading and CLI overrides.

Also serves as the M0 CI smoke test: it exercises the tooling chain (ruff/mypy/pytest)
with pure-Python assertions that need no ML/geo dependencies.
"""

from pathlib import Path

import pytest

from src.config import deep_merge, load_config, parse_overrides


def test_deep_merge_overrides_nested():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 20}, "e": 4}
    assert deep_merge(base, override) == {"a": 1, "b": {"c": 20, "d": 3}, "e": 4}


def test_deep_merge_does_not_mutate_base():
    base = {"b": {"c": 2}}
    deep_merge(base, {"b": {"c": 99}})
    assert base == {"b": {"c": 2}}


def test_parse_overrides_nested_and_coercion():
    result = parse_overrides(
        ["train.lr=0.001", "train.epochs=10", "train.amp=true", "model.name=fc_siam_diff"]
    )
    assert result == {
        "train": {"lr": 0.001, "epochs": 10, "amp": True},
        "model": {"name": "fc_siam_diff"},
    }


def test_parse_overrides_rejects_malformed():
    with pytest.raises(ValueError):
        parse_overrides(["no_equals_sign"])


def test_parse_overrides_scalar_conflict():
    with pytest.raises(ValueError):
        parse_overrides(["train=1", "train.lr=0.1"])


def test_load_config_with_overrides(tmp_path: Path):
    pytest.importorskip("yaml")
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("seed: 42\ntrain:\n  lr: 0.01\n  epochs: 5\n", encoding="utf-8")
    cfg = load_config(cfg_file, overrides=["train.lr=0.1"])
    assert cfg["seed"] == 42
    assert cfg["train"]["lr"] == 0.1
    assert cfg["train"]["epochs"] == 5
