"""Configuration loading and CLI-override utilities.

Kept dependency-light: the pure-dict helpers (`deep_merge`, `parse_overrides`) have no
third-party imports, and PyYAML is imported lazily inside `load_config` so the rest of the
module is usable (and unit-testable) even where PyYAML is not installed.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins).

    Nested dicts are merged key-by-key; any non-dict value replaces the base value.
    ``base`` is never mutated.
    """
    result = deepcopy(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def _coerce(value: str) -> Any:
    """Coerce a string CLI value to bool/int/float/None where unambiguous."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse ``a.b.c=value`` strings into a nested dict with light type coercion."""
    result: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not in key=value form")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Override '{item}' has an empty key")
        parts = key.split(".")
        node: dict[str, Any] = result
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Override key '{key}' conflicts with a scalar value")
            node = child
        node[parts[-1]] = _coerce(raw)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config file and apply optional ``key=value`` overrides."""
    import yaml  # lazy: keeps this module importable without PyYAML

    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config: dict[str, Any] = dict(loaded)
    if overrides:
        config = deep_merge(config, parse_overrides(overrides))
    return config
