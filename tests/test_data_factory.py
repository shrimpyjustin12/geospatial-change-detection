"""build_dataset factory dispatch (src.data). Pure dispatch logic — the loaders are stubbed, so
this needs no torch/torchgeo/rasterio and runs anywhere (incl. the light CI env)."""

import sys
import types

import pytest

from src.data import build_dataset


def _install_fake_loader(monkeypatch, module_name: str, attr: str) -> dict:
    """Register a fake loader module so build_dataset's lazy ``from ... import`` resolves to it."""
    captured: dict = {}

    class _Fake:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    mod = types.ModuleType(module_name)
    setattr(mod, attr, _Fake)
    monkeypatch.setitem(sys.modules, module_name, mod)
    return captured


def test_dispatches_levircd(monkeypatch):
    captured = _install_fake_loader(monkeypatch, "src.data.levircd", "TiledLEVIRCD")
    build_dataset({"name": "levircd", "root": "/d", "tile_size": 128}, split="val", augment=False)
    assert captured == {"root": "/d", "split": "val", "tile_size": 128, "augment": False}


def test_dispatches_oscd(monkeypatch):
    captured = _install_fake_loader(monkeypatch, "src.data.oscd", "TiledOSCD")
    build_dataset({"name": "oscd", "root": "/o"}, split="train", augment=True)
    assert captured == {"root": "/o", "split": "train", "tile_size": 256, "augment": True}


def test_defaults_to_levircd(monkeypatch):
    captured = _install_fake_loader(monkeypatch, "src.data.levircd", "TiledLEVIRCD")
    build_dataset({"root": "/d"}, split="test", augment=False)
    assert captured["root"] == "/d"


def test_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown dataset"):
        build_dataset({"name": "nope", "root": "/x"}, split="train", augment=False)
