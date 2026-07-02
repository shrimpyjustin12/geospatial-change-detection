"""Small training helpers. Pure functions here are torch-free (unit-tested in CI)."""

from __future__ import annotations


def scale_lr(base_lr: float, effective_batch: int, reference_batch: int) -> float:
    """Linear LR scaling rule (Goyal et al. 2017): ``lr = base_lr * eff / ref``.

    With 4-GPU DDP the effective batch is ``per_gpu_batch * world_size``; scaling the base LR
    from the reference batch keeps the update magnitude comparable. Logged into the manifest.
    """
    if reference_batch <= 0:
        raise ValueError("reference_batch must be > 0")
    return base_lr * (effective_batch / reference_batch)


def warmup_factor(step: int, warmup_steps: int) -> float:
    """Linear warmup multiplier in ``(0, 1]``; returns 1.0 once past ``warmup_steps``."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def even_feature_layers(num_layers: int, k: int) -> list[int]:
    """``k`` evenly-spaced transformer hidden-state indices ending at the last layer.

    ``transformers`` hidden states have length ``num_layers + 1`` (index 0 is the embedding output,
    ``i`` the output of layer ``i``). Used by the DINOv2 FM tier to tap a pseudo-multi-scale set of
    ViT layers. Ending the schedule at ``num_layers`` guarantees every layer is upstream of a *read*
    output, so no LoRA/decoder parameter is left unused (which would otherwise trip DDP without
    ``find_unused_parameters``).
    """
    k = max(1, min(k, num_layers))
    idx = {min(num_layers, max(1, round(num_layers * (j + 1) / k))) for j in range(k)}
    idx.add(num_layers)
    return sorted(idx)


def set_determinism(seed: int) -> None:
    """Seed python / numpy / torch. Called on every rank BEFORE model init (leonardo.md)."""
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
