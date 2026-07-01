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
