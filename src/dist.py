"""Torch-distributed (DDP) helpers.

DDP engages automatically when the launcher provides a world size > 1 (``torchrun`` or SLURM
``srun``). One rank per GPU (leonardo.md). All helpers degrade to single-process no-ops so the
same code path runs on a laptop / login node without a process group.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def resolve_dist_env() -> tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)`` from torchrun or SLURM env (else 0,1,0)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (
            int(os.environ["RANK"]),
            int(os.environ["WORLD_SIZE"]),
            int(os.environ.get("LOCAL_RANK", 0)),
        )
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        return (
            int(os.environ["SLURM_PROCID"]),
            int(os.environ["SLURM_NTASKS"]),
            int(os.environ.get("SLURM_LOCALID", 0)),
        )
    return 0, 1, 0


def setup_distributed() -> tuple[int, int, int]:
    """Init the NCCL process group if world_size > 1. Returns ``(rank, world_size, local_rank)``."""
    rank, world_size, local_rank = resolve_dist_env()
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist():
        dist.barrier()


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce across ranks (no-op single-process)."""
    if is_dist():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def cleanup() -> None:
    if is_dist():
        dist.destroy_process_group()
