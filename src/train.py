"""Config-driven DDP training for change detection (PRD §6.4, §7; leonardo.md).

CPU smoke:    python -m src.train --config configs/levircd_baseline_smoke.yaml
4-GPU DDP:    srun ... python -m src.train --config configs/levircd_baseline.yaml

DDP engages automatically when the launcher sets world_size > 1. Checkpoints are rank-correct
(saved by rank 0, loaded by all ranks); ``--resume-if-exists`` auto-resumes the latest so a
walltime cut never loses progress. Effective batch size and the scaled LR are logged to the
run manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from src import dist as D
from src.config import expand_env, load_config
from src.data import build_dataset
from src.losses import BceDiceLoss
from src.metrics import prf1_iou
from src.models import build_model
from src.utils import scale_lr, set_determinism, warmup_factor

_SHOULD_STOP = False


def _install_signal_handlers() -> None:
    def handler(signum: int, frame: FrameType | None) -> None:
        global _SHOULD_STOP
        _SHOULD_STOP = True

    for sig in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(sig, handler)


def resolve_git_sha(repo_root: Path) -> str:
    """Best-effort git SHA: env GIT_SHA -> REVISION file -> git -> 'unknown'."""
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"]
    revision = repo_root / "REVISION"
    if revision.exists():
        return revision.read_text().strip()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        )
        return out.strip()
    except Exception:
        return "unknown"


def build_loader(
    dataset: Any,
    batch_size: int,
    world_size: int,
    rank: int,
    *,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
) -> tuple[DataLoader, Any]:
    """Build a DataLoader with a DistributedSampler under DDP, else a plain sampler."""
    sampler: Any
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )
    else:
        sampler = RandomSampler(dataset) if shuffle else None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )
    return loader, sampler


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically save on rank 0 only (write temp then rename)."""
    if not D.is_main():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.rename(path)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    autocast_dtype: torch.dtype | None,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Validation: accumulate change-class TP/FP/FN, all-reduce across ranks, then P/R/F1/IoU."""
    model.eval()
    counts = torch.zeros(3, dtype=torch.float64, device=device)  # tp, fp, fn
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                logits = model(images)
        else:
            logits = model(images)
        pred = torch.sigmoid(logits.float()) >= threshold
        target = masks.bool()
        counts[0] += (pred & target).sum()
        counts[1] += (pred & ~target).sum()
        counts[2] += (~pred & target).sum()
    D.all_reduce_sum(counts)
    tp, fp, fn = (int(counts[0]), int(counts[1]), int(counts[2]))
    return prf1_iou(tp, fp, fn)


def build_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int, kind: str
) -> torch.optim.lr_scheduler.LRScheduler:
    """Warmup + cosine (or constant) multiplier on top of the scaled base LR."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return warmup_factor(step, warmup_steps)
        if kind == "cosine" and total_steps > warmup_steps:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-if-exists", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg = expand_env(load_config(args.config, args.set))

    rank, world_size, local_rank = D.setup_distributed()
    seed = int(cfg.get("seed", 1337))
    set_determinism(seed)  # same seed on all ranks BEFORE model init (leonardo.md)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    autocast_dtype = torch.bfloat16 if (use_cuda and cfg["train"].get("amp", True)) else None

    tcfg = cfg["train"]
    dcfg = cfg["data"]
    per_gpu_batch = int(tcfg["batch_size"])
    effective_batch = per_gpu_batch * world_size
    reference_batch = int(tcfg.get("lr_reference_batch", per_gpu_batch))
    lr = scale_lr(float(tcfg["lr"]), effective_batch, reference_batch)

    # ---- data
    train_split = dcfg.get("split_for_smoke", "train")
    augment = bool(dcfg.get("augment", train_split == "train"))
    train_ds = build_dataset(dcfg, split=train_split, augment=augment)
    val_ds = build_dataset(dcfg, split="val", augment=False)
    num_workers = int(dcfg.get("num_workers", 4))
    train_loader, train_sampler = build_loader(
        train_ds,
        per_gpu_batch,
        world_size,
        rank,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader, _ = build_loader(
        val_ds,
        per_gpu_batch,
        world_size,
        rank,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    # ---- model / optim / sched / loss
    model = build_model(cfg["model"]).to(device)
    ckpt_path = Path(cfg["logging"]["log_dir"]) / cfg["run_id"] / "checkpoints" / "last.pt"
    best_path = ckpt_path.with_name("best.pt")

    resume_state: dict[str, Any] | None = None
    if (args.resume_if_exists or tcfg.get("resume_if_exists")) and ckpt_path.exists():
        resume_state = torch.load(ckpt_path, map_location=device)  # load on ALL ranks
        model.load_state_dict(resume_state["model"])

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if use_cuda else None,
            find_unused_parameters=bool(tcfg.get("ddp_find_unused_parameters", False)),
        )

    lossfn = BceDiceLoss(
        bce_weight=float(cfg["loss"].get("bce_weight", 1.0)),
        dice_weight=float(cfg["loss"].get("dice_weight", 1.0)),
    ).to(device)
    # Optimize only params that require grad: identical to the full set for M1/M2 (nothing frozen),
    # but for the FM tier this excludes the frozen DINOv2 backbone so AdamW only tracks the LoRA
    # adapters + decoder (the trainable-param count the comparison table reports).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=float(tcfg.get("weight_decay", 0.0))
    )

    epochs = int(tcfg.get("epochs", 1))
    max_steps = tcfg.get("max_steps")
    limit_batches = tcfg.get("limit_batches")
    steps_per_epoch = (
        len(train_loader) if limit_batches is None else min(len(train_loader), int(limit_batches))
    )
    total_steps = int(max_steps) if max_steps is not None else epochs * steps_per_epoch
    warmup_steps = int(tcfg.get("warmup_steps", max(1, round(0.05 * total_steps))))
    scheduler = build_scheduler(
        optimizer, total_steps, warmup_steps, str(tcfg.get("scheduler", "cosine"))
    )

    start_epoch, global_step, best_f1 = 0, 0, -1.0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state["epoch"])
        global_step = int(resume_state["global_step"])
        best_f1 = float(resume_state.get("best_f1", -1.0))

    # ---- manifest + tensorboard (rank 0)
    writer: Any = None
    if D.is_main():
        run_dir = Path(cfg["logging"]["log_dir"]) / cfg["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_id": cfg["run_id"],
            "git_sha": resolve_git_sha(repo_root),
            "world_size": world_size,
            "per_gpu_batch": per_gpu_batch,
            "effective_batch": effective_batch,
            "base_lr": float(tcfg["lr"]),
            "scaled_lr": lr,
            "seed": seed,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "device": str(device),
            "amp_dtype": str(autocast_dtype),
            "config": cfg,
            "env": {k: os.environ.get(k, "") for k in ("SLURM_JOB_ID", "SLURM_NTASKS", "HOSTNAME")},
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(run_dir))
        except Exception as exc:  # TensorBoard optional — training proceeds without it
            print(f"[train] TensorBoard unavailable ({exc}); continuing without TB", flush=True)
        print(
            f"[train] run={cfg['run_id']} world={world_size} eff_batch={effective_batch} "
            f"lr={lr:.2e} total_steps={total_steps} device={device}",
            flush=True,
        )

    _install_signal_handlers()
    ckpt_every = float(tcfg.get("ckpt_every_min", 30)) * 60.0
    last_ckpt_t = time.monotonic()

    def snapshot(epoch: int) -> dict[str, Any]:
        base = (
            model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        )
        return {
            "model": base.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_f1": best_f1,
            "effective_batch": effective_batch,
            "scaled_lr": lr,
        }

    # ---- training loop
    stop = False
    for epoch in range(start_epoch, epochs):
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)  # rank-correct shuffling per epoch
        model.train()
        for i, batch in enumerate(train_loader):
            if limit_batches is not None and i >= int(limit_batches):
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if autocast_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    loss = lossfn(model(images), masks)
            else:
                loss = lossfn(model(images), masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            if D.is_main() and writer is not None and global_step % 10 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                print(
                    f"[train] epoch={epoch} step={global_step} loss={loss.item():.4f}", flush=True
                )

            if time.monotonic() - last_ckpt_t > ckpt_every:
                D.barrier()
                save_checkpoint(ckpt_path, snapshot(epoch))
                last_ckpt_t = time.monotonic()

            if _SHOULD_STOP or (max_steps is not None and global_step >= int(max_steps)):
                stop = True
                break
        # end epoch: validate + checkpoint
        metrics = evaluate(
            model,
            val_loader,
            device,
            float(cfg["eval"].get("threshold", 0.5)),
            autocast_dtype,
            max_batches=int(limit_batches) if limit_batches is not None else None,
        )
        if D.is_main() and writer is not None:
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, global_step)
            print(f"[train] epoch={epoch} val={metrics}", flush=True)
        D.barrier()
        save_checkpoint(ckpt_path, snapshot(epoch + 1))
        last_ckpt_t = time.monotonic()
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_checkpoint(best_path, snapshot(epoch + 1))
        if stop:
            break

    if D.is_main():
        print(f"[train] done. best_f1={best_f1:.4f} stopped_by_signal={_SHOULD_STOP}", flush=True)
        if writer is not None:
            writer.close()
    D.barrier()
    D.cleanup()
    sys.stdout.flush()
    if _SHOULD_STOP:
        sys.exit(0)  # let SLURM requeue via the sbatch


if __name__ == "__main__":
    main()
