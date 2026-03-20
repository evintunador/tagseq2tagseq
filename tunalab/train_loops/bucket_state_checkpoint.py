"""
Atomic feature: checkpoint_best_model + bucket_state persistence + optional
per-step timing.

Extends checkpoint_best_model by:

  1. Writing the current BucketedPackDataset position into every checkpoint's
     metadata dict under ``"bucket_state"``, enabling exact resume.

  2. Optionally measuring per-step wall-clock time broken down into:
       backward_s  — fwd+bwd including DDP allreduce overlap
       wait_s      — stall at dist.barrier() waiting for slower ranks
       total_s     — backward_s + wait_s
     Results are printed per-step (no rolling average) and saved to
     ``step_timing_rank{N}.csv`` in output_dir.

When ``step_timing_all_ranks=True`` is passed alongside ``bucket_state_fn``,
smart_train selects this single feature rather than trying to LLM-compile a
combination of bucket_state_checkpoint + step_timer.  This is the intended
usage: the step_timer feature continues to exist for cases where bucket_state
checkpointing is not needed.

Usage in main.py
----------------
    atomic_feature_kwargs['bucket_state_fn'] = dataset.get_state
    # Also add step_timing_all_ranks=True to enable timing (optional).

Resume path (main.py)
---------------------
    ckpt = torch.load(resume_from, weights_only=False)
    bs = ckpt.get('metadata', {}).get('bucket_state')
    start_state = BucketState(**bs) if bs else None
    dataset = BucketedPackDataset(..., start_state=start_state)
"""

import csv
import os
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

import tunalab.checkpointer as checkpointer
from tunalab.distributed import is_main_process, cpu_barrier
from tunalab.train_loops.checkpoint_best_model import _eval_loss


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    # ---- checkpoint_best_model kwargs ----------------------------------------
    save_best_model: bool = False,
    output_dir: Optional[str] = None,
    val_loader=None,
    val_interval: int = 10,
    # ---- bucket_state extension ----------------------------------------------
    bucket_state_fn: Optional[Callable] = None,
    # ---- per-step timing (step_timer interface) ------------------------------
    step_timing_all_ranks: bool = False,
    step_timing_csv: bool = True,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, Any]:
    """Training loop with bucket_state checkpointing and optional step timing.

    Args:
        model:                  nn.Module; forward(batch) returns scalar loss.
        optimizer:              PyTorch optimizer.
        train_loader:           Training data iterable.
        save_best_model:        Enable best-val-loss checkpoint saving.
        output_dir:             Directory for checkpoint and timing CSV files.
        val_loader:             Validation iterable (required when save_best_model).
        val_interval:           Steps between validation runs.
        bucket_state_fn:        Zero-arg callable → object with ``__dict__``
                                (typically ``BucketedPackDataset.get_state``).
                                When provided, its result is embedded in every
                                saved checkpoint's metadata for exact resume.
        step_timing_all_ranks:  If True, every DDP rank prints per-step timing
                                and writes its own CSV.  If False, only rank 0.
                                Setting this to True selects this combined feature
                                instead of triggering LLM compilation.
        step_timing_csv:        Write per-step timings to step_timing_rank{N}.csv.
        device:                 Device string used for cuda.synchronize().
        **kwargs:               Forwarded to checkpoint metadata["config"].
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    _dev = torch.device(device)
    timing_enabled = step_timing_all_ranks or rank == 0

    model.train()
    best_val_loss = float("inf")
    result: Dict[str, Any] = {"model": model}
    optimizer.zero_grad(set_to_none=True)
    step_count = 0

    # ---- timing CSV setup ---------------------------------------------------
    csv_file = None
    csv_writer = None
    if step_timing_csv and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"step_timing_rank{rank}.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["step", "total_s", "backward_s", "wait_s", "it_per_s", "loss"]
        )

    if timing_enabled:
        print(
            f"\n{'step':>6}  {'total_s':>8}  {'bwd_s':>7}  {'wait_s':>7}  "
            f"{'it/s':>7}  loss",
            flush=True,
        )

    # ---- checkpoint helper --------------------------------------------------
    def _maybe_save():
        nonlocal best_val_loss
        if not save_best_model:
            return
        if val_loader is None or output_dir is None:
            raise ValueError(
                "val_loader and output_dir must be provided when save_best_model=True."
            )
        val_loss = _eval_loss(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            result["best_val_loss"] = best_val_loss
            raw_model = model.module if hasattr(model, "module") else model
            if is_main_process():
                metadata: Dict[str, Any] = {
                    "val_loss": val_loss,
                    "step": step_count,
                    "config": kwargs.get("config", {}),
                }
                if bucket_state_fn is not None:
                    metadata["bucket_state"] = bucket_state_fn().__dict__
                checkpointer.save_checkpoint(
                    filepath=os.path.join(output_dir, "checkpoints", "best_model.pt"),
                    metadata=metadata,
                    model=raw_model,
                    optimizer=optimizer,
                )
            cpu_barrier()

    # ---- training loop ------------------------------------------------------
    for batch in train_loader:
        t_start = time.perf_counter()

        loss = model(batch)
        loss.backward()

        # Sync after backward: waits for this rank's CUDA work + DDP allreduce
        torch.cuda.synchronize(_dev)
        t_after_bwd = time.perf_counter()

        # Barrier: waits until ALL ranks have completed allreduce.
        # Time spent here is stall from slower ranks (density imbalance on live
        # training; near-zero on density-bucketed precomputed training).
        if is_distributed:
            dist.barrier()
        t_after_barrier = time.perf_counter()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if save_best_model and step_count % val_interval == 0:
            _maybe_save()

        # ---- record timing --------------------------------------------------
        bwd_s = t_after_bwd - t_start
        wait_s = t_after_barrier - t_after_bwd
        total_s = t_after_barrier - t_start
        it_per_s = 1.0 / total_s if total_s > 0 else float("inf")
        loss_val = float(loss.detach().cpu().item())

        if csv_writer is not None:
            csv_writer.writerow(
                [step_count, f"{total_s:.4f}", f"{bwd_s:.4f}",
                 f"{wait_s:.4f}", f"{it_per_s:.3f}", f"{loss_val:.4f}"]
            )

        if timing_enabled:
            print(
                f"{step_count:>6}  {total_s:>8.3f}  {bwd_s:>7.3f}  "
                f"{wait_s:>7.3f}  {it_per_s:>7.2f}  {loss_val:.4f}",
                flush=True,
            )

        step_count += 1

    # Final validation if the last step wasn't a val step
    if save_best_model and step_count % val_interval != 0:
        _maybe_save()

    if csv_file is not None:
        csv_file.close()
        if timing_enabled:
            print(f"\nStep timing saved → {csv_path}", flush=True)

    return result
