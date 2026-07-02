"""
Atomic feature (ts2-local): checkpoint_best_model + BucketedPackDataset resume.

A strict superset of ``checkpoint_best_model`` (single ``val_loader``): identical
best-val-loss checkpointing, plus persistence of the BucketedPackDataset schedule
position so a precomputed-epoch run resumes at the exact pack it left off.

This is the *single-val-loader* counterpart to ``multi_val_bucketed`` (plural
``val_loaders``, the production path).  Both keep bucket knowledge in ts2: the
position is stored as a lazy callable under ``metadata["bucket_state"]`` and
resolved at save time by ``checkpointer.save_checkpoint``, so the persisted value
reflects the actual checkpoint moment rather than when the loop was assembled.

For per-step timing use the standalone ``step_timer`` atomic feature — this
feature does checkpointing only (atomic features stay atomic).

Usage in main.py
----------------
    atomic_feature_kwargs['bucket_state_fn'] = dataset.get_state

Resume path (main.py)
---------------------
    ckpt = torch.load(resume_from, weights_only=False)
    bs = ckpt.get('metadata', {}).get('bucket_state')
    start_state = BucketState(**bs) if bs else None
    dataset = BucketedPackDataset(..., start_state=start_state)
"""

import os
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn

import tunalab.checkpointer as checkpointer
from tunalab.distributed import is_main_process, cpu_barrier
from tunalab.train_loops.checkpoint_best_model import _eval_loss


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    save_best_model: bool = False,
    output_dir: Optional[str] = None,
    val_loader=None,
    val_interval: int = 10,
    bucket_state_fn: Optional[Callable] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Best-val-loss checkpointing that also persists BucketedPackDataset position.

    Args:
        model:            nn.Module; forward(batch) returns scalar loss.
        optimizer:        PyTorch optimizer.
        train_loader:     Training data iterable.
        save_best_model:  Enable best-val-loss checkpoint saving.
        output_dir:       Directory for checkpoint files.
        val_loader:       Validation iterable (required when save_best_model=True).
        val_interval:     Steps between validation runs.
        bucket_state_fn:  Zero-arg callable → object with ``__dict__`` (typically
                          ``BucketedPackDataset.get_state``).  When provided, the
                          dataset position is embedded (lazily, resolved at save
                          time) in every saved checkpoint's metadata for exact
                          resume.
        **kwargs:         Forwarded to checkpoint metadata["config"] via "config".
    """
    model.train()
    best_val_loss = float("inf")
    result: Dict[str, Any] = {"model": model}
    optimizer.zero_grad(set_to_none=True)
    step_count = 0

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
                    # Lazy: checkpointer resolves the callable at save time so the
                    # persisted position reflects this exact checkpoint moment.
                    metadata["bucket_state"] = lambda: bucket_state_fn().__dict__
                checkpointer.save_checkpoint(
                    filepath=os.path.join(output_dir, "checkpoints", "best_model.pt"),
                    metadata=metadata,
                    model=raw_model,
                    optimizer=optimizer,
                )
            cpu_barrier()

    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if save_best_model and step_count % val_interval == 0:
            _maybe_save()

        step_count += 1

    # Final validation if the last step wasn't a val step.
    if save_best_model and step_count % val_interval != 0:
        _maybe_save()

    return result
