"""Atomic feature (ts2-local): multi_val + BucketedPackDataset resume state.

A strict superset of the ``multi_val`` feature: identical multi-loader
validation and best-model checkpointing, plus persistence of the
BucketedPackDataset schedule position so a precomputed-epoch run resumes at the
exact pack it left off.

Why a separate feature (not a kwarg on ``multi_val``)
-----------------------------------------------------
smart_train selects a feature only when ALL its declared kwargs are present, and
its dropped-kwarg guard warns about any kwarg no *selected* feature declares.  So
``bucket_state_fn`` must be a DECLARED kwarg of the selected feature — it cannot
ride ``multi_val`` as an undeclared passenger (that would trip a false "silently
inactive" warning) nor be added to ``multi_val``'s signature (that would stop
non-bucketed runs from selecting ``multi_val`` at all).  A superset feature is
selected only when ``bucket_state_fn`` is also supplied, and subsumes plain
``multi_val`` via smart_train's strict-subset rule; non-bucketed runs are
unaffected.

Bucket knowledge lives here in ts2.  tunalab stays generic: the bucket position
is stored as a lazy callable under ``metadata["bucket_state"]`` and resolved at
save time by ``checkpointer.save_checkpoint`` (which calls any callable metadata
value), so the persisted position reflects the actual checkpoint moment rather
than when the training loop was assembled.

Kwargs:
    val_loaders (dict[str, DataLoader]):  mapping of name -> loader.
    val_interval (int):                   steps between validation passes.
    save_best_model (bool):               save on new best mean val loss.
    output_dir (str):                     checkpoint directory (required when
                                          save_best_model=True).
    bucket_state_fn (callable):           zero-arg callable -> object with
                                          ``__dict__`` (BucketedPackDataset
                                          .get_state); its dict is embedded under
                                          metadata["bucket_state"] for exact resume.
"""
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

import tunalab.checkpointer as checkpointer
from tunalab.distributed import is_main_process, cpu_barrier


@torch.no_grad()
def _eval_loss(model: nn.Module, loader) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        loss = model(batch)
        total += float(loss.detach().cpu().item())
        count += 1
    if was_training:
        model.train()

    if dist.is_available() and dist.is_initialized():
        # Reduce summed loss and count, then divide once: (Σ total) / (Σ count).
        # Reducing per-rank means would give mean/val_steps (deflation bug).
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cuda")
        t = torch.tensor([total, float(count)], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t[0].item() / max(t[1].item(), 1.0)

    return total / max(count, 1)


def _full_optimizer_state(optimizer):
    """Collective (ALL ranks must call): world-size-portable, name-keyed
    optimizer state via ``optimizer.state_dict_full()``, or ``None`` for a
    plain optimizer that lacks it.

    The distributed Muon optimizer shards its momentum across ranks, so a
    rank-0-only ``state_dict()`` captures just a fraction keyed by positional
    index — neither complete nor portable.  ``state_dict_full`` runs an
    all_gather to union every rank's shard into one name-keyed dict, so it MUST
    be invoked by every rank together (outside the ``is_main_process`` guard) or
    the non-rank-0 processes will deadlock on the missing collective.
    """
    fn = getattr(optimizer, "state_dict_full", None)
    if fn is None:
        return None
    return fn()


def _save_ckpt_full(raw_model, optimizer, metadata, filepath, bucket_state_fn=None):
    """Save a checkpoint whose optimizer state resumes EXACTLY at any world_size.

    All ranks first compute the collective full optimizer state; rank 0 then
    embeds it (already resolved) under ``metadata['optimizer_state']`` and writes
    the file.  The model is passed to the checkpointer as a stateful object; the
    optimizer is NOT passed as a stateful object because its positional
    ``state_dict()`` is redundant with — and only a fraction of — the full state.
    """
    full_state = _full_optimizer_state(optimizer)  # collective — every rank
    if is_main_process():
        md: Dict[str, Any] = dict(metadata)
        if full_state is not None:
            md["optimizer_state"] = full_state
        if bucket_state_fn is not None:
            # Lazy: checkpointer resolves the dataset position at save time.
            md["bucket_state"] = lambda: bucket_state_fn().__dict__
        checkpointer.save_checkpoint(filepath=filepath, metadata=md, model=raw_model)
    # All ranks wait for rank 0 to finish writing before proceeding.
    cpu_barrier()


def _save_best(raw_model, optimizer, val_loss, step, output_dir, bucket_state_fn, config):
    """Save best checkpoint (full, portable optimizer state) on new best val loss."""
    _save_ckpt_full(
        raw_model, optimizer,
        {"val_loss": val_loss, "step": step, "config": config},
        os.path.join(output_dir, "checkpoints", "best_model.pt"),
        bucket_state_fn=bucket_state_fn,
    )


def _save_latest(raw_model, optimizer, val_loss, step, output_dir, bucket_state_fn, config):
    """Overwrite the single ``latest.pt`` regardless of val — so a kill loses at
    most ``save_latest_interval`` steps, not everything since the last val gain."""
    _save_ckpt_full(
        raw_model, optimizer,
        {"val_loss": val_loss, "step": step, "config": config},
        os.path.join(output_dir, "checkpoints", "latest.pt"),
        bucket_state_fn=bucket_state_fn,
    )


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    val_loaders: Optional[Dict[str, Any]] = None,
    val_interval: int = 10,
    save_best_model: bool = False,
    output_dir: Optional[str] = None,
    bucket_state_fn: Optional[Any] = None,
    save_latest_interval: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """multi_val training loop that also persists BucketedPackDataset position.

    Behaves exactly like ``multi_val``; additionally, when ``bucket_state_fn`` is
    provided, the dataset schedule position is embedded in every best-model
    checkpoint for exact resume.

    When ``save_latest_interval`` is set, also overwrites a single ``latest.pt``
    every that-many optimizer steps (independent of val), so preemption/kill
    loses at most ``save_latest_interval`` steps rather than everything since the
    last val improvement.  Both best_model.pt and latest.pt carry the full,
    world-size-portable optimizer state for exact Muon+AdamW resume.
    """
    model.train()

    if not val_loaders:
        val_loaders = {}

    if save_best_model and (not val_loaders or output_dir is None):
        raise ValueError(
            "val_loaders and output_dir must be provided when save_best_model=True."
        )
    # 0 / None both mean "disabled" (main.py always passes an int, since this is
    # a declared kwarg required for feature selection).
    if save_latest_interval and output_dir is None:
        raise ValueError("output_dir must be provided when save_latest_interval is set.")

    histories: Dict[str, List[float]] = {name: [] for name in val_loaders}
    best_val_loss = float("inf")
    config = kwargs.get("config", {})
    last_val_mean = float("nan")  # most recent mean val loss, stamped into latest.pt

    def _run_val(step: int) -> None:
        nonlocal best_val_loss, last_val_mean
        losses = {}
        for name, loader in val_loaders.items():
            losses[name] = _eval_loss(model, loader)
            histories[name].append(losses[name])

        if not losses:
            return

        mean_loss = sum(losses.values()) / len(losses)
        last_val_mean = mean_loss
        if save_best_model and mean_loss < best_val_loss:
            best_val_loss = mean_loss
            raw_model = model.module if hasattr(model, "module") else model
            _save_best(raw_model, optimizer, mean_loss, step, output_dir,
                       bucket_state_fn, config)

    step_count = 0
    for batch in train_loader:
        loss = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if val_loaders and step_count > 0 and step_count % val_interval == 0:
            _run_val(step_count)

        # Periodic latest.pt (collective — every rank calls _save_latest).
        if (save_latest_interval and step_count > 0
                and step_count % save_latest_interval == 0):
            raw_model = model.module if hasattr(model, "module") else model
            _save_latest(raw_model, optimizer, last_val_mean, step_count,
                         output_dir, bucket_state_fn, config)

        step_count += 1

    # Final validation pass if not already run at the last step.
    if val_loaders and (step_count == 0 or step_count % val_interval != 0):
        _run_val(step_count)

    # Final latest.pt so the very end of a run is always recoverable — skipped if
    # the last step already landed on a save_latest_interval boundary (avoids an
    # immediate redundant multi-GB write + collective gather).
    if save_latest_interval and step_count % save_latest_interval != 0:
        raw_model = model.module if hasattr(model, "module") else model
        _save_latest(raw_model, optimizer, last_val_mean, step_count,
                     output_dir, bucket_state_fn, config)

    result: Dict[str, Any] = {"model": model}
    for name, hist in histories.items():
        result[f"val_loss_history_{name}"] = hist
    if histories:
        latest = [h[-1] for h in histories.values() if h]
        if latest:
            result["val_loss"] = sum(latest) / len(latest)

    return result
