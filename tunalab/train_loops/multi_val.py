"""Atomic feature: multiple named validation loaders, evaluated periodically.

Replaces the single-loader ``val_loader`` / ``val_interval`` pattern when
multiple held-out sets need to be tracked simultaneously.  Each loader is
evaluated independently; losses are logged under its name.

Also handles best-model checkpointing (subsumes checkpoint_best_model when
val_loaders is used instead of val_loader).

Kwargs:
    val_loaders (dict[str, DataLoader]):  mapping of name → loader.
    val_interval (int):                   steps between validation passes.
    save_best_model (bool):               save checkpoint on new best mean val loss.
    output_dir (str):                     directory for checkpoint (required when
                                          save_best_model=True).
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from tunalab.train_loops.checkpoint_best_model import _save_best


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


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    val_loaders: Optional[Dict[str, Any]] = None,
    val_interval: int = 10,
    save_best_model: bool = False,
    output_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Training loop with multiple named validation loaders.

    Each entry in ``val_loaders`` is evaluated every ``val_interval`` steps.
    Loss histories are keyed as ``val_loss_history_{name}`` in the result.

    When ``save_best_model=True``, saves a checkpoint whenever the mean val
    loss across all loaders improves, using the same ``_save_best`` helper as
    ``checkpoint_best_model``.
    """
    model.train()

    if not val_loaders:
        val_loaders = {}

    if save_best_model and (not val_loaders or output_dir is None):
        raise ValueError(
            "val_loaders and output_dir must be provided when save_best_model=True."
        )

    histories: Dict[str, List[float]] = {name: [] for name in val_loaders}
    best_val_loss = float("inf")

    def _run_val(step: int) -> None:
        nonlocal best_val_loss
        losses = {}
        for name, loader in val_loaders.items():
            losses[name] = _eval_loss(model, loader)
            histories[name].append(losses[name])

        if not losses:
            return

        mean_loss = sum(losses.values()) / len(losses)
        if save_best_model and mean_loss < best_val_loss:
            best_val_loss = mean_loss
            raw_model = model.module if hasattr(model, "module") else model
            _save_best(raw_model, optimizer, mean_loss, step, output_dir, kwargs)

    step_count = 0
    for batch in train_loader:
        loss = model(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if val_loaders and step_count > 0 and step_count % val_interval == 0:
            _run_val(step_count)

        step_count += 1

    # Final validation pass if not already run at the last step.
    if val_loaders and (step_count == 0 or step_count % val_interval != 0):
        _run_val(step_count)

    result: Dict[str, Any] = {"model": model}
    for name, hist in histories.items():
        result[f"val_loss_history_{name}"] = hist
    if histories:
        latest = [h[-1] for h in histories.values() if h]
        if latest:
            result["val_loss"] = sum(latest) / len(latest)

    return result
