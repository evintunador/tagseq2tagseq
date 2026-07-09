"""Regression test for the val-loss deflation bug.

The distributed reduction in ``_eval_loss`` used to all-reduce each rank's
*mean* loss and divide by the summed count, yielding
``(Σ_rank mean) / (Σ_rank count) = mean / val_steps`` — a val loss deflated by
roughly ``val_steps``. The fix reduces the summed loss and count separately, so
the result is the true token-batch-weighted global mean
``(Σ_rank total) / (Σ_rank count)``.

These tests run real gloo (CPU) process groups so the ``dist.all_reduce`` path
is actually exercised, with *unequal* per-rank losses and counts — the regime
where the old and new formulas diverge most. The single-process val wiring of
the compiled loops is covered by the ``__specific_tests__`` in
``tunalab/train_loops/test_multi_val{,_bucketed}.py``, which smart_train's
compilation harness runs but cannot exercise multi-process.
"""
import os

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from tunalab.train_loops.multi_val import _eval_loss as _eval_loss_multi_val
from tunalab.train_loops.multi_val_bucketed import _eval_loss as _eval_loss_bucketed


class _ConstLossModel(nn.Module):
    """Model whose forward returns a preset per-batch loss, ignoring input."""

    def __init__(self, losses):
        super().__init__()
        # A real parameter so ``next(model.parameters()).device`` works (CPU).
        self.p = nn.Parameter(torch.zeros(1))
        self._losses = list(losses)
        self._i = 0

    def forward(self, batch):
        loss = torch.tensor(self._losses[self._i], dtype=torch.float32)
        self._i += 1
        return loss


# Per-rank loss lists: rank 0 sees 2 batches, rank 1 sees 4 — deliberately
# unequal counts so a count-weighted mean differs from a mean-of-means, and so
# the buggy ``mean / count`` formula is far from correct.
_RANK_LOSSES = {
    0: [4.0, 2.0],              # rank 0: total=6, count=2
    1: [1.0, 3.0, 5.0, 3.0],   # rank 1: total=12, count=4
}
# Correct global token-weighted mean: (6 + 12) / (2 + 4) = 18 / 6 = 3.0
_EXPECTED_GLOBAL_MEAN = 3.0


def _worker(rank, world_size, fn_name, return_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        fn = {
            "multi_val": _eval_loss_multi_val,
            "bucketed": _eval_loss_bucketed,
        }[fn_name]
        model = _ConstLossModel(_RANK_LOSSES[rank])
        loader = [object()] * len(_RANK_LOSSES[rank])  # forward ignores batch
        result = fn(model, loader)
        return_queue.put((rank, result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fn_name", ["multi_val", "bucketed"])
def test_eval_loss_distributed_reduction(fn_name):
    """Every rank returns the true count-weighted global mean, not mean/count."""
    world_size = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(r, world_size, fn_name, q))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in range(world_size):
        rank, val = q.get(timeout=60)
        results[rank] = val
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    # All ranks agree, and agree on the correct global mean.
    for rank in range(world_size):
        assert results[rank] == pytest.approx(_EXPECTED_GLOBAL_MEAN, abs=1e-9), (
            f"rank {rank} got {results[rank]}, expected {_EXPECTED_GLOBAL_MEAN}"
        )

    # Guard against the historical bug: the deflated formula
    # (Σ mean)/(Σ count) would give (3.0 + 3.0)/6 = 1.0 here, not 3.0.
    buggy_value = (
        sum(sum(v) / len(v) for v in _RANK_LOSSES.values())
        / sum(len(v) for v in _RANK_LOSSES.values())
    )
    assert not any(
        results[r] == pytest.approx(buggy_value, abs=1e-9) for r in range(world_size)
    ), "reduction still matches the deflated mean/count formula"


def test_eval_loss_single_process_is_plain_mean():
    """Without a process group, result is the plain per-rank mean."""
    losses = [4.0, 2.0, 3.0]
    model = _ConstLossModel(losses)
    loader = [object()] * len(losses)
    assert not (dist.is_available() and dist.is_initialized())
    assert _eval_loss_multi_val(model, loader) == pytest.approx(3.0)
