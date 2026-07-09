"""
Specific tests for the multi_val_bucketed atomic feature.
Registered in __specific_tests__ so smart_train's compilation pipeline can
validate them (signature: (run_training_fn, device)).

Single-process, so they guard the compiled loop's val wiring — each named
loader's loss is the mean per-batch loss, recorded in the history. The
distributed all_reduce math (the val-loss deflation bug) is covered separately
by tests/train_loops/test_eval_loss_reduction.py, which the single-process
compilation harness cannot exercise.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


class _FixedLossModel(nn.Module):
    """Forward returns a batch-derived loss independent of params, so the val
    loss is stable across training steps yet gradients still flow."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return batch.float().mean() + 0.0 * self.p.sum()


class _FixedDataset(IterableDataset):
    def __init__(self, rows, device):
        self._data = [torch.tensor(r, device=device) for r in rows]

    def __iter__(self):
        for x in self._data:
            yield x


def test_bucketed_multi_val_reports_mean_batch_loss(run_training_fn, device):
    """Each loader's reported val loss is the mean of its per-batch losses."""
    model = _FixedLossModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train = DataLoader(_FixedDataset([[0.0, 0.0]] * 4, device), batch_size=None)
    # Batch means 2.0 and 4.0 → expected val loss (2 + 4) / 2 = 3.0.
    val = DataLoader(_FixedDataset([[1.0, 3.0], [4.0, 4.0]], device), batch_size=None)

    out = run_training_fn(
        model=model,
        optimizer=optimizer,
        train_loader=train,
        val_loaders={"v": val},
        val_interval=1,
    )

    hist = out.get("val_loss_history_v")
    assert hist, "val_loss_history_v missing or empty"
    assert abs(hist[-1] - 3.0) < 1e-5, f"expected val loss 3.0, got {hist[-1]}"


__specific_tests__ = [
    test_bucketed_multi_val_reports_mean_batch_loss,
]
