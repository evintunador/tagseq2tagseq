"""
Specific tests for the bucket_state_checkpoint atomic feature.
Registered in __specific_tests__ so smart_train's compilation pipeline
can validate them.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


# ---------------------------------------------------------------------------
# Minimal helpers
# ---------------------------------------------------------------------------

class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, batch):
        x = batch.float()
        return self.linear(x).mean()


class _SimpleDataset(IterableDataset):
    def __init__(self, n=20):
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            yield torch.randn(4)


@dataclass
class _FakeState:
    epoch_idx: int
    global_accum_step: int
    bucket_consumed: dict


# ---------------------------------------------------------------------------
# Specific tests
# ---------------------------------------------------------------------------

def test_bucket_state_saved_in_checkpoint(run_training_fn, device):
    """bucket_state_fn result is stored under metadata['bucket_state'] in the checkpoint."""
    state = _FakeState(epoch_idx=0, global_accum_step=7, bucket_consumed={0: 4, 1: 2})

    model = _SimpleModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(_SimpleDataset(20), batch_size=None)
    val_loader = DataLoader(_SimpleDataset(5), batch_size=None)

    with tempfile.TemporaryDirectory() as tmp:
        run_training_fn(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            save_best_model=True,
            output_dir=tmp,
            val_loader=val_loader,
            val_interval=5,
            bucket_state_fn=lambda: state,
        )
        ckpt_path = os.path.join(tmp, "checkpoints", "best_model.pt")
        assert os.path.exists(ckpt_path), "Checkpoint file not written"
        ckpt = torch.load(ckpt_path, weights_only=False)
        meta = ckpt.get("metadata", {})
        assert "bucket_state" in meta, (
            f"bucket_state not found in checkpoint metadata. Keys: {list(meta.keys())}"
        )
        bs = meta["bucket_state"]
        assert bs["epoch_idx"] == 0
        assert bs["global_accum_step"] == 7
        assert bs["bucket_consumed"] == {0: 4, 1: 2}


def test_no_bucket_state_fn_backward_compat(run_training_fn, device):
    """Without bucket_state_fn the checkpoint saves normally (no bucket_state key)."""
    model = _SimpleModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = DataLoader(_SimpleDataset(20), batch_size=None)
    val_loader = DataLoader(_SimpleDataset(5), batch_size=None)

    with tempfile.TemporaryDirectory() as tmp:
        run_training_fn(
            model=model,
            optimizer=optimizer,
            train_loader=loader,
            save_best_model=True,
            output_dir=tmp,
            val_loader=val_loader,
            val_interval=5,
        )
        ckpt_path = os.path.join(tmp, "checkpoints", "best_model.pt")
        assert os.path.exists(ckpt_path), "Checkpoint file not written"
        ckpt = torch.load(ckpt_path, weights_only=False)
        meta = ckpt.get("metadata", {})
        assert "bucket_state" not in meta, (
            "bucket_state unexpectedly present when bucket_state_fn was not provided"
        )


__specific_tests__ = [
    test_bucket_state_saved_in_checkpoint,
    test_no_bucket_state_fn_backward_compat,
]
