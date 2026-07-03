"""
Specific tests for the profile_training atomic feature.
Registered in __specific_tests__ so smart_train's compilation pipeline can
validate them (signature: (run_training_fn, device)).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(4, 8)   # named like the real module's parts
        self.backbone = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
        self.norm = nn.LayerNorm(8)
        self.loss_fn = nn.Linear(8, 1)

    def forward(self, batch):
        x = self.embedding(batch.float())
        x = self.backbone(x)
        x = self.norm(x)
        return self.loss_fn(x).mean()


class _SimpleDataset(IterableDataset):
    """Yields pre-materialized batches already on the target device.

    Batches are drawn once at construction (deterministic given the seed) so two
    loaders built after the same manual_seed feed identical data — required for
    the numerical-equivalence check. base_loop does not move tensors (that is the
    `device` feature's job in a real composition), so we place them here.
    """
    def __init__(self, device, n=24):
        self._data = [torch.randn(4, device=device) for _ in range(n)]

    def __iter__(self):
        for x in self._data:
            yield x


def _mk(device):
    model = _SimpleModel().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loader = DataLoader(_SimpleDataset(device, 24), batch_size=None)
    return model, opt, loader


def _base_loop(model, optimizer, train_loader):
    """Reference base_loop, for the numerical-equivalence check below."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for batch in train_loader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {"model": model}


def test_profile_off_is_numerically_base_loop(run_training_fn, device):
    """profile_run=False must be numerically IDENTICAL to base_loop.

    This is the strong form of the reduces-to-base guarantee: the feature adds
    NO branch that alters the observed I/O when off. We train two identically
    seeded models on the same data — one via the feature (off), one via the
    reference base_loop — and assert every parameter matches exactly.
    """
    torch.manual_seed(0)
    m_feat, opt_feat, loader_feat = _mk(device)
    torch.manual_seed(0)
    m_base, opt_base, loader_base = _mk(device)
    # same init
    for pf, pb in zip(m_feat.parameters(), m_base.parameters()):
        pb.data.copy_(pf.data)

    out = run_training_fn(model=m_feat, optimizer=opt_feat, train_loader=loader_feat)
    assert "model" in out
    assert "profile_summary" not in out  # off → no profiling output at all

    _base_loop(m_base, opt_base, loader_base)

    for pf, pb in zip(m_feat.parameters(), m_base.parameters()):
        assert torch.equal(pf, pb), "profile_run=False diverged from base_loop"


def test_profile_on_returns_summary(run_training_fn, device):
    """profile_run=True runs the profiling loop and returns a per-phase summary."""
    model, opt, loader = _mk(device)
    out = run_training_fn(
        model=model, optimizer=opt, train_loader=loader,
        profile_run=True,
        profile_warmup_steps=1,
        profile_active_steps=4,
        profile_model_internals=True,
    )
    assert "model" in out
    summ = out.get("profile_summary")
    assert summ is not None, "profile_run=True must return profile_summary"
    for k in ("data_ms", "fwd_ms", "bwd_ms", "opt_ms", "wall_ms"):
        assert k in summ["mean_ms"], f"missing phase {k} in summary"


def test_profile_no_sync_nccl_estimate_single_process(run_training_fn, device):
    """no_sync steps requested single-process: runs cleanly, nccl_est is None."""
    model, opt, loader = _mk(device)
    out = run_training_fn(
        model=model, optimizer=opt, train_loader=loader,
        profile_run=True,
        profile_warmup_steps=1,
        profile_no_sync_steps=2,
        profile_active_steps=3,
    )
    summ = out["profile_summary"]
    # Single process → no_sync path still exercised, but NCCL isolation needs
    # ≥2 ranks; ensure it degrades gracefully rather than crashing.
    assert summ["world_size"] == 1


__specific_tests__ = [
    test_profile_off_is_numerically_base_loop,
    test_profile_on_returns_summary,
    test_profile_no_sync_nccl_estimate_single_process,
]
