"""
Tests for world-size-portable, name-keyed optimizer-state resume
(optimizers/muon.py: state_dict_full / load_state_dict_full).

These exercise the (de)serialization + gather/reshard logic WITHOUT running the
Muon step (whose Polar-Express kernels need CUDA): optimizer state is fabricated
directly in ``optimizer.state`` on CPU, exactly mirroring the shapes/dtypes the
real step would create.  This isolates the checkpoint-resume machinery — the
thing the "full parity resume" work is about — from the GPU compute path.

Covered:
  - single-process round-trip (Muon momentum + second moment + bf16 mantissa + AdamW)
  - name-keying survives param-group reordering (the untie-split failure mode)
  - shape-mismatch entries are skipped, not misapplied
  - distributed gather + reshard across world_size changes (2->1, 1->2, 2->2)
    via gloo, with fabricated per-rank shards

Real-GPU end-to-end (NCCL, actual optimizer steps, field-by-field bit-exact across
world-size changes) lives in the standalone tests/smoke_optimizer_resume_multigpu.py.
"""

import os

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from tunalab.train_loops.multi_val_bucketed import _save_ckpt_full
from optimizers.muon import (
    MuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam,
    _FULL_STATE_FORMAT,
)


# ---------------------------------------------------------------------------
# Helpers: build an optimizer over a tiny model and fabricate optimizer state
# ---------------------------------------------------------------------------

def _tiny_model():
    """A model spanning both optimizer paths and both Muon reduction axes:
    a bf16 Muon weight (so the bf16 mantissa buffer is exercised), an fp32 Muon
    weight with the opposite red_dim, an embedding + bias on AdamW."""
    m = nn.Module()
    m.backbone = nn.Module()
    m.backbone.w = nn.Parameter(torch.randn(8, 6, dtype=torch.bfloat16))  # Muon, bf16 (mantissa), M>=N
    m.backbone.tall = nn.Parameter(torch.randn(4, 10))      # Muon, fp32, M<N -> red_dim=-2
    m.embedding = nn.Module()
    m.embedding.weight = nn.Parameter(torch.randn(20, 6))   # AdamW
    m.bias = nn.Parameter(torch.zeros(6))                   # AdamW
    return m


def _build_optimizer(cls, model):
    muon, adamw = [], []
    id_to_name = {}
    for name, p in model.named_parameters():
        id_to_name[id(p)] = name
        if "backbone" in name and p.ndim >= 2:
            muon.append(p)
        else:
            adamw.append(p)
    groups = [
        dict(params=muon, use_muon=True, lr=0.02, momentum=0.95, weight_decay=0.1, beta2=0.95),
        dict(params=adamw, use_muon=False, lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.1),
    ]
    opt = cls(groups)
    opt._id_to_name = id_to_name
    return opt


def _fabricate_state(opt, seed=0):
    """Populate opt.state for every param exactly as the real step would, but
    with deterministic random values so we can assert exact round-trip."""
    g = torch.Generator().manual_seed(seed)
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state[p]
            if group["use_muon"]:
                st["momentum_buffer"] = torch.randn(p.shape, generator=g, dtype=torch.float32)
                M, N = p.shape[-2], p.shape[-1]
                red_dim = -1 if M >= N else -2
                buf_shape = (M, 1) if red_dim == -1 else (1, N)
                st["second_momentum_buffer"] = torch.rand(buf_shape, generator=g, dtype=torch.float32)
                st["red_dim"] = red_dim
                if p.dtype == torch.bfloat16:
                    st["mantissa"] = torch.randint(0, 65535, p.shape, dtype=torch.uint16)
            else:
                st["exp_avg"] = torch.randn(p.shape, generator=g)
                st["exp_avg_sq"] = torch.rand(p.shape, generator=g)
                st["step"] = 42


def _assert_state_equal(opt_a, opt_b, names_a, names_b):
    """Assert opt_b's state for each param equals opt_a's, matched by name."""
    a_by_name = {names_a[id(p)]: opt_a.state[p] for g in opt_a.param_groups for p in g["params"] if opt_a.state.get(p)}
    b_by_name = {names_b[id(p)]: opt_b.state[p] for g in opt_b.param_groups for p in g["params"] if opt_b.state.get(p)}
    assert set(a_by_name) == set(b_by_name), (set(a_by_name), set(b_by_name))
    for name, sa in a_by_name.items():
        sb = b_by_name[name]
        for k, va in sa.items():
            vb = sb[k]
            if isinstance(va, torch.Tensor):
                assert torch.equal(va, vb), f"{name}.{k} mismatch"
            else:
                assert va == vb, f"{name}.{k} mismatch ({va} != {vb})"


# ---------------------------------------------------------------------------
# Single-process
# ---------------------------------------------------------------------------

def test_single_process_round_trip():
    m = _tiny_model()
    opt = _build_optimizer(SingleDeviceMuonWithAuxAdam, m)
    _fabricate_state(opt, seed=1)

    full = opt.state_dict_full()
    assert full["format"] == _FULL_STATE_FORMAT
    # keys are parameter names, not integer indices
    assert set(full["state"]) == {
        "backbone.w", "backbone.tall", "embedding.weight", "bias",
    }

    # Fresh optimizer over a fresh model -> restore -> state matches exactly.
    m2 = _tiny_model()
    opt2 = _build_optimizer(SingleDeviceMuonWithAuxAdam, m2)
    restored, skipped = opt2.load_state_dict_full(full)
    assert set(restored) == {"backbone.w", "backbone.tall", "embedding.weight", "bias"}
    assert skipped == []
    _assert_state_equal(opt, opt2, opt._id_to_name, opt2._id_to_name)


def test_name_keying_survives_param_group_reorder():
    """The untie-split failure mode: param ORDER differs between save and load,
    but name-keying restores each param's state correctly regardless."""
    m = _tiny_model()
    opt = _build_optimizer(SingleDeviceMuonWithAuxAdam, m)
    _fabricate_state(opt, seed=2)
    full = opt.state_dict_full()

    # Rebuild with AdamW params in reversed order (bias before embedding),
    # simulating a param-group layout change across the untie split.
    m2 = _tiny_model()
    muon = [m2.backbone.w, m2.backbone.tall]
    adamw = [m2.bias, m2.embedding.weight]  # reversed vs. build order
    id_to_name = {id(p): n for n, p in m2.named_parameters()}
    groups = [
        dict(params=muon, use_muon=True, lr=0.02, momentum=0.95, weight_decay=0.1, beta2=0.95),
        dict(params=adamw, use_muon=False, lr=3e-4, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.1),
    ]
    opt2 = SingleDeviceMuonWithAuxAdam(groups)
    opt2._id_to_name = id_to_name

    restored, skipped = opt2.load_state_dict_full(full)
    assert skipped == []
    _assert_state_equal(opt, opt2, opt._id_to_name, opt2._id_to_name)


def test_shape_mismatch_is_skipped_not_misapplied():
    m = _tiny_model()
    opt = _build_optimizer(SingleDeviceMuonWithAuxAdam, m)
    _fabricate_state(opt, seed=3)
    full = opt.state_dict_full()

    # Target model where embedding has a DIFFERENT shape (e.g. vocab changed).
    m2 = _tiny_model()
    m2.embedding.weight = nn.Parameter(torch.randn(30, 6))  # was (20, 6)
    opt2 = _build_optimizer(SingleDeviceMuonWithAuxAdam, m2)
    restored, skipped = opt2.load_state_dict_full(full)
    assert "embedding.weight" in skipped
    assert "embedding.weight" not in restored
    # The other params still restored fine.
    assert {"backbone.w", "backbone.tall", "bias"} <= set(restored)
    # And the mismatched param has NO fabricated state applied.
    assert not opt2.state.get(m2.embedding.weight)


def test_missing_name_is_skipped():
    m = _tiny_model()
    opt = _build_optimizer(SingleDeviceMuonWithAuxAdam, m)
    _fabricate_state(opt, seed=4)
    full = opt.state_dict_full()
    # Drop one param's saved state.
    del full["state"]["bias"]

    m2 = _tiny_model()
    opt2 = _build_optimizer(SingleDeviceMuonWithAuxAdam, m2)
    restored, skipped = opt2.load_state_dict_full(full)
    assert "bias" in skipped
    assert "bias" not in restored


def test_load_rejects_wrong_format():
    m = _tiny_model()
    opt = _build_optimizer(SingleDeviceMuonWithAuxAdam, m)
    with pytest.raises(ValueError):
        opt.load_state_dict_full({"format": "something_else", "state": {}})


def test_serialize_requires_id_to_name():
    m = _tiny_model()
    opt = _build_optimizer(SingleDeviceMuonWithAuxAdam, m)
    opt._id_to_name = None
    with pytest.raises(RuntimeError):
        opt.state_dict_full()


# ---------------------------------------------------------------------------
# Distributed: gather + reshard across world_size changes (gloo, CPU)
# ---------------------------------------------------------------------------

def _dist_worker(rank, world_size, tmpdir, phase, seed):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        m = _tiny_model()
        opt = _build_optimizer(MuonWithAuxAdam, m)

        if phase == "save":
            # Fabricate ONLY this rank's shard of the Muon params (mirrors step()'s
            # ownership: position i owned by rank i % world_size), plus full AdamW.
            g = torch.Generator().manual_seed(seed)  # same seed all ranks -> identical AdamW
            for group in opt.param_groups:
                params = group["params"]
                if group["use_muon"]:
                    for i, p in enumerate(params):
                        if i % world_size != rank:
                            continue
                        st = opt.state[p]
                        # deterministic per-NAME values so any rank produces the same
                        gp = torch.Generator().manual_seed(seed + 1000 + i)
                        st["momentum_buffer"] = torch.randn(p.shape, generator=gp, dtype=torch.float32)
                        M, N = p.shape[-2], p.shape[-1]
                        red_dim = -1 if M >= N else -2
                        buf_shape = (M, 1) if red_dim == -1 else (1, N)
                        st["second_momentum_buffer"] = torch.rand(buf_shape, generator=gp, dtype=torch.float32)
                        st["red_dim"] = red_dim
                else:
                    for p in params:
                        st = opt.state[p]
                        st["exp_avg"] = torch.randn(p.shape, generator=torch.Generator().manual_seed(seed + hash(id(p)) % 7))
                        st["exp_avg_sq"] = torch.rand(p.shape, generator=torch.Generator().manual_seed(seed + 3))
                        st["step"] = 7

            full = opt.state_dict_full()
            if rank == 0:
                torch.save(full, os.path.join(tmpdir, "full.pt"))
            dist.barrier()

        else:  # phase == "load"
            full = torch.load(os.path.join(tmpdir, "full.pt"), weights_only=False)
            restored, skipped = opt.load_state_dict_full(full)
            # This rank must own state for exactly its Muon shard + all AdamW params.
            muon_params = opt.param_groups[0]["params"]
            expected_muon = {
                opt._id_to_name[id(muon_params[i])]
                for i in range(len(muon_params)) if i % world_size == rank
            }
            expected_adamw = {opt._id_to_name[id(p)] for p in opt.param_groups[1]["params"]}
            got = set(restored)
            assert got == (expected_muon | expected_adamw), (rank, world_size, got)
            assert skipped == []
            # Spot-check a restored Muon buffer matches the deterministic source.
            for i, p in enumerate(muon_params):
                if i % world_size == rank:
                    gp = torch.Generator().manual_seed(seed + 1000 + i)
                    ref = torch.randn(p.shape, generator=gp, dtype=torch.float32)
                    assert torch.equal(opt.state[p]["momentum_buffer"], ref)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_dist(save_ws, load_ws, tmpdir, seed=11):
    # Save phase at save_ws.
    mp.spawn(_dist_worker, args=(save_ws, tmpdir, "save", seed), nprocs=save_ws, join=True)
    # Load phase at load_ws.
    mp.spawn(_dist_worker, args=(load_ws, tmpdir, "load", seed), nprocs=load_ws, join=True)


@pytest.mark.parametrize("save_ws,load_ws", [(2, 1), (1, 2), (2, 2)])
def test_distributed_reshard_across_world_size(save_ws, load_ws, tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    _run_dist(save_ws, load_ws, str(tmp_path))


# ---------------------------------------------------------------------------
# Training-loop save helpers: full optimizer state lands in the checkpoint;
# plain optimizers (no state_dict_full) degrade gracefully.
# ---------------------------------------------------------------------------

def test_loop_save_embeds_full_optimizer_state(tmp_path):
    m = _tiny_model()
    opt = _build_optimizer(SingleDeviceMuonWithAuxAdam, m)
    _fabricate_state(opt, seed=5)

    fp = str(tmp_path / "checkpoints" / "latest.pt")
    _save_ckpt_full(m, opt, {"step": 10, "val_loss": 1.5, "config": {}}, fp)
    assert os.path.exists(fp)

    ckpt = torch.load(fp, weights_only=False)
    md = ckpt["metadata"]
    assert md["step"] == 10
    full = md["optimizer_state"]
    assert full["format"] == _FULL_STATE_FORMAT
    assert set(full["state"]) == {"backbone.w", "backbone.tall", "embedding.weight", "bias"}

    # Restore into a fresh optimizer straight from the saved checkpoint.
    m2 = _tiny_model()
    opt2 = _build_optimizer(SingleDeviceMuonWithAuxAdam, m2)
    restored, skipped = opt2.load_state_dict_full(full)
    assert skipped == []
    _assert_state_equal(opt, opt2, opt._id_to_name, opt2._id_to_name)


def test_loop_save_omits_optimizer_state_for_plain_optimizer(tmp_path):
    """A plain torch optimizer lacks state_dict_full — _save_ckpt_full must still
    write the checkpoint (model + metadata), just without optimizer_state."""
    m = _tiny_model()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    fp = str(tmp_path / "checkpoints" / "best_model.pt")
    _save_ckpt_full(m, opt, {"step": 3, "config": {}}, fp)

    ckpt = torch.load(fp, weights_only=False)
    assert "optimizer_state" not in ckpt["metadata"]
    assert ckpt["metadata"]["step"] == 3
