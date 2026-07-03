"""Atomic feature (ts2-local): DDP-aware training profiler.

Replaces the drift-prone standalone ``profile_training.py`` script that used to
live at the repo root.  Because this is a real atomic feature discovered by
``smart_train`` (via its unique trigger kwarg ``profile_run``), it exercises the
*exact* model / optimizer / dataloader that main.py builds — it can never drift
away from the real training path the way a parallel script did.

Design note — how the profile_run=False path stays base-loop-equivalent
-----------------------------------------------------------------------
There is a SINGLE ``run_training`` loop.  It IS the base loop
(``for batch: forward → backward → step → zero_grad``); every profiling action
is layered on as a ``if profile_run:`` conditional around that unchanged core,
exactly like ``grad_norm_clip`` layers clipping via ``if norm_clip_value``.  So
with ``profile_run=False`` the observable inputs/outputs — the model's post-train
parameters and the returned ``{"model": ...}`` dict — are identical to
``base_loop`` (no separate helper, no early return).  This also lets the
smart_train LLM compiler interleave this feature's conditionals with other
features' (grad_accum, validation, tqdm, …) over one shared loop skeleton,
instead of trying to merge two divergent code paths.

What it measures (profile_run=True)
-----------------------------------
Per training step, broken out so a bottleneck is unambiguous:

    data_load    — CPU wall time blocked on next(loader) (traversal, memmap I/O,
                   packing, collate).  High → input-bound.
    mask_create  — CPU wall time building the FlexAttention/Triton block mask.
    forward      — GPU time, model(batch) up to loss (CUDA events).
    bwd+NCCL     — GPU time, loss.backward() on SYNC steps (incl. DDP allreduce).
    bwd only     — GPU time, loss.backward() on no_sync steps (compute only).
    NCCL est.    — derived (sync_bwd − nosync_bwd): the grad-sync cost.
    optim_step   — GPU time, optimizer.step() + zero_grad.
    step_wall    — total end-to-end wall time.

Model-internal breakdown (``profile_model_internals=True``): forward hooks
record CUDA events around the eager top-level components — ``embedding``,
``backbone`` (whole compiled stack), ``norm``, ``loss_fn`` — showing how GPU
time splits WITHOUT touching the compute graph (hooks fire in eager around the
compiled backbone call; no Python timers inside the compiled region, which would
force a graph break / recompile).  For per-kernel detail *inside* the backbone
(attention vs. MLP vs. Newton-Schulz), set ``profile_trace=True`` for a chrome
trace.

Diagnosing the multi-rank DDP hang: ``profile_all_ranks=True`` makes every rank
print its own per-phase line, so a rank frozen mid-step reveals which rank/phase
stalled; ``profile_nccl_bench_steps>0`` first confirms raw collective health.

Unique trigger kwarg
--------------------
profile_run : bool  (default False → the loop is base_loop; True → profiled).
All other ``profile_*`` knobs are declared keyword-only so smart_train registers
them; the LLM compiler composes this feature with the others (grad_accum,
validation, checkpointing, …) as one unified loop.
"""

import os
import time
import csv
import statistics
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from tunalab.device import to_device as _to_device


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    *,
    profile_run: bool = False,              # semantic trigger
    profile_warmup_steps: int = 3,          # steps discarded before timing
    profile_no_sync_steps: int = 0,         # no_sync steps (isolate NCCL cost)
    profile_active_steps: int = 20,         # timed steps after warmup+no_sync
    profile_model_internals: bool = True,   # forward-hook component breakdown
    profile_trace: bool = False,            # emit torch.profiler chrome trace
    profile_trace_steps: int = 6,           # active window for the trace
    profile_nccl_bench_steps: int = 0,      # raw all_reduce micro-benchmark reps
    profile_all_ranks: bool = False,        # every rank prints (not just rank 0)
    output_dir: Optional[str] = None,       # chrome-trace / CSV destination
    device: str = "cuda",                   # device to move batch tensors to
    **kwargs,
) -> Dict[str, Any]:
    """Training loop with optional per-phase profiling (see module docstring).

    With ``profile_run=False`` the loop is base_loop: its observable I/O (model
    post-train params, returned dict) is identical, and no profiling machinery
    runs.  With ``profile_run=True`` the same loop additionally times each phase.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # -- profiling setup (only when profiling; otherwise all no-ops) ------------
    cuda = torch.cuda.is_available()
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    world = dist.get_world_size() if is_dist else 1
    should_print = profile_run and (rank == 0 or profile_all_ranks)

    prof = None
    hook_handles: List[Any] = []
    comp_names = ["embedding", "backbone", "norm", "loss_fn"]
    comp_events: Dict[str, List] = {n: [] for n in comp_names}
    _pending: Dict[str, Any] = {}
    _collect = [False]
    _mask_times: List[float] = []
    _orig_bmc = None
    inner = model.module if hasattr(model, "module") else model
    no_sync_bwd: List[float] = []
    sync_rows: List[Dict[str, float]] = []
    total_steps = profile_warmup_steps + profile_no_sync_steps + profile_active_steps

    if profile_run:
        pdev = torch.device(device if cuda else "cpu")
        if should_print:
            print(f"[profile rank={rank}] start  warmup={profile_warmup_steps} "
                  f"no_sync={profile_no_sync_steps} active={profile_active_steps} "
                  f"world={world} internals={profile_model_internals} "
                  f"trace={profile_trace} device={pdev}", flush=True)

        # NCCL micro-benchmark: raw all_reduce health before any model op.
        if is_dist and profile_nccl_bench_steps > 0 and cuda:
            buf = torch.zeros(256 * 1024 * 1024 // 4, dtype=torch.float32, device=pdev)
            for mb in (1, 16, 64, 256):
                sub = buf[: mb * 1024 * 1024 // 4]
                dist.all_reduce(sub); torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(profile_nccl_bench_steps):
                    dist.all_reduce(sub)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / profile_nccl_bench_steps * 1000
                if rank == 0:
                    print(f"[profile nccl_bench] {mb:>4}MB all_reduce: {ms:>7.2f}ms "
                          f"({mb * 2 / (ms / 1000) / 1024:.1f} GB/s)", flush=True)
            del buf

        # Wrap block_mask_creator to time CPU-side mask construction.
        _orig_bmc = getattr(inner, "block_mask_creator", None)
        if _orig_bmc is not None:
            def _timed_bmc(*a, _o=_orig_bmc, **kw):
                t0 = time.perf_counter()
                r = _o(*a, **kw)
                _mask_times.append(time.perf_counter() - t0)
                return r
            inner.block_mask_creator = _timed_bmc

        # Forward hooks: CUDA events around eager top-level components.
        if profile_model_internals and cuda:
            def _mk_pre(name):
                def pre(_m, _a):
                    if _collect[0]:
                        ev = torch.cuda.Event(enable_timing=True); ev.record()
                        _pending[name] = ev
                return pre

            def _mk_post(name):
                def post(_m, _a, _o):
                    if _collect[0] and name in _pending:
                        end = torch.cuda.Event(enable_timing=True); end.record()
                        comp_events[name].append((_pending.pop(name), end))
                return post

            for name in comp_names:
                mod = getattr(inner, name, None)
                if isinstance(mod, nn.Module):
                    hook_handles.append(mod.register_forward_pre_hook(_mk_pre(name)))
                    hook_handles.append(mod.register_forward_hook(_mk_post(name)))

        # Optional chrome trace (fine-grained, incl. inside compiled backbone).
        if profile_trace and cuda:
            from torch.profiler import ProfilerActivity, schedule as _sched, profile as _profile

            def _on_ready(p):
                dest = output_dir or "."
                os.makedirs(dest, exist_ok=True)
                out = os.path.join(dest, f"profile_trace_rank{rank}.json.gz")
                p.export_chrome_trace(out)
                if should_print:
                    print(f"[profile rank={rank}] chrome trace → {out}", flush=True)

            prof = _profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=_sched(wait=1, warmup=1, active=max(1, profile_trace_steps), repeat=1),
                on_trace_ready=_on_ready, record_shapes=False, with_stack=False,
            )

    # -- the training loop — base_loop core, profiling layered as conditionals --
    trace_ctx = prof if prof is not None else nullcontext()
    step = 0
    with trace_ctx:
        for batch in train_loader:
            if profile_run and step >= total_steps:
                break

            # phase boundaries (only meaningful when profiling)
            is_warmup = profile_run and step < profile_warmup_steps
            is_no_sync = profile_run and (
                profile_warmup_steps <= step < profile_warmup_steps + profile_no_sync_steps)
            _collect[0] = profile_run and (not is_warmup) and (not is_no_sync) \
                and profile_model_internals

            if profile_run:
                t0 = time.perf_counter()
                batch = _to_device(batch, pdev, non_blocking=True)
                t_data = time.perf_counter()
                e_fs, e_fe, e_be, e_oe = (
                    [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    if cuda else (None, None, None, None))
                if cuda:
                    e_fs.record()

            # --- base_loop core: forward → backward → step → zero_grad ---
            sync_ctx = (model.no_sync() if (is_no_sync and hasattr(model, "no_sync"))
                        else nullcontext())
            with sync_ctx:
                loss = model(batch)
                if profile_run and cuda:
                    e_fe.record()
                loss.backward()
            if profile_run and cuda:
                e_be.record()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if profile_run:
                if cuda:
                    e_oe.record()
                    torch.cuda.synchronize()
                t_end = time.perf_counter()
                if not is_warmup:
                    row = dict(
                        data_ms=(t_data - t0) * 1000,
                        mask_ms=(_mask_times[-1] * 1000) if _mask_times else 0.0,
                        fwd_ms=e_fs.elapsed_time(e_fe) if cuda else 0.0,
                        bwd_ms=e_fe.elapsed_time(e_be) if cuda else 0.0,
                        opt_ms=e_be.elapsed_time(e_oe) if cuda else 0.0,
                        wall_ms=(t_end - t0) * 1000,
                    )
                    (no_sync_bwd.append(row["bwd_ms"]) if is_no_sync
                     else sync_rows.append(row))
                    if should_print:
                        print(f"[profile rank={rank}] step={step:>3} "
                              f"{'no_sync' if is_no_sync else 'active'}  "
                              f"loss={float(loss.detach()):.4f}  "
                              f"wall={row['wall_ms']:.0f}ms  data={row['data_ms']:.0f}  "
                              f"mask={row['mask_ms']:.0f}  fwd={row['fwd_ms']:.0f}  "
                              f"bwd={row['bwd_ms']:.0f}  opt={row['opt_ms']:.0f}", flush=True)
                if prof is not None and not is_warmup and not is_no_sync:
                    prof.step()

            step += 1

    if not profile_run:
        return {"model": model}

    # -- profiling teardown + summary -------------------------------------------
    if _orig_bmc is not None:
        inner.block_mask_creator = _orig_bmc
    for h in hook_handles:
        h.remove()

    result: Dict[str, Any] = {"model": model}
    if not sync_rows:
        if should_print:
            print(f"[profile rank={rank}] no active steps timed "
                  f"(increase profile_active_steps).", flush=True)
        return result

    keys = ["data_ms", "mask_ms", "fwd_ms", "bwd_ms", "opt_ms", "wall_ms"]
    mean = {k: statistics.mean([r[k] for r in sync_rows]) for k in keys}
    nccl_est = (max(0.0, mean["bwd_ms"] - statistics.mean(no_sync_bwd))
                if no_sync_bwd else None)
    comp_mean = {name: statistics.mean([s.elapsed_time(e) for s, e in evs])
                 for name, evs in comp_events.items() if evs}

    result["profile_summary"] = {
        "mean_ms": mean, "nccl_est_ms": nccl_est, "component_ms": comp_mean,
        "world_size": world,
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"profile_rank{rank}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + keys)
            for i, r in enumerate(sync_rows):
                w.writerow([i] + [f"{r[k]:.4f}" for k in keys])

    if rank != 0:
        return result

    sep = "=" * 68
    print(f"\n{sep}\n  profile_training summary  (world={world}, "
          f"{len(sync_rows)} active steps)\n{sep}", flush=True)
    for k in keys:
        print(f"  {k:<12} {mean[k]:>8.1f} ms", flush=True)
    if nccl_est is not None:
        print(f"  {'NCCL est.':<12} {nccl_est:>8.1f} ms  (sync_bwd − nosync_bwd)", flush=True)
    if comp_mean:
        print(f"{'-' * 68}\n  model-internal breakdown (GPU, mean per step):", flush=True)
        for name in comp_names:
            if name in comp_mean:
                print(f"    {name:<12} {comp_mean[name]:>8.1f} ms", flush=True)
    print(sep, flush=True)

    return result
