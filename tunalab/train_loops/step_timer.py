"""
Atomic feature: per-step timing log — standalone version.

Records per-step wall-clock time broken down into data_load_s / backward_s /
wait_s / total_s and writes a CSV per rank.  Useful for profiling training
throughput and diagnosing data starvation on the real training path (bucketed
or live dataset + backend I/O).

    data_load_s — wall time blocked on next(loader) before compute (starvation)
    backward_s  — fwd-start → cuda.synchronize (backward + DDP allreduce)
    wait_s      — dist.barrier stall waiting for slower ranks
    total_s     — data_load_s + backward_s + wait_s (true end-to-end step time)

This feature is timing-only — atomic features stay atomic.  For persisting the
BucketedPackDataset schedule position in checkpoints, use the
``bucket_state_checkpoint`` (single val_loader) or ``multi_val_bucketed``
(multiple val_loaders) features.

Unique trigger kwarg
--------------------
step_timing_log : bool  (default False)
    Setting this to True selects step_timer as a single atomic feature.
    ``step_timing_all_ranks`` controls whether every rank or only rank 0
    prints/writes; it is accepted via **kwargs.
"""

import csv
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist


def run_training(
    model,
    optimizer,
    train_loader,
    *,
    step_timing_log: bool = False,     # unique kwarg — triggers feature selection
    step_timing_csv: bool = True,
    output_dir: Optional[str] = None,
    device: str = "cpu",
    **kwargs,
) -> Dict[str, Any]:
    """Training loop with per-step timing breakdown (standalone, no checkpointing).

    Triggered by passing ``step_timing_log=True``.  Pass
    ``step_timing_all_ranks=True`` in kwargs to have every rank print/write
    rather than only rank 0.

    Each step measures:
      data_load_s — wall time blocked on next(loader) before compute (starvation)
      backward_s  — fwd-start → cuda.synchronize (backward + DDP allreduce)
      wait_s      — dist.barrier stall (how long this rank waits for others)
      total_s     — data_load_s + backward_s + wait_s (true end-to-end step time)
    """
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    is_main = rank == 0

    step_timing_all_ranks = kwargs.get("step_timing_all_ranks", False)
    should_print = is_main or step_timing_all_ranks

    _dev = torch.device(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # CSV writer setup
    csv_file = None
    csv_writer = None
    if step_timing_csv and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"step_timing_rank{rank}.csv")
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ["step", "total_s", "data_load_s", "backward_s", "wait_s", "it_per_s", "loss"]
        )

    if should_print:
        print(
            f"\n{'step':>6}  {'total_s':>8}  {'data_s':>7}  {'bwd_s':>7}  {'wait_s':>7}  "
            f"{'it/s':>7}  loss",
            flush=True,
        )

    # Iterate manually so the data-fetch (iterator next) is timed separately from
    # compute — the true data-starvation signal on the real training path.
    _loader_iter = iter(train_loader)
    step = 0
    while True:
        # ---- data fetch ----
        t_data0 = time.perf_counter()
        try:
            batch = next(_loader_iter)
        except StopIteration:
            break
        t_start = time.perf_counter()
        data_load_s = t_start - t_data0   # wall time blocked on the dataloader

        loss = model(batch)
        loss.backward()

        # Sync after backward: waits for this rank's own CUDA work + allreduce
        torch.cuda.synchronize(_dev)
        t_after_bwd = time.perf_counter()

        # Barrier: waits until ALL ranks have finished their backward+allreduce.
        # Any time spent here is stall time on this rank.
        if is_distributed:
            dist.barrier()
        t_after_barrier = time.perf_counter()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # ---- timings ----
        # total_s spans the full step including the data fetch, so it_per_s
        # reflects true end-to-end throughput.
        bwd_s = t_after_bwd - t_start
        wait_s = t_after_barrier - t_after_bwd
        total_s = t_after_barrier - t_data0
        it_per_s = 1.0 / total_s if total_s > 0 else float("inf")
        loss_val = float(loss.detach().cpu().item())

        if csv_writer is not None:
            csv_writer.writerow(
                [step, f"{total_s:.4f}", f"{data_load_s:.4f}", f"{bwd_s:.4f}",
                 f"{wait_s:.4f}", f"{it_per_s:.3f}", f"{loss_val:.4f}"])

        if should_print:
            print(
                f"{step:>6}  {total_s:>8.3f}  {data_load_s:>7.3f}  {bwd_s:>7.3f}  "
                f"{wait_s:>7.3f}  {it_per_s:>7.2f}  {loss_val:.4f}",
                flush=True,
            )

        step += 1

    if csv_file is not None:
        csv_file.close()
        if should_print:
            print(f"\nStep timing saved → {csv_path}", flush=True)

    return {"model": model}
