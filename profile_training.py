"""Standalone DDP-aware training profiler for TAGSeq2TAGSeq.

Instruments each phase of the training step separately:
  data_load    — CPU wall time (graph traversal, memmap I/O, packing)
  mask_create  — CPU wall time (FlexAttention BlockMask construction)
  forward      — GPU CUDA event time
  bwd+NCCL     — GPU CUDA event time (sync steps: includes all-reduce)
  bwd only     — GPU CUDA event time (no_sync steps: compute only)
  NCCL est.    — derived (sync_bwd - nosync_bwd)
  optim_step   — GPU CUDA event time
  step_wall    — total wall time

Launched via launch_slurm.py (--script profile_training) or standalone:

    # Single-GPU smoke test
    python profile_training.py \\
        --config configs/baseline.yaml \\
        --data.dataset_dir data/pretokenized_datasets/simplewiki \\
        --model.compile false \\
        --profile.warmup_steps 2 \\
        --profile.profile_steps 5 \\
        --profile.no_sync_steps 0

    # Multi-node via SLURM
    python launch_slurm.py --nodes 2 --gpus-per-node 8 \\
        --script profile_training \\
        --config configs/stack_100m_32k.yaml \\
        --data.dataset_dir data/pretokenized_datasets/stack_100m
"""

import os
import time
import statistics
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as tdist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity

import tiktoken

from tunalab.configuration import compose_config
from tunalab.distributed import DistributedManager, setup_signal_handlers
from tunalab.reproducibility import ReproducibilityManager
from tunalab.optimizers.muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam

from model import TS2TSTrainingModule
from model.graph_traversal.block_mask_creator import (
    make_mask_creator_callable,
    make_mask_creator_callable_from,
)
from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
from model.graph_traversal.python_import_detector import PythonImportDetector
from data.dataset import GraphIndex, PretokShardedBackend
from data.packed_dataset import PackedSequenceDataset
from data.layout import make_layout_policy
from data.pack_sampler import PackBatchSampler
from data.traversal import (
    BFSStrategy,
    DFSStrategy,
    RandomSelectionStrategy,
    RandomWalkStrategy,
)


def _log(rank: int, msg: str) -> None:
    """Per-rank timestamped debug print, always flushed."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[profile_training rank={rank} {ts}] {msg}", flush=True)


def main(cfg: Dict[str, Any], dist: DistributedManager, rep: ReproducibilityManager):
    setup_signal_handlers()
    _log(dist.rank, "entered main()")

    # -------------------------------------------------------------------------
    # 1. Profiling config
    # -------------------------------------------------------------------------
    prof_cfg         = cfg.get('profile', {})
    warmup_steps     = int(prof_cfg.get('warmup_steps',        5))
    no_sync_steps    = int(prof_cfg.get('no_sync_steps',       10))
    profile_steps    = int(prof_cfg.get('profile_steps',       30))
    torch_prof_steps = int(prof_cfg.get('torch_profiler_steps', 10))
    trace_dir        = Path(prof_cfg.get('trace_dir', 'artifacts/profiler_traces'))
    total_steps      = warmup_steps + no_sync_steps + profile_steps

    # DDP/compile knobs — vary these to isolate the source of NCCL non-overlap.
    ddp_static_graph   = bool(prof_cfg.get('static_graph',   True))
    ddp_bucket_cap_mb  = int(prof_cfg.get('bucket_cap_mb',   256))
    use_optimize_ddp   = bool(prof_cfg.get('optimize_ddp',   True))
    nccl_debug         = str(prof_cfg.get('nccl_debug',      'OFF'))

    if nccl_debug != 'OFF':
        os.environ['NCCL_DEBUG'] = nccl_debug

    if dist.is_main_process:
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[profile_training] Starting: "
            f"warmup={warmup_steps}  no_sync={no_sync_steps}  profile={profile_steps}  "
            f"world={dist.world_size}  "
            f"compile={cfg['model'].get('compile', True)}  "
            f"static_graph={ddp_static_graph}  bucket_cap_mb={ddp_bucket_cap_mb}  "
            f"optimize_ddp={use_optimize_ddp}  nccl_debug={nccl_debug}",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # 2. Data setup (mirrors main.py)
    # -------------------------------------------------------------------------
    _log(dist.rank, "loading GraphIndex")
    dataset_dir = Path(cfg['data']['dataset_dir'])
    graph_index = GraphIndex(dataset_dir)
    backend     = PretokShardedBackend(graph_index)
    _log(dist.rank, "GraphIndex loaded")

    enc = tiktoken.get_encoding(graph_index.metadata.get('tokenizer', 'gpt2'))

    layout_policy_name = cfg.get('data', {}).get('layout_policy', 'null')
    layout_policy = make_layout_policy(
        name=layout_policy_name,
        encode_fn=enc.encode_ordinary,
    )

    strategy_name = cfg.get('data', {}).get('strategy', 'random')
    if strategy_name == 'random':
        strategy_factory = lambda: RandomSelectionStrategy()
    elif strategy_name == 'random_walk':
        strategy_factory = lambda: RandomWalkStrategy(edge_mode='outgoing', restart_prob=0.05)
    elif strategy_name == 'bfs':
        strategy_factory = lambda: BFSStrategy(edge_mode='outgoing')
    elif strategy_name == 'dfs':
        strategy_factory = lambda: DFSStrategy(edge_mode='outgoing')
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    base_seed = cfg.get('seed', 42)
    pack_sampler = PackBatchSampler(
        graph=graph_index,
        strategy_factory=strategy_factory,
        token_budget=cfg.get('model', {}).get('max_seq_len', 2048),
        doc_budget=cfg.get('data', {}).get('doc_budget'),
        overflow_policy='truncate',
        doc_level_trim_side='tail',
        pack_level_trim_side='head',
        max_candidates_per_component=1000,
        seed=base_seed + dist.rank,
        order_mode=cfg.get('data', {}).get('order_mode', 'prefer_targets_first'),
        layout_policy=layout_policy,
    )
    dataset = PackedSequenceDataset(
        graph=graph_index,
        backend=backend,
        pack_sampler=pack_sampler,
        layout_policy=layout_policy,
        as_2d=True,
    )
    data_iter = iter(dataset)
    _log(dist.rank, "dataset ready")

    # -------------------------------------------------------------------------
    # 3. Model setup (dynamic=True — required for multi-node, see MEMORY.md)
    # -------------------------------------------------------------------------
    seq_len    = cfg['model']['max_seq_len']
    mask_type  = cfg.get('model', {}).get('mask_type', 'doc_causal')

    if mask_type == 'cross_doc_link':
        link_detector_name = cfg.get('model', {}).get('link_detector')
        if not link_detector_name:
            raise ValueError(
                "model.link_detector must be set to 'markdown' or 'python' "
                "when model.mask_type is 'cross_doc_link'"
            )
        if link_detector_name == 'markdown':
            detector = MarkdownLinkDetector(decode_fn=enc.decode)
        elif link_detector_name == 'python':
            detector = PythonImportDetector(decode_fn=enc.decode)
        else:
            raise ValueError(f"Unknown link_detector: {link_detector_name!r}")
        _mcfg = cfg.get('model', {})
        block_mask_creator = make_mask_creator_callable_from(
            CrossDocLinkMaskCreator(
                link_detector=detector,
                max_grants=_mcfg.get('max_grants', 64),
                max_grants_start=_mcfg.get('max_grants_start'),
                max_grants_warmup_steps=int(_mcfg.get('max_grants_warmup_steps', 0)),
            )
        )
    else:
        block_mask_creator = make_mask_creator_callable(mask_type)

    tokenizer_name = graph_index.metadata.get('tokenizer', 'gpt2')
    vocab_size = 50257 if tokenizer_name == 'gpt2' else cfg['model'].get('vocab_size', 50257)

    _log(dist.rank, "building model")
    model = TS2TSTrainingModule.from_config(
        vocab_size=vocab_size,
        num_layers=cfg['model']['num_layers'],
        model_dim=cfg['model']['model_dim'],
        num_heads=cfg['model']['num_heads'],
        max_seq_len=seq_len,
        dropout=cfg['model'].get('dropout', 0.0),
        drop_path_rate=cfg['model'].get('drop_path_rate', 0.0),
        block_mask_creator=block_mask_creator,
        fp8=cfg['model'].get('fp8', False),
        weight_tying=cfg['model'].get('weight_tying', True),
        ignore_index=cfg['model'].get('ignore_index', -100),
        dtype=getattr(torch, cfg['model'].get('dtype', 'bfloat16')),
        activation_checkpointing=cfg['model'].get('activation_checkpointing', False),
    ).to(dist.device)
    _log(dist.rank, "model built")

    # Build optimizer param groups BEFORE compile/DDP so named_parameters() gives
    # clean names and weight-tied tensors are counted only once.
    muon_params, adamw_params = [], []
    seen_ids: set = set()
    for name, param in model.named_parameters():
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        if 'backbone' in name and param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    OptCls = MuonWithAuxAdam if dist.is_distributed else SingleDeviceMuonWithAuxAdam
    optimizer = OptCls([
        dict(
            params=muon_params,
            use_muon=True,
            lr=cfg['optimizer']['muon_lr'],
            momentum=cfg['optimizer'].get('momentum', 0.95),
            weight_decay=cfg['optimizer']['wd'],
        ),
        dict(
            params=adamw_params,
            use_muon=False,
            lr=cfg['optimizer']['adamw_lr'],
            betas=(cfg['optimizer'].get('beta1', 0.9), cfg['optimizer'].get('beta2', 0.95)),
            weight_decay=cfg['optimizer']['wd'],
        ),
    ])

    if cfg['model'].get('compile', True):
        _log(dist.rank, "torch.compile starting")
        torch._dynamo.config.optimize_ddp = use_optimize_ddp
        model.backbone = torch.compile(
            model.backbone,
            dynamic=True,  # CRITICAL: prevents per-shape recompile on multi-node (see MEMORY.md)
            mode=cfg['model'].get('compile_mode', 'default'),
        )
        _log(dist.rank, "torch.compile done (wrapping only; kernel compile deferred to first fwd)")

    if dist.is_distributed:
        _log(dist.rank, "DDP wrapping")
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            static_graph=ddp_static_graph,
            find_unused_parameters=False,
            bucket_cap_mb=ddp_bucket_cap_mb,
        )
        _log(dist.rank, "DDP wrap done")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    # -------------------------------------------------------------------------
    # 3b. NCCL micro-benchmark: measure raw all_reduce latency/bandwidth
    #     before any model ops so we know the baseline collective speed.
    # -------------------------------------------------------------------------
    nccl_bench_steps = int(prof_cfg.get('nccl_bench_steps', 5))
    if dist.is_distributed and nccl_bench_steps > 0:
        # Warm up with one call, then time nccl_bench_steps calls.
        _sizes_mb = [1, 16, 64, 256, 560]   # MB — covers parameter-sized blobs
        buf = torch.zeros(560 * 1024 * 1024 // 4, dtype=torch.float32,
                          device=dist.device)
        for _s in _sizes_mb:
            _n = _s * 1024 * 1024 // 4
            _buf = buf[:_n]
            # warm-up
            tdist.all_reduce(_buf)
            torch.cuda.synchronize()
            # timed
            _t0 = time.perf_counter()
            for _ in range(nccl_bench_steps):
                tdist.all_reduce(_buf)
            torch.cuda.synchronize()
            _elapsed = (time.perf_counter() - _t0) / nccl_bench_steps * 1000
            if dist.is_main_process:
                _bw = _s * 2 / (_elapsed / 1000)   # MB/s  (×2 for reduce+scatter)
                print(f"[nccl_bench] {_s:>4}MB all_reduce: {_elapsed:>7.1f}ms  "
                      f"({_bw/1024:.1f} GB/s effective)",
                      flush=True)
        del buf

    # -------------------------------------------------------------------------
    # 4. Monkey-patch block_mask_creator for CPU timing
    #    BlockMask construction is CPU-side (computes block-sparsity indices over
    #    doc_spans before dispatching the CUDA flex_attention kernel).
    # -------------------------------------------------------------------------
    _inner = model.module if dist.is_distributed else model
    _orig_bmc = _inner.block_mask_creator
    _mask_times: List[float] = []

    def _timed_bmc(**kwargs):
        t0 = time.perf_counter()
        result = _orig_bmc(**kwargs)
        _mask_times.append(time.perf_counter() - t0)
        return result

    _inner.block_mask_creator = _timed_bmc

    # -------------------------------------------------------------------------
    # 5. Profiling loop
    # -------------------------------------------------------------------------
    no_sync_timings: List[Dict[str, float]] = []
    sync_timings:    List[Dict[str, float]] = []

    def _on_trace_ready(p: torch.profiler.profile) -> None:
        out = trace_dir / f"rank{dist.rank}.json.gz"
        p.export_chrome_trace(str(out))
        if dist.is_main_process:
            print(f"[profile_training] Chrome trace → {out}", flush=True)

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=2, warmup=2, active=torch_prof_steps, repeat=1),
        on_trace_ready=_on_trace_ready,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for step in range(total_steps):
            is_warmup  = step < warmup_steps
            is_no_sync = warmup_steps <= step < warmup_steps + no_sync_steps
            is_profile = step >= warmup_steps + no_sync_steps

            # --- Phase 1: Data loading (pure CPU: graph traversal, memmap, packing) ---
            t_step_start = time.perf_counter()
            batch = next(data_iter)
            t_data_done = time.perf_counter()

            # Move tokens to device; H2D transfer intentionally excluded from data_load time.
            batch['tokens'] = batch['tokens'].to(dist.device, non_blocking=True)

            # --- GPU phases via CUDA events (stream-ordered, no forced serialisation) ---
            e_fwd_start, e_fwd_end, e_bwd_end, e_opt_end = [
                torch.cuda.Event(enable_timing=True) for _ in range(4)
            ]

            e_fwd_start.record()

            # no_sync suppresses all-reduce on backward; lets us isolate NCCL cost.
            ctx = (
                model.no_sync()
                if (dist.is_distributed and is_no_sync)
                else nullcontext()
            )
            with ctx:
                loss = model(batch)       # forward (includes mask creation via monkey-patch)
                e_fwd_end.record()
                loss.backward()           # backward + NCCL all-reduce (suppressed with no_sync)

            e_bwd_end.record()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            e_opt_end.record()

            # Synchronise so all CUDA event times are finalised before reading them.
            torch.cuda.synchronize()
            t_step_end = time.perf_counter()

            # Collect timings (skip warmup steps)
            if not is_warmup:
                entry: Dict[str, float] = dict(
                    data_ms=  (t_data_done  - t_step_start) * 1000,
                    mask_ms=  _mask_times[-1] * 1000 if _mask_times else 0.0,
                    fwd_ms=   e_fwd_start.elapsed_time(e_fwd_end),
                    bwd_ms=   e_fwd_end.elapsed_time(e_bwd_end),
                    opt_ms=   e_bwd_end.elapsed_time(e_opt_end),
                    wall_ms=  (t_step_end   - t_step_start) * 1000,
                )
                if is_no_sync:
                    no_sync_timings.append(entry)
                else:
                    sync_timings.append(entry)

                if dist.is_main_process:
                    phase = "warmup" if is_warmup else ("no_sync" if is_no_sync else "profile")
                    print(
                        f"[profile_training] step={step}  phase={phase}  "
                        f"loss={loss.item():.4f}  "
                        f"wall={entry['wall_ms']:.0f}ms  "
                        f"fwd={entry['fwd_ms']:.0f}ms  "
                        f"bwd={entry['bwd_ms']:.0f}ms  "
                        f"opt={entry['opt_ms']:.0f}ms",
                        flush=True,
                    )

            # Advance the profiler only during the profile phase so the schedule
            # window aligns correctly with the steps that actually matter.
            if is_profile:
                prof.step()

    backend.close()

    # -------------------------------------------------------------------------
    # 6. Per-rank stats
    # -------------------------------------------------------------------------
    if not sync_timings:
        if dist.is_main_process:
            print("[profile_training] No sync timing data. Increase profile_steps.", flush=True)
        return

    def _mean_std(values: List[float]):
        if not values:
            return 0.0, 0.0
        mean = statistics.mean(values)
        std  = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, std

    keys = ['data_ms', 'mask_ms', 'fwd_ms', 'bwd_ms', 'opt_ms', 'wall_ms']
    sync_mean = {k: _mean_std([e[k] for e in sync_timings])[0] for k in keys}
    sync_std  = {k: _mean_std([e[k] for e in sync_timings])[1] for k in keys}

    nosync_bwd_mean, nosync_bwd_std = _mean_std([e['bwd_ms'] for e in no_sync_timings])
    nccl_est_ms = max(0.0, sync_mean['bwd_ms'] - nosync_bwd_mean)

    # Flat tensor for cross-rank all_gather: indices map to columns in the summary.
    # [0]=data  [1]=mask  [2]=fwd  [3]=sync_bwd  [4]=nosync_bwd  [5]=opt  [6]=wall
    rank_vec = torch.tensor([
        sync_mean['data_ms'],
        sync_mean['mask_ms'],
        sync_mean['fwd_ms'],
        sync_mean['bwd_ms'],
        nosync_bwd_mean,
        sync_mean['opt_ms'],
        sync_mean['wall_ms'],
    ], dtype=torch.float32, device=dist.device)

    # -------------------------------------------------------------------------
    # 7. Cross-rank aggregation
    # -------------------------------------------------------------------------
    if dist.is_distributed:
        gathered = [torch.zeros_like(rank_vec) for _ in range(dist.world_size)]
        tdist.all_gather(gathered, rank_vec)
        all_ranks = torch.stack(gathered, dim=0)   # (world_size, 7)
    else:
        all_ranks = rank_vec.unsqueeze(0)           # (1, 7)

    # -------------------------------------------------------------------------
    # 8. Print summary (rank 0 only)
    # -------------------------------------------------------------------------
    if not dist.is_main_process:
        return

    L     = cfg['model']['num_layers']
    D     = cfg['model']['model_dim']
    world = dist.world_size

    def _cross(col: int) -> str:
        vals = all_ranks[:, col]
        mn   = vals.min().item()
        md   = vals.median().item()
        mx   = vals.max().item()
        return f"{mn:>6.0f} / {md:>6.0f} / {mx:>6.0f}ms"

    nccl_per_rank = (all_ranks[:, 3] - all_ranks[:, 4]).clamp(min=0.0)
    nccl_mn = nccl_per_rank.min().item()
    nccl_md = nccl_per_rank.median().item()
    nccl_mx = nccl_per_rank.max().item()

    sep = "=" * 72

    print(f"\n{sep}", flush=True)
    print(f"  profile_training: step timing summary", flush=True)
    print(f"  config: {L}L/{D}D  seq={seq_len}  world={world}", flush=True)
    print(f"  warmup={warmup_steps}  no_sync={no_sync_steps}  profile={profile_steps}", flush=True)
    print(f"  compile={cfg['model'].get('compile', True)}  mask={cfg.get('model',{}).get('mask_type','?')}  "
          f"static_graph={ddp_static_graph}  bucket_cap_mb={ddp_bucket_cap_mb}  "
          f"optimize_ddp={use_optimize_ddp}", flush=True)
    print(sep, flush=True)
    print(
        f"{'Phase':<22} | {'This Rank (mean ± std)':<28} | {'All Ranks (min / med / max)'}",
        flush=True,
    )
    print(f"{'-'*22}-+-{'-'*28}-+-{'-'*32}", flush=True)

    rows = [
        # (label,                col, mean,                    std,                   tag)
        ("data_load",            0,   sync_mean['data_ms'],    sync_std['data_ms'],   "[CPU]"),
        ("mask_create",          1,   sync_mean['mask_ms'],    sync_std['mask_ms'],   "[CPU]"),
        ("forward",              2,   sync_mean['fwd_ms'],     sync_std['fwd_ms'],    "[GPU]"),
        ("bwd+NCCL (sync)",      3,   sync_mean['bwd_ms'],     sync_std['bwd_ms'],    "[GPU]"),
        ("bwd only (nosync)",    4,   nosync_bwd_mean,         nosync_bwd_std,        "[GPU]"),
        ("optim_step",           5,   sync_mean['opt_ms'],     sync_std['opt_ms'],    "[GPU]"),
        ("step_wall",            6,   sync_mean['wall_ms'],    sync_std['wall_ms'],   "     "),
    ]

    for label, col, mean, std, tag in rows:
        this_str  = f"{mean:>6.0f}ms ± {std:>4.0f}ms {tag}"
        cross_str = _cross(col)
        print(f"{label:<22} | {this_str:<28} | {cross_str}", flush=True)

    # NCCL estimate (derived, not gathered directly)
    nccl_this = f"{nccl_est_ms:>6.0f}ms (derived)      "
    nccl_all  = f"{nccl_mn:>6.0f} / {nccl_md:>6.0f} / {nccl_mx:>6.0f}ms"
    print(f"{'NCCL est.':<22} | {nccl_this:<28} | {nccl_all}", flush=True)

    print(sep, flush=True)
    if no_sync_steps == 0:
        print("  Note: no_sync_steps=0 — NCCL estimate unavailable", flush=True)
    if profile_steps > 0:
        print(f"  Chrome traces → {trace_dir}/rank*.json.gz", flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    import argparse
    import datetime

    parser = argparse.ArgumentParser(add_help=False)
    cfg = compose_config(parser)

    run_id  = datetime.datetime.now().strftime("profile_%Y%m%d_%H%M%S")
    run_dir = Path(__file__).parent / "runs" / run_id

    dist_mgr = DistributedManager()
    rep = ReproducibilityManager(
        output_dir=str(run_dir),
        is_main_process=dist_mgr.is_main,
    )
    with dist_mgr, rep:
        main(cfg, dist_mgr, rep)
