"""Probe cdb_bim_v18 correctness on the exact packs thestack training sees.

Simulates rank R of a world_size W DDP run and checks correctness of the
training kernel against flex for every step in [start, end].  Useful for
finding mask structures in the real dataset that corrupt the kernel without
needing to reproduce the full training loop.

Does NOT model weight dynamics — q/k/v are random.  A failure here is a
mask/kernel bug; absence of failure doesn't rule out weight-driven instability.

Usage:
    CUDA_VISIBLE_DEVICES=1 python benchmarks/thestack_nan_probe.py \\
        --parquet schedules/thestack_bfs/epoch_0/packs.parquet \\
        --steps 0 199 \\
        --rank 0 \\
        --world-size 16 \\
        --max-grants 256

    # Also test rank 3, steps around known NaN window
    CUDA_VISIBLE_DEVICES=1 python benchmarks/thestack_nan_probe.py \\
        --parquet schedules/thestack_bfs/epoch_0/packs.parquet \\
        --steps 90 160 \\
        --rank 3 \\
        --world-size 16 \\
        --max-grants 256 \\
        --no-backward
"""
from __future__ import annotations

import argparse
import collections
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.attention_harness import (
    MaskInputs,
    _build_cross_doc_masks,
    _build_flex_cross_doc_block_mask,
    _clone_requires_grad,
    _compiled_flex,
    _p99,
    _spans_to_cu_seqlens,
    _spans_to_document_ids,
    _to_bhnd,
    _to_thd,
)
from data.bucketed_pack_dataset import _make_bucket_sequence
from data.epoch_precompute import PackRecord, _table_to_records

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Pack loading
# ---------------------------------------------------------------------------

def load_bucket_lists(parquet_path: str) -> Dict[int, List[PackRecord]]:
    """Load parquet and group records by bucket_id, sorted by pack_id."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    records = _table_to_records(table)
    bucket_lists: Dict[int, List[PackRecord]] = collections.defaultdict(list)
    for r in records:
        bucket_lists[r.bucket_id].append(r)
    for b in bucket_lists:
        bucket_lists[b].sort(key=lambda r: r.pack_id)
    return dict(bucket_lists)


def iter_rank_packs(
    bucket_lists: Dict[int, List[PackRecord]],
    n_buckets: int,
    world_size: int,
    rank: int,
    start_step: int,
    end_step: int,
    epoch_seed: int = 0,
) -> Iterator[Tuple[int, int, PackRecord]]:
    """Yield (step, bucket_id, pack) for the given rank at each step in [start, end].

    Replicates BucketedPackDataset's bucket selection exactly:
    - _make_bucket_sequence determines the bucket at each accum step.
    - rank r draws pack at index bucket_consumed[b] + r.
    - bucket_consumed[b] advances by world_size after each accum step.
    No fallback logic needed for the first few hundred steps (buckets have
    thousands of packs, only a handful are consumed).
    """
    bucket_seq = _make_bucket_sequence(n_buckets, seed=epoch_seed)
    bucket_consumed: Dict[int, int] = collections.defaultdict(int)

    for step in range(end_step + 1):
        chosen = bucket_seq[step % len(bucket_seq)]
        consumed = bucket_consumed[chosen]
        packs = bucket_lists.get(chosen, [])
        idx = consumed + rank
        if idx >= len(packs):
            raise RuntimeError(
                f"step {step}: bucket {chosen} exhausted "
                f"(need index {idx}, only {len(packs)} packs)"
            )
        pack = packs[idx]
        bucket_consumed[chosen] += world_size
        if step >= start_step:
            yield step, chosen, pack


# ---------------------------------------------------------------------------
# MaskInputs from PackRecord
# ---------------------------------------------------------------------------

def pack_to_mask_inputs(
    record: PackRecord,
    max_grants: int,
    device: torch.device,
) -> Tuple[MaskInputs, int]:
    """Build MaskInputs from a PackRecord.

    Returns (mask_inputs, n_grants_encoded).
    Skips the [T,T] dense_mask (1 GB at T=32768) since we use flex as the
    reference rather than naive math.  doc_ids are remapped to pack-order
    0-indexed values matching cu_seqlens offsets.
    """
    # doc_spans with 0-indexed doc_id
    pos = 0
    doc_spans = []
    for i, eff_len in enumerate(record.effective_lens):
        doc_spans.append(SimpleNamespace(doc_id=i, start=pos, end=pos + eff_len))
        pos += eff_len
    seq_len = pos

    # Remap raw GraphIndex doc IDs → 0-indexed pack positions
    raw_to_idx = {raw: i for i, raw in enumerate(record.doc_ids)}
    link_to_target: Dict[int, List[int]] = {}
    for link_pos, targets in zip(record.link_end_positions, record.link_target_doc_ids):
        remapped = [raw_to_idx[t] for t in targets if t in raw_to_idx]
        if remapped:
            link_to_target[int(link_pos)] = remapped

    cu_seqlens, max_seqlen = _spans_to_cu_seqlens(doc_spans, device)
    document_ids = _spans_to_document_ids(doc_spans, seq_len, device)

    # Build bitmasks only (skip dense_mask)
    n_chunks = max(1, (max_grants + 63) // 64)
    q_bm_list = [torch.zeros(seq_len, dtype=torch.int64, device=device) for _ in range(n_chunks)]
    kv_bm_list = [torch.zeros(seq_len, dtype=torch.int64, device=device) for _ in range(n_chunks)]
    grant_idx = 0
    for link_pos, target_doc_ids in sorted(link_to_target.items()):
        for target_doc_id in target_doc_ids:
            link_span = next((s for s in doc_spans if s.start < link_pos <= s.end), None)
            target_span = next((s for s in doc_spans if s.doc_id == target_doc_id), None)
            if link_span is None or target_span is None:
                continue
            gs, ge = link_pos, min(seq_len, link_span.end)
            ts, te = max(0, target_span.start), min(seq_len, target_span.end)
            if gs >= ge or ts >= te or grant_idx >= max_grants:
                continue
            chunk = grant_idx // 64
            bit_pos = grant_idx % 64
            bit = (1 << bit_pos) if bit_pos < 63 else -(1 << 63)
            q_bm_list[chunk][gs:ge] |= bit
            kv_bm_list[chunk][ts:te] |= bit
            grant_idx += 1
    q_bitmasks = torch.stack(q_bm_list)
    kv_bitmasks = torch.stack(kv_bm_list)

    # Doc-causal block mask
    def _dc_mod(b, h, qi, ki):
        return (qi >= ki) & (document_ids[qi] == document_ids[ki])
    flex_doc_causal_bm = create_block_mask(
        _dc_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )

    # Cross-doc flex block mask (needed for flex reference)
    flex_cross_doc_bm = None
    if grant_idx > 0:
        flex_cross_doc_bm = _build_flex_cross_doc_block_mask(
            seq_len, document_ids, q_bitmasks, kv_bitmasks, device,
        )

    # BIM for the standard 64-block BIM (legacy; not used by v18)
    bim = None
    try:
        from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
        c = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
        c.triton_block_size = 64
        c._n_chunks = n_chunks
        bim = CrossDocLinkMaskCreator._build_block_interaction_mask(
            c, seq_len, document_ids, list(q_bitmasks), list(kv_bitmasks), device,
        )
    except Exception:
        pass

    mask_inputs = MaskInputs(
        seq_len=seq_len,
        doc_spans=doc_spans,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        document_ids=document_ids,
        dense_mask=None,          # skipped — use flex as reference
        q_bitmasks=q_bitmasks,
        kv_bitmasks=kv_bitmasks,
        flex_doc_causal_block_mask=flex_doc_causal_bm,
        flex_cross_doc_block_mask=flex_cross_doc_bm,
        bim=bim,
    )
    return mask_inputs, grant_idx


# ---------------------------------------------------------------------------
# Impl runners (build BIMs inline — no shared cache, no id() collision risk)
# ---------------------------------------------------------------------------

def _run_flex(q, k, v, mask_inputs: MaskInputs, scale: float) -> torch.Tensor:
    """FlexAttention with cross_doc BlockMask.  Falls back to doc_causal if no grants."""
    if mask_inputs.flex_cross_doc_block_mask is not None:
        bm = mask_inputs.flex_cross_doc_block_mask
    else:
        bm = mask_inputs.flex_doc_causal_block_mask
    q4, k4, v4 = _to_bhnd(q), _to_bhnd(k), _to_bhnd(v)
    out4 = _compiled_flex(q4, k4, v4, block_mask=bm, scale=scale)
    return _to_thd(out4)


def _run_v18(q, k, v, mask_inputs: MaskInputs, scale: float) -> torch.Tensor:
    """cdb_bim_v18 with freshly-built BIMs (no shared cache)."""
    from kernels.cross_doc_bitmask_bim_v17 import _build_bim_64
    from kernels.cross_doc_bitmask_bim_v12 import _build_bim_128
    from kernels.cross_doc_bitmask_bim_v18 import triton_attn_cross_doc_bitmask_bim_v18
    n_chunks = mask_inputs.q_bitmasks.shape[0]
    dev = mask_inputs.document_ids.device
    bim128 = _build_bim_128(
        mask_inputs.seq_len, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, dev, n_chunks,
    )
    bim64 = _build_bim_64(
        mask_inputs.seq_len, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, dev, n_chunks,
    )
    return triton_attn_cross_doc_bitmask_bim_v18(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, bim64, scale,
    )


# ---------------------------------------------------------------------------
# Per-step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step: int
    bucket: int
    kv_block_count: int
    seq_len: int
    n_docs: int
    n_grants: int
    fwd_max_err: float = 0.0
    fwd_p99_err: float = 0.0
    bwd_max_err: float = 0.0
    bwd_p99_err: float = 0.0
    v18_fwd_nan: bool = False
    v18_fwd_inf: bool = False
    v18_bwd_nan: bool = False
    v18_bwd_inf: bool = False
    error: Optional[str] = None

    @property
    def any_nan_inf(self) -> bool:
        return self.v18_fwd_nan or self.v18_fwd_inf or self.v18_bwd_nan or self.v18_bwd_inf


# ---------------------------------------------------------------------------
# Per-step check
# ---------------------------------------------------------------------------

def check_step(
    record: PackRecord,
    max_grants: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    check_backward: bool,
    step: int,
    bucket: int,
) -> StepResult:
    mask_inputs, n_grants = pack_to_mask_inputs(record, max_grants, DEVICE)
    scale = head_dim ** -0.5

    result = StepResult(
        step=step,
        bucket=bucket,
        kv_block_count=record.kv_block_count,
        seq_len=mask_inputs.seq_len,
        n_docs=len(mask_inputs.doc_spans),
        n_grants=n_grants,
    )

    # Random q/k/v — deterministic per step for reproducibility
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(step)
    T = mask_inputs.seq_len
    q = torch.randn(T, num_heads, head_dim, dtype=dtype, device=DEVICE, generator=gen).requires_grad_(True)
    k = torch.randn(T, num_heads, head_dim, dtype=dtype, device=DEVICE, generator=gen).requires_grad_(True)
    v = torch.randn(T, num_heads, head_dim, dtype=dtype, device=DEVICE, generator=gen).requires_grad_(True)

    # Reference: flex
    q_flex = _clone_requires_grad(q)
    k_flex = _clone_requires_grad(k)
    v_flex = _clone_requires_grad(v)
    try:
        out_flex = _run_flex(q_flex, k_flex, v_flex, mask_inputs, scale)
        if check_backward:
            out_flex.backward(torch.ones_like(out_flex))
            dq_flex = q_flex.grad.clone()
            dk_flex = k_flex.grad.clone()
            dv_flex = v_flex.grad.clone()
    except Exception as e:
        result.error = f"flex error: {str(e)[:120]}"
        return result

    # v18
    q_v18 = _clone_requires_grad(q)
    k_v18 = _clone_requires_grad(k)
    v_v18 = _clone_requires_grad(v)
    try:
        out_v18 = _run_v18(q_v18, k_v18, v_v18, mask_inputs, scale)
    except Exception as e:
        result.error = f"v18 fwd error: {str(e)[:120]}"
        return result

    result.v18_fwd_nan = bool(torch.isnan(out_v18).any())
    result.v18_fwd_inf = bool(torch.isinf(out_v18).any())

    fwd_diff = (out_v18.float() - out_flex.detach().float()).abs()
    result.fwd_max_err = fwd_diff.max().item()
    result.fwd_p99_err = _p99(fwd_diff)

    if check_backward:
        try:
            out_v18.backward(torch.ones_like(out_v18))
            dq_v18 = q_v18.grad
            dk_v18 = k_v18.grad
            dv_v18 = v_v18.grad
        except Exception as e:
            result.error = f"v18 bwd error: {str(e)[:120]}"
            return result

        result.v18_bwd_nan = bool(
            torch.isnan(dq_v18).any() or torch.isnan(dk_v18).any() or torch.isnan(dv_v18).any()
        )
        result.v18_bwd_inf = bool(
            torch.isinf(dq_v18).any() or torch.isinf(dk_v18).any() or torch.isinf(dv_v18).any()
        )

        dq_diff = (dq_v18.float() - dq_flex.float()).abs()
        dk_diff = (dk_v18.float() - dk_flex.float()).abs()
        dv_diff = (dv_v18.float() - dv_flex.float()).abs()
        result.bwd_max_err = max(dq_diff.max().item(), dk_diff.max().item(), dv_diff.max().item())
        result.bwd_p99_err = max(_p99(dq_diff), _p99(dk_diff), _p99(dv_diff))

    return result


# ---------------------------------------------------------------------------
# Main probe loop
# ---------------------------------------------------------------------------

def run_probe(
    parquet_path: str,
    start_step: int,
    end_step: int,
    rank: int,
    world_size: int,
    max_grants: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    check_backward: bool,
    n_buckets: int = 32,
    epoch_seed: int = 0,
) -> List[StepResult]:
    print(f"Loading parquet: {parquet_path}")
    bucket_lists = load_bucket_lists(parquet_path)
    total_packs = sum(len(v) for v in bucket_lists.values())
    print(f"  {total_packs} packs across {len(bucket_lists)} buckets")
    print(f"  rank={rank}  world_size={world_size}  steps=[{start_step}, {end_step}]  max_grants={max_grants}")
    print(f"  dtype={dtype}  heads={num_heads}  head_dim={head_dim}  backward={'yes' if check_backward else 'no'}")
    print()

    hdr = (f"  {'step':>5}  {'bkt':>3}  {'kv_blk':>7}  {'T':>6}  "
           f"{'docs':>5}  {'grants':>6}  "
           f"{'fwd_max':>9}  {'fwd_p99':>9}  "
           f"{'bwd_max':>9}  {'bwd_p99':>9}  {'flags'}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    results: List[StepResult] = []
    for step, bucket, pack in iter_rank_packs(
        bucket_lists, n_buckets, world_size, rank, start_step, end_step, epoch_seed,
    ):
        r = check_step(pack, max_grants, num_heads, head_dim, dtype, check_backward, step, bucket)
        results.append(r)

        if r.error:
            flags = f"ERROR: {r.error}"
        else:
            flags_parts = []
            if r.v18_fwd_nan: flags_parts.append("FWD_NAN")
            if r.v18_fwd_inf: flags_parts.append("FWD_INF")
            if r.v18_bwd_nan: flags_parts.append("BWD_NAN")
            if r.v18_bwd_inf: flags_parts.append("BWD_INF")
            flags = " ".join(flags_parts) if flags_parts else "-"

        highlight = " ◄" if (r.any_nan_inf or r.error) else ""
        print(
            f"  {r.step:>5}  {r.bucket:>3}  {r.kv_block_count:>7}  {r.seq_len:>6}  "
            f"{r.n_docs:>5}  {r.n_grants:>6}  "
            f"{r.fwd_max_err:>9.2e}  {r.fwd_p99_err:>9.2e}  "
            f"{r.bwd_max_err:>9.2e}  {r.bwd_p99_err:>9.2e}  "
            f"{flags}{highlight}",
            flush=True,
        )

    print(sep)

    n_nan = sum(1 for r in results if r.any_nan_inf)
    n_err = sum(1 for r in results if r.error)
    n_steps = len(results)
    print(f"\n  {n_steps} steps checked  |  {n_nan} NaN/Inf  |  {n_err} errors")
    if n_nan == 0 and n_err == 0:
        fwd_max = max((r.fwd_max_err for r in results), default=0.0)
        bwd_max = max((r.bwd_max_err for r in results if not r.error), default=0.0)
        print(f"  peak fwd_max_err={fwd_max:.2e}  peak bwd_max_err={bwd_max:.2e}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--parquet", required=True,
                   help="Path to packs.parquet (e.g. schedules/thestack_bfs/epoch_0/packs.parquet)")
    p.add_argument("--steps", type=int, nargs=2, default=[0, 199], metavar=("START", "END"),
                   help="Step range [start, end] inclusive (default: 0 199)")
    p.add_argument("--rank", type=int, default=0,
                   help="DDP rank to simulate (default: 0)")
    p.add_argument("--world-size", type=int, default=16,
                   help="DDP world size (default: 16)")
    p.add_argument("--max-grants", type=int, default=256,
                   help="max_grants used during precompute (default: 256)")
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--no-backward", action="store_true",
                   help="Skip backward pass check (faster)")
    p.add_argument("--n-buckets", type=int, default=32)
    p.add_argument("--epoch-seed", type=int, default=0,
                   help="epoch_idx used as seed for _make_bucket_sequence (default: 0)")
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    results = run_probe(
        parquet_path=args.parquet,
        start_step=args.steps[0],
        end_step=args.steps[1],
        rank=args.rank,
        world_size=args.world_size,
        max_grants=args.max_grants,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        check_backward=not args.no_backward,
        n_buckets=args.n_buckets,
        epoch_seed=args.epoch_seed,
    )

    if any(r.any_nan_inf or r.error for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
