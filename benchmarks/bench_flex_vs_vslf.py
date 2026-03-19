"""Benchmark: FlexAttention (doc_causal) vs PyTorch built-in varlen flash attention.

For doc_causal masks, torch.ops.aten._flash_attention_forward with cu_seqlens is a
perfect semantic equivalent — each packed document gets independent causal attention.
This isolates whether FlexAttention's compilation overhead vs a clean flash kernel
justifies writing a custom Triton kernel.

The key question: if flex_attention with a plain doc_causal BlockMask is ~equivalent
in speed to VSLF, the bottleneck during training is elsewhere (mask density / NCCL),
not in FlexAttention's kernel dispatch overhead.

Usage:
    python benchmarks/bench_flex_vs_vslf.py
    python benchmarks/bench_flex_vs_vslf.py --seq-lens 8192 16384 32768 --doc-lens 512 2048
"""

import argparse
import math
import statistics
from typing import List, Tuple

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

DEVICE = torch.device("cuda")

# ---------------------------------------------------------------------------
# Compiled flex_attention (amortises triton JIT across calls)
# ---------------------------------------------------------------------------
_compiled_flex = torch.compile(flex_attention, dynamic=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc_spans_and_cu(seq_len: int, doc_len: int) -> Tuple[List[Tuple[int, int]], torch.Tensor, int]:
    """Uniform-length docs packed into seq_len. Returns (spans, cu_seqlens, max_seqlen)."""
    starts = list(range(0, seq_len, doc_len))
    spans = [(s, min(s + doc_len, seq_len)) for s in starts]
    lengths = [e - s for s, e in spans]
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0).to(DEVICE)
    return spans, cu, max(lengths)


def make_block_mask(seq_len: int, spans: List[Tuple[int, int]]) -> "BlockMask":
    doc_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=DEVICE)
    for i, (start, end) in enumerate(spans):
        doc_ids[start:end] = i

    def doc_causal_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[q_idx] == doc_ids[kv_idx])

    return create_block_mask(
        doc_causal_mod, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=DEVICE
    )


def time_fn(fn, warmup: int, iters: int) -> Tuple[float, float]:
    """Returns (median_ms, stdev_ms) over `iters` GPU-timed runs after `warmup` warm-ups."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Benchmark kernels
# ---------------------------------------------------------------------------

def bench_flex(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_mask,
    scale: float,
    fwd_only: bool,
    compiled: bool,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    # q/k/v shape: (1, H, T, D)
    attn_fn = _compiled_flex if compiled else flex_attention

    def fwd():
        return attn_fn(q, k, v, block_mask=block_mask, scale=scale)

    def fwd_bwd():
        q.grad = k.grad = v.grad = None
        out = attn_fn(q, k, v, block_mask=block_mask, scale=scale)
        out.sum().backward()

    fn = fwd if fwd_only else fwd_bwd
    return time_fn(fn, warmup, iters)


def bench_vslf(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
    fwd_only: bool,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    # q/k/v shape: (T, H, D)
    _vslf = torch.ops.aten._flash_attention_forward

    def fwd():
        return _vslf(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                     0.0, True, False, scale=scale)

    def fwd_bwd():
        q.grad = k.grad = v.grad = None
        out, lse, _, _, _ = _vslf(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                   0.0, True, False, scale=scale)
        out.sum().backward()

    fn = fwd if fwd_only else fwd_bwd
    return time_fn(fn, warmup, iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(
    seq_lens: List[int],
    doc_lens: List[int],
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
):
    scale = head_dim ** -0.5

    sep = "=" * 100
    print(f"\n{sep}")
    print(f"  FlexAttention (doc_causal) vs PyTorch varlen flash attention")
    print(f"  num_heads={num_heads}  head_dim={head_dim}  dtype={dtype}  warmup={warmup}  iters={iters}")
    print(sep)
    print(
        f"  {'seq_len':>7}  {'doc_len':>7}  {'n_docs':>6}  {'sparsity':>8}  "
        f"{'flex_fwd_ms':>12}  {'vslf_fwd_ms':>12}  {'ratio_fwd':>9}  "
        f"{'flex_fwdbwd_ms':>14}  {'vslf_fwdbwd_ms':>14}  {'ratio_fwdbwd':>12}"
    )
    print(f"  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*9}  {'-'*14}  {'-'*14}  {'-'*12}")

    for seq_len in seq_lens:
        for doc_len in doc_lens:
            if doc_len > seq_len:
                continue

            spans, cu_seqlens, max_seqlen = make_doc_spans_and_cu(seq_len, doc_len)
            n_docs = len(spans)
            # Approximate sparsity for doc_causal: sum of triangular areas / full square
            total_attended = sum((e - s) * (e - s + 1) // 2 for s, e in spans)
            sparsity_pct = 100.0 * (1.0 - total_attended / (seq_len * seq_len))

            # Build BlockMask (CPU time excluded from GPU timings)
            block_mask = make_block_mask(seq_len, spans)

            # Create QKV tensors
            # FlexAttention: (B=1, H, T, D)
            qf = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
            kf = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
            vf = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
            # VSLF: (T, H, D)
            qv = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
            kv = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
            vv = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=DEVICE, requires_grad=True)

            # --- Forward only ---
            flex_fwd, _ = bench_flex(qf, kf, vf, block_mask, scale, fwd_only=True,
                                     compiled=True, warmup=warmup, iters=iters)
            vslf_fwd, _ = bench_vslf(qv, kv, vv, cu_seqlens, max_seqlen, scale,
                                     fwd_only=True, warmup=warmup, iters=iters)

            # --- Forward + backward ---
            flex_fb, _ = bench_flex(qf, kf, vf, block_mask, scale, fwd_only=False,
                                    compiled=True, warmup=warmup, iters=iters)
            vslf_fb, _ = bench_vslf(qv, kv, vv, cu_seqlens, max_seqlen, scale,
                                    fwd_only=False, warmup=warmup, iters=iters)

            ratio_fwd = flex_fwd / vslf_fwd
            ratio_fb = flex_fb / vslf_fb

            print(
                f"  {seq_len:>7}  {doc_len:>7}  {n_docs:>6}  {sparsity_pct:>7.1f}%  "
                f"{flex_fwd:>11.1f}  {vslf_fwd:>11.1f}  {ratio_fwd:>8.2f}x  "
                f"{flex_fb:>13.1f}  {vslf_fb:>13.1f}  {ratio_fb:>11.2f}x",
                flush=True,
            )

    print(sep)
    print("  ratio = flex / vslf  (>1.0 means flex is slower)")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--doc-lens", type=int, nargs="+", default=[256, 1024, 4096, 16384])
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("Compiling flex_attention (first call will be slow)...", flush=True)

    # Force an initial compile at smallest config to amortise JIT before benchmarking
    _warmup_seq = min(args.seq_lens)
    _warmup_doc = min(args.doc_lens)
    spans0, cu0, ms0 = make_doc_spans_and_cu(_warmup_seq, _warmup_doc)
    bm0 = make_block_mask(_warmup_seq, spans0)
    q0 = torch.randn(1, args.num_heads, _warmup_seq, args.head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
    k0 = torch.randn_like(q0, requires_grad=True)
    v0 = torch.randn_like(q0, requires_grad=True)
    out0 = _compiled_flex(q0, k0, v0, block_mask=bm0, scale=args.head_dim ** -0.5)
    out0.sum().backward()
    torch.cuda.synchronize()
    print("Compile done.\n")

    run_benchmark(
        seq_lens=args.seq_lens,
        doc_lens=args.doc_lens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
