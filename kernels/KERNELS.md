# Custom Triton Attention Kernels

Custom flash-attention kernels for the cross-document BIM (Block Interaction Mask)
attention pattern used during TAGSeq2TAGSeq training.

Benchmark harness: `python benchmarks/attention_harness.py bench --seq-lens 32768 --doc-lens 512`
Correctness suite: `python benchmarks/attention_harness.py correctness --seq-len 512 --doc-len 128`

---

## Latest Results (A100 80GB, bf16, 16 heads, Dh=64, doc_len=512)

All ratios are relative to compiled FlexAttention (`torch.compile(flex_attention, dynamic=True)`).
Ratio < 1.0 = faster than flex. Memory = peak HBM delta during fwd / fwd+bwd.

| Kernel | T=4k | T=8k | T=16k | T=32k | fwd_MB | bwd_MB |
|--------|------|------|-------|-------|--------|--------|
| flex (ref) | 1.00× | 1.00× | 1.00× | 1.00× | 9→69 | 17→138 |
| v4 (old best) | 1.24× | 1.24× | 1.25× | 1.30× | 4× more | 4× more |
| v11 | 1.03× | 1.00× | 0.99× | 1.00× | same as flex | same as flex |
| **v12 ✓ BEST** | **0.91×** | **0.84×** | **0.78×** | **0.77×** | **same as flex** | **same as flex** |
| v13 | 0.92× | 0.84× | 0.78× | 0.76× | same as flex | same as flex |

At T=32768, doc_len=512 (absolute):

| Kernel | fwd_ms | bwd_ms | fwd_MB | bwd_MB |
|--------|--------|--------|--------|--------|
| flex   | 2.38   | 6.30   | 69     | 138    |
| v4     | 2.24   | 9.06   | 271    | 340    |
| v11    | 1.25   | 7.41   | 69     | 138    |
| **v12**| **1.14** | **5.52** | **69** | **138** |
| v13    | 1.10   | 5.55   | 69     | 138    |

**v12 beats flex by 23% on fwd+bwd and uses identical memory.**
Forward is 2× faster than flex; backward is 12% faster.
Speedup increases with T — at T=4k the gain is smaller but still present.

### Performance across head dimensions (T=32768, doc_len=512, H=16)

| Dh | flex fwd | flex bwd | v12 fwd | v12 bwd | **v12 total ratio** |
|----|---------|---------|--------|--------|---------------------|
| 32  | 1.98ms | 3.95ms  | 0.91ms | 2.50ms | **0.57×** |
| 64  | 2.38ms | 6.30ms  | 1.14ms | 5.54ms | **0.77×** |
| 128 | 2.24ms | 177.3ms | 2.06ms | 16.85ms | **0.11×** |

At Dh=128, flex backward collapses to 177ms (numerically correct, verified).
v12 backward is 10.5× faster (16.85ms). Total speedup: ~9×.
FlexAttention appears to have no optimized backward path for Dh=128 with custom
block masks at T=32k. v12 is the only viable option for Dh=128 workloads.

---

## Kernel Inventory

| File | Description |
|------|-------------|
| `causal_attn.py` | Full causal attention — `triton_causal` |
| `varlen_attn.py` | Doc-causal via cu_seqlens — `triton_varlen` |
| `cross_doc_naive_attn.py` | Cross-doc + dense bool mask — `triton_cross_doc_naive` |
| `cross_doc_bitmask_attn.py` | Cross-doc + bitmasks, OR-reduction block skip — `cdb_v1` |
| `cross_doc_bitmask_bim_v1.py` | BIM CSR forward + backward — `cdb_bim_v1` |
| `cross_doc_bitmask_bim_v2.py` | v1 + diagonal-first backward — `cdb_bim_v2` |
| `cross_doc_bitmask_bim_v3.py` | v2 + full/partial block split — `cdb_bim_v3` |
| `cross_doc_bitmask_bim_v4.py` | v3 + split dKV/dQ backward kernels — `cdb_bim_v4` |
| `cross_doc_bitmask_bim_v5.py` | Dead end: O(T²) bitmask scan — `cdb_bim_v5` |
| `cross_doc_bitmask_bim_v6.py` | Dead end: double-tile via in-kernel CSR merge — `cdb_bim_v6` |
| `cross_doc_bitmask_bim_v7.py` | Dead end: precomputed coarse CSR — `cdb_bim_v7` |
| `cross_doc_bitmask_bim_v8.py` | Dead end: varlen-style same-doc pipelining — `cdb_bim_v8` |
| `cross_doc_bitmask_bim_v9.py` | Dead end: split varlen-style pipelining — `cdb_bim_v9` |
| `cross_doc_bitmask_bim_v10.py` | **bf16 TC matmuls** — `cdb_bim_v10` |
| `cross_doc_bitmask_bim_v11.py` | v10 + **no permute/contiguous copies** — `cdb_bim_v11` |
| `cross_doc_bitmask_bim_v12.py` | v11 + **BIM_BLOCK_SIZE=128** — `cdb_bim_v12` (**BEST**) |
| `cross_doc_bitmask_bim_v13.py` | v12 + **pure-cross dispatch** — `cdb_bim_v13` |

---

## What Made v4→v12 Work

### v10 — Native bf16 Tensor Core matmuls
Remove all `.to(tl.float32)` casts at load sites. Feed bf16 tensors directly to
`tl.dot`; it accumulates into fp32 via Tensor Cores natively. Scale factors that
were pre-multiplied into K/Q (before tl.dot) move to post-multiply on the fp32
output. Intermediates fed back into matmuls (P_T, dLdS_T) are cast to bf16 before
each `tl.dot` call.

Effect: register pressure for K/V/Q/dLdO tiles halved (bf16=2B vs fp32=4B).
Higher SM occupancy → better memory-latency hiding. Explicit upcast instructions
eliminated at every load site.

### v11 — Eliminate permute/contiguous copies
Prior autograd functions did `q.permute(1,0,2).unsqueeze(0).contiguous()` to
convert (T,H,Dh) → (1,H,T,Dh) before each kernel call. The kernels accept
arbitrary strides; pass THD strides `(H·Dh, Dh, 1)` directly instead.

Effect: eliminates ~200 MB of unnecessary HBM memcpy per fwd+bwd call at T=32k.
Memory drops from 271/340 MB to 69/138 MB (flex parity). Also contributes ~0.35ms
forward speedup by removing three full-tensor copies from the critical path.

### v12 — BIM_BLOCK_SIZE=128
With fp32 K/V tiles, BIM_BS=128 was infeasible: K(128,64) fp32=32KB + V=32KB +
dLdK fp32=32KB + dLdV fp32=32KB = 128KB persistent registers, leaving <2 CTAs/SM.
With bf16 from v10, K+V = 32KB total, so persistent pressure = 96KB → ~2.7 CTAs/SM
(versus ~4 for BS=64). The trade-off is worth it: each backward CTA now does 4×
larger matmuls (128×MICRO vs 64×MICRO), achieving far better TC utilization and
cutting the backward from 7.41ms to 5.52ms.

BIM rebuild: `_build_bim_128()` in `cross_doc_bitmask_bim_v12.py`. Separate BIM
structures coexist (BS=64 for v1–v11, BS=128 for v12+).

### v13 — Pure-cross block classification
Adds `q_kv_n_pure_cross` / `kv_q_n_pure_cross` to `BlockInteractionMask` and
a new bitmask-only inner kernel path. Off-diagonal block pairs where both blocks
are single-doc in distinct documents (`blk_is_pure=True`, different doc IDs) have
`same_doc=False` for all position pairs by construction → only bitmask masking
needed, no doc_id load or same_doc computation.

At BIM_BS=128 with aligned doc lengths (e.g. doc_len=512=4×128), 100% of
non-full off-diagonal blocks qualify. In practice the gain is ~1% on forward and
negligible on backward — the masking cost is small vs matmul compute. v12 is
preferred for its simplicity.

---

## Dead Ends (v5–v9)

All five were attempts to increase the backward's matmul tile size by other means,
but each ran into a fundamental blocker before v10's bf16 register reduction made
BIM_BS=128 viable:

- **v5**: replaced BIM-guided traversal with an O(T) bitmask union scan per CTA
  → O(T²) total work at any realistic T. Strictly worse.
- **v6**: merged two adjacent BIM blocks per CTA to get 128-token KV tiles, but
  in-kernel deduplication of the merged CSR lists added overhead exceeding the gain.
- **v7**: precomputed a coarse 128-token CSR to avoid v6's in-kernel merging.
  Added data structure complexity with marginal benefit.
- **v8**: hypothesis that the per-CSR-entry sub-kernel calls prevented software
  pipelining. In fact, the inner `range(num_steps=4)` loop IS pipelined by Triton
  (num_stages=4); the per-entry call overhead is just one tl.load + branch (~5ns).
- **v9**: split version of v8. Same dead end.

---

## BlockInteractionMask CSR Row Ordering

```
q_kv row  (for each Q-block, sorted KV-block indices):
  [full(0..n_full-1),  pure_cross(n_full..n_full+n_pc-1),
   boundary(n_full+n_pc..count-2),  diagonal(count-1)]

kv_q row  (for each KV-block, sorted Q-block indices):
  [diagonal(0),  full(1..1+n_full-1),  pure_cross(1+n_full..1+n_full+n_pc-1),
   boundary(1+n_full+n_pc..count-1)]
```

- **full**: both blocks single-doc, same doc, off-diagonal → all positions attend,
  no masking needed (`_attn_fwd_inner_full_v10`)
- **pure_cross**: both blocks single-doc, different docs, off-diagonal → bitmask
  only, no same_doc check (`_attn_fwd_inner_pure_cross_v13`)
- **boundary**: at least one block straddles a doc boundary → full masking
  `same_doc | in_grant` (`_attn_fwd_inner_cdb_v10`)
- **diagonal**: same block for Q and KV → full masking + causal

---

## Tried and Failed (v14–v16)

These were implemented, tested correct, benchmarked, and scrapped because they did
not improve on v12 at T=32k:

### v14 — Expanded autotune (num_warps=16, num_stages ∈ {2,6})
**REGRESSION** at T=32k: bwd 5.57ms → 7.32ms (+31%).
Root cause: with 60 configs instead of 24, the noisy single-run autotune
selected `BLOCK_SIZE_MICRO=128` for Q_bwd (making `num_micro=128/128=1`,
a degenerate inner loop with no pipelining). The narrow {4,8}×{3,4,5}×{16,32,64,128}
search in v10 found the correct configs; expanding to {4,8,16}×{2,3,4,5,6}×{...}
caused a false selection. v10's narrow autotune is already optimal.

Also discovered: `BLOCK_SIZE_MICRO` must be the **last** positional param before
any autotuned constexprs, because the autotuner passes it as a kwarg. New params
added after it in the signature will collide. See v15/v16 for the correct placement.

### v15 — Persistent-CTA backward (atomic work-stealing, dKV + dQ)
**REGRESSION** at T=32k: bwd 5.58ms → 6.15ms (+10%). Neutral or slight improvement
at T≤8k.
Root cause: persistent CTAs hurt K/V cache utilization. In the original (n_blocks, H)
grid, each CTA loads K[k] and V[k] once (16KB each) and processes it to completion.
With 13 persistent CTAs per head, each CTA claims ~20 KV blocks and loads K+V
20 times in sequence — 20×32KB = 640KB per CTA, vs 32KB in the non-persistent
version. The L2 (40 MB) cannot hold all K/V blocks for all 13 active CTAs
simultaneously, causing DRAM thrashing that outweighs the load-balancing gain.

Implementation notes:
- Bug: `break` inside `for _ in range(runtime_n)` in Triton compiles as a function-
  level early return (stores never fire, all gradients remain uninitialized). Fixed by
  using `if pid < n_blocks:` to guard the body instead.
- New runtime params (`n_blocks`, `max_steps`, `work_counter_ptr`) must appear
  **before** `BLOCK_SIZE_MICRO: tl.constexpr` in the kernel signature, otherwise
  the autotuner's kwarg for BLOCK_SIZE_MICRO collides with positional arg n_blocks.

### v16 — .cg cache hints on bitmask loads
**NO IMPROVEMENT** at T=32k: 0.77× identical.
Root cause: at T=32k, bitmasks (2×256KB = 512KB total) fit comfortably in A100's
40 MB L2 with the default `.ca` policy. The bitmask data is already L2-resident
across heads without explicit hints. The backward is memory-bandwidth-bound on Q/K/V
and dLdO tensors (64 MB each), not on bitmasks.

---

## Kernel development complete

v12 is the production kernel. No further kernel work is planned. The "tried and failed"
log (v14–v16) and the dead-end history (v5–v9) are preserved above for reference.
