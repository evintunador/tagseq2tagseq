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

## Remaining Optimization Ideas

These have not been tried. Ordered roughly by expected impact:

### High priority

1. **Expanded autotune search space for v12 backward**
   Current backward autotune tries `num_warps ∈ {4,8}`, `num_stages ∈ {3,4,5}`,
   `BLOCK_SIZE_MICRO ∈ {16,32,64,128}`. With BIM_BS=128 and bf16 register reduction,
   try adding `num_warps=16` and `num_stages ∈ {2,6}`. The register file is less
   stressed, so more warps might improve latency hiding without hurting occupancy.

2. **Persistent CTAs for the backward dKV kernel**
   At T=32k, n_blocks=256 × H=16 = 4096 CTAs per backward kernel. On A100 with
   108 SMs, that's ~38 CTA waves. KV-blocks early in the sequence have far fewer
   Q-entries in their CSR (heavily triangular), creating load imbalance across
   waves. A persistent kernel with work-stealing (process multiple KV-blocks per
   SM lifetime) would keep all 108 SMs busy throughout and eliminate last-wave
   inefficiency. Estimated gain: 5–15% on backward.

3. **BIM_BLOCK_SIZE=128 with Dh=128** (MHA architectures with larger head dim)
   At Dh=128, BLOCK_SIZE_KV can go up to 128, enabling 128×128×128 = 2M flop
   TC matmuls per CTA (vs 128×64×64 currently). Likely a much bigger win at
   larger head dimensions typical in modern architectures.

### Medium priority

4. **Cache hints on bitmask loads**
   Bitmask arrays (q_bitmasks, kv_bitmasks) are the same for all H heads; they
   get re-loaded once per head per kernel launch. Use `tl.load(..., cache=".ca")`
   (evict-last / L2-persistent) so the first head's loads populate L2 and
   subsequent heads hit cache. Estimated gain: 1–3% on bitmask-heavy workloads
   (dense grant patterns with many cross-doc blocks).

5. **Fuse forward + backward preprocess into a single kernel**
   `_attn_backward_preprocess_cdb` (computes Delta = rowsum(O·dLdO)) is a separate
   kernel launch that reads O and dLdO in full. These tensors are already live in
   L2 immediately after the forward. Fusing Delta computation into the first
   backward kernel (dKV, which also reads dLdO per Q-tile) would eliminate one
   full HBM read of O and one kernel launch. However, dQ also needs Delta — so
   either compute it twice or pass it between the split kernels via a temporary
   buffer (current approach is already optimal for the split case).

6. **Tensor parallel / sequence parallel sharding**
   At T=32k with H=16, the BIM grid is 256×16 = 4096 CTAs. For longer sequences
   (T=65k–131k), this scales well — BIM_BS=128 gives n_blocks=512–1024. No
   changes needed for standard sequence parallelism; the BIM is built per-device
   over the local sequence shard.

7. **BIM precomputation at dataset build time**
   For training, the BIM is rebuilt every batch by `CrossDocLinkMaskCreator`.
   At BS=128 and T=32k it costs ~5ms on CPU. Precomputing and storing it
   alongside the packed sequences (in `epoch_precompute.py`) would eliminate this
   from the training critical path.

### Lower priority / speculative

8. **Producer-consumer warp specialization**
   Split warps within each CTA into a producer group (issues async loads of K/V/Q
   tiles) and a consumer group (runs MMA instructions). Requires Triton's
   `tl.async_copy` / dot pipelining. Complex to implement but could achieve
   near-roofline arithmetic intensity for the backward. Likely already close to
   this with Triton's software pipelining via `num_stages`.

9. **Reorder BIM CSR traversal for better L2 reuse**
   Currently kv_q CSR lists Q-blocks in ascending order. On A100, L2 is 40 MB;
   at T=32k the Q/K/V tensors are 3×32768×16×64×2 = 192 MB total — doesn't fit.
   Reordering to traverse Q-blocks in a cache-friendly pattern (e.g. Z-order or
   grouped by proximity to the KV block) could reduce L2 misses. Complex to
   implement without breaking the diagonal-first invariant.

10. **Half-precision dLdK/dLdV accumulators with Kahan compensation**
    The fp32 accumulators (dLdK, dLdV) take 32KB each at BIM_BS=128. Dropping
    to bf16 with a compensation term would free 32KB per CTA and allow 3–4 CTAs/SM
    vs the current ~2.7. Training gradients tolerate some precision loss, but
    this needs careful validation.
