<!-- PROVENANCE
Written against commit 6134163 (main @ 2026-08-07). This brief describes the code as of
that commit; it is NOT authoritative if the source has changed since. Before trusting it,
check whether the covered sources have drifted:
    git diff --stat 6134163..HEAD -- kernels/
Empty output = brief still current. Non-empty = re-verify the changed parts against source.
Covered sources: kernels/
-->

# CODE BRIEF: Triton attention kernels (agent a0332a4a)
KERNELS.md STALE (says v12 prod); ground truth = triton_v18 (cross-doc), varlen_bim_v2 (doc-causal) per CLAUDE.md + attention.py:199-229. Both wrap v10 core.

## BIM = Block Interaction Mask (BlockInteractionMask, cross_doc_mask.py:69-144)
Precomputed block-level CSR (Q-block,KV-block)→interact?, "analogous to FlexAttn BlockMask WITHOUT B/H dims" (mask same across all heads+layers → built ONCE per batch, reused). Two reps:
- Grant bitmasks: grant k→chunk k//64 bit k%64, [n_chunks,T] int64 q/kv; in_grant=any_c((q_bm[c,i]&kv_bm[c,j])!=0), pointwise no seq reduction. 512KB@32k vs ~1GB dense. bit63=INT64_MIN. kernel OR-over-chunks = tl.static_range unrolled.
- BIM/CSR block table (_build_block_interaction_mask:597-914): CPU numpy ~1-2ms/batch, OR-reduce bitmasks to block granularity, emit q_kv_* (fwd + dQ bwd) + kv_q_* (dK/dV bwd) CSR. Grid (n_blocks,H), each CTA walks only its CSR row → empty pairs never launched. bim.sparsity = fraction causal pairs skipped.
- CSR ORDERED BY BLOCK CLASS → progressively cheaper inner loops: FULL (single-doc same-doc off-diag, NO masking), PURE_CROSS (v13: single-doc diff-doc, bitmask-only no doc_id load), BOUNDARY (straddles, full same_doc|in_grant), DIAGONAL (full+causal). Order [full,pure_cross,boundary,diagonal].
- Ordinal run-index relabel (:636-669): same-doc block-overlap test needs monotonic labels; traversal-order doc_ids NON-monotonic; relabel cumsum(doc-id changes). Without it: straddling block drops same-doc KV → softmax collapse → NaN/dQ~5.7e4 in thestack cross_doc_link training. CITE as correctness fix.

## Speed vs FlexAttention (A100 80GB bf16, 16h Dh64 doc512; KERNELS.md:16-49)
v12: T=4k 0.91×, T=32k 0.77× total (fwd 2× faster 1.14 vs 2.38ms, bwd 12% 5.52 vs 6.30ms). **Dh=128: FlexAttn bwd COLLAPSES to 177ms → v12/v18 bwd 16.85ms ~9-10× faster.** varlen_bim_v2 targets vslf-class (old triton_varlen 31ms fwd vs vslf 0.54ms; vslf bwd unusable on real packed data) — BIM gets vslf-class fwd WITH reliable gradients.
Why faster: (1) v10 native bf16 TC matmuls, no fp32 load, post-mult scale → half register/SMEM → higher occupancy; (2) v11 no permute/contiguous, THD layout direct strides → -200MB HBM/step@32k, peak 271/340→69/138MB (flex parity) +0.35ms fwd; (3) v12 BIM_BLOCK_SIZE=128 larger bwd tiles; (4) structural: precompute block schedule once/batch, specialize per class, no empty-block launch, no per-block mask re-eval.

## Varlen (packed) handling
Legacy varlen_attn.py slow. Production varlen_bim_v2: reuse BIM CSR for PURE doc-causal (grants all-zero), inner kernels check only same_doc+causal, ZERO bitmask loads; off-diag same-doc blocks→full fast path, only diagonal masked. BIM cached by (id(cu_seqlens|doc_ids),seq_len). _from_doc_ids entry handles non-contiguous doc spans + doc_id=-1 gaps.

## fused_relu_sq_mlp + polar_express
- fused squared-ReLU MLP: out=relu(x@W1ᵀ)²@W2, one Triton matmul fp32 grouped-swizzle L2, mode-switched epilogue (fwd save pre-act; bwd (grad@W2ᵀ)·2·relu(pre)); W2 stored TRANSPOSED (no .T.contiguous in bwd), zero-init (muP). fp32 activation. layer.py:47.
- polar_express: Muon orthogonalization REPLACING Newton-Schulz, arXiv:2505.16932. Per-iter tuned coeffs (5 iters). a·X+(b·A+c·A²)·X via 3 custom Triton kernels XXT/XTX/ba_plus_cAA exploiting SYMMETRY (skip ~half tiles + mirror-store). Picks XᵀX vs XXᵀ by tall/wide. Fuses Nesterov(fp32)+orthog(bf16) in one torch.compile. Spectral-norm 2e-2 margin. momentum_t 0-D CPU tensor (avoid recompile). H100-autotuned, functional-not-optimal on A100. [DUP w/ muon brief]

## Autotuning
@triton.autotune narrow hand-curated grids keyed [N,Dh,n_chunks,BIM_BLOCK_SIZE]; fwd 24 configs (KV{32,64}×stages{3,4,5}×warps{4,8}); bwd MICRO{16,32,64} (128 excluded so BS64%MICRO==0). EXPANDING grid was REGRESSION (v14 noisy single-run autotuner false-picked degenerate MICRO=128). .autotune_cache.json persists kernel→config, bypasses search (concurrent multi-rank JIT autotune corrupts compiler).

## Novel/publishable
- BIM: layer/head-shared block-class-partitioned CSR schedule for structured cross-doc masks. Amortize O((T/bs)²) CPU precompute over all layers/heads + per-class dispatch. Genuinely novel vs FlexAttn per-call block-mask eval.
- Bitmask grant encoding O(T) not O(T²), no seq reduction, + block-level OR-union prune.
- Ordinal run-index relabel (correctness fix tied to real NaN).
- Asymmetric fwd(128)/bwd(64) block sizes (v17/18) for A100 SMEM@Dh128 → ~2× bwd.
- Sentinel-LSE NaN guard (v18): flash M=-1e6 sentinel survives for tokens w/ zero valid KV → exp2(+1e6)=inf, inf×0=NaN; fixed post-kernel nan_to_num (zero fwd output anyway). Publishable numerical-stability finding for sparse-mask flash attn.
- Documented dead-ends v5-v9 + failed v14-v16 w/ root cause (rigorous negatives).
- fused sq-ReLU MLP + symmetric-matmul Polar Express = solid eng (methods themselves prior work).

FLAGS: KERNELS.md stale (trust code); v18 exact ratio not cleanly re-benched (v18=v17+NaN guard); varlen 31/0.54/2.3× author-reported docstring; NO GPU run (all numbers as-documented).

## → LIT REVIEW IMPLICATIONS
- A6 kernel slice: Triton (Tillet), FlashAttention kernel design, block-sparse GPU kernels, online softmax (Milakov flash-attn precursor), bitmask/CSR sparse attention.
- Numerical stability of softmax/flash under full masking → could cite.
