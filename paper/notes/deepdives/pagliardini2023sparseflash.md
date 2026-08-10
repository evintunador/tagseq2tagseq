## pagliardini2023sparseflash — Faster Causal Attention Over Large Sequences Through Sparse Flash Attention

arXiv 2306.01160v1 (1 Jun 2023, preprint under review). Pagliardini, Paliotta, Jaggi, Fleuret (EPFL / U. Geneva).
Open-source Triton code: https://github.com/epfml/dynamic-sparse-flash-attention.
Cross-referenced against our code briefs `kernels.md` and `masks.md` (both @ commit 6134163) and verified
against source: `kernels/cross_doc_bitmask_bim_v18.py`, the v10 kernel core, `model/graph_traversal/cross_doc_mask.py`.
This is our brief's designated "closest precedent for a flash kernel consuming a RUNTIME sparse block pattern."

### What the paper actually does

**Core contribution.** The authors extend FlashAttention (dao2022flashattention) into **Sparse Causal Flash
Attention (SCFA)** — a Triton GPU kernel that relaxes FlashAttention's hard assumption that the causal mask
is a *perfect lower-triangular*. FlashAttention exploits triangularity to skip the upper-triangular tiles
(for query block i it only walks key blocks j ≤ i, and applies a local triangular mask on the diagonal
block). Any *dynamic* sparsity that reorders or drops tokens destroys that triangular shape, so a "naive"
implementation must materialize the full attention matrix and apply an arbitrary mask — killing the whole
FlashAttention advantage. SCFA is the kernel that keeps the tiling/online-softmax machinery but drives tile
selection off **runtime index vectors** instead of the block coordinates. Contribution is explicitly
threefold: (1) the SCFA kernel that handles "any sparsity pattern expressible as a range of keys per query,
plus any causal masking inside the resulting sub-blocks"; (2) an *exact* hash-based attention (Hash-sparse)
that fixes Reformer's approximate coverage; (3) fine-grained per-head query/key dropping (QK-sparse).

**The two runtime patterns.** Both work per-head, per-sequence (two sequences in a batch may have totally
different patterns), and reduce to the *same* kernel trick: pass the kernel extra vectors and recompute the
tile loop bounds from them.

- **QK-sparse (§3.1).** Independently per head, decide to keep/drop each key and each query. Physically
  *compact* the tensors — build smaller `Qc, Kc, Vc` by removing dropped rows (a **stable sort** that keeps
  time order, then gather; padded to the per-head max kept-count across the batch). The compacted attention
  matrix `Ac` is still monotone-causal in the *original* indices. The kernel additionally receives `qidx`,
  `kidx` (original positions). Tile `Ti,j` is skippable iff `max(qidx_i) < min(kidx_j)`; the kernel scans key
  blocks to find `jstop` = first block failing that, computes tiles `j ∈ [0, jstop)`, and applies a *local*
  causal mask inside each tile via `qidx[:,None] >= kidx[None,:]`. This recovers FlashAttention-class
  runtime on the reduced matrix.

- **Hash-sparse (§3.2).** Independently per head, assign a **bucket id** to each key and query (LSH). Stable-sort
  Q,K,V along the sequence by bucket id → same-bucket tokens cluster near the diagonal, and *within* a bucket
  the original positions stay monotone. Kernel gets both `qidx/kidx` (original positions, for causality) and
  `qhash/khash` (bucket ids, for the sparsity). It computes **both** a `jstart` (first key block whose buckets
  overlap the query block's) and a `jstop` (first block whose buckets all exceed), then *refines* `jstop` down
  by the position test `max(qidx_i) >= min(kidx_j)`. It computes only tiles `j ∈ [jstart, ĵstop]`, applying a
  fused mask `(qidx[:,None] >= kidx[None,:]) & (qhash[:,None] == khash[None,:])` — causal AND same-bucket.
  (`>=` is swapped to `>` to forbid self-attention, following Reformer.) Hashing uses Reformer's scheme:
  shared query-key space (K = normalized Q), the Andoni et al. 2015 angular-distance LSH, `nb` buckets chosen
  directly. Unlike Reformer's fixed-chunk approximation, SCFA's coverage of same-bucket collisions is **exact**.

**Setup & scale.** 122M-param autoregressive transformer (12 layers, d=768, 12 heads × 64). OpenWebText2 for
LM; enwik8 (char-LM) and sequential-MNIST for the Reformer comparison. Trained 15k iterations on 2–3 A100s,
bf16, data-parallel. T=8192 (batch 96 = 4×8×2) and T=16384 (batch 30 = 2×5×3). K set equal to normalized Q
for *all* models (fairness vs Reformer). Hash-LM uses `nb=16` buckets. Vacuum-timing benchmarks used B=4, 48
heads × 64. The goal is explicitly *not* SOTA perplexity — it's speed at matched perplexity.

**Concrete numbers.**
- Headline: matched perplexity, **2.0× (T=8k) and 3.3× (T=16k)** end-to-end training speedup over F-LM
  (full FlashAttention). Attention-only iter speedups quoted as **1.8× (8k) / 2.3× (16k)**; the larger
  end-to-end factor is because attention is a bigger share of the step at long T. H-LM even *slightly beats*
  the baseline perplexity-per-iteration, and H-LM gets *faster during training* (buckets specialize).
- QK-dropping (D-LM, T=8k): dropping 30% matches F-LM perplexity while ~2× faster; speed factors reported
  ×1.9 / ×2.6 / ×3.5 at 30/50/70% drop. Higher sparsity hurts perplexity — and notably "more sparsity does
  not necessarily mean decreasing perplexity faster."
- **Naive vs SCFA crossover:** a naive (PyTorch SDPA + custom mask) QK-sparse implementation only beats full
  FlashAttention beyond **~70% drop**; SCFA gives speedups "even at relatively low sparsity levels." The
  reshaping overhead is linear in T and is amortized as T grows.
- Reformer comparison (Fig 4/5): both SCFA-hash and Reformer are linear in T, but SCFA is faster per-token
  *and* keeps **100% collision coverage** while Reformer's coverage decays steeply with T. Quality: MNIST ppl
  1.67 vs 1.76; enwik8 (T=4096) **2.29 vs 3.32 bits/char** — SCFA-hash clearly beats Reformer.

### Methodology: theirs vs. ours

**The one-axis summary.** SCFA and TS2TS are the *same kernel idea* — a FlashAttention variant whose tile
loop bounds are recomputed at runtime from index vectors instead of being fixed by triangularity — pointed at
opposite *sources of the pattern*. SCFA's pattern is **content/hash-driven and ephemeral** (recomputed every
forward from the current Q,K via LSH, or from a random drop mask); ours is **graph-link-driven and structural**
(a document A→B hyperlink/import/citation edge, known before the step, identical in pretraining and inference).
This is precisely the brief's framing, and it holds up in detail.

- **Runtime pattern representation.** SCFA passes the kernel *dense per-token index/hash vectors*
  (`qidx, kidx, qhash, khash`) and derives block bounds by an in-kernel **scan** over key blocks
  (`min(kidx_j) <= max(qidx_i)`, plus the bucket-overlap `jstart/jstop`). It never precomputes a block table —
  the tile loop is a contiguous `[jstart, ĵstop]` range because sorting made same-bucket tokens contiguous.
  Ours (`kernels.md`, `masks.md`) precomputes a **block-level CSR "BlockInteractionMask" (BIM)** on CPU once
  per batch (~1–2ms numpy) and reuses it across *all heads and all layers* (our mask is head/layer-independent;
  theirs is per-head). Each CTA walks only its CSR row, so empty (Qblk,KVblk) pairs are never launched at all.
  So: **SCFA = per-head in-kernel index scan over a contiguous sorted range; TS2TS = amortized, layer-shared,
  arbitrary-scatter CSR schedule.** Their contiguous-range trick cannot express our arbitrary A→B block
  rectangles (a linker doc reads a target doc that sits at an arbitrary, non-adjacent packing offset); it only
  works because their sort *makes* the live region contiguous. Our grant rectangles are scattered, which is
  exactly why we need a CSR/bitmask rather than a `[jstart, jstop]` interval.

- **What defines "keep this pair."** SCFA-hash: `same_bucket & causal`, where bucket = LSH(Q). SCFA-QK:
  `both-kept & causal`. Ours (`masks.md` formal semantics): `causal & (same_doc OR in_grant)`, where
  `in_grant` = the bit-packed grant test `OR_c (q_bm[c][q] & kv_bm[c][k]) != 0` — a link A→B grants rows
  `[link_end_pos, A.end) × cols [B.start, B.end)`. Their sparsity *removes* computation relative to dense
  causal; **ours ADDS** computation relative to per-doc causal (a grant opens cross-document pairs that plain
  doc-causal would forbid). We are a super-set of doc-causal; they are a sub-set of dense-causal. Same kernel
  generalization (non-triangular causal sub-blocks), opposite direction of the FLOP delta.

- **Sorting / index monotonicity is load-bearing for both — same failure mode.** SCFA's entire correctness
  rests on **stable sort preserving original-position monotonicity within a bucket**, so the in-tile causal
  mask `qidx >= kidx` is valid. Our directly-analogous device is the **ordinal run-index relabeling**
  (`cross_doc_mask.py:636-669`, both briefs flag it): traversal-order doc_ids are *non-monotonic*, so the
  block-level `same_doc` interval-overlap test is invalid until we relabel `ordinal[i]=cumsum(doc-id changes)`.
  Getting this wrong caused a concrete NaN in thestack cross_doc_link training (LSE collapse, dQ≈5.7e4). Both
  projects independently learned that **the block-skipping optimization is only sound if per-token labels are
  monotone**, and both had to engineer that monotonicity (they by stable sort, we by ordinal relabel).

- **The stranded-query NaN — near-identical bug and fix.** SCFA §3.1/App B (Alg 3) documents that dropping
  keys/queries produces "stranded queries with no keys," which yield `−∞` running max and NaN in the online-
  softmax accumulation; their fix rewrites UPDATE_STATS to replace `−∞` max by 0 and `∞` in `1/ℓ` by 1, so
  stranded queries default to a **0 output**. Our **sentinel-LSE NaN guard (v18)** is the same phenomenon in
  the backward: `kernels/cross_doc_bitmask_bim_v18.py` documents ~155/32767 tokens in a simplewiki BFS pack
  whose forward LSE stays at the init sentinel (≈−1e6) because they attended zero valid KV, so
  `exp2(score+1e6)=∞` in bf16 and `∞×0=NaN` poisons the gradient; fix = `torch.nan_to_num(...,0)` on
  dLdq/dLdk/dLdv, correct because those tokens' output is 0 anyway. **This is direct external corroboration
  that our v18 finding is a real, general hazard of sparse-mask flash attention, not a TS2TS quirk** — the
  brief calls v18 a "publishable numerical-stability finding," and SCFA is the citation that proves the class
  of bug exists in the QK-dropping regime too. Worth citing side-by-side.

- **Train-on-structure vs retrieve-at-inference.** Neither project is a cached-KV / GNN-edge method. SCFA is
  pure attention-matrix sparsification; the pattern is recomputed live every forward from the current
  activations (hash) or a random mask (drop) — there is **no persistent edge, no inference-time semantics**
  (the hash buckets at inference are just whatever LSH produces on the generated tokens). TS2TS's edge is a
  *structural fact of the corpus* used *identically at train and inference*: a generated link deterministically
  pulls its target doc into attention. So on the brief's axis, SCFA sits entirely at "efficiency kernel, no
  inductive bias" — it is the *machinery* precedent (A6 kernel slice in both briefs' LIT REVIEW IMPLICATIONS),
  not a methods competitor. It is the paper to cite for "flash kernel consuming a runtime sparse block pattern"
  and for "how to handle non-triangular causal sub-blocks in flash," full stop.

- **Where we diverge on engineering.** SCFA re-sorts/gathers Q,K,V every forward (linear overhead they
  explicitly amortize over long T) and dispatches per-head; we precompute a block schedule **once per batch,
  shared across layers/heads**, with a block *taxonomy* (FULL / PURE-CROSS / BOUNDARY / DIAGONAL, CSR-ordered
  for progressively cheaper inner loops) that SCFA has no analog to. Our density-aware bucketing (kv_block_count)
  to balance DDP has no SCFA counterpart because their per-sequence cost is roughly constant given `nb`.
  Conversely we do *not* physically compact tensors (they do for QK-sparse); we keep the full THD layout and
  skip via CSR — avoiding their sort/gather/pad overhead entirely.

### Predictions & open questions for our method

- **Speedup grows with sequence length — expect the same shape for us.** Their gains widen 8k→16k (2.0×→3.3×)
  because reshaping overhead is linear while the saved attention is quadratic. Our BIM precompute is a fixed
  ~1–2ms/batch and our savings scale with skipped causal pairs, so our *win over dense/FlexAttention should
  also widen with T* — consistent with the kernels brief's v12 numbers being best at T=32k. Prediction: our
  cross_doc_link kernel advantage over Flex should be near-flat/small at short T and clearly positive at 32k.
- **Low-sparsity regime is where custom kernels earn their keep.** SCFA's key result is that a *naive* masked
  SDPA only wins past ~70% drop, while the custom kernel wins at low sparsity. Our cross_doc_link masks are
  *low-sparsity relative to dense* (grants only add a modest fraction of pairs on top of doc-causal). This
  predicts the BIM/CSR kernel is *necessary* — a naive full-matrix + mask approach (or dense FlexAttention
  fallback, which our CLAUDE.md warns OOMs at 32k) would give no benefit or negative benefit at our sparsity
  level. It validates the whole custom-kernel investment.
- **"More sparsity ≠ better learning" warns our compute-controls.** SCFA-drop found 30% matched baseline but
  higher drop hurt perplexity, and more sparsity didn't speed convergence. Analogy: our concat compute-control
  variants add *more* attention (doc_concat_link, doc_concatenated are FLOP super-sets of cross_doc_link). If
  more attention doesn't monotonically help, then a cross_doc_link *win* over doc_concat_link would be strong
  evidence the gain is the *link-position gating inductive bias*, not raw FLOPs — exactly the isolation those
  controls are designed for (`masks.md` novelty #5).
- **Their hash buckets "specialize" over training (H-LM speeds up).** They observe hash attention getting
  faster as buckets sharpen. Our pattern is fixed by the graph, so we won't see that; but it suggests a probe:
  does our realized live-block count *drift* over training (e.g., because grants that get dropped depend on
  max_grants truncation)? If our density is stable, that's a cleaner story than theirs.
- **Open question we can resolve for them / they for us.** SCFA leaves open whether a *non-random, learned*
  drop/hash pattern helps — they say "better dropping schemes could be devised... outside scope." TS2TS is in
  effect a *principled, non-content* pattern (structural graph edges) rather than content-hash — a data point
  that a *semantically meaningful* sparse pattern (not similarity-hash) can be both fast and useful. Conversely,
  their exact-coverage-vs-Reformer result reassures us that *exactness* of the sparse pattern matters (Reformer
  lost quality by missing collisions); our grants must likewise be applied exactly at train and eval, which is
  our `masks.md` reviewer-attackable #6 (three mask reimplementations must agree; max_grants must match).

### Gotchas

- **Non-triangular-causal flash = NaN factory for zero-KV queries.** Both SCFA (stranded queries) and our v18
  (sentinel LSE) hit it. Any change to our masking that can leave a query with *zero* valid KV (e.g. a doc
  whose only same-doc predecessor is masked, a −1 layout gap, tighter grant gating) re-arms this. Keep the
  `nan_to_num`/sentinel guard; don't "optimize it away." SCFA's independent report is the evidence it's
  fundamental, not incidental.
- **Reshape/sort overhead only amortizes at long T.** SCFA's naive-vs-real crossover shows the linear
  preprocessing can *lose* at short sequences. Our analog is the ~1–2ms CPU BIM build and the density-bucketing
  sort: at short T or tiny batches these could dominate. Any smoke test at reduced T is *not* representative of
  the 32k economics — matches our CLAUDE.md rule that smoke tests must keep real T/compile settings.
- **Fairness hack: K = normalized Q.** SCFA (and Reformer) tie K to normalized Q so hashing is well-defined.
  That is a *modeling constraint that changes the architecture*, and their perplexity numbers are under it. If
  we ever cite their perplexity-neutral claim, note it holds only in the tied-QK setting — not a generic
  transformer. Our model does not tie QK, so their "no perplexity loss" is not directly transferable evidence
  for our masks.
- **Per-sequence dynamic patterns break naive caching/batching assumptions.** SCFA stresses two sequences in a
  batch can have totally different drop patterns → padding to the batch-max kept-count. Our per-batch BIM is
  likewise data-dependent; do not assume a cached kernel schedule across batches (the `.autotune_cache.json`
  caches *kernel configs*, not the CSR — good). Watch DDP imbalance: our briefs already flag ~6× live-block
  variance; SCFA's padding-to-max is the same imbalance in a different guise.
- **Small-scale, short-training evidence.** All SCFA results are 122M params, 15k iters, ppl not SOTA. Treat
  their "matched perplexity" as a *speed* claim at small scale, not a scaling-law statement. Don't over-read it
  as proof sparse patterns are quality-neutral at our scale.

### Missed citations worth adding

Checked against `paper/bib/refs.bib` (grep). **`pagliardini2023sparseflash` itself is already present**
(line 1833) and correctly cited in `related_work_notes.md` §7 as our runtime-sparse-pattern precedent — do not
re-add. Also already present: reformer/kitaev2020, child2019sparsetransformers, zaheer/bigbird, beltagy/
longformer, katharopoulos, choromanski/performers, tillet2019triton, dao2022flashattention, andoni (only as an
author surname collision — see below). Scanning SCFA's *own* reference list for works relevant to OUR project
and genuinely missing:

- **kitaev2020reformer** — Reformer: The Efficient Transformer, ICLR 2020 (arXiv 2001.04451). *Why:* grep shows
  exactly one "reformer"/"kitaev" hit and it is inside the SCFA entry's own text, **not a standalone bib entry**.
  Reformer is the canonical LSH/hash-attention + reversible-layers long-context model and the direct antecedent
  SCFA improves on; our kernel/long-context related-work (A6 slice) should cite it directly. HIGH — verify no
  standalone key exists before adding.
- **andoni2015lsh** — Andoni, Indyk, Laarhoven, Razenshteyn, Schmidt, "Practical and Optimal LSH for Angular
  Distance," NeurIPS 2015 (arXiv 1509.02897). *Why:* the actual LSH scheme behind hash-attention. grep "andoni"
  matches only "Andonian" (an unrelated author). Only worth adding if we discuss hashing/LSH machinery; LOW–MED.
- **kim2022learnedtokenpruning** — Kim, Shen, Thorsley, Gholami, Kwon, Hassoun, Keutzer, "Learned Token Pruning
  for Transformers," KDD 2022 (arXiv 2107.00910). *Why:* SCFA's QK-dropping antecedent; the "learned to drop
  tokens/keys" line. Relevant only if we position our grant-gating against token-pruning; LOW.
- **child2019sparsetransformers** — already in refs (line 358); NOT missing. (Listed here only to note it's
  covered.)

Lower priority / out of scope from their refs: Longformer, BigBird, Linformer, Performer, Synthesizer, RFA,
head-pruning (Michel/Voita), Long Range Arena — either already in refs or not relevant to our graph-link thesis.

---
Confirmed from the PDF (arXiv 2306.01160v1, all 24 pages incl. App. B Alg. 1–3): the SCFA kernel mechanics,
both sparsity modes, `jstart/jstop` tile-bound derivation, the stranded-query/NaN UPDATE_STATS fix, the
K=normalized-Q fairness setting, and the 2.0×/3.3× (end-to-end) & 1.8×/2.3× (attention) speedups, enwik8
2.29 vs 3.32 bpc, MNIST 1.67 vs 1.76, nb=16, 122M/12L/768/12h, 15k iters. Our side is grounded in `kernels.md`,
`masks.md`, and verified against `kernels/cross_doc_bitmask_bim_v18.py` and `cross_doc_mask.py:636-669`. The
v18 sentinel-LSE / stranded-query parallel is confirmed from both sources, not inferred. Done.
