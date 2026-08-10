## wang2024flashmask — FlashMask: Efficient and Rich Mask Extension of FlashAttention

Wang, Zeng, Xiao, Wu, Yang, Zheng, Chen, Bian, Yu, Wang (Baidu). arXiv 2410.01359 (v1 Oct 2024,
v2 Mar 2025). **ICLR 2025.** Open-sourced in PaddlePaddle / PaddleNLP.
Cross-referenced against our code briefs `kernels.md` and `masks.md` (both @ commit 6134163) and the
source they cite (`model/graph_traversal/cross_doc_mask.py`, `kernels/`), plus the one-line entry that
already exists for this work in `related_work_notes.md` (§ kernel slice).

### What the paper actually does

**Problem.** FlashAttention(-2) is fast because it never materializes the N×N score matrix, but its
public API only bakes in a *causal* flag. Any richer mask (document/packing mask, sliding window,
prefix-LM, shared-question for RM/DPO, etc.) has to be passed as a **dense N×N boolean/additive bias**,
which reintroduces O(N²) memory and HBM traffic and kills the IO-awareness that made flash fast. That
dense bias is the thing FlashMask replaces.

**Core idea — a column-wise *interval* representation of the mask.** Instead of an N×N matrix, FlashMask
stores, per key/column j, at most **two contiguous half-open row-intervals** of masked entries: one in the
lower-left triangle and one in the upper-right triangle relative to the diagonal. Four length-N integer
vectors encode this:
- **LTS** (Lower-Triangular Start) / **LTE** (Lower-Triangular End): masked rows `[LTS_j, LTE_j)` below the diagonal.
- **UTS** (Upper-Triangular Start) / **UTE** (Upper-Triangular End): masked rows `[UTS_j, UTE_j)` above the diagonal.

The masked set of column j is the *union of those two intervals*. Their worked example: column 5 masks
`[7,10) ∪ [2,4)`. Memory drops from **O(N²) to O(N)** (four length-N vectors; a causal mask needs only
LTS, other types use subsets). The representation rests on the assumption that real masks have a
"continuous nature" — each column's masked region is a run, not scattered points.

**Expressible mask zoo (their Fig. 1a, 12 types).** Causal, sliding-window/local, causal document mask,
(bidirectional) document mask, shared-question mask (RM/DPO prompt sharing), global+sliding-window,
causal blockwise, prefix-LM causal, prefix document, QK-sparse, hash-sparse, random-eviction. They claim
this covers "the majority of mainstream Transformer modeling requirements."

**Stated limitation (the crux for us, verbatim).** "It cannot represent arbitrary masks, particularly
those with irregular masked regions within a single column"; "extreme cases, such as completely random
masks, pose challenges." The representation gives *at most two contiguous intervals per column* (one per
triangle) — not arbitrary disjoint segments.

**Kernel (extends FlashAttention-2).** Two phases. (1) *Preprocess*: tile the four vectors into T_c
column-blocks and compute per-block min/max → 8 derived vectors (LTStart^{max,min}, LTEnd^{max,min},
UTStart^{max,min}, UTEnd^{max,min}), each of size ⌈N/B_c⌉. (2) *Real-time block-skip classification* (their
Eq. 4): each (query-block, key-block) tile is one of three classes — **fully masked** → skip the tile
entirely; **partially masked** → do element-wise masking from the interval bounds; **unmasked** → run the
matmul with *no* mask check at all. Computational cost becomes O((1−ρ)·T_r·T_c) with ρ the block-sparsity;
they report a linear latency-vs-sparsity relationship. Backward pass gains extra reuse because dK/dV are
column-parallel and the column-vectors are exactly the per-column bounds. Correctness: **bit-level
identical** loss curves vs dense-mask FlashAttention under deterministic mode (Llama-3.1-8B).

**Results (as reported).**
- End-to-end training speedup **1.65×–3.22×** vs the dense-mask FlashAttention path, across SFT / LoRA /
  DPO / RM on Llama-2 7B/13B/70B.
- Kernel TFLOPs/s **+12.1% to +60.7%** over FlexAttention; **37.8%–62.3%** of A100 theoretical peak.
- Supports **>100B-param** models at contexts up to **128K**; Llama-2-7B LoRA reaches **544K** tokens vs
  64K for the dense-mask baseline (at 64K the dense mask alone costs ~8 GB).
- Kernel sweep: 12 mask cases × seq {8K, 32K, 128K} × head-dim {64, 128}, hidden 4096, BF16.

The compare-set they position against: dense FlashAttention (memory O(N²)), **FlexAttention** (compiler
`BlockMask` from a `mask_mod` closure — arbitrary masks but retains O(N²/(B_r·B_c)) block-mask memory),
and xFormers (diagonal-offset + cu_seqlens document masks, O(N) but a narrow mask family).

### Methodology: theirs vs. ours

**One-sentence axis.** FlashMask and our BIM are *the same kind of object* — a compressed, block-aware
sparse-mask representation that extends flash-style attention and is shared across heads/layers — but they
compress along an axis (contiguous per-column intervals) that **structurally cannot express our A→B grant
rectangles**, which is precisely why we invented the bit-packed grant encoding instead of reusing an
interval scheme. This is the cleanest "we considered the obvious encoding and it doesn't fit" citation in
the kernel section.

- **What is being masked.** FlashMask targets *packing/windowing* masks whose masked region per column is a
  run (causal doc mask = "everything before my document + everything after," a single interval; sliding
  window = one band). Our `cross_doc_link` mask (masks brief §Formal semantics) is
  `M=(q≥k)&(same_doc OR in_grant)`, where a grant is a **rectangle** placed by an explicit link: source
  code `cross_doc_mask.py:~482-493` sets `cross_doc_mask[grant_start:grant_end, target_start:target_end]=True`
  with `grant_start = link_pos` (row = the linking position onward within the source doc) and
  `[target_start,target_end)` = the *linked-to document's* column span. Crucially the granted **column
  interval is chosen by link identity**, not by any fixed function of the query row — doc A at rows
  [100,200) may grant columns [5,60) (doc B) while doc A' at rows [201,260) grants columns [800,850)
  (doc B'). A single key column j can therefore be granted by several *disjoint* row-runs coming from
  different linkers, and a single query row can be granted several *disjoint* column-runs to different
  targets.

- **Why the interval encoding fails on us.** FlashMask's per-column store is (LT interval, UT interval) =
  two runs max. A packed 32k sequence with hundreds of realized links produces, for one key column,
  arbitrarily many disjoint granting row-ranges — exactly the "irregular masked regions within a single
  column" the paper says it cannot represent. Symmetrically, per query row we allow up to `max_grants=256`
  (masks brief §max_grants — class default 64 but production wires 256, `model.py`) disjoint target column
  intervals. Neither the row-view nor the column-view is a bounded number of intervals. So the interval
  representation is not merely lossy for us; it is inexpressive by construction.

- **What we do instead — bit-packed grants (masks brief §Bit-packed grants; kernels brief §BIM).** We
  assign each of the ≤256 grants a bit: grant k → chunk k//64, bit k%64, over `[n_chunks, T]` int64
  q- and kv-bitmasks (256 grants = 4 chunks). Membership is *pointwise, no sequence reduction*:
  `in_grant = OR_c( q_bm[c][q] & kv_bm[c][k] ) != 0`. Cost O(T·n_chunks) ≈ KB–MB vs a dense O(T²) ≈ 1 GB
  matrix. This encodes an **arbitrary union of rectangles** — the exact generality FlashMask trades away.
  It is our answer to the same memory problem FlashMask solves, but for a mask family their scheme excludes.

- **Block-skip: convergent, at different granularities.** Both papers precompute a block schedule once and
  specialize the inner kernel by block class. FlashMask: per-tile {fully / partially / unmasked} from
  min/max of the interval bounds. Ours (kernels brief §BIM CSR): a block-level CSR built on CPU (~1–2 ms/
  batch), reused across *all heads and layers* (mask is head/layer-independent), with a **richer taxonomy**
  ordered for progressively cheaper inner loops: FULL (single-doc off-diagonal, no masking), PURE_CROSS
  (single-doc different-doc, bitmask-only, no doc_id load), BOUNDARY (straddling, full elementwise),
  DIAGONAL (full + causal). Empty (Q-block,KV-block) pairs are never launched. FlashMask's "fully masked →
  skip" is our "pair absent from CSR"; their "partially masked → elementwise" is our BOUNDARY/DIAGONAL;
  their "unmasked → no check" is our FULL. The difference is that our mask predicate is
  `same_doc | (bitmask grant) & causal` rather than an interval test, so our "partial" work is a bitmask AND
  rather than an interval-containment compare.

- **Shared-across-heads-and-layers.** Both exploit that the mask is identical over B/H (FlashMask stores
  head-independent vectors; our BIM is explicitly "analogous to FlexAttn BlockMask WITHOUT B/H dims,"
  built once per batch). Same amortization insight, both distinct from FlexAttention's per-call block-mask
  build.

- **vs FlexAttention (our production inference/eval mask path).** FlashMask beats FlexAttention on
  throughput precisely by giving up FlexAttention's arbitrary-`mask_mod` generality. We are in the opposite
  corner: we *need* FlexAttention-class generality (arbitrary rectangle unions) and recover the speed with
  the domain-specific bitmask+CSR rather than by restricting the mask family. So FlashMask is the paper
  that most sharply frames our design trade — "restrict the mask to intervals for O(N)" — against which our
  contribution is "keep arbitrary rectangles, get O(T·n_chunks) anyway via bit-packing."

- **Train-on-structure vs retrieve-at-inference.** FlashMask is orthogonal on this axis — it is a pure
  kernel/systems paper, mask-agnostic, no retrieval, no training-signal claim. It shares nothing with our
  linking *inductive bias*; the entire overlap is at the mask-representation + kernel layer. That is the
  correct and narrow scope to cite it in (A6 kernel slice), not in the modeling/retrieval discussion.

### Predictions & open questions for our method

- **Achievable A100 utilization sets our bar.** FlashMask reaches 37.8%–62.3% of A100 peak on interval
  masks at 8K–128K. Our kernels brief reports v12/v18 hitting FlexAttention parity or better (fwd ~2×
  faster, Dh=128 bwd ~9–10× faster than FlexAttention's collapsing bwd). FlashMask's numbers predict our
  achievable utilization ceiling is in that same 40–60% band; if our `cross_doc_link` kernel sits far below
  it at high density, the bitmask-AND inner loop (not the block schedule) is the suspect. Their linear
  latency-vs-sparsity curve is the shape we should also see when we sweep realized grant density — a
  *superlinear* blowup in ours would flag the bit-packed membership test (OR over chunks) as the cost, not
  the block skipping.

- **Sparsity is the lever, so measure realized ρ.** Their speedup is entirely a function of block-sparsity
  ρ. Our analytic O(#blocks) density metric (masks brief; density-aware bucketing via kv_block_count) is
  the same quantity. Prediction: our end-to-end speedup over the dense/concat baselines will track realized
  grant sparsity almost linearly — and because we bucket by kv_block_count to fight the ~6× live-block DDP
  imbalance, the *variance* in ρ across packed sequences is our analog of FlashMask's per-mask-type spread.
  Report speedup against measured ρ, not against nominal max_grants.

- **Backward pass is where the win concentrates.** FlashMask emphasizes bwd gains from column-parallel
  dK/dV reuse of the column-vectors. Our kernels brief independently found the bwd is the hard part
  (asymmetric fwd@128 / bwd@64 tiles for A100 SMEM; FlexAttention Dh=128 bwd collapses to 177 ms). Two
  papers converging on "the mask kernel's payoff and its danger both live in the backward pass" strengthens
  our decision to hand-write and separately tile the bwd; cite FlashMask as external corroboration that
  bwd-specific mask handling is where efficient sparse-mask attention is won or lost.

- **Open question we resolve that they raise.** FlashMask explicitly parks arbitrary/irregular-per-column
  masks as out of scope ("cannot represent … extreme cases such as completely random masks"). Our
  bit-packed grant encoding is a *concrete constructive answer* for one important non-interval family
  (arbitrary rectangle unions from a link graph) with bounded cost O(T·n_chunks). We can frame our
  encoding as extending the reach of efficient sparse-mask attention past the interval boundary FlashMask
  drew — a genuine "their open limitation, our mechanism" pairing for the kernel section.

- **Open question they resolve for us.** Their bit-level-equivalence deterministic-loss check (matching a
  dense reference to bit level) is the exact validation protocol our masks brief flags we *need*: it warns
  our "Flex vs Triton vs dense-viz = 3 mask reimplementations must agree." FlashMask's methodology (verify
  the fast sparse kernel against a dense reference on identical loss curves) is the template for our
  three-way mask-agreement test.

### Gotchas

- **Bit-level equivalence is achievable and expected — hold ourselves to it.** FlashMask demonstrates a
  sparse-mask flash kernel that matches dense to the bit. If our `triton_v18` diverges from a dense
  reference by more than fp-reassociation noise, that is a bug, not "kernels differ." This directly serves
  the masks-brief reviewer-attackable point that our three mask implementations must agree; FlashMask sets
  the standard that agreement should be near-exact under deterministic mode.

- **Numerical stability under full masking is a real trap (we already hit it).** FlashMask's block-skip
  skips *fully masked* tiles, which sidesteps a class of NaN we ran into head-on: our kernels brief §v18
  Sentinel-LSE guard — a query row with zero valid KV keeps the flash M=−1e6 sentinel, `exp2(+1e6)=inf`,
  `inf×0=NaN`. Because our grant rectangles can leave a row with no in-grant, non-same-doc, causal key,
  we *cannot* always skip the block the way FlashMask can; hence our post-kernel `nan_to_num`. Lesson: an
  interval mask rarely produces an all-masked query row, but a rectangle-union mask does — do not assume
  the FlashMask block-skip removes the empty-row NaN for us. (Also our BIM diagonal guard
  `np.fill_diagonal(same_doc, True)`.)

- **The "continuous nature" assumption is exactly what breaks on link graphs.** FlashMask's whole
  efficiency rests on masks being column-contiguous. Any attempt to shoehorn our grants into an
  interval/xFormers-style cu_seqlens scheme (a tempting "simpler" refactor) will silently mis-mask
  multi-target rows. The masks brief already lists "multiple grants compose by UNION, no weighting" and
  "DAG gate: only backward links granted" — both produce non-contiguous per-column masks. Treat interval
  encodings as a known dead-end for our mask family, and say so.

- **max_grants truncation is a density knob with a bias.** FlashMask has no cap — its cost is set by ρ
  alone. Ours caps at 256 and drops overflow **positionally** (later links lose first; masks brief
  §max_grants). When benchmarking speedup vs sparsity, remember our ρ is partly a *policy* artifact of the
  cap, not just corpus structure; a fair FlashMask-style sparsity sweep must hold max_grants fixed.

- **PaddlePaddle numbers don't transfer 1:1.** Their TFLOPs/s are from a Paddle CUDA implementation on
  A100; our Triton kernels are H100-autotuned and "functional-not-optimal on A100" (kernels brief). Do not
  quote their absolute utilization as our target without noting the framework/hardware gap; use the *shape*
  of their results (linear-in-sparsity, bwd-dominated, 40–60% band), not the absolute cells.

### Missed citations worth adding

Checked against `paper/bib/refs.bib`. **Already present** and NOT missing: `wang2024flashmask` (this
paper, line 1835), `dao2022flashattention`, `dao2023flashattention2`, `shah2024flashattention3`/
`dao2024flashattention3`, `flexattention_blog2024`, `flexattention_paper2024`, `tillet2019triton`,
`milakov2018onlinesoftmax`, `rabe2021selfattention`, `gray2017blocksparse`, `child2019sparsetransformers`,
`beltagy2020longformer`, `zaheer2020bigbird`, `pagliardini2023sparseflash`, `lefaudeux2022xformers`,
`liu2023ringattention`, `liu2023blockwise`, `korthikanti2022sequence`, `jacobs2023ulysses`,
`brandon2023striped`, `li2023distflashattn`, `hsu2024liger`. Our kernel/attention slice is already very
well covered, so genuinely-missing candidates from FlashMask's own reference list are few:

- **PaddlePaddle / PaddleNLP** — the framework FlashMask ships in (Ma et al., "PaddlePaddle: An Open-Source
  Deep Learning Platform…", or the PaddleNLP toolkit). *Why:* only worth a cite if we contrast framework
  ecosystems when positioning our Triton/PyTorch kernel; low priority, likely a software/URL cite with no
  clean arXiv id. Not in refs (grep: 0 hits for "paddle"). Optional.

- **Prefix-LM / UL2-style bidirectional-prefix masking** (Raffel et al. T5 2019; Tay et al. UL2, arXiv
  2205.05131) — FlashMask lists prefix-LM and prefix-document masks as target mask types. *Why:* tangential
  to us (we don't do prefix-LM), include only if we discuss the general mask zoo; probably out of scope.
  (Verify presence of any T5/UL2 key before adding — I did not grep these specifically.)

Net: **no high-value missing kernel/attention citation** surfaces from FlashMask that we don't already
have. The paper's value to us is not new references but as the sharpest foil for the *design rationale* of
our bit-packed grant encoding (interval-encoding limitation → our arbitrary-rectangle answer). I did not
find a genuinely-missing must-add cite; flagging only the two optional ones above rather than padding the
list.

---
Confirmed from the arXiv abstract (2410.01359) and the v2 HTML (arxiv.org/html/2410.01359v2): the
LTS/LTE/UTS/UTE column-vector definitions, the union-of-two-intervals semantics and its worked example,
the explicit "cannot represent irregular per-column / random masks" limitation, the 12 mask types, the
three-way block-skip classification extending FlashAttention-2, O(N) memory, and the speedup/utilization/
scale numbers (1.65×–3.22×; +12.1%–60.7% vs FlexAttention; 37.8%–62.3% A100; >100B params; 128K/544K
tokens). Everything about OUR method is grounded in the `kernels.md`/`masks.md` briefs and the cited source
lines in `model/graph_traversal/cross_doc_mask.py` (grant rectangle assignment) and `kernels/` (BIM/
bitmask/CSR), which I read directly. Done.
