# yuan2025nsa — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

DeepSeek, Yuan et al., 2025. arXiv 2502.11089 (v2, 27 Feb 2025). Grounded against the paper's
HTML/abstract on arXiv and our code briefs (kernels.md, masks.md, packing_density.md) plus the
consolidated related_work_notes.md. Numbers below are as-reported by the paper; I flag what is
inferred vs. confirmed.

## What the paper actually does

NSA is a **natively trainable** sparse-attention mechanism: the sparse pattern is present during
pretraining, not bolted on for inference. For each query token it runs **three parallel branches**
over its own key/value cache and blends them with a learned per-token, per-branch gate
`g_t^c ∈ [0,1]` (an MLP + sigmoid on the query features):
`o*_t = Σ_{c∈{cmp,slc,win}} g_t^c · Attn(q_t, K̃_t^c, Ṽ_t^c)`. Each branch has its **own** K/V
projections to prevent shortcut learning between branches.

- **Compression (cmp):** contiguous KV blocks (block length `l`, sliding stride `d<l` to avoid
  fragmentation) are aggregated into one block-level key/value each via a learnable MLP with an
  intra-block position encoding. Gives cheap coarse global context.
- **Selection (slc):** fine-grained blockwise **top-`n`** token selection. Importance scores are
  **reused from the compression branch's attention** (`p_t^cmp = softmax(q_tᵀ K̃_t^cmp)`); when the
  selection block size `l'` equals `l=d` the scores are used directly, otherwise summed over
  overlapping compression blocks (their Eq. 9). Crucially, scores are **summed across the H query
  heads sharing a GQA group** (Eq. 10) so every head in a group selects the *same* blocks — this is
  what makes the KV load contiguous and small at decode time.
- **Sliding window (win):** the last `w` tokens verbatim, giving explicit local context and keeping
  the other two branches from having to relearn local patterns.

**Hardware co-design.** cmp and win reuse FlashAttention-2 kernels; **selection needs a custom
Triton kernel**. Design principles: (1) **group-centric data loading** — load all query heads of a
GQA group for a query block into SRAM at once, since they share the selected KV blocks; (2) load the
shared, *contiguous* selected KV blocks (block size divides `l'`); (3) put the query/output loop on
the Triton grid scheduler because the inner (selected-KV) loop length is nearly constant → balanced
work across SMs, near-optimal arithmetic intensity, no redundant KV transfers. **Blockwise** (not
token-level) selection is the whole point: contiguous block reads hit Tensor Cores and avoid the
gather/non-contiguous access that kills token-level methods on GPU.

**Scale / setup.** Backbone = GQA + DeepSeekMoE, **27B total / 3B active** params, 30 layers, hidden
2560; **4 GQA groups, 64 heads**; `d_q=d_k=192, d_v=128`; 72 routed + 2 shared experts, top-6.
Pretrained on ~**260–270B tokens** at 8k length (the paper states both figures in different places),
then continued-trained + SFT to 32k with YaRN. NSA hyperparameters: `l=32, d=16, l'=64, n=16`
(of which 1 forced initial block + 2 forced local blocks), `w=512`.

**Headline results (as-reported).**
- General benchmarks (Table 1): NSA avg **0.456 vs 0.443** full attention, winning 7/9; e.g.
  GSM8K 0.520 vs 0.486, DROP 0.545 vs 0.503, MMLU 0.565 vs 0.567 (essentially tied), HumanEval
  0.348 vs 0.335. So sparse *matches or beats* dense at 27B/3B scale.
- LongBench (Table 2): NSA **0.469** vs full 0.437 vs an exact-top-k baseline 0.423.
- Needle-in-a-haystack: **perfect retrieval at all positions up to 64k**.
- Reasoning after distillation (AIME, Table 3): NSA-R > Full-R (0.121 vs 0.046 at 8k).
- Speed (A100, Triton): up to **9.0× forward / 6.0× backward** at 64k training; decode
  memory-access speedup 4× (8k) → 6.4× (16k) → 9.1× (32k) → **11.6× (64k)**.
- Ablations (Fig. 7, 3B): cluster-based selection (ClusterKV-style) and non-differentiable
  heuristic/aux-loss selection (Quest/InfLLM-style) both give worse loss than native NSA.

## Methodology: theirs vs. ours

**The one-line axis:** NSA's sparsity is **learned, content-derived importance over the same
sequence**; ours is a **hard, binary, externally-graph-dictated key range**. Both parties agree on
the meta-claim that matters most for our paper — *train with the sparse pattern, don't bolt it on* —
and both **co-design a Triton kernel** around block-structured sparsity. Everything else diverges.

- **Where sparsity comes from.**
  - NSA: `q_tᵀ K̃_t^cmp` importance → top-`n` blocks. Data-dependent, differentiable-through-gate,
    recomputed every forward pass per query.
  - Ours: `cross_doc_link` grants a rectangle `[link_end_pos, A.end) × [B.start, B.end)` whenever
    linking doc A references doc B, `M = (q≥k) & (same_doc OR in_grant)` (masks.md; source
    `cross_doc_mask.py:417-423`). The set of visible keys is **dictated by the link graph**, not by
    query-key affinity. It is fixed for a given pack (baked "Option B" graph-edge grants during
    training; masks.md §Option B), and it is **directional/DAG-gated** — only backward links
    (`target_start < link_end_pos`) are granted, so A reads B but never the transpose.
- **Learned vs. structural block choice.** NSA *learns which blocks matter*; we *know which blocks
  matter* from an external edge and encode it as a bitmask. NSA's top-`n` is a soft, per-step ranking
  with a gate that can down-weight a branch; ours is a hard union of grant rectangles (compose by OR,
  no weighting, positional truncation at `max_grants=256`; masks.md). We have **no importance
  scoring and no gate** — the graph is the oracle.
- **Kernel parallels are strikingly close, and worth citing side-by-side.**
  - Group-centric loading (all GQA heads of a group share the selected KV): the exact analog of our
    **head-and-layer-shared BIM** — our Block Interaction Mask is identical across all heads and
    layers, so the block schedule is built **once per batch on CPU** and reused (kernels.md; masks.md
    §Compilation). Both exploit "the mask/selection is shared across heads" to amortize.
  - Both make the *outer* loop the query grid and the *inner* loop the (few) live KV blocks, with a
    CSR-like or index-list of blocks to visit; our BIM CSR "walks only its row → empty pairs never
    launched" (kernels.md), NSA loads only the `n` selected blocks. Same "don't launch empty blocks"
    idea.
  - Both insist on **blockwise** granularity for Tensor-Core-friendly contiguous access; NSA argues
    token-level selection is hardware-hostile, which is exactly why our grants are block rectangles
    and why we run a 64-tok Triton block (kernels.md; note packing_density.md flags the 128-vs-64
    block-size mismatch in our *density proxy*).
  - Divergence: NSA's live-block *count per query is near-constant* (`n` fixed), which is what lets
    it park the query loop on the grid scheduler cleanly. **Ours is highly variable** — link density
    swings ~6× across packs, which is precisely why we need **density bucketing + per-step cross-rank
    density matching** (packing_density.md §CENTRAL CONTRIBUTION). NSA sidesteps the DDP-straggler
    problem by construction; we solve it at the data-scheduling layer. This is a clean contrast to
    draw in the paper.
- **Three branches vs. one mask.** NSA needs compression + selection + sliding-window because a
  learned selector alone loses local context and global gist. We fold the analogous roles into a
  single mask: intra-doc causal attention *is* our "local/sliding" context (full within a doc), and
  the grant edges *are* our "selection" of far context. We have **no compression branch** — we never
  summarize a target doc into block tokens; the linker reads the target's raw tokens through the
  grant. So where NSA's global path is *lossy compression*, ours is *lossless but sparse* (you only
  see the docs you linked to).
- **Inference symmetry.** Both use the trained pattern at inference, but the trigger differs. NSA
  recomputes importance from the running query at decode. Ours is genuinely novel here: a
  *generated* link token dynamically fetches the target doc into the attention context — at
  inference the grant is produced by **text link detection** (masks.md §Option B: "GENERATION uses
  text detection, no graph shortcut"), so the model literally writes a link and thereby pulls a
  document into scope. NSA has no equivalent generative retrieval step; its sparsity is always
  affinity-driven over what's already in the KV cache.
- **Positional encoding.** NSA's compression branch adds an intra-block position encoding and the
  branches operate on ordinary RoPE'd keys. Our reviewer-flagged issue (masks.md §Reviewer-attackable
  #1) is that **RoPE is not reset per doc** — a linker reads its target at a relative offset equal to
  the arbitrary packing distance. NSA doesn't face this because it never re-homes far blocks to a
  linker; it just attends within one contiguous sequence. Worth noting NSA as prior art that keeps
  positions native while still being sparse.

## Predictions & open questions for our method

- **Sparse-trained can match/beat dense — expect the same, not a regression.** NSA's central
  empirical result (avg *up* on general benchmarks, big up on long-context) is the strongest external
  evidence that our `cross_doc_link` mask should not cost quality vs. our `doc_concatenated` /
  `doc_concat_link` compute-control masks (masks.md), and may *win* on the multi-hop / long-context
  slices. If our compute-matched controls beat `cross_doc_link`, that would be surprising given NSA —
  and would point at our link *gating* being wrong rather than sparsity per se.
- **Effect should be strongest on long-context, cross-document retrieval tasks** (their LongBench
  HPQ/2Wiki gains; needle 64k). That maps directly onto our HotpotQA / 2WikiMultiHopQA / MuSiQue eval
  plan (related_work_notes.md §multi-hop). Predict our grant edge helps most exactly where a
  supporting doc must be read across a boundary, and least on single-doc perplexity.
- **Gate/branch analogy → ablation prediction.** NSA shows all three branches are needed; removing
  the sliding-window branch hurts local modeling. Our analog: if we ever weakened intra-doc causal
  attention (the "local" role) in favor of grants, expect degradation. Conversely NSA suggests a
  *learned gate* over "read the linked doc vs. rely on local context" could help us — we currently
  hard-OR grants with no weighting (masks.md #4). Open question our design could probe: **is a hard
  binary grant enough, or do we (like NSA) need a soft gate** balancing linked-context vs.
  local-context? Our compute-control masks give us a way to answer this that NSA cannot.
- **Recall of the important-block set.** NSA argues heuristic/parameter-free selection has *low
  recall* of the truly important blocks (their ablation rationale). Our grants have **perfect recall
  of the linked doc by construction** (the graph is ground truth), but **zero recall of
  semantically-relevant-but-unlinked docs**. Prediction: on tasks where the answer doc is linked,
  we should beat any affinity selector; on tasks where relevance is not encoded as a link, NSA-style
  learned selection would beat us. This delimits where our inductive bias is a feature vs. a
  limitation.
- **Their open question we may resolve:** NSA leaves open whether native sparsity generalizes when
  the *sparse structure is externally specified rather than learned*. We are exactly that experiment
  — structure from a link graph, trained natively. Our result speaks to whether "native trainability"
  is what matters (NSA's thesis) independent of whether the pattern is learned or given.
- **Their result that constrains us:** NSA matched dense at 27B/3B on 260B tokens. We should not
  over-claim a *quality* win from sparsity alone at smaller scale; the honest framing (which NSA
  supports) is "sparse-native training preserves quality while enabling the structural/efficiency
  benefit," and any quality gain must be attributed to the *linking bias* via the compute controls,
  not to sparsity.

## Gotchas

- **Attribution trap.** NSA beats *both* full attention and an exact-top-k baseline — i.e. they were
  careful to separate "sparsity helps" from "our particular selection helps." That is precisely our
  `doc_concatenated` / `doc_concat_link` compute-control design (masks.md #5). Lesson: report the
  matched-FLOP control prominently, or a reviewer will assume any win is just more/less compute.
- **Two token-count figures.** The paper itself states both 260B and 270B pretraining tokens in
  different places — a reminder to pin exact numbers from tables, not prose, when we cite it (I did
  not resolve which is canonical).
- **Speedups are regime-dependent and partly *expected* (analytical), not all wall-clock.** Their
  decode speedups (4×–11.6×) are memory-access-based *expected* numbers; forward/backward 9×/6× are
  measured at **64k**. At our shorter/denser regimes the multiple would be much smaller. This mirrors
  our own honesty flag: our density proxy is analytic (~1ms) and only *rank-preserving*, not the
  literal executed block count (packing_density.md #1, #2). Don't quote a peak speedup as typical.
- **Non-differentiable selection breaks gradient flow.** NSA's whole justification for the
  compression-attention importance trick is that hard top-k is non-differentiable; they route
  gradients through the gate and the compression branch. Our mask is *also* non-differentiable
  (hard binary), but we don't need gradients through it because the structure is fixed/given — a
  point to state explicitly so a reviewer doesn't ask "how do gradients flow through selection?"
- **Masking full rows → NaN.** NSA forces 1 initial + 2 local blocks always-selected, partly so no
  query has an empty KV set. We hit the same failure from the other side: our **sentinel-LSE NaN
  guard** (kernels.md, v18) exists precisely because a token with zero valid KV under a sparse mask
  produces `exp2(+1e6)=inf, inf×0=NaN`. Independent confirmation that empty-KV rows are a real sparse-
  attention footgun; our fix is a genuine numerical-stability contribution to cite alongside theirs.
- **GQA-shared selection is load-bearing for their kernel speed.** If our BIM ever stopped being
  head/layer-shared, our once-per-batch amortization collapses — NSA's design underlines how central
  "sparsity shared across heads" is to *both* our kernels.

## Missed citations worth adding

Checked against bib/refs.bib. Note: `yuan2025nsa` (assigned key) **and** `yuan2025nativesparse`
are BOTH in refs.bib for this same paper — a **duplicate entry that should be de-duped** (keep
`yuan2025nsa`, the more complete/assigned key). The following NSA references are genuinely absent
and matter to us:

- **tang2024quest** — "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference," Tang,
  Zhao, Zhu et al., arXiv **2406.10774**. NSA's named foil for blockwise query-aware KV selection.
  The closest *inference-time* analog to our per-query block relevance; sharpens the
  train-with-it (us/NSA) vs. select-at-inference contrast. (The "Quest" already noted in our
  related_work_notes line 20 is a different data-*selection* Quest — this one is uncited.)
- **xiao2023streamingllm** — "Efficient Streaming Language Models with Attention Sinks," Guangxuan
  Xiao et al., arXiv **2309.17453**. Attention-sink / always-keep-initial-tokens; directly parallels
  NSA's forced-initial-block and our EOS/layout-token handling and empty-row guard. A standard
  reference for fixed-plus-recent sparse patterns we currently lack.
- **wu2024retrievalhead** — "Retrieval Head Mechanistically Explains Long-Context Factuality," Wu,
  Wang, Xiao, Peng, Fu, arXiv **2404.15574**. NSA cites it for the claim that a sparse set of
  "retrieval heads" carries long-context factuality and is vulnerable to post-hoc pruning. Highly
  relevant to us: it's mechanistic evidence for *why* cross-document reading concentrates in specific
  heads — a natural probe target for whether our grant edge is used by such heads.

Lower priority / likely out of scope (NSA cites them but they overlap our existing efficient-attn
slice): H2O (heavy-hitter KV eviction), InfLLM, ClusterKV, MInference. Mentioning them by name in the
"learned/heuristic KV-selection at inference" sentence would round out the contrast, but only the
three above are strong standalone adds.

---
Confirmed: NSA method/branches/kernel/scale and the specific result numbers were extracted from the
arXiv v2 HTML; the three recommended cite-adds and the `yuan2025nativesparse` duplicate were verified
against refs.bib and arXiv abstract pages. Inferred (not GPU-verified here): the kernel/BIM parallels
rely on our code briefs as-written.
