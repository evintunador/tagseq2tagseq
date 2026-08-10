## lu2025moba — MoBA: Mixture of Block Attention for Long-Context LLMs

arXiv 2502.13189 (Moonshot AI / Kimi; Lu, Jiang, Liu, Du, et al., Feb 2025). Code: github.com/MoonshotAI/MoBA.
Cross-referenced against our code briefs `kernels.md` and `masks.md` (both @ commit 6134163) and the
`related_work_notes.md` §3 entry that already pairs MoBA with `yuan2025nsa` as "natively trained block
sparsity … with learned importance or a router over the same sequence, versus our graph-dictated key ranges."
Numbers below confirmed from the arXiv HTML (v1); where the HTML rendered only citation keys I say so.

### What the paper actually does

**Core idea.** MoBA applies the Mixture-of-Experts routing idea *to attention itself*: partition the KV
sequence into contiguous positional **blocks** and let each query token pick, via a **parameter-less top-k
gate**, the few blocks it will actually attend to. Softmax runs only over the keys/values in the selected
blocks. The design principle is stated as "less structure" — do not hard-code sink/window patterns; let the
model *learn* where to attend. The KV blocks are the "experts"; the gate is the router.

**The gate (the mechanism that matters for us).** Context of length N is split into `n` blocks of size
`B = N/n`, block *i* spanning tokens `[(i−1)B+1, iB]`. For a query *q*:
- Affinity score `s_i = ⟨q, mean_pool(K[block i])⟩` — inner product of the query with the **mean-pooled keys**
  of the block (block "representation" = average key). This is the only "routing" computation; it adds **no
  parameters** (reuses the existing K projection).
- `g_i = 1` iff `s_i` is among the **top-k** highest scores across blocks, else 0. The selected index set
  `I = ∪{block i : g_i=1}`; softmax is normalized only over `K[I], V[I]`.

**Causality, enforced by three rules** (this maps almost one-to-one onto our mask predicates):
1. **No future blocks:** for any block strictly ahead of the query (`pos(q) < iB`), set `s_i = −∞`, `g_i = 0`.
2. **Current block always selected, with a causal mask:** the block containing the query is force-routed
   (`g=1`) and attended `causal=True`, so mean-pooling can't leak future tokens inside it. The paper likens
   this to the **shared expert** in modern MoE.
3. **Past blocks fully attended:** strictly historical selected blocks are computed `causal=False`.

**Switching to full attention (a headline selling point).** MoBA is a "flexible substitute for full
attention" via two recipes: (a) **MoBA/Full hybrid training** — train on MoBA for the first ~90% of tokens,
then flip to full attention for the last ~10% ("loss nearly identical to full, no spikes at the switch");
(b) **layer-wise hybrid** — keep the last few layers full attention (large-model eval: "last three layers
full, other 29 MoBA"), which also fixes SFT's sparse-gradient problem from prompt-token loss masking.

**Setup & scale.** Scaling-law suite: five models 568M→2.1B params, seqlen 8K (32K for trailing-loss tests),
block 512 / top-3. Hybrid: three 1.5B models, 30B tokens, 32K. Large eval: start from **Llama-3.1-8B-Base**,
extend 128K→256K→512K→1M by position interpolation, MoBA active for 100B tokens, block 4096 / top-12; SFT
context grows 32K→1M. Efficiency probed to **10M tokens** (query-head-level tensor parallel, K/V broadcast).

**Result numbers (exact).**
- **Scaling-law LM loss (8K):** MoBA `2.625·C^−0.063` vs Full `2.622·C^−0.063`; val-loss gap within 1e−3.
- **Trailing loss (32K, last 2K tokens):** MoBA `1.546·C^−0.108` vs Full `1.464·C^−0.097` — MoBA slightly
  worse at long-range trailing positions, but "the gap is progressively narrowing" with compute.
- **Benchmarks (MoBA vs Full):** RULER@128K 0.7818 vs 0.7849 (at 62.5% sparsity); LongBench@32K 0.4828 vs
  0.4821; MMLU 0.4903 vs 0.4904; GSM8K 0.7278 vs 0.7142; HellaSwag 0.8262 vs 0.8279; AGIEval 0.5144 vs
  0.5146; HumanEval pass@1 0.6951 vs 0.7012. Needle-in-Haystack "satisfactory" to 1M.
- **Efficiency:** up to **6.5×** prefill speedup at 1M tokens; **16×** attention-compute reduction at 10M.
  Sparsity: 8K→81.25%, 32K→95.31%, 1M→95.31% (64 blocks, top-3).

### Methodology: theirs vs. ours

One-sentence axis: **MoBA and TS2TS both do natively-trained, block-granular sparse attention that can
revert to full attention — but MoBA's block selection is a *learned, content-similarity, top-k router over
positional blocks of the same sequence*, whereas ours is a *deterministic, hard, graph-edge-dictated grant*
of read-access from a linking token into a specific target document.** MoBA is a router; we are a mask.

Concrete comparison points:

- **Router over positional blocks vs. grant along a graph edge.** MoBA's "block" is a contiguous range of
  *past positions* in one sequence, and the query chooses among them by learned relevance
  (`⟨q, mean_pool(K)⟩`). Our block-interaction is defined by the graph: a detected link A→B grants a
  *rectangle* — rows `[link_end_pos, min(T,A.end))` × cols `[B.start,B.end)` — that is **asymmetric and
  never transposed** (`masks.md` §Formal semantics; `cross_doc_mask.py:417-423`). MoBA picks *which past
  window is relevant*; we dictate *which specific document a link points to*. Their selection is by
  similarity; ours is by identity.

- **The predicate shape is the same; the membership test is the divergence.** MoBA computes
  `causal ∧ (block ∈ top-k ∨ block = current)`. Our `cross_doc_link` mask is
  `M = (q≥k) ∧ (same_doc ∨ in_grant)` (`masks.md` §Formal semantics). Line them up:
  their "current block, causal" == our `same_doc` diagonal always-attend; their learned top-k `in top-k`
  == our structural `in_grant`. **Both are "causal AND (self OR a selected-elsewhere set)."** The entire
  contrast is what fills the second disjunct: a *learned soft router* vs a *graph edge resolved by exact
  identifier match* (Option B baked grants, `masks.md` §Option B).

- **Parameter-less, both of us — worth stating jointly.** MoBA advertises a *parameter-less* gate (reuses
  K, no router weights). Our mask likewise adds **no learned edge bias and no message passing** (related-work
  §2 framing). This is a shared virtue against the graph-transformer line (`ying2021graphormer` learned
  edge-bias terms): neither MoBA nor TS2TS pays extra parameters for structure. But MoBA's selection is
  still *learned end-to-end through the gradient into K* (the gate is differentiable via the selected
  blocks); ours is **not learned at all** — the grant is fixed by the link graph, identical in train and
  inference, with no gradient shaping *which* blocks interact.

- **Static-per-batch shared schedule vs. dynamic-per-token-per-head routing (kernel consequence).** This is
  the sharpest systems divergence. Our BIM is a block-level CSR computed **once per batch on CPU and reused
  across all heads and all layers** because the mask is head/layer-independent (`kernels.md` §BIM;
  `masks.md` §Compilation). MoBA's gate fires **per query token and per head** — every token routes to its
  own top-k blocks, so the sparse pattern is *dynamic and cannot be precomputed once or shared across
  heads/layers*; MoBA instead permutes/gathers query-to-block assignments and runs a varlen FlashAttention
  each forward. We amortize an `O((T/bs)²)` schedule build over the whole network; MoBA re-derives routing
  every layer. Our sparsity is **data-defined but static**; theirs is **content-defined and dynamic**.

- **Top-k budget vs. union-of-grants.** MoBA caps attention at a fixed *k* blocks per query (a compute
  budget → its clean 75–95% sparsity numbers). Our grants compose by **UNION (OR), not top-k**, with a
  *positional* cap `max_grants=256` (`masks.md` §max_grants, §Reviewer-attackable #4): links beyond the cap
  are dropped by link-order, not importance. So MoBA's budget is per-query and importance-ranked; ours is
  per-sequence and order-ranked — a link that matters but arrives late can be silently truncated, whereas
  MoBA would still rank it if its affinity were high.

- **Mean-pooled key representation vs. exact block membership.** MoBA summarizes each block by an *average
  key* — a lossy, order-agnostic block signature (the paper notes Quest is "MoBA with a smaller block size
  and a min/max-pooling block representation"). We have no pooled representation: membership is exact
  bit-packed grant testing, `in_grant = OR_c(q_bm[c][q] & kv_bm[c][k]) ≠ 0`, pointwise with no sequence
  reduction (`kernels.md` §Grant bitmasks). MoBA's pooling is where its "routing" lives; our equivalent
  "enabling trick" is the bitmask encoding, which is about *cheaply representing a known pattern*, not
  *scoring an unknown one*.

- **Train-on-structure vs. retrieve-at-inference.** Both are firmly *train-on-structure* (native sparse
  attention trained end-to-end, not an inference bolt-on) — this is exactly why the notes group MoBA/NSA
  apart from the H2O/StreamingLLM/Quest inference-time KV-eviction crowd. TS2TS goes one step further on
  train/inference symmetry: the *same* mask runs at inference so a **generated** link deterministically
  fetches its target into attention (`masks.md` §Option B: generation uses text detection, same
  `link_to_target` semantics). MoBA has no inference-time "fetch a specific document" analog — its router
  just re-selects relevant positional blocks already in the window.

- **Reverting to full attention.** MoBA needs an *explicit* transition (90/10 token hybrid, or full last-3
  layers) and warns it is "not a drop-in" — it must *continue-train* from a full model. Our compute-control
  masks make the same axis a *first-class experimental variable*: `doc_causal < cross_doc_link ≤
  doc_concat_link ≤ doc_concatenated` is a strict superset ladder (`masks.md` §Formal semantics), where
  `doc_concatenated` is (nearly) full attention within a component. We don't need a training-schedule switch
  to reach full attention — we run matched-compute variants side by side.

### Predictions & open questions for our method

- **Sparse ≈ full on LM loss is achievable — but their sparsity is *learned to cover relevant blocks*.**
  MoBA matches full-attention loss within 1e−3 at 75–95% sparsity. Encouraging that a block-sparse trained
  mask need not hurt LM loss. **But the caveat is load-bearing for us:** MoBA's router can attend to *any*
  relevant past block; our grant lets a token attend *only* to same-doc + explicitly-linked target docs. Where
  relevant context is present but **not linked**, MoBA would still route to it and we cannot. Prediction: our
  edge helps precisely on *link-mediated* dependencies (multi-hop QA gold-supporting docs, cross-file imports)
  and gives *nothing* where relevance is unlinked — a sharper, narrower win than MoBA's general recall.
  This is the strongest argument for reporting link-conditioned Δnll, not corpus-average loss.

- **Trailing/long-range positions carry a small sparse penalty.** MoBA's trailing-loss exponent is worse
  than full (`1.546·C^−0.108` vs `1.464·C^−0.097`). Prediction: at the *cross-document* positions right
  after a link fires, we may see a small penalty vs the full-attention `doc_concatenated` control unless the
  grant actually captures the needed target. Track loss *at and just after link positions* specifically —
  that's where our mechanism must pay off, mirroring MoBA's trailing-token probe. Their "gap narrows with
  compute" also predicts our edge/concat gap should be scale-dependent — check it doesn't vanish at scale.

- **Block granularity is a tunable with real consequences.** MoBA sweeps block 512→4096 and top-k, keeping
  ~75% sparsity, and finds finer blocks + more-selected give better selection at fixed sparsity. Our BIM
  block size is **128** (`masks.md`/`kernels.md`), chosen for kernel efficiency not selection quality. Since
  our selection is graph-exact (not similarity-ranked), finer blocks mainly reduce boundary waste, not
  selection error — but MoBA's ablation predicts that if we ever coarsen blocks for speed, a
  *learned*-selection method would degrade while ours should be robust (the grant rectangle is exact
  regardless of block size, modulo BOUNDARY straddle blocks). Worth an ablation: vary BIM_BLOCK_SIZE and
  confirm loss is flat, isolating that our sparsity is exact, not approximate.

- **Hybrid full layers may matter for us too.** MoBA keeps the **last 3 of 32 layers full attention**,
  citing SFT sparse-gradient problems. Prediction/experiment: a TS2TS variant with a few full-attention
  (`doc_concatenated`) layers, or full attention late in training, could recover any residual gap — and if
  it *doesn't* help, that's evidence our graph grant already delivers the needed connectivity. Cheap ablation
  that directly borrows their finding.

- **Open question we can resolve for them.** MoBA's thesis is "less structure — let the model learn where to
  attend." TS2TS is the *maximally-structured* counter-experiment on the same block-sparse-attention
  substrate: fixed, exact, graph-dictated selection with **zero learned routing**. If our graph grant matches
  or beats a learned router on link-mediated tasks, that's direct evidence that *predefined structural bias*
  can beat learned selection *when the structure is real* — the question MoBA leaves open by design.
  Conversely, MoBA suggests an extension we deliberately forgo: a *learned* gate over candidate target docs
  (soft link-resolution) instead of exact identifier match — relevant if our deterministic detector's recall
  is the bottleneck.

### Gotchas

- **"Not a drop-in" / continue-train dependence.** MoBA's clean loss-parity comes from *continue-training a
  full-attention Llama*, not from-scratch sparse training. We train from scratch with the mask baked in from
  step 0 — so MoBA's "matches full within 1e−3" is **not** a from-scratch guarantee. Don't cite their parity
  as evidence that our from-scratch sparse mask is free; run the matched-compute ladder ourselves.
- **The pooled-key causal leak — our analog is the ordinal-relabel NaN.** MoBA had to force the current block
  causal because mean-pooling keys would otherwise leak future tokens. Their correctness fix rhymes with our
  two documented sparse-mask correctness fixes: the **ordinal run-index relabel** (non-monotonic traversal
  doc_ids break block-overlap `same_doc`, causing thestack `cross_doc_link` NaN / dQ≈5.7e4) and the
  **sentinel-LSE NaN guard** for fully-masked rows (`kernels.md` §Novel, `masks.md` §Novel #2). Lesson: any
  block-level summarization or masking primitive has a subtle future-leak / degenerate-row failure mode;
  ours are already found and fixed, but re-audit if the mask changes.
- **Mask-mismatch across code paths.** MoBA warns that switching attention modes (MoBA↔full, train↔SFT)
  needs care. Our version of this risk is real and already flagged: **three** mask reimplementations (Flex
  BlockMask, Triton BIM, dense viz) must agree, and `max_grants` must match train vs eval or the effect is
  understated (`masks.md` §Reviewer-attackable #6), plus **generation uses text-detected links while
  train/graph-eval use baked Option-B grants** (`masks.md` §Option B) — a genuine train/inference mask
  divergence to validate, exactly the kind of switch MoBA cautions about.
- **Sparsity-vs-quality knee.** MoBA at low sparsity → full attention; at high sparsity trailing loss grows.
  Our analog knob is `max_grants` (positional truncation) and block size. Too-aggressive grant capping is our
  "too-small top-k": pick it from realized link density, not a round number, and report realized vs intended
  connectivity.
- **Benchmark deltas are tiny and noisy.** MoBA's benchmark table swings both directions within ~±0.01
  (GSM8K +0.014, HumanEval −0.006). At that scale these are within-noise. This validates our related-work §6
  preference for **paired continuous Δnll with bootstrap CIs** over a few-point accuracy delta
  (`biderman2024lessons`, `schaeffer2023mirage`); don't let a 1-point RULER/LongBench move carry the claim.
- **Efficiency numbers are inference/prefill, not training.** MoBA's 6.5×@1M / 16×@10M are *prefill* speedups
  at extreme lengths with 95% sparsity. Our operating point is **T=32K training** with graph-dictated
  (lower, variable) sparsity, and our own kernel numbers are vs FlexAttention at 0.77–0.91× total at 32K
  (`kernels.md` §Speed) — a different regime. Don't import their speedup multipliers as expectations for us.

### Missed citations worth adding

Checked against `paper/bib/refs.bib`. **Already present:** `lu2025moba` itself (line 920), `yuan2025nsa`
(918; NSA, the sibling native-sparse paper — MoBA's HTML did *not* cite NSA, they are contemporaneous),
`gao2024seerattention` (922), `beltagy2020longformer`, `zaheer2020bigbird`, `child2019sparsetransformers`,
`ainslie2020etc`, `deepseekv3`, `gu2023mamba`, `peng2023rwkv`, `sun2023retnet`, `katharopoulos2020linear`,
`choromanski2021performer`, `poli2023hyena`. Genuinely **missing** MoBA references relevant to us
(arXiv ids to be verified by the maintainer — I have not confirmed them against arXiv and may be wrong):

- **tang2024quest** — "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference" (ICML 2024;
  arXiv ~2406.10774 — *verify*). *Why:* MoBA explicitly calls Quest "MoBA with a smaller block size and a
  min/max-pooling block representation." It is the closest *inference-time* block-level sparse-attention
  method to our block-interaction mask — a direct contrast point (our block selection is a graph grant, not
  a query-aware pooled-key estimate) and a natural neighbor to our BIM discussion. Highest-value add.

- **xiao2023efficient** (StreamingLLM) — "Efficient Streaming Language Models with Attention Sinks"
  (arXiv ~2309.17453 — *verify*). *Why:* the canonical *fixed-pattern* (sink + window) baseline MoBA argues
  against under its "less structure" principle; our related-work §3 fixed-pattern list
  (child2019/beltagy2020/zaheer2020) is missing the attention-sink result, which is also relevant to our
  RoPE-no-reset / boundary-token discussion (`masks.md` §Reviewer-attackable #2: −1/EOS pseudo-doc).

- **zhang2024h2o** (H2O) — "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs"
  (arXiv ~2306.14048 — *verify*). *Why:* the reference inference-time KV-eviction / heavy-hitter method; the
  foil for "native trained sparsity (us, MoBA, NSA) vs. post-hoc eviction." Sharpens the train-on-structure
  axis in §3.

- **shazeer2017outrageously** — "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
  Layer" (arXiv ~1701.06538 — *verify*). *Why:* the conceptual parent of MoBA's "MoE-for-attention" router,
  and the reference for top-k gating in general. Only worth adding if we write the "router-over-blocks vs
  graph-edge grant" contrast explicitly (which this deepdive recommends); otherwise low priority. Its
  companion `fedus2022switch` (Switch Transformer) is the same lineage, lower priority.

- **lu2024longheads** (Longheads) — MoBA describes it as "MoBA with a top-1 gating network"
  (arXiv ~2402.10685 — *verify*). *Why:* a training-free multi-head block-selection method — the top-1
  degenerate case of block routing, and a tidy contrast to our multi-grant UNION. Medium/low priority.

Lower priority / out of scope: `fedus2022switch`, `lepikhin2020gshard`, `dai2024deepseekmoe` (MoE systems),
`jiang2024minference`, `fu2024moa`, `oren2024tova`, `ge2023fastgen` (more inference-time sparse-KV methods) —
add only if we build out a dedicated MoE-routing or KV-eviction paragraph.

---
Confirmed from the arXiv HTML (2502.13189v1): gating formula (mean-pooled-key affinity, top-k, current-block
causal shared-expert rule, no-future-block masking), block/top-k values (512/3, 2048/3, 4096/12), the two
full-attention transition recipes, the scaling-law/trailing-loss coefficients, the benchmark table, and the
6.5×/16× efficiency and 81/95% sparsity figures quoted above. Our-method claims are grounded in the two named
code briefs and the source lines they cite. Candidate-citation arXiv ids are unverified and flagged as such.
Done.
