## wu2022memorizing — Memorizing Transformers (Wu, Rabe, Hutchins, Szegedy; ICLR 2022 spotlight, arXiv 2203.08913)

This is the "mystery paper" the author recalled as the cached-per-doc-KV Google work. It is the
clearest instance of the frozen-precomputed-KV family in `related_work_notes.md` (§"Precomputed
corpus memory and the cached-KV frozen-store family"). Below, everything under *What the paper does*
was confirmed from the arXiv abstract and the ar5iv HTML of the full paper; the compare-and-contrast
is grounded in `code_briefs/masks.md`, `code_briefs/generation_retrieval.md`, and the sources they cite.

### What the paper actually does

**Idea.** Extend a decoder-only Transformer LM with a large *external, non-differentiable* memory of
past `(key, value)` attention pairs, queried by approximate **kNN lookup** at a single designated
layer. At each step the local (key,value) pairs are appended to the memory (a FIFO ring — oldest
dropped once the memory reaches size M). This lets the model "remember" facts, function definitions,
and theorems it saw earlier in a long document without those tokens being in the local context window.

**Where the edge lives (mechanism, confirmed).**
- One layer is kNN-augmented; default the **9th of 12** layers. Ablation over layers {3,6,9,12}: middle
  layers work best (layer 6 slightly best at 2.36 ppl), layers adjacent to input/output worst.
- **Same query** feeds both local attention and the memory lookup. For each query, kNN retrieves the
  **k=32** nearest keys from memory (32 ≈ 128 ≈ 256 in ablation, so k is not sensitive); memory
  attention is computed over just those retrieved (k,v) pairs, in parallel with ordinary local
  attention over the current segment + a Transformer-XL sliding cache.
- **Gate:** the two attention results are combined by a *learned per-head scalar* `g = σ(b_g)`:
  `V_a = V_m ⊙ g + V_c ⊙ (1−g)`. The gate is **content-independent** (a bias, not a function of the
  token). Most heads learn `g≈1`, attending almost exclusively to memory.
- **Positions:** T5 relative-position bias is applied to **local** attention only. **No positional
  bias on retrieved memories** — the paper reports long-range relative position "did not matter,"
  so memory keys are position-free.
- **Staleness:** keys/queries are **normalized** to keep magnitudes consistent as parameters drift
  across the many steps a long document spans; for very large memories (131K+) they pretrain with a
  small memory then finetune up, because stale keys otherwise destabilize training.

**Scale & setup.** Base model ~200M params (12 layers, d=1024, 8 heads × 128, FFN 4096); also scaled
to 1B and 8B. Documents split into **512-token subsequences** (also tested context 2048). Datasets:
arXiv Math, PG-19 (books), C4 (webtext), GitHub (code), Isabelle (formal theorems). 500K steps.

**Numbers that matter (perplexity, lower better; ctx 512 unless noted):**

| Config | arXiv | PG19 | C4 | GitHub | Isabelle |
|---|---|---|---|---|---|
| No memory, no XL | 3.29 | 13.71 | 17.20 | 3.05 | 3.09 |
| + memory 8192 | 2.49 | 12.29 | 14.42 | 2.09 | 2.19 |
| + memory 65K + XL | 2.31 | 11.62 | 14.04 | 1.87 | 2.06 |
| ctx 2048, mem 65K + XL | 2.26 | 11.37 | 13.64 | 1.80 | 1.99 |

- **Memory > parameters:** a Memorizing Transformer with only **8K** memory matches a vanilla model
  with **5× more trainable parameters**. This is the headline result.
- **Scales with memory:** monotone improvement from 1536 → 262K memory tokens; best around 65K, further
  gains to 262K after arXiv finetuning.
- **Retrofit:** a *pretrained non-memory* 1B model finetuned to use memory closes **85% of the gap in
  20K steps (4% of pretraining)**, fully closed by 100K. Memory can be bolted onto an existing model.
- The gains are largest on **code and formal math** — domains where a newly-defined function/theorem
  must be reused verbatim far later — which is exactly the cross-file-reuse regime TS2TS targets.

### Methodology: theirs vs. ours

The right axis (per `_DEEPDIVE_BRIEF.md`): **train-on-structure vs. retrieve-at-inference; attention edge
vs. GNN vs. cached-KV vs. training-pair signal.** Memorizing Transformers sits at **cached-KV**; TS2TS
sits at **trained attention edge**. Point by point:

- **What enters memory.** Theirs: compressed *representations* — per-token `(key, value)` vectors from
  one layer, stripped of position. Ours: **verbatim document tokens** re-materialized into the packed
  sequence, attended at *every* layer with full RoPE positions (`generation_retrieval.md`: retrieval is
  literal insertion of the target doc's tokens into the packed token list, `_docs.insert(idx)`, then
  `build_sequence()` recomputes offsets). Their memory is a lossy KV snapshot; ours is the real text.

- **Differentiable?** Theirs: **non-differentiable, frozen** memory — no gradient flows into stored
  (k,v); the model is trained only to *consume* a fixed store. Ours: the cross-doc grant is a hard mask
  over ordinary self-attention, so gradients flow through the attended target tokens exactly as through
  same-doc tokens — the link edge is trained end-to-end, not read-only.

- **How the target is selected.** Theirs: **approximate kNN / MIPS** by embedding similarity — a soft,
  learned-similarity, top-k retrieval with approximation error. Ours: **deterministic graph-edge /
  identifier resolution** — a link detector fires and `index_doc_span(node)` resolves the exact target
  node via an exact hashmap (`document_corpus.py`; `masks.md` §Detected link→grant). No embeddings, no
  ANN index, no approximation error. This is the same distinction `related_work_notes.md` draws for the
  whole cached-KV family: "deterministic graph-edge/identifier resolution rather than a learned
  similarity search."

- **Granularity.** Theirs: **token granularity** (k=32 individual token KV pairs). Ours: **document/node
  granularity** — a grant opens a whole target-doc column range `[B.start,B.end)` to the linking rows
  (`masks.md`: grant rect per link A→B). We attend an entire retrieved document, not 32 scattered tokens.

- **Where the edge acts.** Theirs: **one layer** (the 9th), one attention head-group, gated in. Ours:
  **all layers**, because the target's tokens are physically in the sequence; every layer's self-attention
  sees them subject to the same block-sparse mask. There is no separate "memory attention" module and no
  per-head gate — the inductive bias is entirely in the mask geometry.

- **Train/inference symmetry.** Theirs: memory used at both train and test, but always as an *external*
  store consulted via kNN — the local model never differs structurally. Ours: the **same** link
  machinery (detector + `index_doc_span` match key + grant-from-`link_end_pos` + DAG ordering) runs in
  pretraining and in generation; `generation_retrieval.md` calls this the strongest claim — inference
  retrieval *is* the training cross-doc mask realized by materializing the linked doc into the packed
  seq. Their symmetry is "same frozen store"; ours is "same trained attention primitive."

- **Position handling — a genuine convergence worth citing.** They found long-range relative position
  "did not matter" and applied **no position bias to memory keys**. TS2TS does the *opposite*: it keeps
  **global RoPE positions across packed docs and does not reset per doc** (`masks.md` Reviewer-attackable
  #1; `related_work_notes.md` §"Boundary masking and position handling"). So a linking doc reads its
  target at a relative offset equal to the (arbitrary, traversal-order-dependent) packing distance. Their
  result is direct evidence that a model *can* learn to use content-addressed long-range reads without
  meaningful relative-position signal — supportive of our no-reset choice, but note their memory is
  position-*free* whereas ours has arbitrary-but-present positions; the reviewer question ("is the RoPE
  offset semantically meaningful?") is one they sidestepped by dropping position entirely.

- **KV reuse / efficiency.** Theirs: memory *is* a KV cache — the entire point is cheap reuse of
  precomputed KV via kNN, avoiding recompute. Ours: **no KV cache at all** — `build_sequence()` +
  `forward_inference()` recompute from scratch every token, O(T²) per step (`generation_retrieval.md`
  §"KV CACHE = NONE"), because insertion/eviction shifts RoPE positions and makes cached KV incorrect.
  This is the sharpest efficiency contrast: they are on the reuse pole, we deliberately forgo reuse for
  train/inference correctness.

**Shared with us:** long-document code/math as the target domain; the intuition that verbatim earlier
content must be *fetched*, not memorized in weights; and the empirical claim that fetch beats parameters.
**Where we diverge:** frozen non-diff token-KV via approximate kNN at one layer (theirs) vs. a trained,
exact, direction-gated, document-granularity attention mask at all layers (ours).

### Predictions & open questions for our method

- **Fetch should beat parameters (their headline).** Their 8K-memory-≈-5×-params result predicts our
  cross-doc edge should let a *small* TS2TS model match a much larger flat-concat baseline on cross-doc
  tasks. If our matched-compute controls (`masks.md` §5: `doc_concatenated`, `whole_doc_grant`) show the
  linking bias winning at fixed FLOPs, that mirrors their memory-beats-scale finding — and our exact-edge
  variant should, if anything, beat their approximate-kNN one on tasks where the right target is known.

- **Domain profile.** Their gains were largest on **code and formal theorems** (verbatim-reuse regimes),
  smaller on webtext/books. Prediction: TS2TS's cross-doc effect should be **strongest on the import-graph
  code corpus and citation/wikilink hops**, weakest on single-document controls (consistent with the
  single-doc control benchmarks in `related_work_notes.md` §"Single-document controls"). Expect a *large*
  Δnll on the code/HotpotQA settings and ~0 on HellaSwag/ARC/PIQA.

- **Memory-size scaling ↔ our packing/depth budget.** Their perplexity improved monotonically with memory
  up to 262K tokens. Analogue for us: effect should grow with how much linked context is reachable — i.e.
  with pack density (how many topologically-close docs land in the 32k window) and, at inference, with
  `max_link_depth` / `max_auxiliary_documents` (`generation_retrieval.md`). Predict diminishing-returns
  curve in reachable-neighbor budget rather than raw sequence length.

- **Layer placement.** They found *middle* layers best for the memory edge; input/output-adjacent worst.
  We apply the grant at all layers, so this doesn't map directly — but it raises a testable question:
  does the cross-doc grant carry signal uniformly across depth, or would a *depth-restricted* grant
  (only middle layers see cross-doc keys) recover most of the gain at lower cost? Their result suggests
  the effect might concentrate in middle layers — a cheap ablation for us.

- **Retrofit / continual.** Their finetuning-in-memory result (85% of gap in 4% of steps) predicts we may
  not need to pretrain the cross-doc mask from scratch: a flat-concat checkpoint could be *finetuned* to
  use the link grant cheaply. Relevant to `ibrahim2024continual` / WSD-resume machinery already in the
  repo. Open question our design could resolve for them: they had to freeze the store to make it tractable;
  our exact-resolution + full-recompute shows what the *non-frozen, non-approximate* upper bound looks like.

- **Open question of theirs we may answer.** They explicitly leave open whether a *learned* retriever or
  structural signal would beat kNN-on-embeddings. TS2TS's deterministic graph edge is one answer: on
  corpora where an exact edge exists (imports, citations, hyperlinks), you can skip similarity search
  entirely. Conversely, their result that *content-independent* per-head gates suffice hints that our hard
  binary mask (also content-independent) is not leaving much on the table vs. a learned soft edge bias.

### Gotchas

- **Staleness is real and bit them at scale.** Stored keys drift as weights update within a long document
  spanning many steps; they needed key/query normalization and a small-memory-then-finetune schedule
  above 131K. **We largely sidestep this** because we recompute KV every step (no stale store) — but the
  *training-time* analogue is our own NaN failure: `masks.md` Novel #2 documents that naive (non-monotonic)
  doc labeling collapsed the LSE and blew up dQ on thestack cross_doc_link training. Both are "long-range
  attention numerically unstable" failures; watch for LSE/normalizer pathologies whenever the cross-doc
  span is large.

- **Position-free memory hides a question we can't hide.** Because their memory keys have no position, they
  never had to answer "is the retrieved token at a sensible relative distance?" We do — global RoPE offsets
  to a fetched doc are arbitrary and traversal-order-dependent. Their success with position-free memory is
  encouraging but is *not* proof our arbitrary-offset design is safe; test explicitly (the reviewer will).

- **Perplexity ≠ downstream.** All their headline numbers are language-model perplexity. Perplexity gains
  from memory can be dominated by verbatim copying (exactly their code/theorem-reuse story), which inflates
  the metric without demonstrating reasoning. `related_work_notes.md` (§"Evaluation-methodology backbone",
  `biderman2024lessons`, `schaeffer2023mirage`) already pushes us toward paired Δnll and away from
  metric artifacts — good, because a raw-perplexity cross-doc win could be pure copy-through.

- **The gate learned ≈1 (memory dominates).** Most heads attended almost entirely to memory. If we ever add
  a *soft* variant of our grant, expect it to saturate similarly — and expect that to make ablating the
  edge's contribution harder (the model routes around the control). Our hard binary mask + matched-compute
  controls avoids this, but it's a warning against a "learned gate over the grant" design.

- **kNN approximation error.** Their retrieval is approximate MIPS; they never fully separate "memory helps"
  from "memory retrieval is noisy." Our exact hashmap resolution removes that confound entirely — a strength
  to state, but it also means we cannot borrow their robustness-to-noisy-retrieval framing; a single wrong
  `index_doc_span` resolution puts the *whole wrong document* in context, not one bad neighbor.

### Missed citations worth adding

Checked against `bib/refs.bib`. Memorizing Transformers is already present (`wu2022memorizing`,
`refs.bib:93`). Its most relevant references that appear **genuinely absent**:

- **Sukhbaatar et al., "Not All Memories Are Created Equal: Learning to Forget by Expiring" (Expire-Span), ICLR 2021, arXiv:2105.06548.** A cited MT reference and the natural complement to our inference-time `DocumentContext` eviction (`generation_retrieval.md`: `drop_oldest`/`stop_new`, `make_room`): Expire-Span *learns* which memories to forget, versus our topology/depth-based drop. Not found in refs (grep `expire`/`expirespan` empty).
- **Fan et al., "Addressing Some Limitations of Transformers with Feedback Memory" (Feedback Transformer), arXiv:2002.09402.** MT-cited recurrent-memory work; fits the "Memory-augmented and recurrent-memory transformers" cluster alongside `dai2019transformerxl`/`rae2020compressive`. Grep for feedback returns only `hwang2024transformerfam` — absent.
- **Rae et al. / Hutchins et al. Block-Recurrent** is already covered (`hutchins2022blockrecurrent`), so skip.

(Lower-confidence, verify relevance before adding: **Polu & Sutskever, "Generative Language Modeling for
Automated Theorem Proving," arXiv:2009.03393** — MT's Isabelle/formal-math motivation and an antecedent to
"generate an identifier that fetches content," but it is theorem-proving-specific and may be out of scope
for TS2TS. Not in refs. I did not confirm it is cited by MT specifically — flagging as a maybe, not a claim.)

Note: `lample2019productkey`, `guo2020scann` (ScaNN, the ANN backend), `sukhbaatar2019adaptivespan`,
`dai2019transformerxl`, `rae2020compressive`, `roy2021routing`, and `kitaev2020reformer` are all already in
refs.bib, so they are **not** listed above despite being MT-adjacent.

Confirmed: paper details verified against arXiv abstract + ar5iv full-text; codebase contrasts grounded in masks.md and generation_retrieval.md; bib-absence claims verified by grep against refs.bib.
