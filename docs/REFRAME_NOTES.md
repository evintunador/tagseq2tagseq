# TAGSeq2TAGSeq — Re-framing notes (theory-first preprint)

Started 2026-11-02. A working dump of decisions, arguments, candidate formal results, design
questions and future-empirics seeds. Written to be split into branching conversations: each
`##` section is meant to stand on its own. Status tags: **[DECIDED]**, **[LEANING]**,
**[OPEN]**, **[OUT OF SCOPE]**, **[FUTURE EMPIRICS]**.

---

## 0. Why the reframe

- The previous approach was empirical-first with no experiment plan written ahead of time. The
  result was a lot of runs, several bug-invalidated result families (val-loss deflation,
  source forgetting, max_grants mismatch, the 2048 community-pack budget, …), and a paper whose
  central claim (fetch-by-citation) was explicitly unmeasured.
- The real pitch is **theory, ergonomics, completeness, beauty**. A researcher should be
  convinced of the design's merit on the argument alone. Empirics come later, planned
  (what to measure, why, how) before any GPU time is spent.
- Compute access in the future is not guaranteed.

## 1. Decisions so far

- **[DECIDED]** Venue: arXiv preprint.
- **[DECIDED]** **Lead claim = inference efficiency.** A knowledge base is precomputed once as
  per-document KV and recomputed only on *edit*, not on every *read*. That's the abstract hook.
- **[DECIDED]** Co-pillars:
  1. The mask as a **graph-structured likelihood** (the "maximal causal mask" framing, §3).
  2. **Grounding / human-editable knowledge** (§6).
- **[DECIDED]** The algorithm is open to redesign, even if that obsoletes existing checkpoints.
- **[DECIDED]** Keep full dataset scope (wiki, arxiv, 10 code languages, fineweb). Diversity of
  edge modalities is part of the generality argument.
- **[DECIDED]** Old empirical results are set aside. They may come back selectively after bug
  checks and a decision on how to frame them.
- **[DECIDED]** Synthetic demonstrations are acceptable if they're genuinely demonstrative
  (they show a mechanism, not "a number went down").
- **[DECIDED]** Theory style: mostly rigorous design rationale, plus a few lightweight formal
  propositions where they make an argument crisp (§3).
- **[DECIDED]** Counterfactual-document training (perturbing target facts to force reliance)
  is a separate experiment. **[OUT OF SCOPE]** for this paper; mention as future work.

## 2. The pitch, abstract-level (draft)

> Corpora carry their own dependency structure: hyperlinks, imports, citations. We train a
> decoder-only LM whose attention follows that structure. A document may read a linked document
> only from the point where it cites it, and only causally. This is the maximal mask under which
> the model's link-following is itself generative. Its consequence is at inference: because a
> document's representation depends only on itself and what it links to, every document's
> key/value state can be computed once and reused by every request that cites it. The knowledge
> base is "compiled" in one topological sweep and recompiled only where it is edited. The model
> fetches by writing a reference, with no retriever or fusion module, and its knowledge becomes
> partly external, inspectable and editable without touching weights.

Lead with cost (read-many/edit-few amortization). Follow with why it's *exact* (training
already used the same isolation), unlike KV-reuse-for-RAG methods, which are approximate.

### 2.1 The framing line (user, 2026-11-03) [DECIDED]
> Flattening images into an MLP gave way to convolutions, which matched the data's true shape.
> TAGSeq2TAGSeq does the same for LLMs. Corpora are not flat sequences, they are
> **text-attributed directed acyclic graphs that evolve over time**, and the architecture
> (attention mask, positions, cache, training data) should be built around that shape.

The inductive-bias analogy is the intro's opening move. Repo histories as native training data
(§5.3) are the "over time" half of the shape.

## 3. Theory: candidate formal results

Lightweight propositions, each one sentence plus a short proof sketch. F0, F1 and F3 are
must-haves.

### F0. Axiomatic characterization of the mask **[DECIDED: centerpiece of the method section]**
Worked through 2026-11-03.

**Setup.** A context is an ordered list of documents, with the reader A (the one being written)
last. A detector maps tokens to links (p, s) and a resolver maps s to a context document. A mask
policy maps (context, tokens) to M(q, k). Multi-layer composition means q's *information set* is
the closure of chains of mask edges, so the axioms must hold for the closure, not just one row.

**Axioms.**
- **A1. Token causality:** M(q, k) ⇒ k ≤ q.
- **A2. Self-determination:** row q is a function only of x_{≤q} (and the documents before A).
  What A may read is decided by what A has written so far. **[DECIDED] Positions count**: the
  relative offsets the reader sees must also be prefix-determined (see the positional leak
  below). This is what buys the append-only cache result.
- **A3. Entitlement (direct) [DECIDED]:** a cross-doc read from q in A into B requires a link ℓ
  in A resolving to B with decision time τ_ℓ ≤ q. Information from B's own ancestors reaches A
  only through B's states (F7/F8). No transitive direct grants.
- **D. Detector prefix-stability (precondition):** a link found in a prefix is found, with the
  same position and target, in every continuation. τ_ℓ = the first token index whose prefix
  makes the link decided (the "decision token").

**Detector abstraction [DECIDED]:** the *terminator* (the decision token) is an explicit part
of the detector interface. Each detector declares what closes a link (`)` for markdown,
newline/`,`/` as` for imports, …) and returns τ_ℓ directly, instead of mask code inferring it
from `link_end_pos`.

**Theorem 1 (characterization).** A1–A3 are each preserved under unions of masks, so a unique
greatest admissible mask exists. It equals `cross_doc_link` with grants from τ_ℓ. Near-
definitional: present it in prose, with the axiom table as the point.

| Design | A1 | A2 | A3 | Maximal |
|---|---|---|---|---|
| standard causal packing | ✓ | ✓ | ✗ unrelated prior docs | – |
| doc_causal / IntraDoc | ✓ | ✓ | ✓ | ✗ grants nothing |
| **cross_doc_link** | ✓ | ✓ | ✓ | **✓ unique** |
| doc_concat_link | ✓ | ✗ reads B before citing | ✓ | – |
| doc_concatenated | ✓ | ✗ | ✗ unlinked siblings | – |
| LinkBERT / CDLM | ✗ bidirectional | – | – | – |
| RAG / Block-Attention | ✓ | ✗ retriever decides beforehand | ✗ | – |

**Theorem 2 (proper likelihood + leakage). Proof-worthy.** Under A1+A2 the pack loss is a
normalized NLL (chain rule). Without A2, it isn't. Counterexample: A is one token, either "link
B" or "other". The grant exists iff the token is "link B", so the model can put ~1 on whatever
happened, and the "likelihood" sums to ~2. So the concat controls' NLL is partly **leakage**
(link tokens become cheap), not just incomparable. That retroactively explains part of their
old strength.

**Theorem 3 (train = inference identity). Proof-worthy.** Under A1–A3 + D, with positions in
A2, the cdl pack loss for A equals the teacher-forced NLL of the link-following generation
process when the resolver returns exactly the pack's documents. With F2, training is the
generation process run with a random subset of targets available.

**Corollary (cacheability, links to F3).** A1+A3 ⇒ a doc's states depend only on its tokens and
its entitlement closure. The axioms that make it generative also make it cacheable.

**What the axioms exposed in the code:**
1. *Grant start / decision token.* Markdown sets `link_end_pos = j+1`, where j is the token
   containing `)`. For markdown, j is already the decision token, since `)` disambiguates
   "Apple" from "Apple Inc.", so a grant from j is safe and j+1 is one position conservative.
   For delimiter-free syntaxes (`import a.b` can continue `.c`) the decision token is the
   terminator (newline, `,`, ` as`). There a grant at the last identifier token would be
   premature. **Rule: grant from the decision token. Audit each detector.** (TODOS)
2. Grant cap is consumed in link-position order (`cross_doc_mask.py:522`), so A2 is preserved.
   Any other order would break A2 silently.
3. *Positional leak under global RoPE.* With targets-first packing, A's later-linked targets
   sit between A and its earlier-linked targets, so the RoPE distance from A's early queries to
   B1 includes |B2|. That's information about a future link. The generation loop inserts each
   fetch immediately before A (`generation_loop.py:333`, giving [B1, B2, A]), so every insert
   shifts earlier targets. That's why it needs full recompute.
   **Fix: anchor A at position 0 and prepend each new fetch in front of the existing targets
   at increasingly negative positions.** Nothing computed ever moves, so the reader's KV is
   **append-only** during link-following generation. Training needs the matching layout
   (reverse link order, recursively). Readers that share targets can't all get their ideal
   layout from one pack (the diamond problem again), so either accept approximation or use
   per-grant virtual offsets.

**Presentation [DECIDED leaning]:**
- Main body: plain-language axioms, each motivated by a *requirement* (A2 ← "the model decides
  what to read by writing"; A3 ← cacheability and attribution). Then the table, the one-line
  leakage counterexample, and a one-sentence statement of Theorems 2–3.
- Appendix: full proofs.
- Don't dress Theorem 1 up. Theorems 2 and 3 have real downstream consequences (what concat-
  style training measured; the append-only cache), so they earn proofs.

**[OUT OF SCOPE → future work]**
- Section/anchor-scoped grants (`Title#Section`), an efficiency refinement.
- "Declared"/ambient edges (same directory, category). These are A2-safe if given by metadata
  at doc start, but domain-specific and edge-case-prone (e.g. a live doc reading itself via a
  directory rule).

### F1. Factorization
- Under the DAG gate, the `cdl` pack loss is the exact NLL of a valid autoregressive
  distribution. It is a Bayesian-network-like factorization over documents: each document is
  conditioned on its parents, and only after the point where it cites them.
- `doc_causal` is a product of marginals. `concat` is the joint distribution of each connected
  component.
- Bayes-optimal NLL ordering (conditioning can't increase entropy):
  concat ≤ concat_link ≤ cdl ≤ doc_causal. Combined with F0, this explains *why* "cdl must beat
  concat on NLL" was never the right test. The controls condition on oracle future information.
  **The pitch has to be structural, not NLL.**
- The DAG gate and targets-first ordering stop looking like hacks. They are what make the
  factorization valid.
- [OPEN] Cycles. The gate currently drops a link to a later document. Principled options:
  - condense strongly connected components;
  - partial grants (read the prefix of B written before B links back);
  - accept the drop and state the implied distribution.

### F2. Packing marginalizes over which parents are present
- The real training objective is E_{S ⊆ Pa(A)} [ −log p(x_A | x_<, x_S) ], where the
  distribution over subsets S is induced by the traversal, co-packing, the grant cap and edge
  dropout.
- The model sees which grants exist, so the optimum is the Bayes predictor for each subset.
  That gives **robustness to missing or unresolved targets by construction**, which inference
  needs (unresolved links, eviction, out-of-corpus refs).
- It also reframes the dropped grants, the cap and edge dropout as controlling the subset
  distribution. That's the theoretical home of the old density result.

### F3. Cache exactness and invalidation
- **Shift-invariance lemma:** RoPE is relative. A document whose attention stays inside itself
  (or a bundle whose internal layout is fixed) has hidden states that are independent of
  absolute placement. Its V is unchanged and its K differs only by a rotation. So you cache keys
  before rotation, or rotate by the insertion offset when splicing (elementwise). **Caching is
  exact under the current global-RoPE training. No positional redesign needed.**
- **Exactness vs training:** see §4.2. Splicing a single cached doc that already carries its
  ancestors' information matches a configuration that training produces.
- **Invalidation:** an edit to C invalidates every cache entry whose computation consumed C,
  i.e. the reverse-reachable set of C up to the composition depth (§4.3).

### F4. Cost model
- Fetching k documents of length m: **0 prefill FLOPs** for the fetched content, vs ≈ 2·N·k·m
  for RAG or long-context prefill. What remains is reading the KV (memory traffic) and storing
  it.
- Storage per token ≈ 2 · L · d_kv · bytes. The current 24L/1024d model, bf16, no GQA, is
  ≈ 98 KB/token. Mitigations: GQA, MLA or latent KV, KV quantization, caching only the hot set.
- **Break-even reads/edit ratio:** derive the amortized cost per read as a function of hit
  rate, churn, document length and k. This is the formal version of "where it pays" (§4.5).

### F5. Placement and permutation invariance
- With within-document shift invariance, a target's cached state doesn't depend on placement.
- [OPEN] Whether the reader's output is invariant to the *order* of its targets depends on the
  position scheme across grants (§4.4). Under global RoPE it is not, because distance matters.
  State exactly what is and isn't invariant.

### F6. Attribution and edit locality
- Attention mass on grant blocks gives a per-document attribution of each prediction.
- Removing a document has an exact counterfactual at splice cost (drop its KV and re-run the
  reader only).
- When a document is edited, changes in output are confined to the documents that read it.

### F7. Corpus compilation as dynamic programming over the DAG **[DECIDED: headline]**
Worked through 2026-11-04/05.
- **Definition:** in topological order, S(v) = Φ_θ(x_v, {S(u) : u ∈ Pa(v)}), i.e. v's per-layer
  KV computed with grants into its parents' *cached* KV.
- **Lemma (DP = joint forward):** S(v) equals v's KV in one forward over v's whole ancestor
  closure in topological order. Proof: induction over topo order and layers, using A1+A3. So
  compiling the corpus is equivalent to running it as one giant pack, at local cost: one pass of
  projections/MLP per token (≈2N·|corpus|) plus attention Σ_v |v|·(|v| + Σ_Pa |u|).
- **Receptive field = L hops (corrects "unbounded").** Information from an ancestor h hops up
  enters A's residual stream no earlier than the end of layer h. **The stack is an L-round
  message-passing GNN over the corpus DAG: an L-hop receptive field at 1-hop context cost.**
  Deeper hops arrive in later layers (cf. "hopping too late", Biran et al. 2024).
  - User prior: useful depth is far below L (doubt anything useful 12+ hops back). L is an upper
    bound, not an expectation.
- **Invalidation is bounded by L:** an edit to C changes a descendant at distance h only in
  layers > h, and not at all if h ≥ L.
  - Layer-partial recompute (top L−h layers, needs residual checkpoints) → **[FOOTNOTE / future
    work]**. Published code may be suboptimal as long as it conveys the idea.
  - Approximate invalidation (stop at the depth where state change falls below a threshold) is
    an eventual-consistency option.
- **Diamonds disappear:** A→{B, C}→D. S(B) and S(C) each read the single S(D), exactly the joint
  forward.
- **Exactness needs reader-relative cross-grant positions.** One global layout can't give B and
  C both their canonical offset to D. **[DECIDED] NoPE across grants** (RoPE only within a doc)
  satisfies the lemma exactly and makes the layout question vanish (no anchoring, no reverse-
  order packing, no diamond approximation). Per-grant virtual offsets are the alternative. Cost:
  the reader tells targets apart only by content (identifier cards help).
- **Cycles [DECIDED]:** the DP needs a DAG. The principled answer is **time**: if each version
  links only to versions strictly earlier, the corpus is acyclic by construction. Cycles inside
  one git commit are an artifact of snapshotting; a perfect record of edits in the order they
  were written would have none. In the paper, present time-versioned TAGs as the principle and
  a static snapshot as the degenerate case, with a one-line deterministic cycle-breaking
  fallback (what the packer does today; optimal = min feedback arc set, NP-hard).
  - Time dimensions available per source: GitHub (commits), Wikipedia (full revision history
    dumps, TB-scale), arXiv (submission dates and versions; citations point mostly backward
    already), web (crawl timestamps, coarse). Legwork: easy for code, hard for general web.
  - SCC condensation / fixed-point iteration: mention only.
- **Train/inference gap:** compiled S assumes all parents are present recursively, but a depth-h
  chain is fully co-packed with probability ~ρ^h. Fixed by repo-step training (§5.2).

### F8. Summary for citers (reader-shaped messages) **[DECIDED]**
- ∇θ = ∇θ^self + ∇θ^reader, where the reader term is the path from ℓ_A through the computation
  of S(B). Weights are shared, so detaching S(B) doesn't stop A *learning to read*. It stops
  **B learning to encode for its readers**. That's why the stop-gradient ablation is clean.
- Optimum is amortized over citers: m*(B) = argmin ℓ(B) + E_{A∼citers(B)} ℓ(A | m(B)). A summary
  for citers that lives in the representation, not the text.
- **Key contrast (user loves this):** standard causal packing also has reader gradients, but
  from *random* following text, so the incentive is to encode things useful to arbitrary text
  (≈ nothing; plausibly the distraction/contamination effect). cdl aims those gradients at the
  **actual citers**.
- Literature hook: the "two-hop curse" (models fail to compose A→B, B→C latently without CoT).
  F8 is a training-time mechanism for composition along declared edges.
- **Synthetic test [DECIDED: first item of the eventual empirical plan; not run yet]:**
  - Setup: C holds a planted key→value fact. B links C, and B's own text never needs the value.
    A links B (not C) and must produce C's value.
  - Arms: cdl with reader gradients, cdl with stop-gradient on S(B), doc_causal-trained
    evaluated with grants, upper bound where A also links C.
  - Sweeps: chain depth h = 1…L+2; linear probes on B's states.
  - Predictions: reader-grad ≫ stop-grad; accuracy decays with h with a cliff at h ≥ L; probe
    signal concentrated on B's positions after its link.

## 4. Inference / cache design

### 4.1 Per-document caching (the basic thing)
- Every corpus document's KV is computed once per model checkpoint. It is per *model* (KV is a
  function of the weights), but shared across every user, session and request of that model.
- Prefix caching only helps identical prefixes. This cache is **composable**: any set of
  documents can be spliced in any order the position scheme allows.

### 4.2 "A single cached doc carries its ancestors" (the dope realization)
- If B's KV was computed with C in context (B granted into C), B's hidden states from layer 1 up
  already contain whatever B read from C. A reader A that splices in **only B's KV** gets C's
  information indirectly, without paying context for C.
- **This is train-consistent.** Training has exactly this configuration: B and C co-packed, B
  links C, A links B but not C. A cannot attend C (grants aren't transitive) but reads B's
  C-informed states. Not an extrapolation.
- Caveats:
  - Lossy. A only gets what B chose to pull from C, at the positions after B's link.
  - Information from C reaches A one layer "late" (it has to pass through B's attention first).
  - The *depth* of composition at inference (F7: unbounded) exceeds what training ever saw
    (bounded by what fits in a 32k pack). That's an out-of-distribution axis. See §5.2 for the
    training fix.
- [OPEN] Is the indirect information actually useful downstream, vs splicing the full
  k-closure? It's an empirical question, but the cost difference is large: O(1 document) vs
  O(closure) context per hop.

### 4.3 Bundles, depth k, invalidation
- The user's original serving design: cache each doc on its own, then with each doc it links to,
  and so on up to k levels (guess: k ≤ 4 is the useful range). Context runs out beyond a few
  hops, so the inference loop is depth-bounded anyway.
- With F7-style compilation the bundles may be unnecessary: one entry per doc that carries its
  ancestors. Both are options. Bundles give direct attention to deep ancestors; F7 gives depth
  at no extra context cost.
- Invalidation: an edit to C invalidates every entry that consumed C. Under bundles that's
  entries reaching C within k hops. Under F7 it's all descendants (unbounded).
  - Possible mitigation: **eventual consistency**, i.e. tolerate stale descendants and recompute
    lazily or by priority. This mirrors a lagging cache (§5.2) and Scion's eager vs on-demand
    derived data.
- **Diamonds [OPEN, real theory question]:** B and C both link D. B's bundle places D at one
  offset and C's at another, so composing them for A duplicates D or makes its offsets
  inconsistent. Options: dedup with a canonical layout, accept the approximation, bound how
  bundles may overlap. Under F7 (no bundles) the problem mostly disappears: A splices B and C,
  each already D-informed.
- Generated auxiliary documents (recursive "generate the missing doc" fallback) are specific to
  one request, so they can't be cached across requests. They can be cached incrementally within
  a request. The interesting depth question is here: when does recursive generation stop, and
  how deep does it help?

### 4.4 Positions
- Current: global RoPE over the packed index, no per-document reset. Targets-first ordering
  keeps targets before linkers. **The user's prior: this may well be the final best option.**
- F3 makes it cache-compatible (rotate when splicing).
- What remains open:
  1. **Linker-to-target distance distribution.** Training: variable, determined by the packing,
     with other docs in between. Inference: we choose where to splice. Match the training
     distribution, or pick the most favorable placement (immediately before the linker)? This is
     a train/inference matching question, not whether the method works. Relates to the old
     packing-distance diagnostic TODO.
  2. **Total context.** Composed caches still have to fit the trained position range, so the
     window limit applies to cached context too. (F7 helps: fewer docs per reader.)
- **Leading option [LEANING]:** keep RoPE, anchor the reader at 0, and grow fetched docs leftward
  (F0 positional leak). Reader KV becomes append-only. "Take the slight hit" on shared-target
  approximation in training.
- Creative alternatives to brainstorm and narrow down (maybe test, maybe not):
  - RoPE within the primary doc, plus a learned per-slot / per-distance / "uniqueness"
    embedding for aux docs. To keep caches composable, apply it at attention time (a key bias
    or query-side term), not to the cached hidden states.
  - **RoPE on an orthogonal axis** (M-RoPE-style, cf. Qwen2-VL's multi-axis rotary): split head
    dims. One subspace rotates by doc-local token position, the other by a document-level
    coordinate (link order, hop depth). For grant blocks, the kernel uses only the doc-axis
    rotation (store pre-rotation K, rotate per block type; BIM already dispatches by block type).
  - NoPE across grants (content-addressed cross-doc reads). **User is enthusiastic about this one.**
  - ALiBi-style per-document bias.
- Alternatives if global RoPE doesn't hold up: doc-local RoPE + NoPE across grants (kernel-
  friendly, since BIM already distinguishes grant blocks from diagonal blocks); per-grant fixed
  virtual offset (rotate Q per grant block). **[OPEN / possible deep dive or future work.]**

### 4.5 Where caching pays
- Not general Wikipedia-scale web knowledge: the hit rate is too sparse for the storage cost
  (maybe cache only the Zipfian head).
- Yes: a **company monorepo**, and an **agent working in a framework with a large, fast-moving
  documentation surface**. In both, reuse is high, churn is localized (most changes touch a few
  files), and the same files are read over and over.
- Formalize with F4's break-even ratio.

### 4.6 Scion synergy (Anthropic's in-house VCS, blog post 2026-10-13; details from secondary coverage)
- Scion's "derived data": every non-stored value is computed recursively from the parent
  changesets' values through one generic mechanism, eagerly or on demand. Content-addressed blob
  pointers.
- The per-doc KV cache is a derived-data type:
  - **key:** (model hash, file blob hash, hashes of what it consumed);
  - **behavior:** Merkle-style reuse across changesets, so a merge recomputes only entries whose
    consumed set changed;
  - **policy:** eager for hot files, lazy for the tail.
- Motivating story for the intro: 30k merges a day, mostly agent-authored. Agents repeatedly
  read slowly-changing files, which is the ideal regime. Present it as an *illustrative fit*,
  not a claim about Scion internals.

### 4.7 Architecture: KV-efficient attention is a requirement, not an optimization **[DECIDED direction]**
- Storage makes or breaks the lead claim. Today's model is ~98 KB/token. DeepSeek V4.1-Flash
  reportedly reaches ~890 B/token (from secondary coverage; check the tech report "Pushing the
  Limits of KV Cache Compression"). That's ~100× smaller. A 1B-token monorepo goes from ~98 TB
  to ~0.9 TB.
- Mechanism-by-mechanism compatibility (independent of the exact V4 details):
  - **Latent / low-rank KV (MLA, V2 onward):** compatible. Cache the latent. MLA's decoupled
    RoPE key is a small shared component, so re-rotation when splicing (F3) touches only those
    dims. The latent itself is position-free. **Strong synergy.**
  - **Token-wise KV compression (merging adjacent tokens' KV, ratio ~2):** compression groups
    must not straddle document boundaries, so align groups to doc spans and pad. Within the
    reader, the link gate becomes granular to the group size. Targets are fully in the past,
    so they're unaffected. Synergy with F8: the compression is *trained* for being read by
    linkers.
  - **Learned sparse selection (lightning indexer / top-k, CSA):** the mask defines the
    candidate set and the indexer picks top-k within the granted keys. Reading a big spliced
    target becomes sublinear in its size, which shrinks F4's memory-traffic term. Indexer keys
    are per-doc and cacheable too. Selected indices are an explicit attribution signal (F6).
    The kernel has to compose BIM block sparsity with top-k.
  - **Cross-layer KV sharing (Reuse/Reindex modes, projected decoder KV):** cuts storage.
    Compatible as long as the shared KV is still a function of (doc, parents).
  - **Sliding-window layers:** purely local, restricted to the doc by the mask. Nothing to cache
    for targets.
  - **FP4/FP8 KV quantization:** orthogonal, compatible.
- Practical: V4-class models are hundreds of billions of params (MoE), far too big to
  continually pretrain on our budget. Options:
  - (a) adopt the *architecture* (MLA + compression + sparse selection) in our own
    from-scratch/small model;
  - (b) a small open MLA model, e.g. DeepSeek-V2-Lite or Moonlight-16B-A3B (verify what's
    current);
  - (c) convert a GQA model to MLA (TransMLA / MHA2MLA-style) and then continually pretrain.
- **[DECIDED 2026-11-02]** The preprint stays analytical: storage and recompute estimates use
  a DeepSeek V4-class model as the worked example; the architecture switch is deferred to the
  empirics phase.
- For the *preprint* the storage argument can be parametric: bytes/token is a model property,
  so cite MLA/V4 numbers for feasibility and give the break-even analysis (F4) as a function
  of it. Switching the code is needed only for empirics.

## 5. Training algorithm ideas

### 5.1 Raise ρ (realized grant density)
- ρ = the fraction of (link, target) pairs where the target is actually co-packed and granted.
  F2 and §6 make ρ the key knob: more reliance on context, and more robustness training.
- Levers:
  - longer sequences (sequence or context parallelism, ring attention over BIM kernels; also
    unblocks arXiv);
  - RoPE length extrapolation to fit more docs;
  - packing as an optimization problem (maximize realized grants per pack: set cover or
    knapsack instead of plain BFS);
  - the grant cap;
  - **cached-target training (§5.2), which could push ρ close to 1.**
- **[FUTURE EMPIRICS]** ρ is probably the best single lever to explore if compute returns.

### 5.2 Lagging-forward cache training (the user's idea)
- Periodically recompute the corpus-wide KV cache with current weights (forward only, stop
  gradient: a "lagging forward signal"). Train packs where targets are supplied **as cached KV,
  not as tokens**.
- Wins:
  1. Targets cost no pack context, so ρ → ~1 and packs hold mostly readers.
  2. Training now sees deep composition (cached targets that themselves consumed cached
     ancestors). That closes the F7 depth train/inference gap.
  3. Training and inference become the same computation (cached splice). Strong for the
     "train = inference mirror" story.
- Costs and risks:
  - Staleness. Cached KV comes from old weights, a bias like REALM's stale index or MoCo's
    momentum encoder. The lag interval is a knob.
  - No gradient flows into targets' cached KV, but targets are trained when they appear as
    readers themselves.
  - Storage for the training corpus KV is TB-scale. Restrict to frequently-linked docs.
- Prior art to cite and position against: REALM (async index refresh), Memorizing Transformers
  (stale kNN memory), TRIME, MoCo.
- **[OPEN]** Mix ratio of token-targets vs cached-targets. A mixture is *required* by F8
  (cached targets carry no gradient), and probably also to stay good at reading fresh docs.
- **Refresh spectrum:**
  - (a) **Live write-back.** Whenever a doc appears in a micro-batch, store the KV that the
    forward just produced (detached). Zero extra FLOPs. Staleness ≈ the doc's revisit interval
    (~1 epoch).
  - (b) **Periodic full sweep** (e.g. at eval cadence). Costs ≈ ⅓ of an epoch's training FLOPs
    per refresh and bounds staleness by the interval.
  - (c) **DINO-style.** Recompute the targets fresh each step with an EMA "teacher" copy, no
    grad. No staleness in the data, only in the teacher weights. Saves the pack's context but
    not compute.
  - Likely best: a hybrid. Train with token-targets only during early training (weights move
    fast and a random-init cache is garbage), then switch on (a), with occasional (b) refreshes.
- **Write bandwidth is a real cost for (a):** ~98 KB/token × 262k tokens/step ≈ 25 GB/step if
  every token is cached. Cache only docs with in-links, quantize, keep in host RAM or NVMe.
- **Traversal under caching:** traversal stops mattering for ρ (any doc can splice its cached
  parents). It still matters for (i) keeping with-gradient token-target pairs (F8) and
  (ii) freshness: a topological/BFS training order refreshes parents just before children read
  them. That cuts against i.i.d. batches. Note the merged source-forgetting bug, where
  correlated ordering caused real damage.
- Staleness analogies (one-line versions):
  - **REALM:** a precomputed doc-embedding index refreshed every few hundred steps by a stale
    encoder. Closest analog to (b).
  - **MoCo:** a queue of keys produced by an EMA ("momentum") encoder over past batches.
    Closest to (a).
  - **DINO:** an EMA teacher recomputes outputs fresh every step, so staleness is only in the
    weights. Analog of (c).

- **Repo-step training (user refinement, 2026-11-05) [LEANING: replaces lagging-forward where
  units are small]:** process a consistency unit (a repo; for web/wiki, a community or time
  slice) in time/topological order, with the current batch writing the cache the next one
  reads. **Apply the optimizer step only at the end of the whole unit**, so every cache entry
  in the unit comes from the same weights and there's no staleness inside the unit.
  - **Exact DAG backprop (makes it even better):** forward sweep in topo order storing S(v);
    backward sweep in *reverse* topo order. For each doc, recompute its forward (doc-granular
    activation checkpointing) and backprop its own loss plus the accumulated ∂L/∂S(v) from its
    readers, which yields ∂L/∂S(parents) to accumulate. That's the **exact gradient of the
    unit's loss, including F8 reader gradients**, with no stale or detached caches. Memory =
    caches + their grads for the unit. Compute ≈ 2 forwards + 1 backward (like activation
    checkpointing). Candidate algorithmic contribution.
  - Costs: the effective batch = unit size, which varies hugely (10k-token repos vs 100M).
    Bucket or pack small units into one step, chunk huge ones at time boundaries (accept
    staleness there). DDP: rank = unit, so each step mixes many repos (mitigates the
    correlated-gradient / source-forgetting risk).
- **Write cost with DeepSeek-style KV:** ~1–4 KB/token (bf16 latent in training) × 262k
  tokens/step ≈ 0.3–1 GB/step, about 10–40 ms over PCIe to host RAM, so negligible vs multi-second
  steps. Small units can keep the cache on-GPU. (Current MHA ~98 KB/token ≈ 25 GB/step:
  significant but overlappable.)

### 5.3 TAGs that change over time
- Train on the *history* of a graph, e.g. a GitHub repo's commit history: a sequence of graph
  versions that share most of their nodes.
- Connects to §4.6 (serving a versioned corpus) and §6 (knowledge that updates).
- [OPEN] Edges across versions (doc@t links doc@t−1? diffs as documents?). Likely future work.
- **Immutable-version redesign (user idea, 2026-11-03) [OPEN, potentially fundamental]:** make
  every file version (or commit) a new content-addressed node, linked to its predecessor.
  - An edit becomes an *append*. No cache is ever invalidated, since old entries stay valid for
    their version. Like a persistent data structure / Merkle DAG; pairs with Scion (§4.6).
  - Recompute doesn't vanish, it becomes **lazy and optional**. A@t still holds its old
    B-informed state. A fresh reader that links B' directly gets the latest B' (direct
    entitlement), while A's F8 "message" about B stays stale until someone materializes A@t+1.
    So: define consistency levels (latest-direct vs. fully re-derived). **The user flagged this
    stale-A problem as a key design driver for the eventual over-time model.**
  - Generating a new version = a document linking to its previous version, so editing becomes
    conditional generation along a version edge, and repo history becomes training data.
  - Would need new datasets (repo histories), kernels, and cache handling. Small KV
    (V4.1-class ~1 KB/token) makes storing every version plausible.

### 5.4 Document-level GNN / next-node prediction (user idea, 2026-11-05) **[FUTURE — possibly the unifying piece]**
- A document-level model running and training alongside the LLM that predicts the *high-level
  content of the next node* (which doc gets written or linked next), rather than only the next
  token given a prefix and aux docs. A competitor to next-concept-prediction lines (Large
  Concept Models, etc.) with a much larger concept unit: the whole document.
- Needs document embeddings. Candidate: the EOS-token state (a causal summary of the whole doc),
  given an embedding-style objective and kept separate from the general KV cache.
- Possible integration points:
  - **Tiered cache:** doc embedding (tiny) vs full KV (big). An embedding-level GNN carries
    cheap, coarse, *many-hop* information, while the KV path stays rich and L-bounded. That
    changes the depth calculus.
  - **Resolution:** the model's own doc embeddings as the index for links that don't exact-match
    (replacing the fuzzy title index).
  - **Planning:** predict which docs to fetch before writing. This is A2-safe if the prediction
    uses only x_{≤q}.
- Tension: it's an auxiliary objective, which the paper currently argues against (§7 analysis,
  "no aux loss"). Keep out of this paper unless it turns out simple; mention as v2 direction.

## 6. Grounding, alignment, pluralism

- **Vision (discussion section):** models' knowledge is currently a shadow on the cave wall, a
  parametric compression with no handle. If a model relies on its graph, humans can update the
  documents over time and improve a *static* model's outputs. Different groups could maintain
  different graphs reflecting different worldviews over one model. Not fundamentally different
  from what RAG allows, but built in more deeply and hopefully with a stronger effect.
- **The honest gap:** knowledge conflict. Models often trust their weights over their context.
  We can't promise heavy reliance.
- **The claim we *can* make (training incentive, not guaranteed outcome):** for a fact stated
  in B and used in A, `doc_causal` can only predict it from the weights. `cdl` can read it from
  B in a fraction ρ of training occurrences. So the gradient pressure to memorize linked facts
  drops by ~(1−ρ), and the read-from-context path gets explicit supervision with a clear
  attention route. Ties to F2 and §5.1. Higher ρ (§5.2) should strengthen the effect.
- Distinguish this from the old density result. That result showed *reading capability* is
  largely generic (a doc_causal model can exploit grants at eval). This claim is about
  *preference under conflict*, which is a different property.
- F3 makes swapping a whole graph cheap. F6 makes outputs attributable to a specific graph.
- **[FUTURE EMPIRICS]** Edit-following rate: perturb a fact in B and check whether A's
  predictions follow, cdl vs doc_causal. Cheap, needs no extra training, works on synthetic
  data, and tests the alignment vision directly.
- **[OUT OF SCOPE]** Counterfactual-document training to *force* deference (future work).
- [OPEN] The prompt-link fabrication issue (TODOS): a cited-but-missing doc gets *generated*
  and then attended as if real. In the grounding story, generated aux docs must be clearly
  marked as model-authored. A feature (explicit, auditable hallucination) or a liability?

## 7. Mask design space (presentation device)

- One parameterization of "who may read what". Each edge type gets a rule for:
  - query span: from the link onward / the whole doc / a section;
  - key span: the whole target / an anchor or section;
  - a symmetry flag and a transitivity flag.
- The four current masks are four settings. One-hop isolation, set semantics, anchor links and
  transitive grants are more settings, not special cases.
- F0 then reads as "within this space, these settings satisfy mask causality, and cdl is the
  maximal one".
- **[LEANING]** Use it if it makes the design space look complete. Drop it if it's clutter.

## 8. Prior art to position against (inference/caching side especially)

- **KV reuse for RAG:**
  - CacheBlend (recomputes a token subset to patch cross-attention);
  - EPIC (position-independent caching);
  - **Block-Attention (Ma 2025): fine-tunes toward block isolation. Closest prior art, must
    address directly.**
  - Prompt Cache, RAGCache;
  - PagedAttention and RadixAttention (prefix-only).
  - Our difference: exact by construction, because pretraining used the same isolation, plus
    link-directed fetching and DAG composition (F7).
- Precomputed corpus memory: EMAT, TOME / Mention Memory, LUMEN, ReadTwice, Memorizing
  Transformers.
- Stale index refresh: REALM (and MoCo) for §5.2.
- Packing: In-Context Pretraining, SPLiCe, zhao2024 (IntraDoc), cross-contamination-free
  packing.
- Graph pretraining: LinkBERT, CDLM, DRAGON, K-BERT.
- Active retrieval: FLARE, Self-RAG, IRCoT, Toolformer-style (fetch-by-generation without a
  tool API is our distinction).
- Knowledge conflict / context faithfulness literature for §6 (find canonical refs).
- [TODO] Search for 2025–2026 work on "corpus KV precompute", "KV cache for knowledge bases",
  "document-isolated pretraining for cache reuse". The space may have moved since the related
  work was written.

## 9. Current paper: keep / demote / drop

- **Keep (reframed):**
  - TAG definition and online link detectors (the ergonomics story: any corpus with a detector
    becomes a TAG, one implementation for train and inference);
  - traversal packing (as F2's sampling distribution);
  - masks (as F0/F1);
  - generation loop (as the inference algorithm, extended with §4).
- **Demote to appendix:** Triton kernels and the density-aware scheduler ("it's tractable"),
  dataset construction, recipe lineage.
- **Drop or rewrite:**
  - "matched-compute controls isolate the contribution" (F0/F1 show the controls aren't
    comparable generators);
  - the abstract's Δnll numbers;
  - the density dose-response as a contribution;
  - "no KV cache, full recompute" as a limitation (it's now the headline feature).
- Existing analysis paragraphs worth keeping in spirit: no auxiliary structure loss (trained
  integration), scale as an open question, RoPE discussion (rewrite around F3).

## 10. Future empirics (plan before running anything)

- **[LEANING: likely centerpiece if compute returns] Continual pretraining of an off-the-shelf
  open LLM under the cdl mask.** Reasons:
  - A competent generator makes fetch-by-citation and RAG comparisons meaningful. At 350M from
    scratch, generation is incoherent.
  - It costs much less compute than from-scratch pretraining.
  - Baselines on the same base model:
    - (a) unchanged + RAG (docs pasted in the prompt);
    - (b) continual pretraining under doc_causal;
    - (c) continual pretraining under cdl;
    - (d) the same as (c) with cached-KV inference.
  - Block-Attention is evidence that fine-tuning toward isolation works.
  - Engineering notes:
    - Kernels currently assume a fused Wqkv with no GQA. Modern open models use GQA and
      Dh=128, so the kernels need GQA support. The KERNELS.md Dh=128 numbers are favorable.
    - Data must be re-pretokenized with the base model's tokenizer. Link detection is
      tokenizer-agnostic (decoded strings + offsets), which helps.
    - The base model was trained with cross-document bleed under full causal packing, so the
      mask change is a distribution shift to watch for.
- Measurements to design:
  - cache-exactness check (spliced KV vs full forward, which should match up to float error);
  - edit-following rate (§6);
  - link-emission and resolution rates under teacher forcing (the old TODO);
  - indirect-ancestor utility (§4.2);
  - linker–target distance sensitivity (§4.4);
  - serving cost vs RAG at matched quality;
  - ρ sweeps (§5.1).
- Synthetic demos (OK if demonstrative): synthetic graphs with planted facts at known depth,
  to show F7 composition carrying information across hops, edit propagation, and attribution.
- Old results possibly reusable after bug audit: HotpotQA cross-doc Δnll, RepoBench ports,
  density experiments. Decide framing first.

## 11. Open questions (seeds for branching conversations)

1. Exact family definition for F0 so "cdl is maximal" is true and non-trivial.
2. Cycles: SCC condensation vs partial grants vs drop (F1).
3. Diamonds in bundle composition (§4.3), and whether F7 makes bundles unnecessary.
4. Positions: is global RoPE + rotation when splicing final? What splice placement at inference?
5. How useful is indirect ancestor information vs direct attention (§4.2)?
6. Lagging-forward training: lag interval, cached/token target mix, storage budget (§5.2).
7. Eventual-consistency serving: how stale can descendants be? Is there a bound on the error?
8. Generated aux docs: stopping rule, marking, cacheability, the prompt-fabrication behavior.
9. Code link *resolution* on multi-repo corpora (TODOS: detection works, resolution doesn't).
   Needed for the inference algorithm to be "complete".
10. Constrained decoding of identifiers (trie over corpus ids) so generated links resolve.
    Part of a complete fetch-by-citation algorithm?
11. Time-varying TAGs (§5.3): in scope as a discussion item, or future work?
12. Storage economics: is GQA/MLA/quantization part of the design or just a footnote?
13. Is the §7 mask-space formalism worth the page?
14. What does the paper's *figure 1* look like? (Probably: compile once in topological order,
    splice at read time, recompute only the edit's reverse-reachable set.)
