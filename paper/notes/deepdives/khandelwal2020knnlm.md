## khandelwal2020knnlm — Generalization through Memorization: Nearest Neighbor Language Models

Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis. ICLR 2020. arXiv:1911.00172.
Assigned as the cache-as-datastore exemplar and the sharpest contrast to our no-KV-cache,
full-recompute, train-on-structure inference.

### What the paper actually does

**Method.** Take a *frozen* pretrained autoregressive LM (Baevski & Auli's adaptive-input
Transformer, 247M params, 16 layers, 1024-dim). For every training-set token position,
run one forward pass and record a (key, value) pair: the **key** is the ~1024-dim context
representation feeding the *final* feedforward layer (the input to the last FFN, after
self-attention + layernorm — chosen by ablation in their Table 5, 16.06 dev ppl, beating
other layer taps), and the **value** is the actual next token at that position. This is the
"datastore" — one entry per training token, built with a single forward pass over the
corpus and **no training whatsoever**.

At inference, the same LM produces a query context vector `f(x)`. They find the *k* nearest
keys under **squared L2 distance**, form a kNN next-token distribution by softmax over
*negative* distances with mass aggregated per vocab item —
`p_kNN(y|x) ∝ Σ_i 1[y=v_i] exp(−d(k_i, f(x)))` — and linearly interpolate with the LM's own
softmax: `p = λ·p_kNN + (1−λ)·p_LM`. On WikiText-103, **λ=0.25**, **k=1024** (even k=8 already
sets SOTA); for domain adaptation **λ=0.65**. Retrieval is FAISS with product quantization
(4096 centroids from 1M sampled keys, 64-byte codes, 32 probes); full-precision L2 recompute
of the retrieved candidates lifts ppl from 16.5 → 16.06 vs. pure quantized search.

**Concrete results (the ones that matter):**
- *WikiText-103, same-data* (Table 1): base LM dev/test **17.96 / 18.65**; +kNN-LM **16.06 / 16.12**;
  +kNN-LM +continuous cache **15.81 / 15.79** (the headline 15.79 *includes* Grave's continuous
  cache; kNN-LM alone is 16.12). The datastore holds **~103M entries** (one per training token).
  Building it: ~2h on one CPU; validation ~25 min.
- *Retrieval beats training on the same tokens* (Table 3, WIKI-3B/WIKI-100M): a model trained on
  the full 3B tokens gets test **15.17**; a model trained on only 100M tokens gets **19.59**; that
  same 100M-trained model with a datastore built over the 3B tokens gets **13.73** — beating the
  model that was *trained* on all 3B. A ~1.6B-token datastore already passes the fully-trained model.
- *Domain adaptation with no training* (Table 4): a WIKI-3B model on Books scores test **34.84**;
  add a Books datastore and it drops to **20.47** (~14-point gain), purely by swapping the store.
  In-domain Books (Table 2): 11.89 → 10.89.

**Thesis:** interpolating a nonparametric memory helps *even when the datastore is the exact data
the LM was trained on*. The authors argue learning a good similarity function over contexts is
easier than mapping a context to a next-token distribution — memorization improves generalization,
especially on the long tail (rare factual patterns).

### Methodology: theirs vs. ours

The single cleanest axis: **kNN-LM retrieves at inference into the output distribution; we train on
structure and retrieve at inference into the attention context.** Detail by detail:

- **What is stored / the "edge."** kNN-LM stores a dense **KV-like datastore** — (hidden-state key →
  next-token value) — and the "edge" is an *embedding-similarity* hop found by approximate NN search
  at inference. Ours stores nothing precomputed; the edge is a **deterministic graph link** resolved
  by exact identifier match (`document_corpus.py` 3-tier `_resolve_target`: raw-id, then the
  `index_doc_span` detector key — the *same key training uses* — then a fuzzy title index). No
  embeddings, no FAISS, no approximation error. Our brief's contrast to FAISS/HNSW infra
  (`related_work_notes.md` "Dense retrieval backbones") is exact here: target resolution is a hashmap
  lookup, not MIPS.

- **Where the retrieved thing enters the model.** kNN-LM never lets the neighbor touch the
  transformer — it only shifts the *final probability vector* via a scalar-weighted mixture. There is
  **no attention** over the retrieved context; the neighbor's own text is never re-encoded. Ours does
  the opposite: the fetched target document is materialized *into the packed token sequence*
  (`_handle_link` inserts it immediately before the linking doc, `_docs.insert(idx)`), and the
  `CrossDocLinkMaskCreator` grants the linking positions genuine self-attention into the target's
  tokens from the link position onward. Retrieval-by-insertion into attention vs. retrieval-by-logit-mixture.

- **Train-time vs. inference-only.** kNN-LM does **zero** retrieval training — the LM is frozen and the
  datastore is bolted on post hoc (this is why it's grouped with REPLUG/in-context-RALM at the
  inference-only pole in our notes). Ours applies the *identical* link machinery — detector, match key,
  grant geometry, DAG ordering — in **both** pretraining and inference (`generation_retrieval.md`
  "Train/inference mirror", the strongest claim, backed by shared code not analogy). The cross-doc
  attention edge is a *pretraining objective*, not an inference add-on.

- **KV cache.** This is the assigned contrast. kNN-LM is a **KV-cache-as-datastore**: it precomputes and
  freezes per-token KV-analogue vectors and reuses them across all queries — the datastore *is* a giant
  frozen cache read at every step. We do the **opposite extreme**: no KV cache at all. `build_sequence()`
  + `forward_inference` recompute the full packed sequence from scratch **every generated token**
  (grep for `kv_cache`/`past_key`/`use_cache` finds nothing); fetched docs' KV are recomputed each step.
  This is forced, not lazy: pure RoPE + insertion/eviction shift absolute positions, so a naive KV cache
  would be *incorrect* after any insert (this is the `hu2024epic`/EPIC position-shift problem in our
  notes). Cost is ~O(T²) per forward × O(T) steps. So on the memory axis kNN-LM and TS2TS are polar:
  kNN-LM caches *everything and trains nothing*; we cache *nothing and train the edge*.

- **Granularity.** kNN-LM's unit is a single token (context → next token). Ours is a whole **document/node**
  fetched as first-class content with its own `DocSpan`, layout prefix, and attention grants, masked
  exactly like a training neighbor.

**What we share.** Both are *semiparametric* — neither claims the parameters hold all the knowledge; both
reach outside the weights at inference for the long tail. Both keep retrieved information *verbatim*
(kNN-LM copies actual next-token identities; we copy actual document tokens) rather than compressing into
summary state, unlike the recurrent-memory family. And both let you change knowledge by swapping the
external store without retraining — kNN-LM swaps its datastore; we swap the corpus backend
(`PretokShardedBackend`) the link resolver reads from.

### Predictions & open questions for our method

- **"Retrieval beats training on the same tokens" (Table 3) is the single most encouraging result for us.**
  A 100M-trained model + 3B datastore (13.73) beat a 3B-trained model (15.17). Our analogue: a model
  pretrained with the cross-doc attention *and then given corpus read-access at inference* should beat a
  compute-matched concat model, because the linked document supplies exactly the long-tail token our
  parameters didn't memorize. Because we go *further* than logit interpolation — the target's tokens flow
  through attention and condition all downstream computation, not just the final softmax — we should expect
  a *larger* gain than kNN-LM's on the fraction of predictions that genuinely depend on the linked content.

- **Where the effect should be strong: the long tail / rare factual tokens.** kNN-LM's gains concentrate on
  rare patterns and named-entity/factual continuations. This predicts our cross-doc edge should help most on
  citation/hyperlink targets that carry facts (names, numbers, definitions) rather than on generic fluent
  text — argues for slicing our Δnll by token rarity / entity-hood, not just aggregate perplexity. It also
  predicts the effect is strong exactly on our multi-hop QA constructions (HotpotQA supporting docs) and weak
  on the single-document controls (HellaSwag/ARC/PIQA), which is already our control design.

- **Datastore/context-size scaling.** kNN-LM improves monotonically with datastore size and their λ rises with
  store quality. Analogue: more/better link coverage and longer allowed corpus-doc budgets
  (`max_corpus_doc_tokens`) should help monotonically — until eviction (`max_auxiliary_documents`,
  `max_context_length`) forces drops. Predict a saturation/turnover once the packed window is dominated by
  fetched material, i.e. an optimum in how much of the target we head-truncate in.

- **A question their design leaves open that ours resolves.** kNN-LM cannot make the retrieved context
  *interact* — a neighbor never influences another neighbor, and the model can't reason *across* two fetched
  items (it's a per-token logit mixture). Our recursive, bounded multi-hop fetch (`max_link_depth`, a fetched
  doc's own links re-fired at depth+1) is precisely cross-neighbor interaction through attention. If our method
  shows multi-hop gains where kNN-LM plateaus at one hop, that is a clean "attention edge > logit mixture"
  story. Conversely, kNN-LM raises a question for *us*: they show interpolating helps *even retrieving from the
  identical training data* — do we get lift from letting a document attend to a *co-training-set* neighbor it
  already saw, or only from genuinely-unseen inference-time targets? Their same-data result predicts yes.

- **The λ knob has no analogue for us — and that's a risk as well as a virtue.** kNN-LM tunes a single scalar to
  down-weight retrieval when it's noisy. We have no soft gate: a resolved link *always* injects its target into
  attention. If resolution is wrong (fuzzy-title false positive) or the target is off-topic, we have no λ→0
  escape hatch; the model must learn to ignore a bad grant. Predict failure modes concentrated where the
  detector/resolver misfires (see Gotchas).

### Gotchas

- **The headline number is contaminated by a second mechanism.** 15.79 = kNN-LM **+ continuous cache**; kNN-LM
  alone is 16.12. If we ever cite kNN-LM's improvement, use the 18.65→16.12 delta (2.5 pts from kNN alone), not
  2.86. Their own framing here is a caution for our paper: don't let a stacked add-on (e.g. our generation-fallback
  synthesis) get credit that belongs to the core edge — matched-compute isolation is exactly why our concat
  controls exist.

- **Same-data lift can masquerade as generalization but is partly re-exposure.** Their 18.65→16.12 gain retrieving
  from the *training set* shows the store re-surfaces training tokens the parametric model under-weighted. For us,
  if evaluation documents' linked targets overlap the training corpus, a chunk of our "cross-doc reasoning" gain
  may be verbatim re-exposure of memorized text rather than genuine link-following. Our SHA1/n-gram dedup and the
  HotpotQA-2017 leakage caveat (notes §6, `yang2023rephrased`) are the right guardrails; report Δnll on
  contamination-controlled splits (PhantomWiki-style) too.

- **kNN-LM does *not* help open-ended generation** (their line already tracked in our notes:
  `wang2023knnlmgeneration` shows it fails there; `drozdov2022neighbors`, `xu2023whyknnlm` analyze when/why it works).
  Since our generation-fallback path (`corpus_then_generate`) *is* open-ended multi-doc synthesis, this is a direct
  warning: the retrieval-helps-perplexity result may not transfer to free generation quality. Evaluate the
  link-fetch benefit under teacher-forced Δnll (our primary metric) before claiming generation wins.

- **Representation-tap sensitivity.** Their Table 5 shows the *choice of which hidden vector* is the key materially
  changes results (FFN-input-after-layernorm won). We don't build embedding keys, so we dodge this — but the
  analogue bites at the *grant geometry*: which positions get read-access and from where (link_end_pos-onward vs.
  whole-doc grant) is our version of "which representation," and the flex-vs-triton grant kernels are not yet
  numerically diffed (flagged Reviewer-attackable in the brief). Treat grant geometry as a tuned hyperparameter,
  not a free choice.

- **Cost/latency framing.** kNN-LM is *cheap to build* (one forward pass, no training) but adds a FAISS lookup per
  token and a 103M-entry store. Ours is the inverse: no store, but full O(T²) recompute per token and no KV reuse
  (`generation_retrieval.md` "KV CACHE = NONE"). Reviewers who know kNN-LM will ask why we forgo caching entirely;
  the honest answer — RoPE position shifts on insert make paged/prefix KV *incorrect*, and train/inference symmetry
  demands the full mask — should be stated up front, with kNN-LM's cheap-store/expensive-lookup as the foil.

### Missed citations worth adding

Scanned kNN-LM's own reference list against `refs.bib`. Genuinely missing and relevant to us
(**verify ids/venues before adding — I did not confirm these are absent by exhaustive read, only by targeted grep**):

- **grave2017continuouscache** — Grave, Joulin, Usunier, "Improving Neural Language Models with a Continuous Cache,"
  ICLR 2017 (arXiv:1612.04426, *verify*). This is the exact "continuous cache" that kNN-LM stacks to reach 15.79,
  and the direct precursor of the cache-as-datastore idea — a nonparametric recency memory over recent hidden states.
  Not in refs (grep for "continuous cache" = 0). Strong add for our KV-cache-as-datastore framing paragraph.

- **grave2017unboundedcache** — Grave, Cissé, Joulin, "Unbounded Cache Model for Online Language Modeling with Open
  Vocabulary," NeurIPS 2017 (arXiv:1711.02604, *verify*). The kNN-over-cached-states online LM that is the most
  direct methodological ancestor of kNN-LM itself — retrieves neighbors from an unbounded store of past hidden
  states. Missing; belongs beside `khandelwal2020knnlm` in the kNN-datastore cluster.

- **kaiser2017rareevents** — Kaiser, Nachum, Roy, Bengio, "Learning to Remember Rare Events," ICLR 2017
  (arXiv:1703.03129, *verify*). A *differentiable* kNN memory module trained end-to-end, targeting the long tail —
  the "train the memory" counterpoint to kNN-LM's frozen store, adjacent to our train-the-edge stance. Kaiser is in
  refs only as a Transformer/Reformer coauthor, not this paper. Worth adding to the memory-augmented cluster.

- **sprechmann2018mbpa** — Sprechmann et al., "Memory-based Parameter Adaptation," ICLR 2018 (arXiv:1802.10542,
  *verify*). Uses a nonparametric episodic memory to locally adapt parameters at test time — another
  memorization-for-generalization mechanism cited by kNN-LM; a contrast point for "adapt weights locally" vs. our
  "inject tokens into attention." Optional (weaker fit than the two Grave papers).

(Not flagged: Baevski & Auli adaptive-input is kNN-LM's base LM but is architecture-baseline, not relevant to our
thesis; Merity's pointer-sentinel / WikiText is already `merity2017pointer`.)

---
Confirmed against arXiv full text (ar5iv) for all result numbers and mechanism details; codebase claims verified against `generation_retrieval.md` and `related_work_notes.md`; missing-citation arXiv ids are marked *verify* and not asserted as final.
