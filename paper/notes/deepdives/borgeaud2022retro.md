# borgeaud2022retro — Improving Language Models by Retrieving from Trillions of Tokens (RETRO)

Borgeaud, Mensch, Hoffmann, Cai, Rutherford, Millican, et al. (DeepMind), ICML 2022 (PMLR 162:2206–2240),
arXiv:2112.04426 (v3 fixed reported numbers in Table 14).
Sources: arXiv abstract, ar5iv full text (equations/tables), our code briefs `generation_retrieval.md`,
`masks.md`, and `related_work_notes.md`. Numbers below are confirmed from the paper text unless flagged as inferred.

## What the paper actually does

RETRO ("Retrieval-Enhanced Transformer") is a decoder-only autoregressive LM augmented with a
**chunked cross-attention (CCA)** module that reads text chunks retrieved by *approximate kNN* from a
frozen datastore of up to trillions of tokens. Three components: (a) a **frozen BERT retriever**,
(b) a small trainable bidirectional **neighbor encoder**, and (c) the **CCA** layers interleaved into the
main decoder.

Concrete mechanics (confirmed):
- Sequence length **n = 2048**, split into **l = 32 chunks** of **m = 64 tokens** each.
- For each chunk the retriever pulls **k neighbors** (trained with **k = 2**). The retrieval key is
  `Bert(N)` = time-averaged frozen-BERT embedding; distance is L2, `d(C,N) = ‖Bert(C) − Bert(N)‖²₂`.
- Each retrieved value is `[N, F]`: the neighbor chunk **N** (64 tokens) **plus its continuation F**
  (the next 64 tokens in the source doc), so neighbor length **r = 128**. Retrieved set
  `E ∈ ℝ^(l×k×r×d′)`, encoded by the trainable encoder (d′ = 896, 2 layers, ~19M params).
- ANN over the datastore uses **SCaNN** (guo2020scann); neighbors are **precomputed offline** because
  BERT is frozen — "we can query our 2 trillion token database in 10 ms."
- **Autoregressivity is preserved by chunk offset**: a token in chunk `Cᵤ₊₁` may only cross-attend to
  the *encoded neighbors of the preceding chunk* `Cᵤ`. `Ret(C₁) = ∅` (first chunk retrieves nothing).
  The last token of `Cᵤ` is the first that can see `Eᵤ`; the first m−1 positions of an attending chunk
  fall back to identity. Cross-attention is `Ca(h,Y) = softmax(YKQᵀh)YV`; dependence on earlier chunks'
  neighbors propagates only through ordinary self-attention, avoiding quadratic cross-attention cost.
  CCA is inserted in one Retro-block every 3 blocks, starting at layer 6.

Scale (confirmed): four sizes, baseline→Retro params excluding embeddings — 132M→172M (+30%),
368M→425M (+15%), 1,309M→1,451M (+11%), 6,982M→**7,532M** (+8%, d=4096, 32 layers). Training-retrieval
DB = **600B tokens**; evaluation-retrieval DB up to **1.75T** (2T referenced for timing). Retrieval adds
only ~8% params at 7B because the encoder and CCA are small.

Key results (confirmed):
- **Pile**: Retro 7.5B beats its no-retrieval baseline on all test sets and beats Jurassic-1 (178B) on
  a majority; "comparable to GPT-3 and Jurassic-1 despite 25× fewer parameters." It did **not** beat the
  baseline on `dm_mathematics` and `ubuntu_irc` (reasoning-heavy, retrieval-resistant).
- **Wikitext-103 test perplexity**: their baseline 22.96; their kNN-LM 19.54; Retro (retrieving from
  Wikipedia) 18.97; Retro/C4 10.23; Retro/MassiveText-100% = 3.92 (**partly leakage-driven**).
- **LAMBADA**: Retro > baseline at all sizes (last-word accuracy, greedy).
- **RETRO-fitting**: freeze all pretrained weights, train **only CCA + neighbor encoder** (<10% of a 7B
  model's weights) on **6M sequences (~3% of pretraining)**; recovers most of the from-scratch benefit.
  With retrieval disabled, the original model's performance is *exactly* preserved.
- **Storage contrast**: kNN-LM datastore for Wikipedia ≈ 15 TB; Retro's = 215 GB (Wikipedia), 93 TB
  (MassiveText). RETRO stores tokens, not per-token hidden states.

Leakage analysis (confirmed, important for us): they filter train↔eval overlap with **13-gram MinHash
Jaccard** (drop train docs with ≥0.8 similarity to val/test), then re-score by chunk overlap
`r(C) = s/m` (longest common substring / chunk length) and report **leakage-filtered bits-per-byte**
`bpb(α)` over chunks with `r(C) ≤ α`. Finding: Retro *exploits* leakage more than the baseline (steeper
gain on high-overlap chunks) **but still beats the baseline at all leakage levels down to α = 12.5%**
(chunks sharing <8 contiguous tokens) — so gains are partly copying, partly genuine knowledge use.

## Methodology: theirs vs. ours

The shared thesis with TAGSeq2TAGSeq (TS2TS): *bring related external documents into the model's
computation and let attention read them, integrated into training rather than bolted on.* Everything
else diverges on the axis the brief names — **train-on-structure vs. retrieve-at-inference**, and
**what kind of "edge"** connects a document to its neighbor.

1. **Edge type — learned kNN similarity vs. deterministic graph edge.** RETRO's neighbor relation is a
   *frozen BERT embedding L2-nearest-neighbor* — a soft, content-similarity, approximate (SCaNN) edge
   with no ground-truth structure. TS2TS has **no retriever at all**: the "edge" is a real
   hyperlink/import/citation resolved by an exact match key. Per `generation_retrieval.md`, resolution
   is a 3-tier `_resolve_target` (exact raw id → detector-key `index_doc_span(node)` → fuzzy
   `HashNormTitleIndex`), and the detector match key is **the same key training uses** — an exact hashmap
   lookup, no approximation error, no embedding index (contrast the FAISS/SCaNN infra our lit review
   marks as unnecessary). RETRO's neighbors can be topically-similar-but-unrelated; ours are the actual
   graph target.

2. **Integration mechanism — chunked cross-attention into a separate encoder vs. in-sequence
   self-attention grant.** This is the sharpest architectural contrast. RETRO encodes neighbors with a
   *separate bidirectional encoder* and injects them through a *dedicated CCA module* in ~1/3 of blocks;
   neighbor tokens never enter the main decoder sequence and are never predicted. TS2TS instead
   **materializes the target document into the same packed sequence** and grants the linking positions
   *ordinary causal self-attention* to it via a sparse mask (`masks.md`: `cross_doc_link`,
   `M = (q≥k) & (same_doc OR in_grant)`, grant rectangle `[link_end_pos, A.end) × [B.start, B.end)`).
   No cross-attention module, no second encoder, no extra parameters — the "edge" *is* a masked entry in
   the single self-attention. `generation_retrieval.md` calls this "retrieval-by-insertion into the
   packed seq": the fetched doc gets a first-class `DocSpan`, layout prefix, and grants, masked exactly
   like a training neighbor.

3. **Granularity — fixed 64-token chunks vs. whole documents at node granularity.** RETRO's unit is a
   mechanical 64-token window with a 64-token continuation, decoupled from document/semantic boundaries.
   TS2TS's unit is a *document (graph node)*, head-truncated to `max_corpus_doc_tokens` (keep
   abstract+intro) only when needed. Our grant is directional and per-link (`link_end_pos` onward), not
   a uniform per-chunk broadcast.

4. **Directionality and offset discipline.** Both are careful about not leaking the future.
   RETRO enforces it by the *previous-chunk* offset (chunk u+1 reads neighbors of chunk u). TS2TS
   enforces it by a **DAG gate + causal mask**: `masks.md` shows grants are skipped if
   `target_start ≥ link_end_pos` (backward links only), and causal `q≥k` is never relaxed. At inference
   the fetched target is *prepended* so `span.start < link_pos`, mirroring training where the target
   physically starts before the link — same geometry, opposite realization (offset-in-time vs.
   reorder-in-space).

5. **Train vs. inference symmetry.** RETRO is trained *with* retrieval from scratch (or RETRO-fitted),
   and retrieval is present at inference too — so it does train on the structure. But the retriever is
   **frozen** in both regimes and neighbors are **precomputed**, i.e. the *retrieval decision* is never
   learned or gradient-coupled. TS2TS's strongest claim (`generation_retrieval.md` §"Train/inference
   mirror") is a **single link-machinery implementation** (detector + match key + grant-from-link-position
   + DAG ordering) used *identically* in pretraining and generation; inference retrieval is literally the
   training cross-doc mask realized by inserting the linked doc. RETRO shares the "retrieval at both
   train and test" property but splits the mechanism across two modules; TS2TS unifies it in one mask.

6. **Frozen datastore vs. no datastore / live corpus + generation fallback.** RETRO's neighbors come
   from a static precomputed index. TS2TS fetches from a live corpus backend (`PretokShardedBackend`
   mmap) and, on a corpus miss, has **generation fallback** (`corpus_then_generate`): synthesize the
   linked doc from scratch and recurse — open-ended multi-doc synthesis with no analog in RETRO.

7. **KV / caching.** RETRO precomputes neighbor encodings once (frozen BERT) — its whole efficiency story.
   TS2TS deliberately does the opposite at inference: **no KV cache, full O(T²) recompute every token**
   (`generation_retrieval.md` §"KV CACHE = NONE"), because insertion/eviction shifts pure-RoPE absolute
   positions and any KV reuse would be incorrect. So RETRO is a caching/efficiency exemplar and TS2TS is
   a caching-refusenik; the comparison is a clean framing foil (our lit review already positions RETRO
   there under "Retrieval into attention (contrast)").

8. **Compute-control philosophy.** RETRO's efficiency claim (25× fewer params than GPT-3) conflates the
   retrieval benefit with parameter count. TS2TS explicitly *isolates* the linking inductive bias from raw
   FLOPs with matched-compute masks: `doc_concat_link` (whole-source-doc grant, strict FLOP-superset,
   same connectivity → isolates link-position gating) and `doc_concatenated` (union-find component
   attention, most FLOPs, no inference linking) per `masks.md`. RETRO offers no such compute-matched
   ablation of "is it the retrieval mechanism or just more compute/params."

## Predictions & open questions for our method

- **Leakage will dominate our headline number unless filtered — copy their protocol.** RETRO's
  Wikitext-103 3.92 ppl was "partly leakage." Their `bpb(α)` leakage-filtered curve is exactly the
  discipline we need: report our cross-vs-flat Δnll *as a function of source↔target n-gram overlap*, and
  show the effect survives at low overlap (their α=12.5% floor). Our masks make the target's verbatim
  tokens *directly attendable from the link position onward*, so our copying channel is even more direct
  than RETRO's cross-attention — expect a *steeper* leakage-exploitation slope than theirs. If we don't
  stratify by overlap, a reviewer will (correctly) attribute the win to copying. This reinforces the
  MinHash/n-gram dedup already in our eval methodology (`related_work_notes.md` §contamination).

- **Where the effect should be strong: knowledge-intensive, multi-hop, retrieval-answerable content;
  weak: reasoning/math.** RETRO helped broadly on the Pile but *failed* on `dm_mathematics` and
  `ubuntu_irc`. Predict TS2TS's cross-doc grant helps most on HotpotQA-style multi-hop, Wikipedia
  fact-completion, and cross-file code where the answer lives in the linked node, and helps least on
  single-document reasoning (our HellaSwag/ARC/PIQA single-doc controls should be flat — good, that's
  their purpose). Math/reasoning is a likely null or slight-negative regime for us too.

- **Small models benefit as much or more.** RETRO's gains held from 172M to 7.5B and RETRO-fitting
  recovered most of it. This predicts our linking bias should be measurable at *small scale* (good for a
  compute-limited paper) and should not require billions of params to manifest — the structural edge
  substitutes for parametric memory, just as RETRO's retrieval does.

- **The "distant offset" question.** RETRO chose the *previous-chunk* offset specifically so a chunk's
  read of its neighbor is at a controlled, small relative distance. TS2TS's `masks.md` flags that
  **RoPE is not reset per doc** — the linking position reads its target at a relative offset equal to the
  *packing distance*, which is traversal-order-dependent and arbitrary. RETRO's design implicitly warns
  that *where* the neighbor sits relative to the reader matters. Prediction: TS2TS may show
  sensitivity to packing distance / target position; worth an ablation varying target placement (and it
  connects to our lost-in-the-middle citations). Their fixed-offset cleanliness is a design point we
  traded away for train/inference symmetry.

- **Open question our design may resolve for them.** RETRO's retrieval decision is frozen and unlearned;
  they leave open whether *end-to-end* learned/structured retrieval helps. TS2TS sidesteps the learned
  retriever entirely with a *ground-truth graph edge* — effectively an oracle retriever. If TS2TS's
  trained cross-doc attention beats a RETRO-style similarity-kNN comparator at equal compute, that's
  evidence the *structure of the edge* (not just "some retrieval") is what matters — a question RETRO
  cannot answer because it never had ground-truth edges.

- **Open question they resolve for us: retrieval helps *pretraining*, not just fine-tuning.** RETRO
  trained retrieval from scratch and showed monotone LM gains — direct support that baking cross-doc
  reading into pretraining (not post-hoc RAG) is the right regime, which is our central bet.

## Gotchas

- **Leakage is the #1 eval artifact.** See above; RETRO's own headline Wikitext number is partly
  contamination. Their v3 erratum ("fix incorrect reported numbers in Table 14") is a reminder that
  retrieval-eval bookkeeping is error-prone. Stratify by overlap; publish the filtered curve.
- **Retrieval-resistant subdomains produce regressions, not just null.** RETRO *lost* on math and IRC
  chat. Expect TS2TS's cross-doc mask to occasionally *hurt* on documents whose linked target is
  irrelevant/noisy — our union grants compose by OR with no weighting (`masks.md` reviewer-attackable
  #4), so a bad/uninformative target dilutes attention. Watch for per-domain regressions and don't
  average them away.
- **The efficiency claim is a trap if uncontrolled.** "25× fewer params than GPT-3" mixes retrieval
  benefit with the datastore's stored knowledge. Our compute-control masks exist precisely to avoid the
  symmetric mistake ("cross-doc wins because it sees more FLOPs/tokens"). Keep `doc_concat_link` /
  `doc_concatenated` front and center.
- **Neighbor continuation matters.** RETRO retrieves chunk N *and its continuation F*; ablating F hurt.
  Analog for us: head-truncating a fetched corpus doc to abstract+intro (`max_corpus_doc_tokens`) may
  drop the very span that answers the query. `generation_retrieval.md` already flags silent link drops
  when a corpus doc exceeds the cap and `max_corpus_doc_tokens` is unset (no-op) — a real failure mode.
- **Frozen precompute vs. our full recompute is an efficiency liability to preempt.** RETRO's 10ms/2T-token
  query and 215GB store are its selling points; our O(T²)-per-token, no-KV-cache inference
  (`generation_retrieval.md`) will look expensive by comparison. Frame it as the *cost of exact
  train/inference symmetry and position-correctness*, not an oversight — and cite RETRO as the "if you
  freeze and precompute, here's what you save" reference.
- **Autoregressive-integrity bugs.** RETRO's whole CCA offset exists to not leak the future. Our analog
  is the DAG gate + `link_end_pos` half-open containment (`masks.md` reviewer-attackable #3: off-by-one
  sensitive, all 11 detectors must emit exclusive `link_end_pos`). A single detector emitting the wrong
  boundary silently leaks or drops a grant — test every detector's boundary.

## Missed citations worth adding

The retrieval/memory cluster in `related_work_notes.md` is already very thorough and cites RETRO,
kNN-LM, REALM, DPR, FiD, Atlas, Memorizing Transformers, SCaNN, Contriever, RETRO++
(wang2023shallwepretrain), and RAVEN. From scanning RETRO's own references for works relevant to
*our* project and genuinely absent from `refs.bib` (verified by grep — please re-verify before adding):

- **gao2020pile** — Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling"
  (arXiv:2101.00027). Absent from refs.bib. Relevant only if we report Pile-style per-domain
  bits-per-byte (RETRO's headline eval and a natural LM-perplexity comparison surface for us). Low
  priority / tangential to the linking thesis, but the natural dataset cite if we adopt their eval.
- **rae2021gopher** — Rae et al., "Scaling Language Models: Methods, Analysis & Insights from Training
  Gopher" (arXiv:2112.11446). Absent. This is RETRO's base-model / MassiveText lineage and its main
  scale baseline. Only worth adding if we discuss the MassiveText corpus or use Gopher as a scale
  reference; otherwise skip — it is a scale paper, not a method relevant to graph structure.

Honestly, the two above are the only plausibly-relevant absences, and both are weak (eval-dataset /
scale-baseline, not structural-method) — RETRO's method-relevant references (retrievers, memory
transformers, ANN infra) are already covered. I do **not** recommend padding with Jurassic-1
(lieber2021) or generic ANN/retriever cites; they add nothing our lit review lacks.

---
Confirmation: Deep-dive written to `paper/notes/deepdives/borgeaud2022retro.md`, grounded in the RETRO
paper (arXiv:2112.04426 abstract + ar5iv full text) and verified against our code briefs
`generation_retrieval.md` / `masks.md` and `related_work_notes.md`.
