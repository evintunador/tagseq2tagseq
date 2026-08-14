## dejong2022tome — Mention Memory: Incorporating Textual Knowledge into Transformers through Entity Mention Attention (TOME)

de Jong, Zemlyanskiy, FitzGerald, Sha, Cohen. ICLR 2022. arXiv:2110.06176 (v2, Apr 2022).
Verified against the arXiv PDF (full text, incl. appendices A–D). Numbers below are quoted
from the paper unless marked "infer".

### What the paper actually does

**Core idea.** Precompute a corpus-wide, frozen "Mention Memory": a table of dense vector
representations of *every entity mention* in English Wikipedia (~150M linked mentions).
A reader Transformer (TOME) then attends into this table through internal "memory attention"
layers, so factual knowledge lives in a semi-parametric external store rather than only in
weights. This is exactly the "precomputed per-corpus memory" framing our author recalled.

**Two-stage training (memory is frozen for the reader).** Jointly backpropagating through a
150M-entry memory is infeasible, so:
1. *Mention Encoder pretraining.* A BERT-base encoder with two final SpanEncodingLayers
   produces, for each mention span (s,e), a **key** encoding (dim dK=128) and a **value**
   encoding (dim dV=512) as a learned linear projection W[H_s; H_e] of the mention's start/end
   token reps (following Févry et al. / Baldini Soares et al.; mentions are wrapped in special
   [Estart]/[Eend] marker tokens). It is trained end-to-end with BATCH-TOME, a version that
   attends only to an *in-batch* memory built from related high-entity-overlap Wikipedia
   passages (MARGE/ReadTwice-style two-pass reading). Objectives: aggressive entity-masked MLM
   + an in-batch **coreference resolution** loss (predict whether two masked mentions are the
   same entity from encoding similarity). Trained 1M steps, 128 TPUs, batch 4096, T=128.
2. *Build the memory.* Run the frozen Mention Encoder over all of Wikipedia → MemKey ∈ R^{N×128},
   MemValue ∈ R^{N×512}, plus MemEnt ∈ R^N (Wikipedia entity IDs, used only for auxiliary
   losses, never as model input or retrieval supervision). N ≈ 150M.
3. *TOME pretraining.* The reader trains 500k more steps (from the BATCH-TOME init) over the
   *full* frozen memory (a 38M uniformly-sampled subset during pretraining for cost; full 150M
   at fine-tune/eval), with MLM + a **Mention-Memory entity-coreference / entity-prediction**
   loss: predict the answer entity as the aggregate attention mass over retrieved memories
   sharing that entity ID. A "disallowed same-passage retrieval" rule zeroes attention to
   memories originating from the current passage, to prevent trivial copying.

**Memory attention (the retrieve-and-attend primitive).** For each input mention m, compute a
query encoding via a SpanEncodingLayer; do approximate nearest-neighbor search (MIPS, dot
product) over MemKey to get TopMem(m); softmax-attend over those memory values; update the
mention's token rep: M_s = LayerNorm(H_s + W_U · Value(m)). **Top-K in practice (App. C.1.4):
top-2 per TPU shard, then a global top-128 over shards** for the TOMEBlocks (top-32/shard for
the entity-prediction layer). Memory is sharded across the 128 TPUs; overhead is small (memory =
2.2% of device RAM; ANNS = 22% of step time at BERT-base, dropping to 2% at T5-11B-encoder scale
— the memory cost does *not* grow with reader size).

**TOME-1 vs TOME-2.** TOME-1 = 4-layer InitialTransformerBlock + one TOMEBlock (8 layers).
TOME-2 = same init block + two TOMEBlocks (4 layers each) → two memory-attention rounds enable
**multi-hop** retrieval (retrieve, refine, retrieve again). Total layers held fixed; both ≈
BERT-base trainable params (~220M reported incl. Mention Encoder; the 150M-entry memory is *not*
counted as params, matching retrieve-and-read accounting).

**Results (all single BERT read, #Encoded=1).**
- Claim verification (accuracy): HoVer-test 72.8 (TOME-1) / 73.1 (TOME-2); FEVER-test 67.8/68.1;
  FM2-dev 67.7/68.4. Beats REALM (66.1/67.1/65.8, #Encoded=5) and Entities-as-Experts
  (66.6/63.6/63.5). Notably strong on HoVer (explicitly multi-hop). Does **not** use the gold
  evidence passages that most published results use — retrieval is guided only by task accuracy.
- QA as entity-linking (append a [MASK] question mention; answer = argmax aggregate attention
  mass over entity): TriviaQA-test 61.1/65.8; TQA entity-subset dev 60.3/64.8; CWQ-dev 44.9/47.7;
  EntityQuestions-dev 62.1/66.0. TOME-2 beats REALM on the harder multi-hop CWQ (47.7 vs 46.7)
  and rare-entity EQ (66.0 vs 59.0), and crushes Entities-as-Experts everywhere. FiD (440M,
  #Encoded=100, gold-supervised retrieval) is still higher on plain TriviaQA (77.1) — TOME
  trades raw accuracy for a single cheap read.
- **Memory-size scaling (Fig. 2):** claim-verification accuracy rises *smoothly* with fine-tune
  memory size (0→150M), with diminishing returns at the top (overlapping information across
  mentions). Model can use a *larger* memory at eval than it was pretrained on.
- **Zero-shot to unseen entities (Table 5):** pretrain+finetune with a memory that *excludes*
  the answer entities, then swap in the full memory at eval only — performance does not drop
  (TriviaQA 17.4→17.6, CWQ 16.4→16.7 on the held-out rare-entity subset). Knowledge is
  editable by editing the memory, no retraining.
- **Emergent retrieval:** with only MLM, BATCH-TOME/TOME already attend to same-entity memories
  55%/41% of the time — informative retrieval emerges without retrieval supervision.

### Methodology: theirs vs. ours

The shared DNA is real and worth stating plainly: **both systems put a corpus-external unit of
text into the attention computation of a reader, selected by a structural/semantic signal, and
attend to it inside a single forward pass** — not a GNN message-pass, not a kNN-LM logit
interpolation, not a contrastive relevance label. But the axis of divergence is sharp on almost
every design choice.

- **Train-on-structure vs. retrieve-at-inference.** TOME is fundamentally a *retrieve-at-
  inference* architecture: the memory is built once, frozen, and MIPS-queried at every step, in
  both pretraining and downstream use. Ours is *train-on-structure*: the graph edge is a hard,
  binary, direction-gated **attention mask** applied identically in pretraining and inference
  (`masks.md`; `generation_retrieval.md` §"Train/inference mirror"). TOME shares our
  train=inference symmetry (its memory attention runs the same in both phases) — but the thing
  that is symmetric differs: for TOME it is a *learned soft MIPS retrieval*; for us it is a
  *deterministic exact-lookup + mask grant* with no learned similarity anywhere.

- **What resolves the target.** TOME resolves relevance by *learned dot-product similarity* over
  128-d key encodings (top-2/shard → top-128), soft-attended. We resolve the target by an
  *exact identifier lookup* — the detector match key `index_doc_span(node)`, the SAME key
  training uses (`generation_retrieval.md`; `document_corpus.py:18-20`), with a 3-tier
  exact→detector-key→fuzzy cascade. No ANN index, no approximation error, no learned retriever.
  Where TOME needs ScaNN/FAISS and eats 7–22% ANNS overhead, we do a hashmap get. (Our
  `related_work_notes.md` R2 slice makes this exact point against the FAISS/ScaNN/HNSW infra.)

- **Granularity of the retrieved unit.** TOME's atomic unit is an **entity mention** — a single
  span, compressed to *one 512-d value vector*. It never re-reads the source text; it attends to
  a lossy dense summary. Ours is a **whole document (node)**, inserted as *verbatim tokens* into
  the packed 32k sequence and read at full resolution through the cross-doc grant
  (`generation_retrieval.md` §"Retrieval-BY-INSERTION"; `masks.md` grant rect
  [link_end_pos, A.end) × [B.start, B.end)). This is the same distinction our lit review draws
  for the whole cached-KV frozen-store family: TOME/EMAT/Memorizing-Transformers store
  *compressed dense vectors*; we keep *document-granularity verbatim tokens inside a trained
  attention span*.

- **Attention edge vs. cached-KV vs. entity table.** TOME's memory is a frozen *value/key table*
  the reader only *consumes* (gradient does not flow into it). Our cross-doc grant is a live
  attention edge computed fresh every forward — no cached KV at all
  (`generation_retrieval.md` §"KV CACHE = NONE"; full O(T²) recompute per step). TOME's whole
  reason for freezing is to make 150M-entry attention affordable; we pay full recompute to keep
  train/inference identical (RoPE positions shift on insertion, so KV reuse would be incorrect).
  So on the efficiency axis we are the *opposite* trade: TOME is cheap-per-step but lossy and
  learned; we are expensive-per-step but exact and verbatim.

- **Multi-hop.** TOME-2 stacks two memory-attention rounds to hop (retrieve → refine → retrieve);
  it cannot do beam-search multi-hop retrieve-and-read but it *can* chain within the forward.
  Ours does multi-hop by **recursive link-following with bounded depth** (`--max-link-depth`,
  default 2; `generation_retrieval.md` §"max_link_depth"): a fetched node is itself scanned for
  links and its targets fetched at depth+1. TOME's hops are latent attention refinements; ours
  are explicit, inspectable document fetches. This is a genuinely different multi-hop mechanism
  worth contrasting directly in the paper.

- **Direction / edge semantics.** TOME has no notion of a directed edge — a mention query
  attends to whatever is nearest in embedding space. Ours has an explicit **DAG gate**: only
  backward links are granted (target must start before the link position;
  `masks.md` cross_doc_mask.py:417-423), asymmetric grant, no transpose. Our edge carries
  structure TOME's similarity retrieval discards.

- **Supervision.** Both are unsupervised on *retrieval* (TOME uses no gold evidence; we use no
  relevance labels — the edge is given by the graph). This is a strong shared talking point:
  TOME's headline "learns to retrieve informative mentions with only task accuracy as signal"
  is our story too, except our "retrieval" is not learned at all — it's handed to the model by
  the link structure. We get their unsupervised-retrieval benefit *for free* and exactly.

### Predictions & open questions for our method

- **Memory-size scaling should transfer to corpus/graph size for us.** TOME's smooth accuracy-
  vs-memory-size curve with diminishing returns (Fig. 2) predicts that our benefit from the
  graph should rise with corpus connectivity/size and then saturate as neighbor documents carry
  overlapping information. We should *plot the analogous curve*: performance vs. number of
  reachable neighbor nodes / packed-neighbor count. Expect the cross_doc_link win over
  doc_causal to grow with average in-degree and then flatten. If ours *keeps* rising past where
  TOME saturated, that's a verbatim-vs-compressed win to claim (their diminishing returns may be
  an artifact of 512-d value compression, which we don't suffer).

- **Where the effect should be strongest.** TOME's gains concentrate on *multi-hop / multi-
  source* tasks (HoVer ≫ FEVER; CWQ ≫ TriviaQA) and *rare entities* (EQ). Prediction: our
  linking bias should show its biggest lift precisely on eval items that require information
  from a *linked* node (especially multi-hop chains) and on *rare/long-tail* target documents —
  and be near-neutral on items answerable from the linking document alone. Design eval slices by
  hop-count and target-frequency; a flat curve across hop-count would be a warning sign.

- **Zero-shot / editability is a strength we can inherit and exceed.** TOME's Table 5 shows you
  can add unseen entities by editing the memory with no retraining. Our exact-lookup corpus is
  even more directly editable: add a node to the corpus and any generated link to it resolves,
  with the *full verbatim text* (not a stale frozen encoding) entering attention. This is a
  clean experiment to run: swap in documents unseen at pretraining, show link-following still
  fetches and uses them. Their result predicts ours should work; ours should degrade *less*
  because we don't rely on the frozen encoder having generalized.

- **Ablation outcome prediction (compute controls).** TOME shows memory is *crucial* (ablating
  it collapses performance). Our matched-compute controls (doc_concat_link, doc_concatenated;
  `masks.md`) are the sharper version of that ablation: they hold FLOPs fixed and isolate the
  *link-position gating*. TOME can't separate "more compute on more text" from "the retrieval
  structure" the way our concat controls can — a design advantage of ours to foreground.

- **Open question ours may resolve for them.** TOME must compress each mention to one 512-d
  value; it cannot re-read source text, so it can't answer "how much is lost to compression vs.
  the retrieval mechanism itself?" Our verbatim-token insertion is the controlled answer: same
  retrieval target, full tokens. If we beat a TOME-style compressed-summary variant on the same
  graph, that quantifies the cost of dense-vector memory.

- **Open question of ours their design informs.** TOME's ANNS-overhead table (negligible at
  large reader scale) predicts that *our* full-recompute cost, which is the reverse (grows with
  T², dominant at 32k), is our real scaling liability. TOME sidesteps it by freezing+compressing.
  If our verbatim-attention advantage is modest, the frozen-cached-KV route (their route) is the
  efficiency escape hatch reviewers will point to.

### Gotchas

- **QA-as-entity-linking inflates/depresses numbers oddly.** TOME scores TriviaQA/CWQ/EQ by
  predicting an *entity*, and counts any answer not in the entity vocabulary as wrong (84% of
  TQA answers, 94% of CWQ are entities). This makes their numbers *not* comparable to generative
  QA (FiD's 77.1 vs their 65.8 is partly this framing). Lesson for us: if we ever evaluate
  link-following on QA, be explicit about the answer-space restriction; don't compare a
  restricted-entity metric against free-generation baselines.
- **"Disallowed same-passage retrieval" = contamination guard.** TOME must zero attention to
  memories built from the current passage, or the model cheats by attending to its own
  (unmasked) encoding. Our analogue: a generated link that resolves to a node *already in
  context*, or a target whose text overlaps the source, could let the model "retrieve itself."
  We should confirm our DAG gate + eviction never lets the active document be fetched as its own
  target (`generation_retrieval.md` notes active doc is protected via `exclude`, but verify the
  train-time mask has an equivalent guard).
- **Frozen encoder = staleness.** Because the memory is frozen after stage 1, any drift between
  the reader and the encoder is baked in; they can only edit *contents*, not the encoding
  function. We don't have this failure mode (verbatim text), which is a point *for* us — but it
  also means TOME's "add unseen entity" success depends on the encoder having generalized, a
  caveat when we cite their editability result as precedent.
- **Pretrain-memory ≠ eval-memory size.** They pretrained on a 38M subsample but evaluate on
  150M. They show this *helps*, but it's a train/eval mismatch of exactly the kind our
  `masks.md` warns about (max_grants must match train/eval or you understate the effect). If we
  ever subsample neighbors at train and use full at eval, measure both.
- **Retrieval quality is hard to score.** TOME notes downstream retrieval is hard to evaluate
  because it often retrieves mentions "not in the gold passage but equally informative," so they
  fall back to qualitative examples on the first few HoVer dev samples. Expect the same for us:
  link-following may fetch a *useful* neighbor that isn't the annotated gold source. Don't build
  a retrieval-precision metric that punishes correct-but-unlabeled fetches.

### Missed citations worth adding

Checked against `paper/bib/refs.bib`. dejong2022tome itself, fevry2020eae, wu2022memorizing,
wu2022emat, guu2020realm, lewis2020rag, lewis2020marge, zemlyanskiy2021readtwice,
verga2020factsasexperts, and the entity-linking cluster (wu2020blink, li2020elq,
logeswaran2019zeroshotel, kolitsas2018endtoend) are **already present**. Genuinely missing from
TOME's reference list and relevant to us:

- **Dhingra et al. 2020, DrKIT — "Differentiable Reasoning over a Virtual Knowledge Base"**
  (ICLR 2020, arXiv:2002.10640). The direct ancestor of the "text corpus as a VKB of mention
  encodings, traversed for multi-hop QA" idea TOME builds on. Highly relevant: it is the
  clearest precomputed-mention-store multi-hop retrieval antecedent, and it *traverses* the VKB
  (closer to our graph-hop framing than TOME's in-reader attention). Strong candidate for our
  precomputed-corpus-memory section.
- **Sun et al. 2021, OPQL — "Reasoning Over Virtual Knowledge Bases With Open Predicate
  Relations"** (arXiv:2102.07043). Memory of *relation* mentions with a
  self-supervised relation encoder + FiLM-style access — a virtual-KB memory keyed on edges/
  relations rather than entities, i.e., a structure-aware memory variant worth contrasting with
  our explicit graph edges.
- **Verga et al. 2021, FiLM — "Adaptable and Interpretable Neural Memory over Symbolic
  Knowledge"** (NAACL 2021, arXiv:2007.00849 is Facts-as-Experts; FiLM is the follow-on
  extending EaE with a KB-fact attention layer). Note: our refs has `verga2020factsasexperts`
  (arXiv:2007.00849) — the *FiLM* paper (Verga et al. 2021) that TOME actually cites as the EaE
  extension may be a distinct entry; verify whether it's the same work before adding.
- **FitzGerald et al. 2021, MOLEMAN — "Mention-Only Linking of Entities with a Mention
  Annotation Network"** (ACL-IJCNLP 2021, arXiv:2106.07352). Compares passage-mention encodings
  to a corpus of mention encodings for entity linking *without retrieving* the mentions — the
  mention-encoding-as-index idea, adjacent to our detector match-key resolution. Minor but on-
  topic for the entity-linking-as-resolution angle.
- **Eisenschlos et al. 2021, FM2 — "Fool Me Twice: Entailment from Wikipedia Gamification"**
  (NAACL 2021, arXiv:2104.04725). A claim-verification benchmark with adversarial/hard
  retrieval; relevant only if we adopt claim-verification-style eval.

(Petroni et al. 2021 KILT — arXiv:2009.02252 — is the standard knowledge-intensive benchmark
suite TOME situates against; worth a check if we frame against KILT, but likely out of scope for
our graph-pretraining lit review.)

Confirmed against the arXiv full text and the current code briefs / refs.bib; claims I could not
verify from source are marked infer or "verify."
