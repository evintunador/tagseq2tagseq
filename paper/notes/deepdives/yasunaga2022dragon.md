# yasunaga2022dragon — DRAGON: Deep Bidirectional Language-Knowledge Graph Pretraining

*(Yasunaga, Bosselut, Ren, Zhang, Manning, Liang, Leskovec. NeurIPS 2022. arXiv:2210.09338.
Code: github.com/michiyasunaga/dragon.)*

Sources: arXiv abstract + ar5iv HTML rendering (method, tables, ablations confirmed from
the rendered paper text, not just the abstract). Our-side claims verified against the code
briefs `masks.md` and `link_detectors.md` and their cited sources. Where I mark **[infer]**
I did not confirm the number/detail against the primary text.

---

## What the paper actually does

DRAGON is a **self-supervised joint pretraining** recipe for a fused text+knowledge-graph
encoder. It takes as input **paired** (text segment, KG subgraph) instances and trains one
model on two objectives at once:

**Input construction.** Sample a text segment of up to **512 tokens**; retrieve an aligned
KG subgraph of up to **200 nodes**, following the QA-GNN retrieval:
1. **Entity linking** — a spaCy linker maps text mentions to KG nodes → seed set.
2. **Bridge nodes** — add any entity lying on a 2-hop path between two linked entities.
3. **Prune** to ≤200 nodes (random downsample), keep all edges spanning the kept nodes.
An **interaction token** (text side) and **interaction node** (KG side) are appended as the
cross-modal exchange points. Note: no PageRank — this is local 2-hop neighborhood retrieval.

**Backbone / fusion.** DRAGON reuses the **GreaseLM** encoder: **19 unimodal LM layers**
then **5 cross-modal fusion layers**. Each fusion layer runs (a) an LM transformer layer over
text, (b) a **GNN layer** over the KG (2 attention heads, 200-dim node embeddings), and
(c) an **MInt** module — an MLP that swaps information between the interaction token and the
interaction node. This is where "bidirectional" lives: text→KG and KG→text flow every fusion
layer. Total **~360M parameters**.

**Two objectives, summed: ℒ = ℒ_MLM + ℒ_LinkPred.**
- **MLM**: mask 15% of text tokens, predict originals (standard).
- **KG link prediction**: drop 15% of subgraph edges, score their reconstruction with a
  **DistMult** head, **128 negative samples** (head/tail corruption), margin γ=0. TransE and
  RotatE heads also work. This is a KG-completion loss run jointly with MLM on the *same*
  fused representation, so each modality is denoised using the other ("bidirectional
  self-supervision").

**Two domains, two inits.**
- General: LM init from **RoBERTa-Large**; text = **BookCorpus** (6GB); KG = **ConceptNet**
  (~800K nodes, 2M edges).
- Biomedical: LM init from **BioLinkBERT-Large**; text = **PubMed** (21GB); KG = **UMLS**
  (~300K nodes, 1M edges).

**Pretraining scale.** 20,000 steps, batch size 8,192, LR 2e-5 (LM) / 3e-4 (GNN + heads),
**7 days on 8×A100, FP16**.

**Headline results (accuracy).**
- General (CommonsenseQA / OBQA / RiddleSense): RoBERTa 68.7 / 64.9 / 60.7 → GreaseLM
  74.2 / 66.9 / 67.2 → **DRAGON 76.0 / 72.0 / 71.3**.
- Biomedical (MedQA-USMLE / PubMedQA / BioASQ): BioLinkBERT 44.6 / 72.2 / 94.8 → GreaseLM
  45.1 / 72.4 / 94.9 → **DRAGON 47.5 / 73.4 / 96.4**.
- Abstract-level claims: +5% avg over LM and LM+KG baselines; **+10% on questions with long
  context or multi-step reasoning**; +8% on low-resource OBQA/RiddleSense.

**Ablations (CSQA / OBQA):**
- Joint beats single: MLM+LinkPred **76.0 / 72.0** vs MLM-only 74.3 / 67.2 vs
  LinkPred-only 73.8 / 66.4. *Both objectives are needed.*
- **Bidirectional fusion 76.0 / 72.0 vs "concatenate at end" 74.5 / 68.0** — deep interleaved
  fusion matters, not late concat.
- **"Use graph" 76.0 / 72.0 vs "convert graph to sentence" 74.7 / 70.1** — keeping the graph
  as explicit structure beats verbalizing it into text.
- Link-pred head barely matters (DistMult 76.0, TransE 75.7, RotatE 75.8).
- DRAGON also improves KG completion itself (Hit@3 78.1 vs DistMult 61.3).

---

## Methodology: theirs vs. ours

The single sharpest axis: **DRAGON trains on a symbolic KG subgraph consumed by a GNN with
message passing; we train on a document-hyperlink graph consumed by a binary, directed,
block-sparse attention mask over one 32k autoregressive sequence.** Everything else follows.

| Axis | DRAGON | TS2TS (ours) |
|---|---|---|
| Graph object | symbolic KG (ConceptNet/UMLS triples), nodes = concepts | text-attributed graph: nodes = whole documents, edges = hyperlinks/imports/citations |
| Edge mechanism | GNN message passing (2-head attn) + MInt MLP exchange | **hard binary attention grant**: link A→B opens a rectangle rows `[link_end_pos, A.end) × cols [B.start,B.end)` (masks.md §Formal semantics) — no message passing, no learned edge bias |
| Directionality | link-pred is symmetric-ish (head/tail corruption); fusion bidirectional | **asymmetric & DAG-gated**: only *backward* links granted (target must start before `link_end_pos`, cross_doc_mask.py:417-423); A reads B, never the transpose |
| Objective on structure | explicit **KG link-prediction loss** (DistMult, 128 negs) | **no structural loss at all** — the edge is an attention pattern, and the *only* loss is next-token LM. Structure is an inductive bias on the compute graph, not a supervised target |
| Backbone | encoder (RoBERTa/BioLinkBERT), MLM | decoder-only, causal LM |
| Fusion depth | 19 unimodal + 5 fusion layers | mask applied at **every** attention layer uniformly (head/layer-independent BIM, masks.md §Compilation) |
| Retrieval scope | ≤200 KG nodes per 512-token text, retrieved per instance | up to **256 link grants** per 32k sequence (max_grants=256 in production, model.py; masks.md §max_grants), docs packed by graph traversal |
| Inference | frozen fused model; still needs KG retrieval + GNN at QA time | **same mask used at inference**: a *generated* link fetches the target doc into the attention context (identical mechanism train and test); generation uses live text detection, not baked grants (link_detectors.md §Protocol) |

**What we genuinely share with DRAGON.** (1) The thesis that *pretraining* on relational
structure (not bolting it on at fine-tune/inference) is where the gain comes from — DRAGON's
whole contribution vs GreaseLM/QA-GNN, which fuse only at QA time. (2) The finding that
**keeping structure explicit beats verbalizing it** ("use graph" 76.0 vs "convert to sentence"
74.7) is the direct KG analog of our claim that a link-gated attention edge beats plain
concatenation of the same documents. (3) Joint text+structure denoising helps *both* the text
task and the structure task.

**Where we diverge hardest.** DRAGON's edge is a *soft, learned, undirected* GNN channel with
its own supervised loss; ours is a *hard, unlearned, directed* attention rectangle with **no
edge parameters and no structural loss**. DRAGON must retrieve a KG subgraph at inference and
run the GNN; our link resolution is native to the decoder's own generation — the model emits a
link and that opens the grant. DRAGON's KG is a curated symbolic ontology; our "KG" is the raw
document corpus's own hyperlink topology, so our nodes carry full text rather than concept ids.

**On compute-controls.** DRAGON's "concatenate at end" and "convert to sentence" ablations are
morally *exactly* our compute-control masks (`doc_concat_link`, `doc_concatenated`; masks.md
§Formal semantics, novelty item #5): they hold the information content roughly fixed and vary
only *how* structure enters. DRAGON shows late-concat / verbalization lose 2-4 points — this is
independent evidence for our experimental design's premise, though our controls are stricter
(true FLOP-superset masks that isolate link-position gating from raw attention budget).

---

## Predictions & open questions for our method

- **Structure loss should be strong on multi-hop / long-context, weak on local tasks.** DRAGON's
  biggest margin is the **+10% on long-context / multi-step** questions and its low-resource
  OBQA/RiddleSense (+8%), while dense-benchmark gains are smaller. Predict our link-grant win
  concentrates on **multi-hop QA (HotpotQA) and cross-file code (RepoBench)** — exactly the
  regimes where the answer requires reading *through* an edge — and is near-flat on
  single-document perplexity. Our compute-controls should show the *smallest* gap on local tasks.
- **Both "halves" are needed — for us, that's the direction gate + the causal LM.** DRAGON's
  MLM-only and LinkPred-only both underperform the joint model by ~2-6 points. Analogously,
  predict that dropping the directed grant (→ `doc_concatenated`, symmetric union) *or* removing
  the traversal packing so no related docs co-occur each degrades the effect; the win is the
  conjunction, not either alone.
- **Explicit edge > verbalized edge is a transferable prior.** DRAGON's "use graph vs convert to
  sentence" result predicts our `cross_doc_link` mask should beat a baseline that simply pastes
  the target document's text inline at the link site (a "verbalize the edge" control). If it does
  *not*, that's an important negative result worth reporting.
- **Open question our design may resolve for them:** DRAGON still pays KG retrieval + GNN cost at
  inference and is capped at 200 nodes / 512 tokens. Our approach shows you can push the same
  "train on structure" idea to a **decoder that resolves its own edges at generation time** with
  no external retriever and 32k context — i.e., collapse DRAGON's separate KG module into the LM's
  own attention. Whether a hard binary grant can match a learned GNN channel's expressiveness is
  the reciprocal open question **their** results pressure-test: their link-head ablation
  (DistMult≈TransE≈RotatE) suggests the *specific* edge scoring barely matters, which is mildly
  encouraging for our parameter-free edge.

---

## Gotchas

- **Retrieval/linking noise is baked into the pretraining signal.** DRAGON's KG subgraph comes
  from a spaCy entity linker + 2-hop bridge heuristic + random downsample to 200. Mislinks and
  the random prune inject label noise directly into the LinkPred loss. Our analog is the **link
  detectors**: markdown is hardcoded to GPT-2 token id 16151, arXiv `\cite` matches must be
  byte-identical to titles, `max_grants` truncation is *positional* not importance-ranked
  (masks.md reviewer-attackable #4; link_detectors.md reviewer-attackable). A mis-detected or
  dropped link silently produces an empty/wrong grant — the same "your structure signal is only
  as clean as your extractor" trap DRAGON lives with. Budget an audit of detection recall.
- **Benchmark contamination / small-data variance.** DRAGON's headline datasets (CSQA, OBQA,
  RiddleSense, MedQA) are small and its gains are a few points; such margins are seed-sensitive.
  Their long-context/multi-hop slice is a *subset* re-cut of the same benchmarks. Mirror this
  caution: report our multi-hop gains with seeds/CIs, and prefer contamination-controlled sets
  (the related-work notes already flag MoreHopQA / PhantomWiki for exactly this).
- **"Convert to sentence" is a weak verbalization baseline.** DRAGON's verbalization control just
  linearizes triples; a *stronger* verbalizer might close the gap. If we run the analogous
  "inline the target text" control, make it a genuinely strong baseline, or a reviewer will argue
  our edge only beats a strawman.
- **Fusion-depth coupling.** DRAGON deliberately fuses only in the top 5 layers; that split is
  tuned. Our mask is applied uniformly at every layer — if we ever ablate *where* the grant is
  active per layer, expect the same fiddly, dataset-dependent tuning surface DRAGON's 19+5 split
  implies. Don't assume "all layers" is optimal without checking.

---

## Missed citations worth adding

Checked against `paper/bib/refs.bib` (533 keys). Already present: linkbert, qagnn, greaselm,
kbert, kepler, jaket, colake, kgplm(he2020), kelm(agarwal2021), realm, retro, lewis2020rag,
peters2019knowbert, zhang2019ernie. The following appear in DRAGON's own reference list, are
relevant to us, and are **NOT** in refs.bib (verified by grep — please still re-verify before
adding):

- **xiong2020wklm** — Xiong et al. 2020, *"Pretrained Encyclopedia: Weakly Supervised
  Knowledge-Pretrained Language Model."* (ICLR 2020; arXiv:1912.09637 **[infer id — verify]**).
  Direct antecedent to our thesis: pretrains on **Wikipedia hyperlink/entity structure** via an
  entity-replacement objective. It's a "train on Wikipedia link structure" prior our lit review's
  "pretraining directly on link structure" subsection is currently missing. Note refs.bib's
  `xiong2020` key is `xiong2020layernorm` — a different paper.

- **feng2020mhgrn** — Feng et al. 2020, *"Scalable Multi-Hop Relational Reasoning for
  Knowledge-Aware Question Answering."* (EMNLP 2020; arXiv:2005.00646 **[infer id — verify]**).
  The multi-hop-over-KG GNN (MHGRN) that our multi-hop framing contrasts against; relevant to the
  "reasoning hops = graph edges" argument.

- **lin2019kagnet** — Lin et al. 2019, *"KagNet: Knowledge-Aware Graph Networks for Commonsense
  Reasoning."* (EMNLP 2019; arXiv:1909.02151 **[infer id — verify]**). The seminal
  path-based-GNN-over-KG-for-QA work; the "explicit relational paths as reasoning" ancestor of the
  whole QA-GNN/GreaseLM/DRAGON line, useful for positioning GNN-edge vs attention-edge.

- **rosset2020knowledgeaware** — Rosset et al. 2020, *"Knowledge-Aware Language Model
  Pretraining."* (arXiv:2007.00655, id from DRAGON's reference list). Injects entity knowledge into
  *pretraining* (not fine-tune); belongs alongside our KG-enhanced-LM subsection.

- **yao2019kgbert** — Yao et al. 2019, *"KG-BERT: BERT for Knowledge Graph Completion."*
  (arXiv:1909.03193, id from DRAGON's reference list). The LM-for-KG-completion baseline DRAGON's
  link-prediction head builds on; marginal for us but the natural cite if we discuss structural
  losses vs attention edges.

Lower relevance (KG-QA-internal, likely skip): shen2020 graph-guided (EMNLP 2020), wang2022 "GNN
is a Counter?" (ICLR 2022), sun2021 ERNIE 3.0 (arXiv:2107.02137). Listed for completeness only.

---

Confirmation: wrote deep-dive to
`/fss/evin_t/tagseq2tagseq/paper/notes/deepdives/yasunaga2022dragon.md`; method/results verified
against the ar5iv rendering, our-side claims against masks.md and link_detectors.md, and every
"missing citation" grep-checked against refs.bib.
