# gutierrez2024hipporag — HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models

*(Gutiérrez, Shu, Gu, Yasunaga, Su. NeurIPS 2024. arXiv:2405.14831. Code: github.com/OSU-NLP-Group/HippoRAG.)*

Sources: arXiv abstract + ar5iv HTML rendering (method, tables, ablations read from the
rendered paper text, not just the abstract). Our-side claims verified against the code briefs
`generation_retrieval.md`, `eval_harness.md`, `traversal.md` and their cited sources, plus
`related_work_notes.md`. Where I mark **[infer]** I did not confirm the number/detail against
the primary text.

---

## What the paper actually does

HippoRAG is a **retrieve-then-read** pipeline for multi-hop QA that replaces the flat dense
index of standard RAG with an **LLM-built open knowledge graph** plus a graph-search step. The
neuroscience framing (hippocampal indexing theory: neocortex = LLM stores/processes knowledge,
hippocampus = an index that binds and pattern-completes over it) is a motivation, not an
architectural constraint — mechanically it is OpenIE + a graph + Personalized PageRank (PPR).
Nothing is trained; every LLM call is few-shot prompting of a frozen model.

**Offline indexing (build once per corpus).**
1. For each passage, an LLM (**GPT-3.5-turbo-1106, temp 0, 1-shot**) does **OpenIE**: first
   extract named entities, then extract relation triples (subject, relation, object) seeded by
   those entities so triples reach beyond just named entities.
2. Aggregate all triples into a **schemaless open KG**: nodes N = noun-phrase entities, edges E
   = extracted relations. This is an entity-level KG, *not* a passage graph.
3. Add **synonymy edges E′**: an off-the-shelf retrieval encoder (**Contriever** or
   **ColBERTv2**) connects two entity nodes whose cosine similarity exceeds **τ = 0.8**. This is
   what lets the graph "pattern-complete" over paraphrases / entity variants.
4. Store an **|N|×|P| occurrence matrix P** (which noun phrase appears in which passage) — the
   bridge from entity-node probability mass back to passages.

Corpus scale (1,000-question dev subsets): MuSiQue 11,656 passages / 91,729 nodes / 21,714
edges; 2Wiki 6,119 / 42,694 / 7,867; HotpotQA 9,221 / 82,157 / 17,523.

**Online retrieval (per query, single step).**
1. LLM extracts the **query named entities** (1-shot), e.g. "Stanford", "Alzheimer's".
2. Encode each with the retrieval encoder M; the **query nodes** are the KG nodes with highest
   cosine similarity (argmax link).
3. Weight each query node by **node specificity** sᵢ = |Pᵢ|⁻¹ (inverse # passages containing
   node i) — an IDF-like signal that down-weights promiscuous, uninformative nodes. This is
   folded into the seed (personalization) vector.
4. Run **Personalized PageRank** over the KG, restart distribution = the weighted query nodes,
   **damping factor 0.5**. PPR is the "pattern-completion / associative recall" step: mass
   spreads from query entities along relation + synonymy edges to entities that are graph-close
   even when they never co-occur in a passage.
5. Multiply the converged node-probability vector by **P** to score passages; return top-k.

The whole point: **one PPR pass simulates multi-hop association** without iterating the LLM.

**Results that matter.**
- **Single-step retrieval R@2/R@5** (best baseline vs HippoRAG-ColBERTv2, avg over 3 sets):
  ColBERTv2 53.9/65.6 → **HippoRAG 57.4/72.9**. Huge on the entity-bridging set **2Wiki:
  59.2/68.2 → 70.7/89.1** (Contriever variant 71.5/89.5). On **HotpotQA HippoRAG loses**
  (ColBERTv2 64.7/79.3 vs HippoRAG 60.5/77.7) — see Gotchas.
- **Single-step ≈ iterative.** HippoRAG (one PPR) roughly matches **IRCoT** (multi-step
  LLM-in-the-loop retrieval) on QA: HippoRAG avg EM/F1 **35.9/48.1** vs IRCoT **33.3/44.7**.
- **Complementary to IRCoT.** Using HippoRAG as IRCoT's retriever is the overall best:
  **IRCoT+HippoRAG** QA avg **38.4/51.7**; retrieval avg **62.7/78.2** (R@2/R@5).
- **Efficiency (its headline practical claim).** Online retrieval is **10–30× cheaper and
  6–13× faster than IRCoT**, because IRCoT re-runs the LLM over retrieved passages every step
  whereas HippoRAG's only online LLM call is query NER; PPR is cheap graph math. (The heavy cost
  — OpenIE over every passage — is paid **once, offline**.)
- **Ablations.** PPR clearly beats "query nodes only" (50.7/56.2) and "query nodes + 1-hop
  neighbors" (42.2/59.2) → the graph *walk* is doing work, not just neighbor expansion. Synonymy
  edges matter most on 2Wiki; node specificity helps MuSiQue/HotpotQA. OpenIE LLM matters:
  Llama-3-8B ≈ GPT-3.5, but REBEL (a trained end-to-end RE model) and Llama-3-70B (malformed
  output → ~20% passage loss) degrade sharply. Error analysis on MuSiQue: **NER 48% / OpenIE
  28% / PPR 24%** of failures.

---

## Methodology: theirs vs. ours

Sharpest axis: **HippoRAG builds an external entity KG and runs PPR over it at inference to
*rank passages* for a frozen reader; TS2TS bakes the document-hyperlink graph into pretraining
as a hard attention edge and resolves the next hop *inside the decoder's own forward pass*.**
Their graph is non-parametric memory consulted at query time; ours is an inductive bias burned
into the weights and the attention pattern.

| Axis | HippoRAG | TS2TS (ours) |
|---|---|---|
| When structure is used | **inference only**; model frozen, KG built offline by prompting | **pretraining** (and identically at inference); graph shapes packing + attention every step |
| Graph object | derived **entity KG**: nodes = noun phrases, edges = LLM-extracted relations + encoder synonymy edges (τ=0.8) | **native TAG**: nodes = whole documents, edges = real hyperlinks/imports/citations from the corpus (traversal.md: directed `neighbors_out`/`neighbors_in`) |
| Edge provenance | **fabricated** by an LLM OpenIE pass (noisy, lossy) | **given** by the corpus; a link detector maps a surface token to a target doc (`generation_retrieval.md`: same `index_doc_span` key train + infer) |
| "Hop" mechanism | **Personalized PageRank** — a stationary-distribution random walk with restart to query nodes, damping 0.5, over the *whole* KG in one shot | **BFS/DFS traversal** to co-pack topologically-close docs (`traversal.md`) + **recursive link-following** at generation (`generation_retrieval.md` `max_link_depth`, default 2): a generated link fetches the target doc, which is scanned for *its* links at depth+1 |
| What the graph search returns | a **ranking of passages** handed to a separate reader LLM | nothing external — the fetched doc is **inserted into the packed sequence** (`_docs.insert(idx)` before the linker) and read through the **cross-doc attention grant** |
| Parametric? | **no** — zero training; graph + PPR are non-parametric memory | **yes** — the link edge is realized by a `CrossDocLinkMaskCreator` mask the model was trained under; no external index at all |
| Reader coupling | decoupled: retriever ranks, reader answers over top-k text (classic retrieve-then-read) | **fused**: retrieval = materializing the target's tokens so the linking positions attend to them (`generation_retrieval.md`: "retrieval-BY-INSERTION", KV recomputed every step, no external store) |
| Multi-hop cost | O(1) LLM calls online (NER) + one PPR solve | O(depth) recursive fetches, bounded by `max_link_depth`; **no KV cache**, full recompute each token (generation_retrieval.md) |
| Directionality | PPR is run on a largely undirected/symmetrized association graph (synonymy edges symmetric) | **DAG-gated, asymmetric**: grant fires only if target starts before the link position (`traversal.md`: `cross_doc_mask.py:417-423`; `prefer_targets_first` topo reorder required to realize outgoing links) |

**What we share.** (1) The core thesis that **graph proximity is the right retrieval prior for
multi-hop** — that documents/entities several hops apart, which never co-occur, must be pulled
together. HippoRAG's PPR and our BFS/DFS+recursive-fetch are two ways to cash out the same
belief. (2) Both treat the graph as *memory*: their "hippocampal index" over neocortical
knowledge is conceptually our packed-sequence-as-working-memory with links as the index. (3)
Both exploit **single-step / non-iterative** multi-hop where possible — HippoRAG's whole selling
point vs IRCoT is "one PPR pass beats iterating the LLM," echoing our claim that the answer doc
is reachable in-sequence via the attention grant rather than by an external retrieve-reason loop.

**Where we diverge hardest.** HippoRAG never touches model weights; its graph is a *fabricated*
entity KG queried by a random-walk *ranker*, and the result is passage text spliced into a
prompt for a frozen reader. Ours is a *trained* attention edge over the *real* document graph,
resolved by the decoder itself with no external index, no PPR, and no separate reader. Their
"multi-hop" is a global stationary distribution over an entity graph; ours is a bounded,
directed, recursive traversal that physically re-orders and inserts documents into one sequence.
Their edge scoring is unlearned graph math (PPR); ours is unlearned too (a **binary attention
grant**, `masks.md`) — a genuine commonality worth stating: *neither of us learns edge weights*,
which their PPR-vs-neighbors ablation and their OpenIE-LLM-swap robustness suggest is often fine.

**PPR vs our random walk (a precise, easily-muddled point).** `traversal.md` is explicit that
our `RandomWalk` strategy **restarts to a UNIFORM-RANDOM node, not to the seed**, so it is *not*
RWR / personalized PageRank; `related_work_notes.md:373-376` already flags the packer's
teleport as PageRank-style but seed-agnostic. HippoRAG is the genuine PPR (restart to *query*
nodes, damping 0.5). So HippoRAG is the clean external-inference realization of the exact
algorithm our packer deliberately does *not* use — an ideal contrast paragraph: personalized
teleport for query-conditioned inference retrieval (theirs) vs uniform teleport for
query-agnostic corpus packing (ours).

---

## Predictions & open questions for our method

- **The link win should peak exactly where HippoRAG's does: entity-bridging multi-hop, not
  single-hop-ish HotpotQA.** HippoRAG's spread is enormous on **2Wiki** (+11.5 R@2, +21 R@5)
  where answers require chaining entities that don't co-occur, but it *loses* on HotpotQA where
  a single strong dense hit often suffices. Predict our `cross_doc_link` Δnll gain (eval_harness.md
  paired cross-vs-flat) is largest on genuine bridge questions and near-zero — possibly negative
  — where the supporting docs already sit adjacent or a single doc answers. Our HotpotQA harness
  (`run_hotpotqa_cross_doc`, bridge-type, A→B link) should show the effect; HippoRAG warns it may
  be *small* on HotpotQA specifically.
- **The graph walk beats mere neighbor expansion — so multi-hop depth should matter for us too.**
  HippoRAG's PPR crushes "query-nodes+1-hop-neighbors" (avg 57 vs 42 R@2). Analog: our recursive
  link-following at `max_link_depth=2` should beat depth-1 (fetch only the directly linked doc)
  on multi-hop, but with **diminishing/negative returns** past the true hop count — expect an
  inverted-U in depth, and budget an ablation over `max_link_depth`.
- **Node specificity ≈ our "which links are worth granting" problem.** Their IDF-like sᵢ says
  *down-weight promiscuous nodes*. Our grant selection is **positional, not importance-ranked**
  (`eval_harness.md`/`masks.md`: `max_grants` truncation by position). Prediction: on
  hub-heavy graphs (Wikipedia has ultra-high-degree nodes) our positional truncation will waste
  grant budget on uninformative high-degree links; an IDF-style edge prioritization before the
  `max_grants=256` cut is a cheap, HippoRAG-motivated improvement likely to help most where the
  degree distribution is heavy-tailed.
- **Single-step (us) vs iterative (IRCoT), and the complementarity result.** HippoRAG shows
  single-step graph retrieval ≈ iterative LLM retrieval, *and* that they compose
  (IRCoT+HippoRAG best). This predicts (a) our in-sequence link resolution should rival
  iterative retrieve-reason baselines on multi-hop without an external loop — a strong framing
  point — and (b) an open question: could our trained link edge *also* be stacked under an
  iterative controller for further gains, or does native resolution already capture what IRCoT
  adds? Their result says stacking helps; worth a discussion sentence, not a claim.
- **Open question our design resolves for them.** HippoRAG must fabricate a KG offline (costly
  OpenIE over every passage) and cannot update the reader. Our approach shows the same
  "graph-structure-as-memory" benefit can be **native and parametric** — no OpenIE, no external
  index, no separate reader, and it works for **code import graphs** (RepoBench) where OpenIE
  entity extraction is ill-defined. Conversely, **their** open question pressures us: PPR
  aggregates *global* graph evidence, while our grant only reaches docs we actually packed /
  fetched within `max_link_depth` and the 32k budget — we cannot integrate evidence from a doc
  that was never brought into the sequence. HippoRAG's global-stationary-distribution view is
  precisely the recall ceiling our bounded local traversal risks missing.

---

## Gotchas

- **The graph is only as good as the OpenIE extractor — and it is noisy/lossy.** HippoRAG's own
  MuSiQue error analysis blames **NER (48%) and OpenIE (28%)** for 76% of failures, and shows a
  bad extractor (REBEL, or Llama-3-70B losing ~20% of passages to malformed output) tanks
  results. Direct warning for us: our **link detectors** are the equivalent single point of
  failure (`generation_retrieval.md`, `eval_harness.md`) — markdown hardcoded to a GPT-2 token
  id, `\cite` needing byte-identical titles, `PythonImportDetector` train-vs-infer divergence
  flagged as *unconfirmed*. A missed/mis-resolved link silently yields an empty or wrong grant,
  exactly HippoRAG's "your structure signal is only as clean as your extractor" trap. Budget a
  detection-recall audit.
- **HippoRAG *loses* on HotpotQA — don't overclaim graph retrieval universally.** A strong dense
  retriever (ColBERTv2) beats it where single-hop lexical/semantic overlap suffices. If our
  cross-doc harness shows flat/negative Δ on the HotpotQA bridge subset, that is *consistent with
  the literature*, not a bug — report it, don't bury it. It also cautions against our
  fire-conditioned subset selection (`eval_harness.md` reviewer-attackable #2): HippoRAG-style
  gains can be a small win on a favorable slice.
- **Contamination.** HippoRAG's sets are built from HotpotQA/2Wiki/MuSiQue over Wikipedia; our
  own harness note (`eval_harness.md` #5) already flags HotpotQA 2017-wiki / training overlap.
  Their strong 2Wiki numbers ride on template-y entity-bridge questions; mirror their caution and
  prefer contamination-controlled sets (related_work_notes flags MoreHopQA / PhantomWiki).
- **PPR damping / synonymy threshold are tuned knobs.** Damping 0.5 and τ=0.8 are hand-set; their
  ablations show synonymy edges swing 2Wiki hard. Analog for us: the **Option-B `span.start+1`
  key hack** and `max_grants` are exactly this kind of load-bearing tuned knob (`eval_harness.md`
  reviewer-attackable #3–4). A reviewer who saw HippoRAG's knob-sensitivity will ask for our
  effect's sensitivity curve; have it ready.
- **"All-Recall" / retrieving *all* supporting docs is the honest multi-hop metric.** HippoRAG
  reports a separate All-Recall (all supports retrieved) where its margin over ColBERTv2 is
  largest (29.8/52.0 vs 21.7/37.4). For genuinely multi-hop claims, single-hit recall@k
  understates the multi-hop story; consider an analogous "did we grant *all* required neighbors"
  measure rather than any-hit.
- **Their efficiency win hides an offline cost we also pay.** "10–30× cheaper online" excludes
  the one-time OpenIE-over-every-passage indexing. Our analog: the **traversal precompute +
  packing** cost, and our *inference* cost is the opposite trade — **no KV cache, ~O(T²)/step
  full recompute** (`generation_retrieval.md`). Don't cite HippoRAG's efficiency framing as if it
  transfers; our inference is expensive per token, theirs is cheap online but expensive to index.

---

## Missed citations worth adding

Checked against `paper/bib/refs.bib`. **Already present** (do not re-add): `gutierrez2024hipporag`
(the paper itself, line 1464), `page1999pagerank`, `haveliwala2002topicpagerank` (topic-sensitive
PageRank = PPR ancestor), `izacard2022contriever`, `ni2022gtr`, `khattab2020colbert` /
`santhanam2022colbertv2`, `trivedi2022musique`, `ho2020twowiki`, `yang2018hotpotqa`,
`trivedi2023ircot`, `jiang2023flare`, `asai2024selfrag`, `edge2024graphrag`. The following are in
HippoRAG's own reference list (or are its direct baselines), are relevant to us, and are **NOT**
in refs.bib (grep-verified — please still re-verify before adding):

- **sarthi2024raptor** — Sarthi, Abdullah, Tuli, Khanna, Goldie, Manning, *"RAPTOR: Recursive
  Abstractive Processing for Tree-Organized Retrieval."* ICLR 2024; **arXiv:2401.18059**
  (title/authors/id confirmed via arXiv). A HippoRAG baseline and a **structure-over-corpus
  retrieval** method (recursive summary tree) — the tree-structured counterpart to our
  graph-structured packing; belongs in the graph/structure-aware retrieval discussion.
- **chen2024densex** — Chen, Wang, S. Chen, Yu, Ma, Zhao, H. Zhang, Yu, *"Dense X Retrieval:
  What Retrieval Granularity Should We Use?"* **arXiv:2312.06648** (confirmed). HippoRAG's
  "Proposition" baseline. Directly relevant to our **retrieval-granularity** stance (we retrieve
  whole documents by insertion; they argue for proposition-level units) — a clean granularity
  contrast for our related work.
- **press2023selfask** — Press, M. Zhang, Min, Schmidt, Smith, Lewis, *"Measuring and Narrowing
  the Compositionality Gap in Language Models"* (Self-Ask). Findings of EMNLP 2023;
  **arXiv:2210.03350** (confirmed). The canonical **self-ask** iterative decomposition baseline;
  our notes already cite IRCoT/FLARE/DecompRC but not Self-Ask — the obvious missing member of the
  "inference-time question decomposition" set the generation loop contrasts against.
- **huguetcabot2021rebel** — Huguet Cabot & Navigli, *"REBEL: Relation Extraction By End-to-end
  Language generation."* Findings of EMNLP 2021. **[no arXiv id — verify; it is an ACL
  Anthology paper]**. HippoRAG's alternative OpenIE extractor (ablation); relevant *only* if we
  discuss open-IE / graph-construction quality as the analog of our link-detector recall problem.
  Lower priority.

Lower relevance / likely skip (KG-QA or neuroscience-internal): the hippocampal-indexing-theory
neuroscience cites (Teyler & DiScenna; complementary-learning-systems, McClelland/Kumaran) — only
if we lean into the biological-memory framing, which we currently do not.

---

Confirmation: wrote deep-dive to
`/fss/evin_t/tagseq2tagseq/paper/notes/deepdives/gutierrez2024hipporag.md`; method/numbers
verified against the ar5iv rendering of arXiv:2405.14831, our-side claims against
`generation_retrieval.md`, `eval_harness.md`, `traversal.md`, and every "missing citation"
grep-checked against `refs.bib` with arXiv ids confirmed via arXiv (except REBEL, flagged).
