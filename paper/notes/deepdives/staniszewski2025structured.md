## staniszewski2025structured — Structured Packing in LLM Training Improves Long Context Utilization (SPLiCe)

arXiv 2312.17296 (v9), AAAI'25. Staniszewski, Tworkowski, Jaszczur, Zhao, Michalewski, Kuciński, Miłoś.
Cross-referenced against our code briefs `packing_density.md` and `traversal.md` (both @ commit 6134163),
and the `related_work_notes.md` entry that already lists this work under §3 (sequence packing).

### What the paper actually does

**Core idea.** SPLiCe ("Structured Packing for Long Context") is a *data-construction* method, not an
architecture change. It replaces the standard "sample random documents, glue with BOS/EOS" packing recipe
with one that **collates mutually relevant documents into a single long training example**, so that a
long context actually contains long-range dependencies for the model to learn to exploit. The stated
target is "dependency density" — normal packed corpora have almost no cross-document dependencies, so a
model trained on them never learns to use tokens far back in the window; SPLiCe manufactures those
dependencies by co-locating related docs. The paper is explicit that "the architecture and training
objectives [are] unchanged" — **standard causal attention over the concatenation; no attention mask, no
cross-document mechanism.** Structure enters purely through *which documents co-occur and in what order*.

**Algorithm (Algorithm 1).** Build one training sample as a retrieval tree:
- Sample a random root document from the corpus.
- BFS expansion via a queue: pop a doc, `RETRIEVE` its top-*k* most-similar documents, append the
  not-yet-used ones to the sample *C*, push them onto the queue.
- Stop when the queue empties or the sample reaches length *L*.
- Emit `CONCAT(TRIM(ORDER(C), L))` — flatten the tree by an `ORDER` traversal, trim to *L*, concatenate.

The branching hyperparameter *k* interpolates: **k=1 builds a *path* of related documents** ("simulates a
long document by creating a path of related examples"); large *k* gives a RAG-style shallow-and-wide sample.
Default is **k=1 with identity ordering**.

**Retriever variants (how RETRIEVE is instantiated):**
- **SPLiCe-BM25** — bag-of-words BM25 over full document content. Reported as the slightly best overall;
  strong generalization, cheap.
- **SPLiCe-Cont(riever)** — Contriever-MSMARCO transformer retriever, zero-shot on the first 512 tokens,
  inner-product ranking via Faiss. Tends to produce samples spanning fewer repositories than BM25.
- **SPLiCe-Repo** — **our closest analog.** No learned retriever: it uses the code repository's
  **directory structure** directly, concatenating files by a **depth-first traversal of the directory tree**
  so files in the same directory are grouped. Fastest/simplest, but only applies where directory metadata
  exists (code repos, not general web text).

**Baselines.** *Example Packing (EP)* = GPT-3-style random-doc packing (the main foil). *Within-Domain EP*
= random docs from the same meta-class (Wikipedia, C code, …) — showed no clear gain over EP, an important
negative control. *Random/noisy-retriever* ablation replaces retrieval with random in-domain docs.
Concurrent **shi2024incontext** (In-Context Pretraining) is the nearest neighbor: same at k=1/identity,
but ICP trains from scratch at 8K, whereas SPLiCe is *brief fine-tuning at ≥32K/64K*.

**Training setup.** Fine-tuning (not from-scratch) of already-pretrained models:
OpenLLaMA-3Bv2 (5.4B tokens), OpenLLaMA-7Bv2 and CodeLlama-13B (2B tokens each), plus ~40 custom 270M
models for ablation. Context length fine-tuned at **32K** (270M scaled to 64K/131K/160K). Context-window
extension via Focused Transformer (FoT) or CodeLlama's method (YaRN/naive also tested). Data mixtures:
3B = 50/50 RedPajama + C code; 7B/13B = 50/25/25 RedPajama/StackExchange/C code; code from StarCoder.
LR 1.5e-5, batch 256K tokens (512K for 13B).

**Evaluation + headline numbers.**
- **In-context learning classification (long context):** TREC 3B-FoT@32K 73.9→79.3 (+5.4), @16K 68.9→76.9
  (+8.0); 7B@32K 75.6→79.4 (+3.8); 13B@32K 89.2→92.4 (+3.2). DBpedia 3B@32K 82.9→85.9 (+3.0),
  7B@32K 82.9→84.9 (+2.0).
- **Perplexity (270M, 50/50 RedPajama+C):** SPLiCe-BM25 3.100 vs EP 3.228; gap *widens* with context —
  64K C# 2.88 vs 3.07, 131K 2.60 vs 2.77. Longer context ⇒ larger structured-packing benefit.
- **Qasper & HotpotQA:** paper reports SPLiCe > EP across 3B/7B/13B (per-model tables in Appendix K; exact
  F1 not captured here — do not cite a specific number).
- **Cross-modal transfer (a flagged result):** training on *code* with SPLiCe improved *arXiv/NL*
  perplexity — "training on a corpus of code can enhance performance on natural language tasks."
- **Short-context is preserved:** 13B MMLU/GSM8K/HumanEval not degraded; GSM8K +1.7 (21.4→23.1).
- **Qualitative:** perfect Needle-in-a-Haystack; mitigates lost-in-the-middle key-value retrieval
  (300-pair dict, ~24K tokens). Burstiness: SPLiCe lowers Zipf coefficient (C code 1.512 vs EP 1.593),
  connecting to Chan et al. 2022's data-distributional account of in-context learning.
- **Robustness:** SPLiCe beats EP even with a noisy retriever, only converging to EP at noise p=0.9.

### Methodology: theirs vs. ours

The one-sentence axis: **SPLiCe and TS2TS both use graph/relatedness structure to decide *what co-occurs
in one long sequence*, but SPLiCe stops at ordering under vanilla causal attention, while we add a trained,
direction-gated cross-document attention *edge* on top of the same co-location.** SPLiCe is the
"train-on-structure-via-packing-order-only" pole; we are "train-on-structure-via-packing-order *plus*
attention edge."

Concrete comparison points:

- **SPLiCe-Repo directory DFS vs. our graph traversal.** SPLiCe-Repo orders files by a DFS over the
  *directory tree*. Ours (`traversal.py`, brief §Strategies) orders by a pluggable walk (BFS/DFS/RandomWalk)
  over an explicit **directed document graph** whose edges are realized hyperlinks/imports/citations, not
  filesystem adjacency. Their "structure" is containment (same directory ⇒ near); ours is a *link*
  (doc A imports/cites/links doc B). SPLiCe-Repo cannot express a cross-directory import edge; our graph can.
  Note both share a subtlety: SPLiCe's tree is built by BFS `RETRIEVE`-expansion but flattened by `ORDER`
  (DFS/identity); we likewise *decouple* growth strategy from emission order — our default
  `prefer_targets_first` re-imposes a per-component Kahn topological sort (`pack_sampler.py:476-558`,
  packing brief §Pack layout) after traversal.

- **Ordering is load-bearing for *different* reasons.** SPLiCe orders so that related docs are *adjacent*
  (short relative distance ⇒ causal attention can bridge them). For us, ordering is load-bearing because
  our **DAG attention gate requires the target to start before the link position** (`cross_doc_mask.py:417-423`,
  traversal brief §Mask interaction): under outgoing traversal + causal masking, a linked-to doc that lands
  *after* its linker gets its grant **silently dropped**. So we topo-sort targets *ahead* of linkers — the
  single most important non-obvious design point in our packer. SPLiCe never faces this because it has no
  edge and no gate; adjacency alone is the whole mechanism.

- **The mask is the divergence.** SPLiCe explicitly keeps attention standard. TS2TS's contribution *is*
  the mask: a block-sparse grant that gives a linking document read-access, from the link token onward,
  into the specific target document (packing brief: analytic O(#blocks) density, custom Triton
  `cross_doc_link`/`triton_v18` kernel per CLAUDE.md). SPLiCe therefore also has no attention-cost problem
  to model — it runs dense causal attention and pays no sparse-kernel tax; our whole density-bucketing
  apparatus (`bucketed_pack_dataset.py`, kv_block_count proxy) exists *only because* we add the edge.

- **Retrieve-at-inference vs. train-on-structure.** Both are firmly train-time. SPLiCe touches training
  data only; at inference the model is a stock causal LM with no retrieval hop. TS2TS is stronger on this
  axis: the same mask is used at inference so a *generated* link deterministically fetches its target into
  attention — SPLiCe has no inference-time analog to that. Neither uses a learned retriever at inference
  (SPLiCe's retriever is offline, data-prep only; ours is a deterministic identifier/index resolution).

- **Retriever vs. exact edge.** SPLiCe-BM25/Cont use an *approximate similarity* retriever to *induce*
  relatedness; SPLiCe-Repo uses exact directory metadata. Our edges are exact structural links (already in
  the corpus), most comparable to SPLiCe-Repo — but SPLiCe-Repo discards the actual edge identity and keeps
  only "same dir," whereas we keep the specific A→B pointer and attend along it.

- **Scale & regime.** SPLiCe = brief *fine-tuning* (2–5.4B tokens) of an existing 3–13B model, context
  *extension* to 32K. TS2TS = pretraining a decoder from scratch at 32K with the edge baked in from step 0.
  Their evidence that even short fine-tuning moves long-context metrics is encouraging for us but not a
  from-scratch result.

### Predictions & open questions for our method

- **The long-context benefit should grow with context length.** SPLiCe's perplexity gap over EP *widens*
  from 32K→64K→131K. Prediction: our cross-doc-edge advantage over concat baselines should likewise be
  small at short context and grow at 32K and under 8k→32k RoPE extension. If our effect is *flat* in
  context length, that is a warning sign that the edge isn't being used (e.g., grants silently dropped —
  see traversal brief §Reviewer-attackable).

- **Adjacency alone already buys a lot — so isolate the *edge's* marginal gain.** SPLiCe shows that merely
  co-locating related docs under plain causal attention yields +3 to +8 points on ICL tasks *with no mask*.
  This is essentially our concat compute-control condition. Prediction: a large fraction of any TS2TS gain
  may come from *packing topology* (related docs together), not from the attention edge per se. Our
  concat-variant controls (brief mention: "compute-control masks isolate the linking inductive bias from
  raw FLOPs") are exactly the right instrument — SPLiCe's results predict those controls will themselves
  beat random packing, and the edge must be shown to add *on top of* that. This is the make-or-break
  ablation.

- **Within-domain packing was *not* enough for them.** SPLiCe's WithinDomEP (same meta-class, random within)
  showed "no clear benefit" over EP — relatedness has to be fine-grained (link/retrieval level), not just
  domain-level. Prediction: our traversal-local neighborhoods should beat a "same-corpus/same-repo random"
  control; if they don't, the graph signal isn't fine-grained enough (relevant to our uniform-seed +
  local-growth honesty caveat in the traversal brief — "BFS packing" = uniform-seeded local BFS).

- **Code→NL transfer.** SPLiCe found structured code packing improved *natural-language* perplexity.
  Prediction: our import-graph-trained model may show cross-domain lift onto wiki/arXiv eval, and our
  multi-corpus setup (TheStack + Wiki + ArXiv) could exhibit similar transfer. Worth an explicit
  cross-corpus eval cell.

- **Short-context tasks should not regress.** SPLiCe preserved MMLU/GSM8K/HumanEval. Our single-document
  control benchmarks (HellaSwag/ARC/PIQA etc., per related-work §6) should be flat — matching SPLiCe's
  finding that the intervention is long-context-specific.

- **k=1 path vs. wide branching.** SPLiCe's k=1 "path of related docs" is closest to our linear traversal
  packing. Their finding that k=1/identity is the default-good setting supports our chain-like traversal
  over shallow-wide neighborhoods — but they leave open whether a wider fan-out helps at very long context.
  Our BFS-vs-DFS-vs-RandomWalk strategy axis is a direct probe of the same question (fan-out vs depth),
  and could *resolve their open question* about branching, since we can sweep it while holding the edge fixed.

- **Open question our design may resolve for them:** SPLiCe explicitly notes it changes *nothing* about
  attention and lists "does an attention mechanism help?" implicitly by leaving it out. TS2TS is the direct
  test of whether adding a trained cross-doc *edge* beats mere co-location — the experiment SPLiCe's setup
  motivates but does not run.

### Gotchas

- **The concat baseline is strong — don't under-tune it.** SPLiCe's own numbers show plain related-doc
  packing under vanilla attention gains multiple points. If our concat compute-control is weak (bad
  ordering, no relatedness), we'll over-attribute the gain to the edge. Build the concat control to be as
  strong as SPLiCe-Repo, or reviewers will say the edge is doing nothing that ordering didn't.
- **Metric fragility on long-context ICL.** SPLiCe leans on TREC/DBpedia few-shot accuracy, which is noisy;
  our related-work §6 already prefers continuous Δnll for exactly this reason (biderman2024lessons,
  schaeffer2023mirage). Prefer our paired Δnll gate over accuracy deltas of a few points.
- **Retriever/packing contamination.** SPLiCe co-locates *test-similar* docs by retrieval; with a
  Wikipedia/HotpotQA setup that risks pulling gold-supporting docs together in a way that leaks. Our
  deterministic edges are less prone to this, but our HotpotQA-2017 leakage caveat still applies.
- **Directory-DFS degenerates on flat/huge dirs.** SPLiCe-Repo relies on directory granularity; a repo with
  one giant dir gives no structure. Our analog failure: worker sharding that cuts the graph at shard
  boundaries degenerates BFS to doc-causal (packing brief §Live-vs-precomputed: "BFS hits shard boundary →
  degenerates to doc-causal"; mitigated by Voronoi BFS partition + 1.5× cap). Verify realized link density,
  don't assume it.
- **Silently-dropped structure inflates the "structured" label.** Their trimming (`TRIM`) can cut retrieved
  docs; ours drops grants three ways (DAG gate ordering, `max_grants=64`, out-of-index neighbors —
  traversal brief). Both mean *realized* structure < intended; quantify realized density before claiming
  the corpus is structured.
- **Fine-tune vs. from-scratch mismatch.** SPLiCe's gains are from brief fine-tuning of a model already
  good at short context. We train from scratch, so we can't borrow their "it's cheap" framing; our compute
  story is different and their token counts don't transfer.

### Missed citations worth adding

Checked against `paper/bib/refs.bib`. Already present: shi2024incontext, zhao2024analysing, tworkowski2023focused
(Focused Transformer), peng2023yarn, roziere2023codellama, li2023starcoder, wu2022memorizing. Genuinely
missing candidates from SPLiCe's own reference list that matter to us:

- **chan2022datadistributional** — Chan et al., "Data Distributional Properties Drive Emergent In-Context
  Learning in Transformers," NeurIPS 2022 (arXiv 2205.05055). *Why:* SPLiCe grounds its "burstiness"
  analysis (Zipf coefficient) here — a principled account of *why* co-locating related docs improves
  in-context learning. Directly supports our claim that graph-local packing changes the training
  distribution in a way that drives long-context use. Not in refs (grep: 0 hits).

- **redpajama / togethercomputer2023redpajama** — "RedPajama: An Open Dataset for Training Large Language
  Models" (Together Computer, 2023; there is also a 2024 arXiv writeup, ~2411.12372). *Why:* one of the
  standard open pretraining corpora; if we describe corpus composition / baselines it's a natural cite
  alongside kocetkov2022stack. Not in refs (grep: 0 hits). (Verify exact key/id before adding.)

- **geng2023openllama** — Geng & Liu, "OpenLLaMA: An Open Reproduction of LLaMA" (2023, GitHub/software
  release; no arXiv id). *Why:* the base model family SPLiCe fine-tunes; minor, only worth adding if we
  reference open LLaMA reproductions. Not in refs (grep: 0 hits). (No arXiv id — software cite.)

Lower priority / likely out of scope: FoT context-extension and StarCoder are already covered
(tworkowski2023focused, li2023starcoder / lozhkov2024starcoder2).

---
Confirmed from the paper's HTML (arXiv 2312.17296v9): method/algorithm, variants, baselines, training
setup, and the TREC/DBpedia/perplexity/burstiness numbers above; exact Qasper/HotpotQA F1 values live in
Appendix K and were not captured, so I did not quote them. Everything about our own method is grounded in
the two named code briefs and the source lines they cite. Done.
