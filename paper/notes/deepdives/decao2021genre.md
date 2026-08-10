## decao2021genre — Autoregressive Entity Retrieval (GENRE)

De Cao, Izacard, Riedel, Petroni. ICLR 2021 (spotlight). arXiv:2010.00904.
The direct antecedent of our trie-constrained link-target generation (`eval/link_annotator.py:TrieTitleIndex`,
`eval/title_index.py`). Read against arXiv abstract + ar5iv full-text HTML; code claims verified against
our source at the commit the briefs cover.

### What the paper actually does

**Core idea.** Instead of treating entity retrieval as classification over a fixed label set with dense
entity vectors (DPR/BLINK-style dot-product scoring), GENRE *generates* the entity's unique name
left-to-right, token-by-token, autoregressively. It is "the first system that retrieves entities by
generating their unique names." Scoring is `score(e|x) = ∏_i p_θ(y_i | y_<i, x)` — an exact per-token
softmax with teacher forcing, so there is no negative-sampling / hard-negative-mining step at all. Because
the model is a full seq2seq cross-encoder over (context, name), it captures fine-grained context↔entity
interaction that a single dot product misses.

**Base model.** BART (Lewis 2020a), 406M params (Table 4), fine-tuned with a standard seq2seq objective
(maximize output likelihood, teacher forcing), dropout, label smoothing 0.1, Adam lr 3e-5, fairseq. It is
*not* trained from scratch — it explicitly leverages BART's pretrained LM.

**The constraint — prefix trie (this is our mechanism).** To guarantee the generated string is a valid
entity name, decoding is restricted by a prefix tree 𝒯 (trie) whose nodes are annotated with vocabulary
tokens; a node's children enumerate the *allowed* next tokens. Built over all ~6M Wikipedia titles under
the BART tokenizer, the trie has ~6M leaves, ~17M internal nodes, ~600MB on disk. Decoding is
**Constrained Beam Search** (10 beams for ED/DR, 6 for EL): at each step invalid tokens' log-probs are
masked out and the beam explores only in-trie continuations. Crucially, beam cost is independent of the
number of entities |ℰ| — it depends only on beam size and average title length (~6 BPE tokens).

**Input formats.** Entity disambiguation (ED): the single mention in the source document is flagged with
`[START_ENT]`/`[END_ENT]` markers; target = the entity's textual name. End-to-end EL: *dynamic
markup-constrained decoding* with a 3-state automaton (outside a mention / inside a mention / inside an
entity link), emitting markup like `[Leonardo](Leonardo da Vinci)`. Document retrieval (KILT): input =
query, output = the title of the page to retrieve.

**Results that matter.**
- ED (Table 1): avg micro-F1 **88.8** across AIDA-CoNLL (in-domain) + MSNBC/AQUAINT/ACE2004/WNED-CWEB/
  WNED-WIKI (out-of-domain), +0.8 over the next-best. Ablations: BLINK-data-only 88.1; **no candidate set
  85.1; no trie constraints 79.6** (constraint worth ~+9 F1); numerical IDs instead of names **−20.3** avg
  (names carry compositional signal — a key finding for us).
- End-to-end EL (Table 2, GERBIL): avg **58.2**, with large jumps on some sets (+13 F1 Derczynski, +4.7
  KORE50) but weak on OKE15/16 (needs coreference it wasn't trained for).
- KILT document retrieval (Table 3): avg R-precision **69.7** vs BLINK+flair 56.0, RAG 47.3 — **+13.7**
  R-prec over best baseline, one model trained on BLINK + all KILT tasks jointly.
- **Memory (Table 4, the headline efficiency claim):** GENRE 2.1GB vs BLINK 30.1GB (~14×), DPR 70.9GB
  (~34×), RAG 40.4GB. The retrieval "index" is 17M params vs 15B (DPR/RAG), 6B (BLINK) — because the
  parametric footprint scales with *vocabulary*, not entity count.
- Unseen entities: a new entity is added by simply appending its name to the trie (no re-encoding, no
  re-indexing). Cold-start on 50 new-2020 Wikipedia pages: 19/50 exact-name hits (all correct) + 14/31 of
  the rest. WikilinksNED Unseen-Mentions: 64.4 (seen) vs 63.2 (unseen) accuracy — nearly flat.

### Methodology: theirs vs. ours

**Shared machinery (the real, mechanical overlap).** Our `TrieTitleIndex` (eval/link_annotator.py:154) is
GENRE's constrained decoder, essentially port-for-port: a BPE-level prefix trie built over every corpus
title at construction time (`tokenizer.encode(raw)` inserted as a trie path, `_TrieNode.children` keyed by
token id, first-inserted wins on collision), then at each generation step the next token is restricted to
`node.children` — exactly "mask log-probs of invalid tokens." We even added beam search (`beam_width`) and
Wu-style length normalization (`length_penalty` α, `score = joint_logprob / n^α`) that GENRE's constrained
beam search motivates; our default `beam_width=1` is greedy trie traversal, GENRE ran 10/6 beams. Our
interior-leaf handling (compare P(")") against the best valid child to decide whether to stop at a title
that is itself a prefix of a longer title) is the concrete analogue of GENRE's trie having internal nodes
that are also valid leaves.

**Where we diverge — the load-bearing axis.** GENRE is *retrieve-at-inference and stop*: the generated
title **is the answer**. The whole system's job is to emit the right identifier; nothing reads the target
entity's article text. TS2TS is *retrieve-then-attend, and it's the same edge we trained on*. In our
generation loop the generated title is a **fetch key**: `_handle_link` (generation_loop.py:216) resolves it
via `corpus.get_document`, then `context.add_corpus_doc(before_entry=active)` physically inserts the target
document's tokens *before* the linking doc in the packed sequence, and the next `build_sequence` +
`forward_inference` lets the cross-document mask grant the linking positions read-access into the fetched
content (cf. generation_retrieval.md "prepend into attention context is LITERAL"). The title is a means;
the target's tokens entering attention is the end. GENRE has no such step — this is precisely the sentence
in related_work_notes.md ("the generated title *is* the answer, whereas for us the title is a fetch-key
that pulls content into attention").

Three further mechanistic contrasts:

1. **Trained edge vs. inference-only decoder.** GENRE's trie constraint is a pure *decoding-time* device
   bolted onto a seq2seq model fine-tuned on (context→name) pairs; the model never learns a
   document-to-document attention edge. Our trie is *also* inference-only, but it feeds the
   train/inference-shared link machinery (CrossDocLinkMaskCreator, `index_doc_span` match key —
   link_detectors.md, generation_retrieval.md §"Train/inference mirror"). Our novelty is that the
   fetch-and-attend the trie triggers is the *same* cross-doc grant used in pretraining; GENRE's novelty is
   that generation replaces classification. Orthogonal contributions that compose.

2. **Where the trie lives in our pipeline.** GENRE's trie is the whole retrieval system. Ours is one of
   several link-resolution paths and is *optional*: `TrieTitleIndex` guarantees a valid corpus title on
   success, else falls back (`min_joint_logprob` threshold → free generation + `HashNormTitleIndex`
   cascade: exact/norm/word-overlap/edit-distance, title_index.py). GENRE has no fuzzy fallback — the trie
   is hard. Our design tolerates a title the model wants to emit that isn't in the trie by degrading to
   nearest-match; GENRE would simply never emit it. (This matters because our detectors also handle code
   imports/citations where the "title" is a path or bibkey, not a Wikipedia name — the trie is the wiki
   instance of a more general resolution layer.)

3. **Name-as-identifier finding transfers directly.** GENRE's "numerical IDs −20.3" ablation is the
   strongest external evidence for a design choice we already made: our link targets resolve on
   *human-readable titles / paths / bibkeys* (`index_doc_span` returns strings, link_detectors.md), not
   opaque node ids. GENRE proves the compositional/generalization value of that choice on real retrieval
   numbers. The DSI/NCI line (tay2022dsi, wang2022nci) that generates structured numeric doc-ids is the
   counter-camp; GENRE is on our side of that axis.

**Not comparable:** GENRE is encoder-decoder (BART), symmetric-ish, short-context, single generated name;
we are decoder-only, causal, 32k-context, and the "generation" continues *through* the fetched document.
GENRE has no KV/cache story because it generates ~6 tokens; our inference is O(T³)-ish full recompute with
no KV cache (generation_retrieval.md §KV CACHE=NONE) — a different regime entirely.

### Predictions & open questions for our method

- **The trie constraint is worth a lot, but less for us than for GENRE.** GENRE's "+9 F1 from constraints,
  −20 from IDs" says the constraint is doing heavy lifting *when the generated string is the final answer*.
  For us the title is only a fetch key and there's a fuzzy fallback, so we should expect the trie's marginal
  value to be smaller — its real payoff is *precision of the fetched target* (fewer wrong-document fetches),
  not end-task accuracy directly. Prediction: ablating `TrieTitleIndex` → free-gen+HashNorm should hurt
  *fetch precision / wrong-neighbor rate* more than downstream Δnll, and hurt most on titles with many
  near-duplicate corpus neighbors (disambiguation-heavy wiki).
- **Constraint helps exactly where the corpus is large and title space is dense.** GENRE's gains are
  largest out-of-domain and on rare entities. Expect our trie to matter most on the wiki corpus (6M-ish
  dense title space, ambiguous surface forms) and least on arxiv/code where `index_doc_span` keys
  (bibkeys, import paths) are already near-unique and exact-match resolution dominates.
- **Unseen-target robustness should hold.** GENRE adds new entities by appending names to the trie with no
  re-indexing (64.4 vs 63.2 seen/unseen). Our trie is rebuilt from `raw_identifiers` at construction with
  no learned entity vectors, so we inherit the same "new corpus doc = one more trie path" property for
  free — a genuine advantage to state over embedding-index RAG baselines.
- **Beam width is a lever we've under-explored.** GENRE used 10 beams; our default is greedy (beam_width=1).
  Their comment that a short high-first-token title can be wrongly preferred over a longer correct one is
  exactly the failure our `beam_width>1` + `length_penalty` docstring describes ("'25' beaten by 'New
  Hampshire'"). Prediction: for multi-token ambiguous wiki titles, beam_width 4-10 with α≈0.6 should
  measurably raise correct-fetch rate; for arxiv/code it should be near-neutral.
- **Open question we may resolve for them:** GENRE stops at the name and never reads the entity's
  description at inference (it's baked into BART weights or ignored). Our fetch-and-attend directly tests
  whether reading the *target document's tokens* after generating its title beats memorizing entity facts
  parametrically — a question GENRE explicitly leaves open (their "cross-encode context and entity" is only
  the name, not the article body).

### Gotchas

- **Trie tokenizer coupling.** GENRE's trie is BART-BPE-specific; ours is built via `tokenizer.encode` and
  is only valid for the exact tokenizer used at construction. link_detectors.md already flags that markdown
  detection hardcodes GPT-2 token id 16151 — same class of brittleness. A tokenizer swap silently invalidates
  the trie. Guard: rebuild the trie with the eval tokenizer, never a cached one from another vocab.
- **The trie is memory/time-heavy to build.** GENRE's Wikipedia trie is 600MB / 6M leaves / 17M nodes. Our
  `TrieTitleIndex.__init__` tokenizes *every* corpus title eagerly; title_index.py's own docstring warns
  it's unsuitable "when the corpus is too large to build a trie." For full-wiki eval, expect a real build
  cost and RAM footprint; that's why `HashNormTitleIndex` (lazy hashmap) exists as the large-corpus path.
- **Collision silently drops titles.** Two titles tokenizing to identical id sequences → first-inserted
  wins, rest lost (our code, and structurally the same in GENRE since a trie path is unique). Rare, but on
  Unicode/whitespace-variant titles it means a fetch can never resolve to the shadowed title. Matches the
  arxiv "byte-identical title" brittleness in link_detectors.md.
- **Constraint hides model errors as valid-looking outputs.** GENRE's constrained decoding *always* returns
  a real title even when the model is confused — the string looks legitimate but points at the wrong entity.
  For us this is worse: a confidently-wrong trie title fetches the *wrong document into attention*, silently
  poisoning the context. Their "no constraints 79.6 vs constrained 88.8" is the flip side — but the failure
  mode of a constraint is exactly a plausible wrong answer, not an obvious garbage one. Our
  `min_joint_logprob` prune is the only guard; tune it, and log fetch-provenance for eval.
- **EL coreference blind spot.** GENRE tanked on OKE15/16 because it wasn't trained to resolve pronouns/
  coref to entities. Our link detection fires on explicit surface link syntax (markdown `](`, `\cite{}`,
  imports), so anaphoric "it"/"the paper" links are simply invisible to us too — don't claim coverage there.
- **Contamination via BART pretraining.** The authors explicitly caution that BART may have seen some
  "unseen" entities during pretraining, muddying the seen/unseen gap. We train from scratch, so we're
  cleaner on this axis — a point worth making, not a trap for us, but a reason not to over-cite their
  unseen-entity numbers as a clean generalization result.

### Missed citations worth adding

Verified absent from `paper/bib/refs.bib` (grepped). All are cited *by GENRE* and relevant to our
trie-constrained-generation / entity-linking framing:

- **Hokamp & Liu 2017** — "Lexically Constrained Decoding for Sequence Generation Using Grid Beam Search"
  (ACL 2017; arXiv:1704.07138). The origin of hard lexical constraints in beam search — the direct
  algorithmic ancestor of GENRE's *and our* trie-constrained beam search. Strongest missing cite for our
  constrained-decoding paragraph.
- **Post & Vilar 2018** — "Fast Lexically Constrained Decoding with Dynamic Beam Allocation for NMT"
  (NAACL 2018; arXiv:1804.06609). The efficient-constrained-decoding follow-up; pairs with Hokamp & Liu to
  situate `TrieTitleIndex`'s beam mechanics in the constrained-decoding lineage (currently our lit review
  has only grammar/regex-constrained decoding: picard, gcd, outlines, syncode — not the lexical-constraint
  beam-search origin).
- **Ganea & Hofmann 2017** — "Deep Joint Entity Disambiguation with Local Neural Attention" (EMNLP 2017;
  arXiv:1704.04920). The canonical neural ED baseline GENRE compares against; our entity-linking subsection
  (letitov2018, kolitsas2018, hoffart2011) is missing this foundational one.
- **Raiman & Raiman 2018 (DeepType)** — "DeepType: Multilingual Entity Linking by Neural Type System
  Evolution" (AAAI 2018; arXiv:1802.01021). Type-constrained EL; a distinct "constrain the output space by
  structure" flavor adjacent to our trie idea.
- **van Hulst et al. 2020 (REL)** — "REL: An Entity Linker Standing on the Shoulders of Giants" (SIGIR
  2020). A practical end-to-end EL system baseline; useful if we position against deployed EL pipelines.

Lower priority / probably out of scope: Broscheit 2019 (BERT end-to-end EL), Mulang' et al. 2020 (KG-context
for ED), Nogueira et al. 2020 (seq2seq doc ranking). Note mGENRE (the multilingual GENRE follow-up) is *not*
cited by this paper (later work) — do not add it as a GENRE reference; flagging only so it isn't confused
with an internal cite.

---
Confirmed: method/results drawn from arXiv abstract + ar5iv full-text; code claims verified against
eval/title_index.py, eval/link_annotator.py (TrieTitleIndex), model/graph_traversal/cross_doc_mask.py, and
the two code briefs; missing-citation absences verified by grep over paper/bib/refs.bib.
