# liu2020kbert — K-BERT: Enabling Language Representation with Knowledge Graph

Liu, Zhou, Zhao, Wang, Ju, Deng, Wang. AAAI 2020. arXiv:1909.07606 (v1, 17 Sep 2019).
Code + KGs: https://github.com/autoliuweijie/K-BERT. Grounded against the full PDF (all 8 pp.)
and our `paper/notes/code_briefs/masks.md` (commit 6134163) plus `related_work_notes.md`.

## What the paper actually does

K-BERT bolts a knowledge graph onto a *frozen, off-the-shelf* BERT-base without any extra
pre-training. At fine-tune/inference time it takes an input sentence, looks up KG triples
whose head entity appears in the sentence, and **splices those triples into the sentence as
depth-1 sub-branches**, producing a "sentence tree." The tree is then flattened back into a
token sequence and fed to a modified transformer. Two devices keep the injected triples from
corrupting the sentence ("knowledge noise", KN):

1. **Soft-position embedding.** After flattening, injected branch tokens sit physically
   between the anchor entity and the next original token, which scrambles word order. K-BERT
   fixes this by giving tokens *position indices that reflect the tree, not the flat order*:
   the token after the anchor keeps the position it would have had in the original sentence,
   and each injected branch is numbered as a continuation *off the anchor*. In their running
   example the flattened stream is "[CLS] Tim Cook CEO Apple is visiting Beijing capital China
   is_a City now" but "is" is given soft-position 3 (right after "Cook", pos 2) rather than 5,
   while the injected "CEO Apple" branch also gets positions 3,4. Two different tokens can
   therefore share a position index (hard-position indices are still unique, 0..12).

2. **Visible matrix (the "seeing layer").** A hard 0/−∞ mask `M` where `M_ij = 0` iff tokens
   `i,j` lie **in the same branch** of the sentence tree (`w_i ⊖ w_j`), else `−∞`. It is fed
   into a "mask-self-attention": `S = softmax((Q Kᵀ + M)/√d_k)`, `h = S V`. So an injected
   triple is only visible to the anchor entity and its own branch; e.g. "Apple" (injected off
   "Cook") cannot see "China" (injected off "Beijing"), and "[CLS]" cannot see "Apple"
   directly — but "[CLS]" still absorbs "Apple" *indirectly* across layers via "Cook", since
   information hops one branch-boundary per stacked mask-self-attention layer. This is
   explicitly the mechanism that lets knowledge "enrich the representation of [Cook] without
   directly changing the meaning of the original sentence."

**Setup / scale.** Chinese only, character-level tokens. BERT-base config: L=12, A=12, H=768,
110M params, identical to Google BERT so weights transfer verbatim. Crucially **no KG is
attached during pre-training** (they argue binding two entity names together during MLM
collapses their word vectors / causes semantic loss); the KG is enabled only at fine-tune and
inference. Pre-training corpora: WikiZh (1.2 GB, ~120M sentences) and their own WebtextZh
(3.7 GB Q&A). Three KGs: **CN-DBpedia** (5.17M triples, encyclopedic), **HowNet** (52,576
triples, a language/sememe KG), **MedicalKG** (13,864 triples, self-built medical hypernyms).

**Results that matter.** On 8 open-domain tasks the KG barely moves sentiment classification
(Book review / Chnsenticorp / Shopping / Weibo essentially flat — sentiment needs no world
knowledge), but helps knowledge-shaped tasks: MSRA-NER F1 rises from Google BERT 93.6 → K-BERT
(CN-DBpedia) **95.7** (+2.1), and note that adding the *extra corpus* WebtextZh alone only got
BERT to 94.6 — the KG beats more data. HowNet (language KG) helps semantic-similarity tasks
(XNLI test 75.4→76.1, LCQMC 86.2→87.0) while CN-DBpedia (encyclopedic) helps QA/NER. The real
gains are domain-specific (Table 3, F1): Finance Q&A 83.9→84.9, Law Q&A 86.4→87.5, Finance NER
86.1→87.6, Medicine NER 92.5→93.8 (CN-DBpedia) and **94.2** with the domain MedicalKG — "1~2%"
across the board. **Ablations (Fig. 5)** are the load-bearing part for us: removing the visible
matrix makes K-BERT on Law Q&A *worse than plain BERT*, directly demonstrating knowledge noise
and that the mask is what converts injected structure from harmful to helpful; removing
soft-position also hurts; and K-BERT converges faster (peak F1 at epoch 2 vs BERT's epoch 4).

## Methodology: theirs vs. ours

Same *genus* — "structure enters the model as a hard additive attention mask, applied
identically at train/fine-tune and inference, with no learned edge bias and no message
passing" — but a different *species* on almost every axis that matters to us.

- **Train-on-structure vs retrieve-at-inference.** Both of us are firmly train-on-structure:
  the mask is part of the forward graph in both phases, not a bolt-on retrieval hop. K-BERT's
  KG lookup (K-Query, exact entity-name match against a triple store) is deterministic and
  symbolic — the same *class* of resolution as our deterministic graph-edge / identifier
  resolution (`related_work_notes.md`: "deterministic graph-edge/identifier resolution rather
  than a learned similarity search"), *not* a learned retriever. The sharp difference is that
  K-BERT re-runs its lookup at inference on the *literal input text*, exactly as our
  **generation path uses text detection** (masks.md §"Option B": "GENERATION uses text
  detection (no graph shortcut)"), whereas our *training* path can shortcut with baked
  graph-edge grants (Option B: precomputed `link_to_target`).

- **The edge is an attention mask, not a GNN/KV/pair.** K-BERT is on our side of this axis:
  no GNN aggregation (unlike THU-ERNIE `zhang2019ernie`, GreaseLM, DRAGON), no cached-KV store
  (unlike Memorizing Transformers), no contrastive pair. Just `Q Kᵀ + M`. This is precisely
  why the lit review files it as "the closest attention-mask analog."

- **Symmetric-undirected vs asymmetric-causal-directed.** K-BERT's `M` is **symmetric**
  ("same branch" is a symmetric relation) and there is no causal triangle — it is an encoder
  (bidirectional MLM-style), so both directions of a branch see each other. Our mask is
  fundamentally *causal and directed*: `cross_doc_link: M=(q>=k)&(same_doc OR in_grant)` with
  an **asymmetric grant rectangle** (masks.md §Formal semantics: "ASYMMETRIC (A×B never
  transpose)") and a **DAG gate** that grants only *backward* links (`skip if
  target_start>=link_end_pos` → backward links only). K-BERT never faces link direction because
  a triple has no reading order relative to its anchor within one bidirectional context.

- **Intra-document triples vs cross-document links.** This is the biggest scope gap.
  K-BERT injects **KG triples inside a single short sentence** (tree depth *fixed to 1*, no
  iterative expansion, sequences are sentence-length). We grant **cross-document read access
  across a 32k-token packed traversal of a document graph**, where the "branch" is an entire
  linked node's text, not a `(rel, tail)` pair. Their "branch" ≈ our per-link *grant rectangle*,
  but ours can be up to a full 32k document and there are up to **256 grants** live per sequence
  (masks.md: production `max_grants=256`, bit-packed into 4 chunks), composed by **union/OR with
  no weighting** — the same "no precedence" property as K-BERT's binary same-branch relation.

- **The indirect-multi-hop mechanism is shared and worth stealing framing from.** K-BERT's
  "[CLS] reaches Apple through Cook across layers" is *exactly* our claim that stacked layers
  turn a one-hop visible-mask into effective multi-hop reading. Our masks.md doesn't
  frame it this way; K-BERT gives us a clean precedent for arguing that a depth-1 (or
  single-link) visibility grant yields multi-hop reasoning through layer composition, without
  materializing multi-hop edges in the mask.

- **Position handling — opposite choices, same problem.** K-BERT's *soft-position* deliberately
  **reuses/overlaps position indices** so an injected branch inherits its anchor's position,
  keeping the original sentence's relative geometry intact. We do the *opposite*: masks.md
  §Reviewer-attackable #1 flags that we **do NOT reset RoPE per doc** — "A reads B at relative
  offset = packing distance (arbitrary, traversal-order-dependent), not semantic." K-BERT is a
  direct antecedent for the reviewer concern in `related_work_notes.md` §"Boundary masking and
  position handling": it shows a model that *did* engineer positions so injected content lands
  at a semantically-sensible relative offset, and its ablation shows dropping that engineering
  ("w/o soft-position") measurably hurts. That is evidence *against* our no-reset choice that
  we should confront rather than ignore.

- **Compute-control philosophy.** K-BERT has no matched-FLOP control; it just toggles the KG on
  a fixed backbone. Our design isolates the linking inductive bias from raw FLOPs with
  `doc_concat_link` / `doc_concatenated` (masks.md §5). But K-BERT's *ablation* (w/o visible
  matrix = all tokens mutually visible) is spiritually our `doc_concatenated`-type control: it
  is the "same tokens, unrestricted attention" condition, and it *loses to the masked version
  and even to no-knowledge BERT*. That is the single most encouraging external result for our
  thesis.

## Predictions & open questions for our method

- **The mask should matter most where injected content is noisy or off-topic, and be neutral
  where the target is irrelevant.** K-BERT's cleanest finding: KG helps knowledge-shaped tasks
  (NER, QA, domain) and is ~flat on sentiment. Analogously we should predict our
  `cross_doc_link` win concentrates on multi-hop / cross-document-dependent eval (HotpotQA-style,
  cross-file code completion) and is ~zero on single-document controls
  (`related_work_notes.md` §"Single-document controls": HellaSwag/ARC/PIQA…). If our
  cross-doc mask moved single-doc control accuracy, that would signal a leak/artifact, not a win.

- **Unrestricted cross-document attention can be *net harmful*, and the mask is what rescues it.**
  K-BERT's "w/o visible matrix < BERT" on Law Q&A is independent corroboration of
  `zhao2024analysing` (packed cross-doc attention is harmful) — and predicts that our
  `doc_concatenated` compute-control (most attention, no link gating) may actually *underperform*
  `cross_doc_link` on knowledge-shaped tasks, not merely match it. If instead `doc_concatenated`
  ties or beats `cross_doc_link`, the win is FLOPs/adjacency, not the link edge. K-BERT says the
  *gating* is the active ingredient.

- **Faster convergence.** K-BERT peaks at epoch 2 vs BERT's 4. A weak prediction: the
  link-grant may improve *sample efficiency* on cross-doc-dependent eval, visible earlier in a
  WSD run than a final-loss gap — worth logging Δnll vs step, not just terminal Δnll.

- **KG-type/task match → link-type/eval match.** K-BERT's HowNet-helps-similarity vs
  CN-DBpedia-helps-QA split predicts that *which* edges we pack (import edges vs citations vs
  hyperlinks) will differentially help different eval families; a citation-graph model need not
  help code completion. Encourages reporting per-corpus, not pooled, effects.

- **Open question our design may resolve for them:** K-BERT is capped at depth-1 trees and
  admits it. Our stacked-layer multi-hop-through-one-hop-grant (their own [CLS]→Cook→Apple
  argument) plus recursive link-following generation is the natural test of *how far* a
  single-hop visible grant propagates in a decoder — a question their sentence-length,
  depth-1 setting could not probe. Conversely, **their open question they leave us:** they never
  isolate soft-position from the visible matrix jointly vs a position-reset baseline — exactly
  the RoPE-reset ablation we should run.

## Gotchas

- **Knowledge noise is real and can go negative.** The headline warning: adding structural
  visibility *without* gating made their model worse than the no-knowledge baseline. For us this
  means an un-tuned or overly-permissive grant (e.g. `whole_doc_grant=True` granting a whole
  source doc, our `doc_concat_link` control) risks *degrading* results, and a bug that widens
  grants (off-by-one in `link_end_pos`, masks.md §Reviewer-attackable #3; or union-blowup past
  the 256 cap, #4) could silently flip a win to a loss. Treat any regression as possible
  KN before assuming a code bug.

- **Position engineering is not free.** Their soft-position ablation dropped performance; we
  chose the opposite (no reset) largely for train/inference symmetry and kernel simplicity. If a
  reviewer cites K-BERT's soft-position as evidence that injected content needs a controlled
  relative offset, we need our own ablation (reset vs no-reset RoPE across the grant boundary)
  ready. This is a concrete tuning trap: the effect may be entangled with *where* the fetched
  doc lands positionally, not just *whether* it's visible.

- **Task selection can manufacture or hide the effect.** Four of their eight open-domain tasks
  (all sentiment) show ~nothing; had they reported only those, the method would look dead; had
  they reported only Medicine NER, it would look huge. This is the eval-artifact risk our
  `related_work_notes.md` §"Evaluation-methodology backbone" already worries about
  (`schaeffer2023mirage`, `biderman2024lessons`): pick the eval suite *before* seeing results and
  report the neutral controls, or the cross-doc effect size is not credible.

- **Encoder→decoder transfer is not guaranteed.** Every K-BERT number is BERT-base MLM,
  bidirectional, sentence-length, Chinese, character-level. Their symmetric visible matrix has no
  causal-triangle interaction; our mask must compose visibility *with* causality
  (`M=(q>=k)&…`), and masks.md §Novel #2 documents a *real NaN* (LSE collapse, dQ≈5.7e4 on
  thestack) caused by the interaction of block-level same-doc masking with non-monotonic doc
  labels — a failure mode K-BERT's regime simply cannot exhibit. Do not assume their "it just
  works, load BERT weights" ease transfers to a from-scratch causal 32k regime.

- **"No KG in pre-training" is a deliberate choice with a reason.** They argue co-training the KG
  binds entity vectors and causes semantic loss. We *do* train the link edge during pretraining.
  Their caveat is a hypothesis worth checking: is there any representational collapse from
  training-time cross-doc grants (e.g. linked docs' representations becoming pathologically
  similar)? Probably not at our scale/granularity, but it's their stated reason for a design we
  invert.

## Missed citations worth adding

Checked against `paper/bib/refs.bib` (533 entries). K-BERT's own bibliography is mostly Chinese
task datasets and BERT-optimization papers not relevant to us. Genuinely-missing and relevant:

- **joshi2019spanbert** — SpanBERT, "Improving Pre-training by Representing and Predicting
  Spans", arXiv:1907.10529. *Not in refs.* Masks contiguous random spans and adds a **span
  boundary objective** (predict a span from its boundary tokens). Directly relevant to our
  span/boundary-masking and FIM discussion (`related_work_notes.md` §"Boundary masking",
  §"Fill-in-the-middle") — a span-boundary training signal is a natural neighbor to our
  link-boundary (`link_end_pos`) grant semantics.

- **bosselut2019comet** — COMET, "Commonsense Transformers for Automatic Knowledge Graph
  Construction", arXiv:1906.05317. *Not in refs.* Trains GPT on KG triples-as-text to
  *generate* KG edges. Relevant as the generative-KG contrast: same spirit as `agarwal2021kelm`
  (verbalize the KG) and `taylor2022galactica` (generate a reference/edge), i.e. structure
  memorized in parameters vs our explicit attention edge — a clean addition to the
  "graph-as-text vs graph-as-mask" contrast set already in the KG-LM cluster.

Lower-confidence / probably skip: the foundational **BERT (Devlin et al. 2018/2019, NAACL
2019, arXiv:1810.04805)** appears absent as a standalone entry (refs has GPT-2, GPT-3,
Transformer, but I found no `devlin*bert` key) — K-BERT builds directly on it, and several of
our own cited encoder works (LinkBERT, GraphCodeBERT, SciBERT) presuppose it, so its absence may
be an oversight worth flagging; but since our model is decoder-only it may be a deliberate
omission. Flagging, not asserting. RoBERTa / XLNet / Baidu-ERNIE / SpanBERT's stablemates are
BERT-optimization papers not specific enough to our thesis to add.

---
Confirmed against the full PDF and refs.bib; direction/scope contrasts and result numbers are quoted from the source, and citation-absence claims were grep-verified (please re-verify keys before adding).
