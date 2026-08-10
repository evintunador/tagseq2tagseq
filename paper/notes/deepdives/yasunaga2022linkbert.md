# yasunaga2022linkbert — LinkBERT: Pretraining Language Models with Document Links

Yasunaga, Leskovec, Liang (Stanford), ACL 2022. arXiv:2203.15827. Bib key already in
`paper/bib/refs.bib:181`. This is the canonical "train on hyperlinked document pairs"
antecedent and the sharpest single-paper comparison for our linking inductive bias.

Sources for this deep-dive: arXiv abstract + the ar5iv HTML render of the full paper
(method, tables, ablations); cross-checked against our code briefs
(`masks.md`, `data_pipelines.md`, `traversal.md`) and `related_work_notes.md`. Numbers
below are quoted from the ar5iv render; where a figure could not be independently
re-derived it is marked as such.

## What the paper actually does

**Core idea.** Standard LM pretraining (BERT) treats each document in isolation and never
sees dependencies that span documents. LinkBERT treats the corpus as a *graph of
documents* (nodes = documents, edges = hyperlinks / citations) and builds pretraining
inputs by **placing two documents that are linked into the same context window**, so the
model can attend across the link during masked-LM prediction.

**Input construction.** Each input is a pair (Segment A, Segment B), BERT-style
`[CLS] A [SEP] B [SEP]`, 512 tokens. After sampling anchor Segment A, Segment B is drawn
one of three ways, uniformly (**1/3 each** on Wikipedia and PubMed):
- **Contiguous** — the text that actually follows A in the same document.
- **Random** — a segment from a random other document.
- **Linked** — a segment from a document that A hyperlinks/cites to.
(BookCorpus, which has no links, uses only contiguous/random at 50/50.)

**Which linked neighbor.** Not uniform over neighbors. They sample a linked document with
probability **inversely proportional to its in-degree**, to avoid over-representing hub
pages (their example: everything links to "United States"). This "diversity" step is an
explicit use of graph centrality; ablating it costs ~1%.

**Two objectives, trained jointly.**
1. **MLM** — standard masked-token prediction (self-supervised token-fill).
2. **Document Relation Prediction (DRP)** — a 3-way classification off the `[CLS]`
   representation predicting whether B is *contiguous*, *random*, or *linked* relative to
   A. This is the paper's new objective; it replaces/generalizes BERT's NSP.

The stated motivation is twofold: MLM over a linked pair teaches *multi-hop knowledge*
(A's masked tokens can be recovered using facts in B), and DRP teaches *document-relevance
structure*.

**Architecture / scale.** It is an **encoder** (bidirectional, BERT-family). Sizes:
tiny 4.4M (from scratch), base 110M, large 340M — base/large are *continued* from released
BERT checkpoints, not trained from scratch. Corpus = Wikipedia+BookCorpus (general) with
Wikipedia hyperlinks added; BioLinkBERT uses PubMed abstracts (21GB) with citation links.
Pretraining is short (base/large 40k steps general; biomed base 62.5k steps at batch 8192).

**Results that matter (quoted from ar5iv tables).**
- General GLUE avg (base): 79.2 → **79.6** (large 81.1). GLUE gains are marginal — GLUE is
  single-sentence/pair, so cross-doc structure barely helps.
- MRQA extractive-QA F1 avg (base): 75.2 → **77.8** (large 81.0). Bigger multi-document
  gains: HotpotQA 76.0→**78.2**, TriviaQA 70.3→**73.9**, SearchQA 74.2→**76.8**.
- BioLinkBERT is where it shines: BLURB score 81.10→**83.39** (large 84.30); BioASQ
  87.56→**91.43** (large 94.82); PubMedQA 55.84→**70.20**; MedQA-USMLE 38.1→**40.0**
  (large 44.6) — new SOTA on several BioNLP tasks.
- Abstract headline: "+5% absolute on HotpotQA and TriviaQA" (few-shot) and "+7% on BioASQ
  and USMLE."

**Ablations (the load-bearing ones for us).**
- **Link quality matters, not just co-occurrence.** Replacing hyperlinks with *random*
  links = same as BERT (−4.1% avg, tiny): merely concatenating two docs buys nothing. TF-IDF
  (lexical-similarity) links recover most of the gain (−1.8% vs hyperlinks but +2.3% vs
  BERT); true hyperlinks are best because they carry salient, non-lexical relatedness.
- **DRP helps, concentrated on multi-doc tasks.** Removing DRP: HotpotQA 78.2→76.5,
  SQuAD-distractor 89.6→87.0; single-doc tasks barely move.
- **Diversity (inverse-in-degree) sampling** worth ~1%.

## Methodology: theirs vs. ours

**The shared thesis — and the one-sentence divergence.** Both projects reject the
isolated-document assumption and use *document links to decide what co-occurs in one context
so the model can read across them during pretraining*. LinkBERT establishes empirically that
this beats both isolated pretraining and random co-occurrence. That empirical result is the
foundation our compute-control design rests on. **But LinkBERT's link only controls
co-occurrence (which two docs land in the window); once they are in the window, attention is
fully dense and symmetric. Our link is itself the compute primitive: a directed, causal,
block-sparse attention *grant* applied in both training and inference.** That is the entire
axis of our contribution relative to this paper.

Concretely, mapping onto our code:

- **Selection vs. mechanism.** LinkBERT = *link-as-selection* only. Our
  `data_pipelines.md` / `traversal.md` show link-as-selection too — graph traversal
  (`traversal.py`: BFS/DFS/RandomWalk over `neighbors_out`/`neighbors_in`) picks which docs
  pack into a 32k sequence, and `prefer_targets_first` topo-orders them. But we *additionally*
  apply `cross_doc_link` (`masks.md`), a per-link attention grant: rows
  `[link_end_pos, A.end) × cols [B.start, B.end)` — the linking doc A gets read-access into
  target B *from the link position onward*. LinkBERT has no analogue; its cross-doc reading is
  just default full attention inside the 512-window.

- **Pair vs. graph-packed super-document.** LinkBERT is strictly *pairwise*: exactly one A and
  one B, 512 tokens, one hop. We pack a *traversal* of many topologically-close docs into 32k
  and light up *many* directed edges at once (production `max_grants=256`, `masks.md`), so a
  single sequence can carry a multi-hop neighborhood. Their "multi-hop reasoning" gains come
  from the pretrained model generalizing, not from multi-hop context in a single pretraining
  input.

- **Encoder/MLM/symmetric vs. decoder/causal/directed.** LinkBERT is a bidirectional encoder
  with a symmetric relation (A↔B attend freely) and an MLM+DRP objective. We are decoder-only,
  causal (`M = (q>=k) & ...`), and the grant is **asymmetric** — A reads B, never the
  transpose — with a **DAG gate** (`cross_doc_mask.py:417-423`) that only grants *backward*
  links (target must start before the link position). LinkBERT's DRP is a discriminative
  auxiliary head; we have no relation-prediction loss at all — our "link objective" is
  entirely realized through the attention topology plus the standard next-token loss.

- **Train/inference symmetry.** LinkBERT's link machinery exists *only at pretraining time*;
  downstream, it is a plain fine-tuned encoder with no link mechanism. Ours is used
  *identically at inference* — a generated link deterministically fetches its target node into
  the attention context (`masks.md`: generation uses text detection rather than baked Option-B
  grants, but the same `link_to_target` semantics). This train=inference-linking property has
  no counterpart in LinkBERT.

- **Diversity sampling vs. traversal weighting.** LinkBERT's inverse-in-degree neighbor
  sampling is the closest thing they have to our traversal-strategy knobs. Our RandomWalk has
  `w_in`/`w_out`/`restart_prob` (`traversal.md`) but restart teleports to a *uniform* node
  (not RWR), and our seed is uniform rejection sampling regardless of strategy. Neither of us
  is doing personalized-PageRank; both bias toward diversity, but LinkBERT does it explicitly
  per-edge and we do it implicitly via walk dynamics + uniform seeding.

- **Compute-control alignment.** Our `doc_concatenated` / `doc_concat_link` masks
  (`masks.md`) are matched-FLOP controls that isolate the *linking bias* from raw co-occurrence
  FLOPs. LinkBERT's *random-link* ablation is the moral equivalent of our concat control at the
  *selection* layer: it shows co-occurrence alone (random B) = baseline, and only *relevant*
  links help. We should cite this as prior evidence that the effect is not a co-occurrence
  artifact — but note theirs controls *selection quality*, ours controls *attention
  connectivity at fixed selection*. They are complementary controls, not the same one.

## Predictions & open questions for our method

- **Expect the effect concentrated on genuinely cross-document tasks, near-zero on
  single-doc.** LinkBERT's GLUE (single-sentence) barely moved while MRQA/HotpotQA/BioASQ
  moved a lot. This directly predicts our single-document controls
  (`related_work_notes.md` §6: HellaSwag/ARC/PIQA/WinoGrande/OpenBookQA/BoolQ) should be
  **flat**, and the effect should surface on HotpotQA / 2WikiMultiHop / MuSiQue and on
  cross-file code completion (RepoBench/CrossCodeEval). If we see gains on single-doc
  controls, suspect a leakage/compute confound, not the link edge.

- **Expect large gains in a knowledge-dense, well-linked domain (biomed/citations); modest in
  general web.** BioLinkBERT's jumps (PubMedQA +14, BioASQ +4) dwarf general GLUE. Prediction:
  our **arXiv citation** corpus and **Python import** graph (dense, semantically load-bearing
  edges) should show a *stronger* linking effect than **enwiki** (avg out-degree 8.44 but many
  hub-y, low-salience links; `data_pipelines.md`) — and much stronger than enwikisource
  (avg deg 0.17, almost no realizable structure).

- **Link *quality/salience* dominates link *quantity*.** Their random-link = BERT result is a
  warning: if our realized edges are noisy (danglers, hub links, non-deterministic import
  resolution — `data_pipelines.md` reviewer-attackable #3/#6), the linking bias could wash out
  to the concat baseline. Prediction: our effect size is bounded by *edge precision*, so the
  arXiv `\cite` edges (explicitly resolved, out-of-corpus cites removed) should behave better
  than the wiki edges (resolution fraction *unmeasured*, `fix_mediawiki_links` invents
  non-matching ids → guaranteed danglers).

- **Diversity/hub down-weighting may matter for us too.** LinkBERT's ~1% from inverse-in-degree
  sampling suggests hub targets (Wikipedia "United States"; Python `__init__.py`/generated SDK
  files that dominate our `links_in_repo>=2` filtered set, `data_pipelines.md` #5) add little
  and may crowd out informative edges. Worth an ablation: down-weight high-in-degree targets in
  traversal seeding/growth and see if the cross-doc gain rises.

- **Open question they leave that our design can answer.** LinkBERT cannot express *multi-hop
  context in a single pretraining input* (pairwise, 512 tokens) and cannot follow a link *at
  inference*. Our 32k graph-packed sequence + inference-time link-fetch can test whether
  *true* multi-hop-in-context (not just generalization from pairwise pretraining) yields
  further gains — a question their pairwise setup structurally cannot pose.

- **Open question of ours their result informs.** Does DRP-style structure signal add anything
  beyond the attention grant? LinkBERT shows a *relation-prediction auxiliary loss* helps
  specifically on multi-doc tasks (HotpotQA 78.2→76.5 without it). We have no such loss. It is
  worth considering whether an auxiliary "is this a real edge" signal would add to our
  attention-only formulation, or whether the causal grant already subsumes it.

## Gotchas

- **Co-occurrence alone is a null result — you must beat the concat control.** Their random-link
  ablation lands exactly on BERT. If our `cross_doc_link` does not beat `doc_concatenated` /
  `doc_concat_link` at matched compute, we have only reproduced "put related docs together,"
  which LinkBERT already showed helps *only when links are relevant*. The matched-compute
  controls in `masks.md` are non-optional for the claim.

- **GLUE-style aggregate metrics hide the effect.** LinkBERT's headline GLUE gain is ~0.4 —
  easy to dismiss. The real signal is in the multi-doc subset. For us this argues for the
  paired per-token Δnll on cross-doc-sensitive positions (`related_work_notes.md` §6 eval
  backbone: schaeffer2023mirage / biderman2024lessons) rather than a single averaged accuracy,
  which would dilute a task-localized effect into noise.

- **Wikipedia hyperlink contamination / hub bias.** They needed inverse-in-degree sampling to
  keep hubs from dominating. Our pipeline does *not* down-weight hubs, and our wiki resolution
  fraction is unlogged (`data_pipelines.md` #6). Two risks: (a) hub-dominated traversals waste
  the 256-grant budget on low-salience edges (positional truncation drops later links,
  `masks.md`), (b) unmeasured dangling rate means realized density < raw graph
  (`traversal.md`: out-of-index neighbors silently dropped). **Quantify realized edge density
  before reporting** — LinkBERT's random-link null shows a low-precision edge set can silently
  collapse the effect to baseline.

- **HotpotQA is a 2017 Wikipedia benchmark → leakage.** LinkBERT trains on Wikipedia and
  evaluates on HotpotQA/Wikipedia multi-hop; the same corpus supplies both. Our
  `related_work_notes.md` already flags the HotpotQA-2017 leakage caveat
  (gong2025phantomwiki / schnitzler2024morehopqa). Since we also train on enwiki, any HotpotQA
  gain is confounded with memorization — prefer contamination-free / final-hop-hardened
  variants for the headline claim.

- **Encoder→decoder transfer is not automatic.** All of LinkBERT's evidence is bidirectional
  MLM. DRP reads the `[CLS]` token, which has no decoder-causal analogue. Do not assume their
  DRP finding transfers; our causal, asymmetric grant is a different mechanism and needs its own
  ablation.

- **Short continued-pretraining.** base/large are 40k-step *continuations* of BERT, not
  from-scratch. Their effect is a fine-tune-scale delta on top of a strong init. We train from
  scratch, so our effect must survive the full pretraining regime — do not calibrate expected
  effect size to their (post-hoc, small-budget) deltas.

## Missed citations worth adding

Checked against `refs.bib` (534 entries). The paper's own key references were largely already
present. Confirmed **already in refs.bib**: guu2020realm, beltagy2020longformer, caciularu2021cdlm,
yang2018hotpotqa, and the wiki/biomed corpora infrastructure. The following are cited by LinkBERT
and relevant to us but I could **not** confirm in refs.bib — flagged as *possibly missing* (verify
before adding; I did not exhaustively grep every alias):

- **pubmedbert / gu2021pubmedbert** — "Domain-Specific Language Model Pretraining for Biomedical
  NLP" (Gu et al., 2021, arXiv:2007.15779), the source of the **BLURB** benchmark that
  BioLinkBERT reports on. If we run any biomedical cross-doc citation eval, BLURB + its
  from-scratch-domain-pretraining baseline is the reference point. Grep `blurb`/`pubmedbert`
  first.
- **jin2021medqa / MedQA-USMLE** — "What Disease does this Patient Have?" (Jin et al.,
  arXiv:2009.13081), the USMLE multi-hop medical-QA benchmark where BioLinkBERT's cross-doc
  effect is largest (+6.5 at large). A candidate multi-hop eval for us in the biomed domain.
- **fisch2019mrqa / MRQA-2019 shared task** — the extractive-QA suite LinkBERT reports its
  clearest general-domain gains on (HotpotQA/TriviaQA/SearchQA all live here). If we adopt any
  of these as cross-doc reading evals, the MRQA task definition is the canonical citation.
  (yang2018hotpotqa is present, but the MRQA umbrella may not be.)

These are all *biomed/QA-benchmark* infrastructure rather than mechanism papers, so their
priority is low unless we add a biomedical-citation corpus or an MRQA-style eval track. The
mechanism-level neighbors of LinkBERT (CDLM, DRAGON, REALM, In-Context Pretraining) are already
in our lit review.

---
One-line confirmation: Deep-dive written to
`paper/notes/deepdives/yasunaga2022linkbert.md`, grounded in the ar5iv full-text render and
cross-checked against masks.md / data_pipelines.md / traversal.md and refs.bib.
