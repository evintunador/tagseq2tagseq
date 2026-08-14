## yang2018hotpotqa — HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering

Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning. EMNLP 2018 (D18-1259), arXiv:1809.09600.
This is our headline multi-hop benchmark source: `eval/nlp_benchmarks.py:run_hotpotqa` and
`run_hotpotqa_cross_doc` (covered in code brief `eval_harness.md`). Numbers below are from the
ar5iv HTML of the paper; where I could only confirm the abstract I say so.

### What the paper actually does

**Task & construction.** 112,779 crowd-authored English QA pairs where answering requires reasoning
over *two* supporting Wikipedia paragraphs. Two collection mechanisms produce two question *types*:

- **Bridge questions (~the majority; a 100-example manual sample: 42% chain "Type I" + 15% Type II
  "multiple properties" + 6% Type III).** Built from a directed hyperlink graph: an edge (a,b) means
  the *first paragraph* of article A hyperlinks to article B. B is the **bridge entity** connecting
  the two articles (their canonical example: a Radiohead question that routes through "Thom Yorke" to
  reach a birthday). Bridge entities were constrained to 591 manually curated category pages so the
  hop is meaningful. Crowdworkers see paragraph A and paragraph B and must write a question whose
  answer needs both. **This hyperlink-from-A-to-B is exactly the structure our harness exploits.**
- **Comparison questions (~27% of the sample).** Two entities sampled from the *same* curated list (42
  lists); worker compares a shared property. Includes yes/no variants ("Are Iron Maiden and AC/DC from
  the same country?") and often requires arithmetic (comparing ages from birth dates). No bridge
  hyperlink — the two entities are siblings, not linked A→B.

**Supporting facts.** Beyond the answer span, annotators mark the *sentence-level* set of supporting
facts needed to answer. This is HotpotQA's signature contribution — strong, explainable supervision
and a second evaluation target.

**Two settings.**
- **Distractor:** each question is given its 2 gold paragraphs + 8 distractor paragraphs (10 total),
  distractors retrieved by **bigram TF-IDF** with the question as query, then shuffled. This is the
  reading-comprehension setting.
- **Fullwiki:** no gold paragraphs given; the model retrieves over the first paragraphs of all ~5M+
  Wikipedia articles (inverted-index filter → up to 5000 candidates → top-10 by bigram TF-IDF). This
  is the open-domain setting. Uses a separate test set to avoid leaking gold paragraphs.

**Splits:** train-easy 18,089 (mostly single-hop) + train-medium 56,814 + train-hard 15,661; dev 7,405;
test-distractor 7,405; test-fullwiki 7,405 (all dev/test are hard multi-hop).

**Corpus / dump:** English Wikipedia dump dated **2017-10-01**, processed with WikiExtractor +
Stanford CoreNLP 3.8.0. This is the dump our harness downloads (`enwiki-20171001-...`, see below).

**Metrics:** answer EM/F1, supporting-fact EM/F1, and a **joint** EM/F1 that multiplies the two
(P_joint = P_ans·P_sup; joint EM = 1 only if both are exact). Baseline = a reimplementation of Clark &
Gardner (2017) multi-paragraph BiDAF-style reader with char models, self-attention, bi-attention, and
a 3-way yes/no/span classifier, with supporting facts as an auxiliary supervised loss.

**Headline results (baseline, test):** distractor 45.46 ans-EM / 58.99 ans-F1, 12.04 joint-EM /
41.37 joint-F1; fullwiki 25.23 / 34.40 ans, 2.63 / 17.85 joint. Human distractor ~83.6 EM / 91.4 F1.
Per-type (dev distractor): bridge 43.41 EM / 59.09 F1, comparison 48.55 EM / 55.05 F1; in fullwiki
bridge collapses (retrieving the bridge entity B is the bottleneck) while comparison drops only
marginally (both compared entities usually appear verbatim in the question, so retrieval is easy).

**Shortcut/artifact discussion in the paper itself.** They explicitly motivate HotpotQA against
single-hop-solvable predecessors: SQuAD answers "can be answered by matching the question with a single
sentence"; TriviaQA/SearchQA "can be answered by matching a few nearby sentences in one paragraph."
They split out **train-easy as mostly single-hop**. But their own ablation shows self-attention and
char models (single-hop machinery) still help on the hard set, i.e. single-hop techniques remain
"somewhat effective" — an early admission that HotpotQA is not fully hop-forcing.

### Methodology: theirs vs. ours

**Different question entirely, and we should be explicit about it.** HotpotQA measures whether a
*reader/retriever* can locate and combine two documents to produce a discrete answer, scored by
EM/F1 over answer strings and supporting-fact sentence sets. **We do not do the HotpotQA task at all.**
`run_hotpotqa_cross_doc` repurposes the *bridge pairs* as raw material for a paired
teacher-forced-NLL contrast on the answer tokens (all our eval is mean-NLL over a designated region,
`logit@t` predicts `token@t+1`; see `eval_harness.md` "Core primitives"). We never retrieve, never
predict a span, never touch supporting-fact EM/F1, never touch the distractor or fullwiki settings.
So none of their leaderboard numbers are comparable to ours — HotpotQA is a *source of naturally
hyperlinked A→B document pairs with a question that provably needs both*, which is precisely the
inductive bias our cross-doc-link attention is built to exploit.

**How our harness builds the cross-doc contrast** (`run_hotpotqa_cross_doc`, `nlp_benchmarks.py:1483`):
1. **Bridge-only.** `_hotpotqa_bridge_examples` filters to `type == "bridge"` (~5918/7405 dev). Comparison
   questions are dropped here — they have no A→B hyperlink to fire a grant on. (`run_hotpotqa`, the
   plain-context flat benchmark, keeps both types and reports `n_bridge`/`n_comparison`.)
2. **Reconstruct A and B from the 2017 corpus.** For each example it pulls the *supporting sentences*
   of article A and article B from the downloaded `enwiki-20171001` abstracts corpus (title.lower()
   keyed), using the annotated supporting-fact sent_ids.
3. **Render A's HTML links to markdown.** The corpus stores links as `<a href="url%20title">anchor</a>`;
   `_html_links_to_markdown` rewrites them to `[anchor](Title)` — the *identical* surface form our
   wikitext training pipeline produces from `[[Article Title]]`, so `MarkdownLinkDetector` fires on the
   `](` bigram exactly as in pretraining. B is packed as a plain-text aux DocSpan with links stripped.
4. **THE PRE-FILTER (the crux the brief flags).** `marker = f"]({b_title})"`; the example is *kept only
   if at least one A supporting sentence literally contains that rendered markdown link to B*
   (`nlp_benchmarks.py:1614-1620`). Otherwise the grant can never fire and cross would equal flat, so
   it's skipped (`n_skipped_no_link`). This is the "we pre-filter to A-sentences that literally contain
   the markdown link to B so our grant can fire" mechanism named in the task.
5. **Score.** context = `A_markdown + "\nQuestion: " + question + "\nAnswer: "`, completion = `answer`,
   aux = `[B_text]`, `aux_raw_identifiers=[b_title]`. Cross arm = `score_completion_with_context_docs`
   → `forward_inference(mask_type='cross_doc_link')`, so from the `](B_title)` link position onward A
   (and the question/answer) can attend back into B. Flat arm = same tokens under `doc_causal` with no
   aux (`score_completions_independent_batched`). **Δnll = flat − cross over the paired fired subset
   (`n_cross_doc`)**; `average_nll_flat_linked_only` vs `average_nll_cross_doc_only` is the headline.

**Axis placement.** Ours is *train-on-structure + use-the-same-structure-at-inference*: the same
`cross_doc_link` mask that granted A read-access into B during pretraining fires on the same Wikipedia
hyperlink at eval. HotpotQA's own baselines and successors (EntityGCN, DFGN, HDE, Cognitive Graph in
our refs) are **retrieve-then-read with GNN edges over a per-question graph built at inference** — the
edge is a message-passing channel between node representations, not an attention grant, and it is not
present during LM pretraining. So the shared object is "articles A,B linked by a hyperlink where the
question needs both," but our edge is a *pretrained attention pathway* and theirs is an
*inference-time graph-reader module*. We are closest in spirit to the bridge construction and farthest
in mechanism.

### Predictions & open questions for our method

- **Bridge >> comparison for us, by construction.** Their fullwiki result — bridge collapses without
  retrieval, comparison survives because both entities are named in the question — predicts the mirror
  image for us: our grant *is* the retrieval-free bridge, so the cross-doc benefit (Δnll) should be
  concentrated exactly on bridge questions, which is why the harness bridge-filters. Comparison
  questions would show ~0 grant benefit (no A→B link) and are correctly excluded from the cross arm.
- **Benefit should track how much of the answer lives in B, not A.** The bridge answer is typically a
  property of the *bridge entity B* (Thom Yorke's birthday), which is present in aux B but absent from
  A's supporting sentences. That is the ideal regime for a positive Δnll: the answer tokens are
  low-probability under A-only (flat) and high under A+B (cross). Where the answer is inferable from A
  alone, expect Δnll → 0. This is a sharper, more favorable signal than RepoBench's next-line
  completion and could be our cleanest positive.
- **Their single-hop-solvability ablation predicts a placebo risk for us.** Because single-hop
  machinery remained "somewhat effective" on hard HotpotQA, some answer tokens are guessable from A +
  question surface form alone. If our cross arm gains partly from *any* extra tokens rather than from
  *B specifically*, that would inflate Δnll — which is exactly what the missing placebo arm (below)
  fails to rule out. A derangement-placebo on HotpotQA (swap B's content for another plausible
  Wikipedia intro while keeping the `b_title` identifier so the grant still fires) would resolve this
  and mirror the Tier-2 placebo already built for RepoBench (`tier2.py`, `eval_harness.md`).
- **Open question we can answer for them.** HotpotQA's fullwiki bottleneck is *retrieval of the bridge
  entity*. Our design collapses retrieval into the pretrained link grant: a generated `[...](B)` link
  fetches B into context. That is a candidate answer to "can multi-hop be done without an explicit
  retriever" — but only in the *distractor-free, gold-link-present* regime the harness constructs, not
  the open fullwiki setting.

### Gotchas

- **Contamination is real and the harness's own defense is contrastive, not clean-room.** The corpus
  is the **2017-10-01** Wikipedia dump; our training data is 2025–2026 Wikipedia dumps of the *same
  articles*. The intro paragraphs of famous entities (Radiohead, Thom Yorke) are near-identical across
  years, so the model has very likely seen this text. The docstring's leakage argument
  (`nlp_benchmarks.py:1226`, `1502`) is that the flat-vs-cross contrast is on *identical text*, so pure
  memorization cancels — a real and reasonable defense for the *Δnll*, but note it does **not** clean
  the absolute perplexities, and it assumes memorization is mask-invariant (a memorized continuation
  could be recalled more strongly under one attention layout than the other). Reviewer point #5 in the
  brief. Do not report absolute HotpotQA perplexity as a capability number.
- **Fire-conditioned subset selection.** Δ is computed only over examples where the grant fired
  (`n_cross_doc`), and firing requires (a) both articles present in the abstracts corpus and (b) A's
  supporting sentence literally containing `](B_title)`. This selects for clean, unparenthesized,
  unquoted titles whose intro paragraph happens to link to B. The two documented structural non-fires
  are kept honest and *not* force-fired: parenthesized titles ("Alien (film)" — detector stops at the
  first `)` and extracts "Alien (film") and quoted titles (`"[Animorphs](Animorphs)"` — `"[` tokenizes
  differently from `[`). 26/200 fell to fallback in their run. This is correct behavior (these links
  never fired during training either) but it *is* sample selection and correlates with entity
  popularity — flag it as reviewer point #2.
- **Abstracts-only corpus may miss supporting facts.** `_load_hotpotqa_corpus` defaults to the ~1.55GB
  *abstracts* corpus (intro paragraphs only). Most HotpotQA supporting facts are in intros, but any
  question whose supporting sentence is in a later section silently becomes `n_skipped_no_corpus`.
  `use_full=True` (~7.4GB) exists but is off by default — another subset-selection knob to disclose.
- **Missing placebo = the fairness hole.** As the brief's reviewer point #1 states, the headline
  HotpotQA arm has *no placebo control* (only RepoBench Tier-2 does). The cross arm sees strictly more
  tokens (B) than the flat arm (no B). Given the paper's own finding that single-hop shortcuts partly
  work, "is the gain from the *right* article B or from *any* extra context?" is unanswered here. This
  is the single most important thing to fix before the HotpotQA number goes in the paper.
- **Comparison questions are not evaluable by our mechanism** — don't accidentally report a combined
  number as if it covered all of HotpotQA. Our cross-doc result speaks only to bridge questions.
- **Yes/no answers.** Comparison (and some bridge) answers are "yes"/"no" — a 1–2 token completion where
  NLL is dominated by a near-uniform binary prior; if any leak into the scored set they add noise to
  Δnll. Bridge-only filtering removes most but not all.
- **Their metric ≠ our metric.** Resist the temptation to compare our perplexity delta to their
  EM/F1 leaderboard; they are different quantities on different subsets under different settings.

### Missed citations worth adding

Checked section 9 of `paper/bib/refs.bib` (multi-hop QA) and surrounding sections. Present already:
welbl2018wikihop, ho2020twowiki, trivedi2022musique, ding2019cognitivegraph, tu2019hde, qiu2019dfgn,
decao2019entitygcn, min2019decomprc, talmor2018complexwebquestions, xiong2021mdr, etc. Genuinely missing
and relevant to *our* eval-artifact / methodology story:

- **min2019necessitate** — "Compositional Questions Do Not Necessitate Multi-hop Reasoning" (Min,
  Wallace, Singh, Gardner, Hajishirzi et al.), ACL 2019, arXiv:1906.02900. **Directly load-bearing:**
  shows a large fraction of HotpotQA bridge questions are answerable single-hop (single-paragraph
  models score high), and proposes a single-hop-decomposition adversary. This is *the* canonical
  citation for the shortcut/single-hop-solvable artifact the task asked us to describe, and it directly
  motivates why our design needs a placebo arm. (Verify author list before adding.)
- **jiang2019avoiding** — "Avoiding Reasoning Shortcuts: Adversarial Evaluation, Training, and Model
  Development for Multi-Hop QA" (Jiang & Bansal), ACL 2019, arXiv:1906.07132. Adversarial distractor
  construction that removes reasoning shortcuts in HotpotQA; the counterfactual-eval framing is a close
  cousin of our derangement placebo. Note jiang2020hover (same first author) is already in refs but this
  is a different, more methodologically relevant paper.
- **chen2019understanding** — "Understanding Dataset Design Choices for Multi-hop Reasoning" (Chen &
  Durrett), NAACL 2019, arXiv:1904.12106. Analyzes WikiHop and HotpotQA and shows models exploit
  dataset design shortcuts; relevant to our contamination/artifact section. (Verify arXiv id.)
- **clark2018simple** — "Simple and Effective Multi-Paragraph Reading Comprehension" (Clark & Gardner),
  ACL 2018, arXiv:1710.10723. The architecture the HotpotQA *baseline is a reimplementation of*; worth
  citing if we describe the baseline number. Lower priority (it's a reader, not close to our method).

I did not find these in refs.bib via grep; please verify keys/ids before insertion as instructed.

Confirmation: wrote /fss/evin_t/tagseq2tagseq/paper/notes/deepdives/yang2018hotpotqa.md per the brief.
