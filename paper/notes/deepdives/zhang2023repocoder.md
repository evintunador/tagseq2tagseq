# zhang2023repocoder — RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation

Zhang, Chen, Zhang, Keung, Liu, Zan, Mao, Lou, Chen. EMNLP 2023. arXiv:2303.12570 (v3, 2023-10-20).
Code/benchmark: microsoft/CodeT `/RepoCoder`.

Sources for this note: arXiv abstract + ar5iv HTML full text (method, tables, appendices). Result
numbers below are as reported in the ar5iv render; I did not independently re-verify the PDF tables,
so treat the exact decimals as "as-reported." Everything about OUR method is checked against the two
assigned code briefs (`eval_harness.md`, `generation_retrieval.md`) and the related-work notes.

## What the paper actually does

**Problem.** Repository-level code completion: predict the next line / API call / function body given
the unfinished file *and* the rest of the repo, where the needed context (a helper, an import target,
a sibling module) lives in *other files*. A plain in-file LM never sees it.

**Method — an inference-time iterate loop, no training.** RepoCoder wraps a frozen retriever and a
frozen code LM; "the parameters of both the retriever and the generator remain unchanged throughout
the entire process." One pass:
1. **Retriever.** Sparse bag-of-words **Jaccard index** over token sets, `Jaccard(Sq,Sc)=|Sq∩Sc|/|Sq∪Sc|`.
   (A dense UniXcoder + cosine variant in Appendix B gives comparable results.) The repo is chopped by a
   **sliding window**: window size `Sw`, stride `Ss`, over contiguous lines of every file. Line/API use
   `Sw=20, Ss=10`; function body uses `Sw=50, Ss=10`. Up to `K=10` snippets retrieved.
2. **Prompt assembly.** Retrieved snippets are placed *before* the unfinished code, in **ascending
   similarity order** (most-similar nearest the completion point), each prefixed with its source file
   path as a plain-text comment. Then the model generates.
3. **Iterate.** The gap: on iteration 1 the query is just the last `Sw` lines of the unfinished file X,
   which is a poor proxy for *what the completion will look like*. So on iteration i>1 they rebuild the
   query as `(last Sw−Ss lines of X) ⊕ (first Ss lines of the previous model prediction Ŷ^{i-1})` — i.e.
   use the model's own draft completion to retrieve code that resembles the *target*, not the prefix.
   Re-retrieve, re-prompt, re-generate. Tested up to 4 iterations; best is usually iteration 2–4.

**RepoEval benchmark (their second contribution).** Real GitHub repos created **after 2022-01-01**
(to dodge Codex/CodeGen training leakage), >100 stars, >80% Python, with unit tests. Three settings:
- **Line completion:** 8 repos, 1,600 samples (200/repo). Metrics: Exact Match (EM), Edit Similarity
  (ES = 1 − Lev(Ŷ,Y)/max(|Ŷ|,|Y|)).
- **API-invocation completion:** same 8 repos, 1,600 samples. EM, ES.
- **Function-body completion:** 6 smaller repos, 373 samples. Metric: **unit-test Pass Rate** (execution,
  not string match) — because EM/ES wrongly penalize functionally-correct variants.

**Models.** GPT-3.5-Turbo (4,096-token budget) and CodeGen-Mono at 350M / 2B / 6B (2,048-token budget).
(Note: the v3 ar5iv text uses GPT-3.5-Turbo; earlier versions used Codex `code-davinci-002`.)

**Headline results (as-reported).**
- Beats the In-File baseline by **>10% EM / >8% ES** in all line/API settings, across all model sizes.
- Line EM (GPT-3.5): In-File 40.56 → Iter-1 (= vanilla one-pass RAG) 55.31 → Iter-3 57.00 (Oracle 57.75).
- API EM (GPT-3.5): 34.06 → 47.69 → Iter-4 49.56 (Oracle 50.13).
- Function Pass Rate (GPT-3.5): 23.32 → 38.34 → Iter-2 42.63 (Oracle 42.63).
- With ≥2 iterations RepoCoder **consistently beats vanilla RAG** (iteration 1). Iterating recovers most
  of the gap to an Oracle that retrieves using the *ground-truth* completion.
- CodeGen-350M + RepoCoder ≈ In-File GPT-3.5-Turbo — retrieval substitutes for parameters.
- Recall of ground-truth API calls rises from Iter-1→Iter-2 (86.04%→90.34% for GPT-3.5) and with model
  strength — direct evidence the model's own draft improves the retrieval query.

## Methodology: theirs vs. ours

The single sharpest axis (exactly the brief's framing): **retrieve-at-inference over a frozen model vs.
train the structural edge into the weights and reuse the same machinery at inference.**

- **When structure enters.** RepoCoder never trains on repo structure — both retriever and LM are frozen;
  cross-file signal is injected only as prompt text at generation time. OUR method bakes the import/link
  edge into pretraining: documents are packed graph-topologically into a 32k sequence and a custom
  block-sparse mask (`cross_doc_link` → `triton_v18`, per repo `CLAUDE.md` and `masks.md`) grants the
  linking document read-access into its target from the link position onward. The *same* mask + detector
  + match-key runs at inference (`generation_retrieval.md`: "train/inference mirror … SHARED CODE not
  analogy"). So RepoCoder is the canonical "inference-only" pole against our "trained-edge" pole.

- **What the "edge" is.** RepoCoder's edge is *soft, learned-similarity, lossy*: Jaccard token overlap
  picks the top-K windows, and their content is spliced into the prompt as ordinary tokens the causal LM
  reads flat. There is no notion of *which* file the current file actually imports — retrieval is by
  lexical similarity, and the most-useful snippets empirically come from "Similar Import" / "Current
  Directory" / "Similar Name" heuristic buckets (their Table 5). OUR edge is *hard, deterministic,
  identifier-resolved*: a detected import/link resolves via `index_doc_span` — "the SAME match key
  training uses" (`generation_retrieval.md`) — to an exact target node, which is inserted into the packed
  sequence and attended through the trained mask. No ANN, no approximation; target resolution is an exact
  hashmap lookup (related_work_notes: "unnecessary when target resolution is an exact hashmap lookup").
  In our eval harness the code arm resolves *per-import-path → doc_id* ("precise" grants,
  `score_completion_with_context_docs`, `eval_harness.md`), which is the deterministic analog of their
  fuzzy retrieval.

- **The iteration loop vs. our recursion.** RepoCoder's loop is the closest structural cousin to our
  link-following generation: both use the *model's own generated tokens* to decide what external code to
  pull. But the mechanisms differ fundamentally. RepoCoder re-runs the whole pipeline N times (fixed
  1–4), each pass a full re-retrieve-re-prompt-regenerate over the entire completion; the "signal" is the
  draft completion feeding a lexical retriever. OURS fires *per-token* the moment a link is closed
  (`_handle_link`, ≤1 link/token, `link_end_pos==len(recent)` over a 200-token window), fetches the
  target, and **inserts it into the packed sequence before the linking doc** so the cross-doc mask grants
  attention — "retrieval-BY-INSERTION into packed seq (not context-string concat)"
  (`generation_retrieval.md`). Depth is bounded by `max_link_depth` (default 2); fetched docs are
  themselves scanned for *their* links (genuine multi-hop recursion). RepoCoder's iteration is
  *refinement of one retrieval*; our recursion is *traversal of a graph*.

- **KV / compute.** Neither system reuses KV in a way that beats full recompute, but for opposite
  reasons. RepoCoder pays N independent forward passes (one per iteration) with no incremental decoding
  across iterations. OURS deliberately forgoes KV caching entirely: `build_sequence()` +
  `forward_inference` from scratch every token, because inserting a fetched node shifts RoPE positions and
  makes any paged/prefix KV reuse incorrect (`generation_retrieval.md`: "KV CACHE = NONE … O(T²)×O(T)").
  Both are efficiency-limited; the reviewer-facing framing (from the brief and related_work_notes'
  KV-serving section) is that we trade caching for exact train/inference symmetry.

- **Where they overlap with us.** (a) Both exploit that a *predicted* draft is a better retrieval key
  than the raw prefix — RepoCoder's central trick, and implicitly ours (a generated link token is the
  fetch key). (b) Both are frozen-generator-friendly: RepoCoder needs no fine-tuning; our inference path
  can run a 32k context from an 8k checkpoint via pure RoPE (`generation_retrieval.md`,
  "length-agnostic inference"). (c) Both target the same task family — repo-level completion — and RepoEval
  is a sibling to the RepoBench port that our headline paired-Δnll eval runs (`eval_harness.md`,
  `run_repobench_cross_doc`).

- **Evaluation contrast.** RepoCoder measures generative EM/ES and (for functions) unit-test pass rate —
  end-to-end task metrics on a frozen API model. OUR headline is a **paired same-token cross-vs-flat
  Δnll** (`eval_harness.md`), teacher-forced NLL of the *same* completion tokens with vs. without the
  cross-doc grant, token-parity-enforced, plus a **derangement placebo** (swap donor content, keep
  identifiers so grants still fire) — a memorization-canceling, contamination-robust design RepoCoder
  does not have. Their contamination control is instead *temporal* (post-2022 repos), which is coarser
  but real.

## Predictions & open questions for our method

- **Draft-conditioned retrieval predicts our link-following should help most where the *target
  identifier itself* is generatable.** RepoCoder's iteration gains come from the model drafting code that
  surfaces the right API names, lifting recall (86→90%). Our analog: link-following only fires when the
  model emits a resolvable link/import token. Expect the strongest cross-doc Δnll exactly on completions
  whose first tokens *are* an import/qualified-name (our `use_line` scope in `eval_harness.md`), and near-
  zero benefit where the needed context is used without ever naming its source — matching the observed
  native→use_line flip (go/java ~0 → clearly positive).

- **Duplication-dependence is a shared failure regime.** RepoCoder's Appendix C: gains track code
  duplication (diffusers high → big gains; rl/vizier low → small). For us this predicts the effect will be
  *strongest on high-boilerplate / high-reuse repos and citation-dense arxiv clusters*, and weak on
  low-redundancy code. Worth an explicit stratified slice — and a caution that a headline number averaged
  over repos hides this. Our advantage: because our edge is the *actual* import (not lexical overlap), we
  should degrade *less* than RepoCoder on low-duplication repos, since a correct import edge exists even
  when no similar snippet does. That is a testable, favorable prediction.

- **Diminishing returns / non-monotonic iteration** predicts our multi-hop depth will similarly plateau.
  RepoCoder: iteration 2–4 best, "later iterations both fix and break cases," optimal count
  undeterminable. Our `max_link_depth` recursion is the analog; expect depth 1–2 to capture most of the
  gain and deeper hops to add noise (fetching wrong/misleading docs). Their result argues *against* a
  large default depth — our default of 2 is well-motivated, and we should ablate depth like they ablate
  iterations rather than assume monotone improvement.

- **Retriever-content quality dominates (their Table 4, GT-Code oracle wins).** This is precisely what our
  **placebo** control isolates and RepoCoder cannot: RepoCoder shows *right* code beats *retrieved* code,
  but never separates "right doc" from "any extra plausible tokens." Our derangement placebo answers their
  open question directly — it proves the benefit is the *right* imported code, not extra context. Frame
  this as: our design *resolves* an ablation RepoCoder leaves open.

- **Frozen small model + retrieval ≈ big model** predicts our cross-doc edge should help small models
  disproportionately. RepoCoder: CodeGen-350M + RepoCoder ≈ In-File GPT-3.5. If the edge substitutes for
  parametric memory, our smallest scales should show the *largest* Δnll — a clean scaling prediction (and
  a caution about our "underpowered/undertrained" runs, `eval_harness.md` #6: small models may show the
  effect *more*, not less, if it's genuinely a memory substitute).

## Gotchas

- **Exact-match penalizes correct code — they had to add unit tests.** RepoCoder explicitly found EM/ES
  flag functionally-correct completions as wrong, motivating the function-body Pass Rate. Our headline is
  Δnll (continuous, avoids this), but any EM/ES-style port slice (RepoBench) inherits the artifact; keep
  the continuous metric primary (matches related_work_notes' schaeffer2023mirage / biderman2024lessons
  discipline).

- **Contamination is temporal, and 2022 is old now.** Their post-2022-01-01 cutoff protected Codex/CodeGen
  but is porous for any modern base model. If we ever compare against or reuse RepoEval, the leakage
  argument is weaker today; our SHA1/repo-name dedup (`eval_harness.md`) is the stronger control, but
  note yang2023rephrased (already in refs) — near-dup rephrasing evades n-gram dedup.

- **Draft-in-query can inject noise.** RepoCoder Appendix D: failures come from *misleading retrieved
  snippets* and *noisy query construction* — the model's own wrong draft pulls in wrong code and
  self-reinforces. Our per-token link firing has the same hazard: a hallucinated-but-resolvable link
  fetches a plausible-wrong target. Our `link_but_skip` / `full_skip` retrieval modes
  (`generation_retrieval.md`) exist partly to probe this; expect a firing-conditioned selection bias
  (`eval_harness.md` #2) exactly analogous to their retrieval-noise failures.

- **Iteration/depth count is a hidden hyperparameter with no clean stopping rule.** They could not
  determine optimal iterations a priori. Don't tune `max_link_depth` on the eval set and report the best;
  that reads as the same trap as our Option-B `span.start+1` key hack (`eval_harness.md` #3) — looks like
  tuning until the effect is independently justified.

- **Prompt ordering matters (ascending similarity, nearest to cursor).** RepoCoder deliberately puts the
  most-relevant snippet closest to the completion — an implicit "lost-in-the-middle" mitigation
  (liu2024lostmiddle, in refs). Our packing puts targets *before* the linker (topological, targets-first);
  worth noting that our ordering rationale is structural (DAG/causality) while theirs is positional
  utilization — and that a positional-utilization confound could masquerade as a cross-doc effect if we
  aren't careful about where fetched docs land.

- **Config-mismatch bites long-doc eval** (our own `eval_harness.md`: the 32k→2048 budget bug, max_grants
  64-vs-256 bug). RepoCoder's model budgets are small (2,048 for CodeGen, 4,096 for GPT-3.5); if we cite
  their numbers as a comparison point, note they operate in a much shorter context regime than our 32k,
  so "they only fit K=10 windows" is not a like-for-like context budget.

## Missed citations worth adding

I grepped `refs.bib` for each; these are the ones cited by RepoCoder (or its direct method dependencies)
that are relevant to our project and appear **absent** (verify before adding — do not assume present):

- **svyatkovskiy2020intellicode** — Svyatkovskiy et al., "IntelliCode Compose: Code Generation Using
  Transformer" (arXiv:2005.08025, FSE 2020). Statement/line-level whole-line code completion deployed in
  an IDE; a foundational repository/editor-context completion antecedent to the task our RepoBench port
  and link-following inference address. Absent (grep: 0 hits).
- **robertson2009bm25** — Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
  (Found. Trends IR, 2009). The canonical sparse-retrieval reference; RepoCoder's Jaccard retriever sits
  in this lexical-retrieval family, which is exactly the learned/lexical-similarity retrieval pole our
  exact-hashmap deterministic edge departs from. We cite dense backbones (DPR, Contriever, SPLADE) but no
  BM25/probabilistic-lexical anchor. Absent (grep bm25/robertson: 0 hits).
- **ren2020codebleu** — Ren et al., "CodeBLEU: a Method for Automatic Evaluation of Code Synthesis"
  (arXiv:2009.10297). The standard code-generation metric (weighted n-gram + AST + dataflow match) that
  the repo-completion literature reports; relevant to our evaluation-methodology section as the
  code-specific counterpart to the EM/ES/Δnll choices we discuss. Absent (grep: 0 hits).

(Checked and already present — not missing: guo2022unixcoder [RepoCoder's dense retriever], lu2022reacc
[its retrieval-augmented-completion baseline], liu2024repobench, ding2023crosscodeeval, chen2021humaneval,
lu2021codexglue, nijkamp2023codegen, zheng2023codegeex.)

---
Confirmation: deep-dive written to `paper/notes/deepdives/zhang2023repocoder.md`, grounded in the ar5iv full text and the eval_harness.md / generation_retrieval.md code briefs; result decimals are as-reported (not re-verified against the PDF), and the three suggested citations were grep-checked as absent from refs.bib.
