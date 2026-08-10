# wu2024repoformer — Repoformer: Selective Retrieval for Repository-Level Code Completion

Wu, Di; Ahmad, Wasi Uddin; Zhang, Dejiao; Ramanathan, Murali Krishna; Ma, Xiaofei. ICML 2024.
arXiv:2403.10059 (v2, 2024-06-04). Authors from AWS AI Labs.

Sources for this note: arXiv abstract + the arXiv HTML render (arxiv.org/html/2403.10059v2), from
which method, tables and ablations were extracted; the PDF stream was not machine-readable, so
result decimals below are "as-reported by the HTML render," not independently re-verified against
the typeset PDF. Everything about OUR method is checked against the two assigned code briefs
(`eval_harness.md`, `generation_retrieval.md`) plus `masks.md`, and the related-work notes.

## What the paper actually does

**Problem.** Repository-level code completion — the RepoCoder/CrossCodeEval task family — where the
useful context (a helper, a callee, an API signature) lives in *other files* of the repo. The
standard fix is retrieval-augmented generation (RAG): retrieve cross-file (CC) snippets, splice them
before the unfinished file, generate. Repoformer's thesis: **always retrieving is wrong.** On their
own analysis (§5.1, RepoEval) retrieval *helped* only ~20% of instances, *left unchanged* >60%, and
*hurt* ~20% — a pattern that holds across CodeGen and StarCoder families and across model sizes.
Unconditional retrieval therefore wastes latency and injects harmful noise.

**Core method — self-selective RAG in one model.** A single code LM is fine-tuned to do two jobs in
one left-to-right pass: (1) *self-evaluate* whether cross-file retrieval would improve its own next
completion, and (2) *generate* the completion (with CC if retrieved, without otherwise). The trick
is a small vocabulary extension layered onto fill-in-the-middle (FIM):
- A new token `<eof>` marks the end of the current file and triggers self-evaluation.
- The model then emits `<cc>` to *request retrieval*, or an empty/other token to *abstain*.
- If retrieval fires, the retrieved CC is placed after `<cc>` and generation continues.
This keeps the whole thing a single decoder pass — the retrieval decision is just a token
probability, `P(<cc>)`, read off at the `<eof>` position.

**Label construction (self-supervised).** Binary labels are made by contrast: for each training
instance, generate the completion *with* CC and *without* CC, score both by **Edit Similarity (ES)**
against the ground truth Y, and label "should retrieve" iff CC improves ES by more than a threshold
**T = 0** (i.e. any improvement). StarCoderBase-1B was the labeler. This produced **240k chunk-**
and **120k function-completion** training instances. The training loss combines a self-evaluation
term (L_eval, teaching `P(<cc>)`) and the generation term (L_gen), weighted λ=1.0.

**Inference threshold.** Retrieve iff `P(<cc>) > T`. Reported working points: **T = 0.15 for
function completion, T = 0.2 for other tasks**; at **T = 0.4** performance matches always-retrieve
with ~50% of the retrieval latency. So T is a tunable accuracy/latency knob, not a fixed gate.

**Backbones / scale.** Fine-tuned **StarCoderBase at 1B / 3B / 7B / 16B** → Repoformer-1B/3B/7B/16B
(seq len 2048, lr 2e-5, batch 512, 2 epochs, 8×A100-40G, ≈8/12/20/50 h). A multilingual variant
(Python, Java, C#, TypeScript) was also trained. As comparison/plug-in generators (not all
fine-tuned): StarCoder(Base), CodeGen-Mono 2B/16B, CodeGen25-7B, Code Llama 7B/13B, gpt-3.5-turbo.
Repoformer-1B was also tested as a *plug-and-play retrieval policy* deciding retrieval for larger
frozen generators.

**Retriever.** Sparse **Jaccard similarity** over token sets (chosen for speed), same family as
RepoCoder. Fixed-size sliding-window chunks (window 20 for line/API/chunk, 50 for function; stride =
half window), top **k=10**, with fragment alignment (include the chunk *following* each match).
Robustness also validated with **dense UniXcoder** retrieval. They follow RepoCoder's **RG-1
(single-iteration)** formulation and *skip iterative retrieval*, arguing one iteration "already
achieves the majority of the performance gains from multi-iteration RAG."

**CrossCodeLongEval (second contribution).** A new long-form completion benchmark built from **1500
raw Python repos taken from CrossCodeEval**, adding *chunk* and *function-body* completion (motivated
by RepoEval's limited repo coverage and CrossCodeEval's limited task coverage). Targets Y are sampled
as random code chunks of varied lengths or as whole function bodies — the *same procedure used to
construct the training data*. Combined with RepoEval, the four task types evaluated are **line, API,
function, chunk** completion.

**Headline numbers (as-reported, RepoEval + CrossCodeLongEval, their Table 2).** Metrics: EM/ES for
line/API/chunk; unit-test pass (UT) + ES for function.
- 1B: No-retrieve → Always → SelectiveT.
  - Line EM/ES 43.44/67.77 → 51.19/72.30 → **54.40/76.00**
  - API EM/ES 37.81/66.54 → 43.94/69.17 → **46.10/72.70**
  - Func UT/ES 22.20/47.65 → 25.71/55.64 → **28.79/57.30**
  - Chunk EM/ES 31.08/60.09 → 37.22/63.73 → **41.92/69.97**
  So **selective > always > none** on every task, not merely faster — the noise-avoidance gives an
  accuracy gain over always-retrieve too.
- Repoformer beats same-size StarCoderBase by **>3 absolute ES points** across tasks. **Repoformer-3B
  ≈ always-retrieving 16B StarCoder.** Repoformer-16B is SOTA on RepoEval/CrossCodeEval/CrossCodeLongEval,
  ~3% avg over the best StarCoder baseline.
- **Speedup:** "as much as 70% inference speedup" online — this figure is for *dense* retrieval on
  large repos where retrieval is >80% of pipeline latency, so a ~20% retrieval rate yields >70%
  end-to-end speedup. Using Repoformer-1B as the policy for larger generators cut latency ~25% while
  *improving* accuracy. Greedy (rather than thresholded) selection gives bigger latency wins (~69%
  API, ~61% line at 1B) at ~1.0 ES cost.
- **Abstention quality (§6):** when the model abstained, the prediction was already correct/unimprovable
  for **>80% of instances across all tasks**. Calibration is near-calibrated for line/API but weaker
  for function completion — hypothesized to come from using ES (a string metric) for labels while
  functions are judged by unit tests.

**Ablations (their Table 5).** A1: merging L_eval into L_gen makes `P(<cc>)`≈1 always → cannot select.
A2: removing L_eval barely changes RAG quality → adding the self-evaluation objective is nearly free.
A3: also removing all CC (pure FIM) keeps in-file quality but worsens RAG → CC training is necessary.
A4: relocating `<cc>`/CC after `<fim_middle>` badly hurts function completion → placement interacts
with learned FIM semantics.

## Methodology: theirs vs. ours

Repoformer sits at the **midpoint of the brief's train-vs-inference axis**: unlike RepoCoder (frozen,
pure inference-time retrieval — see `zhang2023repocoder.md`) it *trains* a behavior into the weights,
but the behavior it trains is a **retrieval policy**, and the retrieved content still arrives at
inference as spliced prompt tokens read under ordinary causal attention. OURS trains the *edge
itself* into attention. Three distinctions matter.

- **What gets trained.** Repoformer trains *when to fetch* and *how to consume flat CC tokens*; the
  cross-file signal is never a structural connection, it is top-k Jaccard windows concatenated into
  the prompt (`<cc>` … CC … completion), attended flatly. OURS trains *the connection*: the
  `cross_doc_link` mask grants the linking document read-access into its target from `link_end_pos`
  onward (`masks.md`: rows `[link_end_pos, A.end)` × cols `[B.start, B.end)`, asymmetric, DAG-gated),
  and the *same* mask + detector + `index_doc_span` match-key runs at inference by inserting the
  fetched target into the packed sequence (`generation_retrieval.md`: "train/inference mirror …
  SHARED CODE not analogy"). Their edge is soft/lexical/lossy (ANN over token overlap); ours is
  hard/identifier-resolved/exact (a resolved import → exact doc_id via a hashmap; "precise" grants in
  `score_completion_with_context_docs`, `eval_harness.md`).

- **The selective decision — a direct mirror of our firing-conditioned grant.** This is the paper's
  sharpest relevance. Repoformer's `P(<cc>)>T` gate is a *learned* decision of whether the cross-file
  connection is worth activating. OUR grant is also *conditioned on a firing event* — but ours fires
  on a **detected link/import token** (deterministic detector, `link_end_pos==len(recent)` over a
  200-token window at generation, or Option-B baked graph edges in training/eval; `masks.md`,
  `generation_retrieval.md`), not on a learned probability. So the two systems answer the same
  question ("should this position pull in cross-file context?") with opposite mechanisms: Repoformer
  *learns a soft policy over content quality*; we *apply a hard rule over graph structure*. Their
  finding that retrieval helps only ~20% of instances is the empirical argument *for* conditioning at
  all — and directly motivates why our eval reports Δnll over the **fired subset** rather than a
  blanket average. The flip side is the reviewer risk our own briefs already flag (`eval_harness.md`
  #2, "firing-CONDITIONED subset selection"): conditioning selection on the fire event correlates the
  reported metric with easy-to-parse imports/clean titles. Repoformer *chose* its condition to be a
  quality gate and can therefore report abstention accuracy (>80%); we should be able to characterize
  our fire condition the same way (what fraction of non-fires would have benefited?).

- **Consumption geometry / compute.** Repoformer keeps the completion a single left-to-right decode:
  the retrieval token, CC, and completion all live in one 2048 window, so ordinary KV caching works
  and the "70% speedup" is achievable precisely because it *skips the retriever call* on ~80% of
  instances. OURS forgoes KV caching entirely — `build_sequence()`+`forward_inference` from scratch
  every token because inserting a fetched node shifts RoPE positions and makes prefix/paged KV reuse
  incorrect (`generation_retrieval.md`: "KV CACHE = NONE … O(T²)×O(T)"). So on the efficiency axis we
  are the *opposite* of Repoformer: they save compute by not retrieving; we spend more compute per
  token but retrieve by exact insertion into a trained attention mask. Their whole contribution is an
  efficiency argument we cannot currently make — worth pre-empting in our serving-cost framing.

- **Where we overlap.** (a) Both target the same task family and benchmarks — RepoEval,
  CrossCodeEval, and Repoformer's **CrossCodeLongEval** are siblings of the RepoBench port our
  headline paired-Δnll eval runs (`run_repobench_cross_doc`, `eval_harness.md`). (b) Both build on
  **FIM** (bavarian2022fim, in refs): Repoformer extends FIM with a retrieval token; our packing
  "generalizes the FIM permutation across *linked* documents plus an attention grant"
  (related_work_notes). (c) Both use the **model's own emission as the retrieval trigger** — their
  `<cc>` token, our generated link/import token — a shared "generate-then-fetch" structure with
  RepoCoder's draft-conditioned query as the common ancestor.

## Predictions & open questions for our method

- **Retrieval helps a minority of positions → our fired-subset Δnll should be large but the
  *unconditional* average should be small.** Repoformer's 20/60/20 (help/neutral/hurt) split predicts
  that if we ever computed Δnll over *all* completion tokens (not the fired subset), the mean effect
  would be diluted toward zero. This validates our decision to report over fired examples — but it
  also predicts a *specific number we can check*: on a repo-completion slice, roughly how often does a
  resolvable cross-doc link even exist at the cursor? If that base rate is ~20%, our fire-rate should
  land near it; a much higher fire-rate would suggest we fire on links that don't carry signal.

- **A learnable "should-I-grant" gate is a natural ablation for us.** Repoformer shows a small model
  can *learn* whether cross-file context helps and abstain accurately (>80%). Our grant currently
  fires deterministically on every resolvable link. Repoformer's result predicts that **some fired
  grants are neutral-or-harmful** (their ~20% hurt), which means a learned gate on top of our detector
  — fire the grant only when the model predicts benefit — could *raise* mean quality by suppressing
  bad grants, exactly as SelectiveT beat Always in their Table 2. This is a concrete follow-up: our
  `link_but_skip`/`full_skip` retrieval modes (`generation_retrieval.md`) already give us the
  machinery to A/B a grant against its own suppression per-instance and *measure* the help/neutral/hurt
  split for our edge — something Repoformer had to do to justify selectivity.

- **The placebo we have answers a question they leave open.** Repoformer never isolates "right code"
  from "any plausible extra tokens" — its selective gate reduces *how often* noise enters but not
  *whether the benefit is the correct file*. Our **derangement placebo** (`eval_harness.md`: keep
  identifiers so grants still fire, swap donor content) does exactly that. Prediction: because our
  edge resolves the *actual* import, our placebo separation should be large where Repoformer's
  lexical top-k would frequently retrieve the wrong-but-similar window (their ~20%-hurt cases are
  plausibly retriever-noise). Frame this as: selectivity mitigates noise; our exact edge + placebo
  *proves the signal is the right doc*.

- **Small-model amplification.** Repoformer-3B with retrieval ≈ 16B always-retrieve, and their 1B
  policy lifted larger generators — retrieval substitutes for parameters. This mirrors RepoCoder's
  CodeGen-350M≈GPT-3.5 finding and predicts our cross-doc edge should show the **largest Δnll at our
  smallest scales** — a caution that our "underpowered/undertrained" runs (`eval_harness.md` #6) may
  show the effect *more*, not less, if it is genuinely a memory substitute.

- **Calibration is task-dependent, and the metric you train the gate on leaks into it.** Their
  function-completion calibration was weaker *because they trained labels on ES but evaluated on unit
  tests*. Our analog: whatever signal we use to decide/where we key grants (Option-B `span.start+1`
  key hack, `eval_harness.md` #3) is the "label" our effect is measured against; a mismatch between
  the keying/grant signal and the evaluation region is exactly the kind of leak that made their
  function calibration drift. Keep the grant geometry and the scored region consistent.

## Gotchas

- **Selective ≠ better unless you *train* the gate; a merged loss collapses to always-on.** Their
  ablation A1 is the warning: if the self-evaluation signal isn't a *separate* objective, `P(<cc>)`
  saturates to 1 and selectivity vanishes. Our analog: if we ever add a learned gate on grants, it
  must be trained against the help/neutral/hurt contrast, not folded into the LM loss, or it will
  learn "always grant."

- **ES/EM penalize functionally-correct code — they needed unit tests, and it *still* muddied
  calibration.** Our headline Δnll sidesteps this (continuous, no string-match artifact), but any
  EM/ES port slice (RepoBench) inherits it, and Repoformer's experience shows the metric choice even
  contaminates a *downstream* learned component. Keep the continuous metric primary.

- **Threshold T is a per-task hyperparameter with no universal value** (0.15 vs 0.2 vs 0.4 knee).
  This is the same hidden-hyperparameter trap as RepoCoder's iteration count and our `max_link_depth`
  / max_grants: don't tune the firing threshold/keying on the eval set and report the best point —
  report the accuracy/latency curve. Repoformer's honest move was showing the whole T-vs-latency
  frontier; we should show a fire-rate/Δnll frontier rather than one operating point.

- **"Single iteration captures most of the gain" is their justification for skipping iterative RAG —
  and a direct claim about *our* multi-hop depth.** They deliberately did *not* iterate. This predicts
  our `max_link_depth`>1 recursion will show sharply diminishing returns, and argues our default depth
  of 2 is already generous; ablate depth rather than assume deeper hops help.

- **Benchmark contamination / self-referential construction.** CrossCodeLongEval targets are sampled
  by the *same procedure as the training data* and drawn from CrossCodeEval's repos; the labeler is
  StarCoderBase-1B. If we adopt CrossCodeLongEval as an eval set, note (a) it shares repos with
  CrossCodeEval (dedup against both), and (b) its "hard/helpful" instance distribution is defined by a
  StarCoder-family model's ES gains, so it may be *easier* for StarCoder-lineage models than for ours.
  Their license note also matters: some RepoEval repos are non-permissive and the dataset was not
  redistributed — check availability before building on it.

- **The 70% speedup is regime-specific.** It requires dense retrieval dominating latency (>80% of the
  pipeline) *and* a low retrieval rate. Don't quote it as a general number; and note it is an argument
  we *cannot* make with our no-KV-cache full-recompute inference — our efficiency story has to be
  about train/inference symmetry, not serving latency.

## Missed citations worth adding

I grepped `refs.bib` for each; these are cited by Repoformer, relevant to OUR selective-retrieval /
firing-conditioned framing, and appear **absent** (verify before adding — do not assume present):

- **mallen2023whennottotrust** — Mallen et al., "When Not to Trust Language Models: Investigating the
  Effectiveness of Parametric and Non-Parametric Memories" (arXiv:2212.10511, ACL 2023). The canonical
  "retrieve only when the model doesn't already know" study (retrieval helps on long-tail/unpopular
  facts, hurts on popular ones). Directly upstream of Repoformer's selectivity and a perfect anchor
  for *why* we condition our grant on a firing event rather than granting everywhere. Absent (grep
  mallen: 0 hits).
- **kadavath2022knowwhattheyknow** — Kadavath et al., "Language Models (Mostly) Know What They Know"
  (arXiv:2207.05221). Foundational self-knowledge/calibration result — models can estimate whether
  they'll be right — which is exactly the capability Repoformer's `P(<cc>)` self-evaluation exploits
  and the theoretical basis for any learned grant-gate we might add. Absent (grep kadavath / "know
  what they know": 0 hits).
- **wang2023skr** — Wang et al., "Self-Knowledge Guided Retrieval Augmentation for Large Language
  Models" (SKR, Findings of EMNLP 2023; arXiv:2310.05002). Learns from the model's self-knowledge
  when to retrieve vs. rely on parametric memory — the closest NL-domain sibling to Repoformer's
  selective policy and to our firing-conditioned grant. Absent (grep self-knowledge/skr: 0 hits).
- **zhou2023docprompting** — Zhou et al., "DocPrompting: Generating Code by Retrieving the Docs"
  (ICLR 2023; arXiv:2207.05987). Retrieval-augmented *code* generation that fetches documentation
  rather than code snippets — a distinct retrieval-content type in the retrieval-augmented-code family
  we survey (we have ReACC, RepoCoder, CoCoMIC, RepoFusion, DraCo but not the docs-retrieval variant).
  Absent (grep docprompting: 0 hits).

(Checked and already present — NOT missing: asai2024selfrag, jiang2023flare, drozdov2022neighbors,
he2021efficientknnlm, ram2023incontextralm, shi2024replug, ding2023crosscodeeval, ding2024cocomic,
shrivastava2023repofusion, shrivastava2023rlpg, zhang2023repocoder, lu2022reacc, bavarian2022fim,
li2023starcoder, nijkamp2023codegen, nijkamp2023codegen2, roziere2023codellama, chen2021humaneval.)

---
Confirmation: deep-dive written to `paper/notes/deepdives/wu2024repoformer.md`, grounded in the
arXiv HTML render (method/tables/ablations "as-reported," PDF not re-verified) and the eval_harness.md
/ generation_retrieval.md / masks.md code briefs; the four suggested citations were grep-checked as
absent from refs.bib and wu2024repoformer itself is already present.
