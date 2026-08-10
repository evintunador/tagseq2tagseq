# shi2024incontext — In-Context Pretraining: Language Modeling Beyond Document Boundaries

Shi, Min, Lomeli, Zhou, M. Li, Szilvasy, James, X. V. Lin, Smith, Zettlemoyer, Yih, Lewis.
arXiv:2310.10638 (v1 Oct 2023, v6 Jun 2024; CC-BY-4.0). Code: github.com/swj0419/in-context-pretraining.
Grounded against our code briefs `packing_density.md`, `traversal.md`, `masks.md` and
`related_work_notes.md` (commit 6134163). Paper facts below are confirmed from the arXiv
abstract page and the v6 HTML full text; where I extrapolate to our method I say so.

## What the paper actually does

**One-sentence thesis.** Keep the entire LM pretraining pipeline byte-for-byte identical —
same objective, same architecture, same full-causal attention — and change *only the order in
which documents are concatenated* so that each context window is filled with semantically
*related* documents instead of random ones. That single change ("read and reason across
document boundaries") yields broad downstream gains.

**Method, in two stages.**
1. **Retrieval (build a kNN similarity graph).** Every document is embedded with **Contriever**
   (izacard2022contriever, already in our refs) — mean-pooled last hidden state over the first
   512 tokens. Pairwise **cosine similarity** defines edge weights; each document keeps its
   **top-k = 10** nearest neighbors. Approximate search uses **FAISS** IVFPQ (code size 256,
   32,768 inverted lists, nprobe 64 ≈ 0.2% probed, 62 GB index over 235M × 768-d float32 vectors)
   via their "OIVFBBS" offline big-batch search. Retrieval scores are reused to **deduplicate**
   near-identical documents (critical — see ablations).
2. **Ordering (graph traversal).** They cast context construction as a **maximum traveling
   salesman problem** (Flood 1956) over the kNN graph: a max-weight path visiting every node
   exactly once, then sliced into 8192-token contexts. Exact TSP is intractable at 235M nodes, so
   **Algorithm 1 is a greedy path**: start at a minimum-degree unvisited node, repeatedly hop to
   the highest-weight *unvisited* neighbor; when a node's neighbors are all visited, jump via a
   weight-0 edge to a random minimum-degree node and continue. This guarantees each document
   appears **once** (∪Cᵢ = D, no data repetition) while keeping adjacent documents related.

**Crucially: no attention change.** The paper is explicit that it "only changes the document
ordering and leaves all other aspects of LM pretraining untouched." There is **no cross-document
mask, no boundary reset, no per-doc position reset** — it is plain full-causal attention over the
8192 window. Because standard concatenation is *also* full-causal, the delta versus the baseline
is purely the ordering. This is the single most important fact for our comparison.

**Scale/setup (confirmed).** LLaMA architecture from scratch; 0.3B / 0.7B / 1.5B / 7B params;
**8192 context**; English CommonCrawl, **235M documents ≈ 306B tokens**; AdamW (β 0.9/0.95),
cosine LR; 7B run = 128 A100s / 16 nodes, 4M-token batch, ~9 days, FlashAttention. Offline cost:
retrieval ~6 h on 32 GPUs, traversal ~12 h on 20 CPUs.

**Result numbers that matter (7B unless noted; ICLM = their method vs Standard concat vs kNN
overlap-ordering baseline).**
- **In-context learning** (7 classification tasks, 32 shots): Standard 66.0 → **ICLM 71.3** (+8%);
  kNN 61.8 (kNN ordering *hurts* — overlap/repetition).
- **Reading comprehension** (2-shot avg over RACE/BoolQ/SQuAD/HotpotQA/DROP): 37.6 → **43.2**
  (+~15%). HotpotQA specifically = 21.9.
- **Factuality / faithfulness to context** (EM): NQ-Swap 39.6 → **45.8**; MemoTrap 48.4 → **56.2**
  (+~16%).
- **Retrieval-augmentation** (open-book EM): NQ 28.5 → **32.2**, TriviaQA 48.1 → **51.6** (+~9%);
  closed-book essentially unchanged (the gain is from *using* the prepended docs).
- **Long-context reasoning** (SCROLLS, finetuned): 32.5 → **34.1** (+~5%, the weakest axis).
- **Ablations.** Relevance ladder (1.5B, Wiki PPL): random 8.2 → clustering 7.9 → traversal-links
  7.3 (more relevance = lower PPL). **Dedup** (1.5B Wiki PPL): no-dedup 8.3 → dedup 7.3 (near-dups
  induce copying + training instability). ICLM overtakes Standard after ~150B tokens and the gap
  **does not shrink with scale**; kNN ordering is consistently worse than ICLM because overlapping
  neighborhoods repeat popular docs and overfit.

## Methodology: theirs vs. ours

The two projects sit on **orthogonal axes** of the same design space — which is exactly why this
is the headline packing comparator (`related_work_notes.md` §3 already flags it as "the most
important prior art").

**Axis 1 — What enters the model as the cross-document signal.**
- *Them:* **ordering only.** Attention is held fixed at full dense causal; the sole intervention
  is placing related docs adjacent. They answer: *does adjacency help under unchanged attention?*
- *Us:* **an attention edge.** Our base attention is the opposite of theirs — `doc_causal` is
  **block-diagonal**, `M = (q≥k) & (doc(q)==doc(k))`, i.e. cross-document attention is *forbidden*
  by default (`masks.md`). We then selectively re-open it only along explicit link edges
  (`cross_doc_link`: grant rectangle `[link_end_pos, A.end) × [B.start, B.end)`, DAG-gated to
  backward links, `masks.md`). We answer: *does a targeted link-gated edge help beyond
  concatenation?*

  Consequence: **ICLM is a strict attention *superset* of our default and even of our compute
  controls.** In ICLM every token attends to every prior token in the window regardless of
  document; this is *more* permissive than our `doc_concatenated` control (which restricts to the
  same union-find component, `masks.md`) and far more than `cross_doc_link`. So ICLM is best read
  as "full-causal concat + semantic-similarity ordering," a baseline whose *attention* we deliberately
  restrict and whose *ordering* we replace with a real graph.

**Axis 2 — Where the edge comes from.**
- *Them:* a **learned dense-similarity kNN graph** (Contriever embeddings + FAISS ANN), i.e. an
  *inferred* relatedness graph with approximation error (nprobe 0.2%). Edges are undirected,
  symmetric, weighted by cosine.
- *Us:* the **actual document graph** — hyperlinks / imports / citations — resolved by exact
  hashmap/identifier lookup (`traversal.py`, `dataset.py`: `normed_identifier` keys, directed
  `neighbors_in`/`neighbors_out`, no ANN, no approximation; out-of-index neighbors silently
  dropped). No learned retriever anywhere. Edges are **directed** and drive a **directed,
  asymmetric** attention grant (A reads B, never the transpose — `masks.md`).

**Axis 3 — Graph traversal.** Both literally "order documents by graph traversal," but over
different graphs and with different aims:
- *Them:* greedy **max-weight Hamiltonian-ish path** (max-TSP relaxation) over the similarity
  graph, whose *purpose is coverage without repetition* (visit-once). Restart = weight-0 hop to a
  random minimum-degree node.
- *Us:* **BFS / DFS / random-walk** strategies over the link graph (`traversal.py`), uniform-seeded
  per subwalk, restart = teleport to a **uniform-random** node (not restart-to-seed; not RWR —
  `traversal.md`). Purpose is *local neighborhoods that realize link edges*, not global coverage;
  we allow revisits and multiple subwalks per pack. And uniquely, our packer must **topologically
  reorder** (`prefer_targets_first`, Kahn per connected component, `pack_sampler.py`) so targets
  precede linkers — otherwise the causal DAG gate silently drops the grant (`traversal.md`: "the
  single most important non-obvious design point"). ICLM has no such coupling because it has no
  mask; ordering is free to be pure max-similarity.

**Axis 4 — Train/inference symmetry.** ICLM's ordering trick applies at *pretraining only*; at
inference you feed whatever context you like under the same unchanged full-causal attention. Our
mask is used **identically in pretraining and at inference** — a generated link deterministically
fetches its target node into the attention span (Option B baked grants offline for training,
text-detection at generation; `masks.md`). ICLM never fetches at inference; it relies on the
weights having learned to exploit adjacency.

**What we share.** (1) Both treat "related documents in one context" as the core resource, against
the random-concat baseline (brown2020gpt3 / touvron2023llama, in refs). (2) Both order by a graph
traversal and both worry about **data repetition** — they via visit-once TSP, we via drop_last +
component structure. (3) Both keep **RoPE positions unreset across the packed window** — ICLM is a
strong existence proof that a competent model tolerates cross-doc relative offsets under plain
causal attention, which is directly citable support for our "no per-doc position reset" choice
(`masks.md` reviewer flag #1; `related_work_notes.md` §3.2 kazemnejad/ruoss vs chen2023).

**Where we diverge hardest.** They demonstrate the *ordering* lever with attention fixed; we
demonstrate the *attention-edge* lever with FLOPs fixed (our `doc_concat_link` and
`doc_concatenated` matched-compute controls, `masks.md` novelty #5). Their gains cannot come from
an edge (they added none), and our claimed gains must survive subtracting the ordering effect —
which motivates the next section.

## Predictions & open questions for our method

- **Reading-across-boundaries tasks should move most; single-doc controls should not.** Their
  largest gains are reading comprehension (+15%) and faithfulness (+16%), the axes about consuming
  a *prior* document; ICL and long-context move less. Our edge is an even more targeted version of
  "read the related document," so we should expect our strongest signal on the same family —
  **HotpotQA and the multi-hop suite** (yang2018hotpotqa etc.) and NQ-Swap-style faithfulness — and
  **flat/no effect on single-document controls** (HellaSwag/ARC/PIQA…), which is exactly the
  control design in `related_work_notes.md` §6. If our cross-doc edge moved the single-doc controls,
  that would signal a confound, not our mechanism.
- **The bar our edge must clear is set by ordering alone.** ICLM's headline finding is that
  *reordering under unchanged full-causal attention* already buys +8–16%. Since that regime is
  strictly more attention-permissive than our `doc_concatenated` control, we should **expect
  `doc_concatenated`/`doc_concat_link` to capture a large fraction of the raw gain**, and the
  publishable claim rests on `cross_doc_link` beating those matched-FLOP controls. Prediction: the
  marginal lift of the *link gate over concat* will be smaller than the total lift over
  random-concat — plan the significance budget (bootstrap-CI, `related_work_notes.md` §6) around
  the *edge-vs-concat* contrast, not edge-vs-random.
- **Long-context is where our edge could out-run theirs.** Their long-context gain is only +5% and
  needed SCROLLS *finetuning*; diffuse semantic adjacency apparently helps least when the relevant
  span is far away. Our mechanism inserts a *specific* target's KV via an explicit grant rather
  than relying on proximity, so this is the regime where a targeted edge should beat mere ordering
  — a genuine open question their design cannot answer and ours can (ties to lost-in-the-middle,
  liu2024lostmiddle; RULER/HELMET, `related_work_notes.md` §6).
- **Scaling.** ICLM's gap opens only after ~150B tokens and *does not diminish* through 7B. This
  predicts (a) our effect may be invisible in short smoke runs and small token budgets — don't
  conclude "no effect" early; (b) the effect should persist rather than wash out with model scale,
  a point we can cite affirmatively.
- **kNN-ordering as a cautionary baseline.** Their kNN-overlap ordering *hurt* (ICL 61.8 < 66.0
  Standard) because overlapping neighborhoods repeat popular documents. Our analogue is a
  frontier-restart or degree-skewed random walk over-sampling hub nodes (high out-degree stationary
  bias, `traversal.md`). Prediction: naive relatedness without visit-once discipline can *degrade*
  — our drop_last/component machinery and traversal-strategy choice matter, not just "put related
  docs together."
- **Open question their design resolves for us.** They isolate the *ordering* contribution cleanly.
  If we run an ICLM-style "similarity-ordered full-causal concat" as an *additional* baseline
  alongside our graph-traversal concat, we can decompose our total gain into (ordering signal) +
  (attention-edge signal) — turning their paper into a ready-made ablation arm.

## Gotchas

- **Near-duplicate copying / training instability (their sharpest warning).** Without dedup, related
  docs that are near-identical make the model "merely copy from the prior document," hurting PPL
  (8.3 vs 7.3) and destabilizing training. Our corpora are *full* of exact/near dups along real
  edges: forked/vendored repos and re-exported files in The Stack, multiple arXiv versions or
  boilerplate-heavy citing papers, template Wikipedia stubs. When `cross_doc_link` grants read
  access into a near-duplicate target, the loss can collapse into copying — a plausible cause of a
  spurious "cross-doc win" that is really degenerate copying. We already dedup (SHA1/n-gram,
  `related_work_notes.md` §6) but ICLM shows *semantic* near-dups within a context are the danger;
  worth a targeted check on packs whose target ≈ source.
- **Ordering baseline is easy to under-power.** Their kNN baseline shows a *bad* ordering can look
  worse than random. If our concat control uses a weak traversal it will flatter `cross_doc_link`
  unfairly. Match the ordering across mask conditions (our compute controls already reuse the same
  pack layout — keep it that way; `packing_density.md`).
- **PPL vs downstream divergence.** Their relevance ladder is measured in Wiki PPL, but the
  headline story is downstream accuracy. Held-out perplexity is a weak proxy for cross-doc benefit;
  prefer our paired Δnll / downstream contrast (biderman2024lessons, schaeffer2023mirage, already in
  refs) rather than reporting a PPL improvement as the effect.
- **ICL-eval fragility.** They evaluate ICL with fixed 32 shots and see a plateau after 32; ICL
  accuracy is notoriously sensitive to demonstration order/format/calibration. If we report any
  few-shot ICL numbers, control for it (see missed citation zhao2021calibrate) — otherwise
  ordering-of-demonstrations noise can masquerade as a cross-doc effect.
- **Offline preprocessing is a real cost and a determinism hazard.** Their pipeline is 6 GPU-hours
  + 12 CPU-hours of retrieval/traversal for 235M docs; ours is analogous (offline traversal + link
  detection + density bucketing, `packing_density.md`). Their traversal is deterministic given the
  graph; ours has known determinism fragilities (shared RNG stream, DFS-consumes-RNG, precompute
  restart_prob 0.0 vs live 0.05 — `traversal.md`). If we cite their reproducibility we should not
  overclaim ours.
- **Long-context claims may need finetuning.** Their long-context gains only materialized after
  SCROLLS finetuning; a pure zero-shot pretraining-ordering effect at long range was weak. If we
  claim long-context wins from pretraining alone, expect a reviewer to point here.

## Missed citations worth adding

Checked against `paper/bib/refs.bib`. Already present and *not* missing: `izacard2022contriever`
(their retriever), `lewis2020marge` (their "Pre-training via Paraphrasing"), `guu2020realm`,
`caciularu2021cdlm`, `yasunaga2022linkbert`, `dao2022flashattention`, `touvron2023llama`. Genuinely
missing (verify before adding):

- **levine2022inductivebias** — Levine, Dalmedigos, Ram, Zeldes, Jannai, Muhlgay, Osin, Lieber,
  Lenz, Shalev-Shwartz, Shashua, Leyton-Brown, Shoham, "The Inductive Bias of In-Context Learning:
  Rethinking Pretraining Example Design," ICLR 2022, **arXiv:2110.04541** (believed correct —
  please verify). *Why it matters:* the theoretical/empirical origin of "which documents you place
  in the same pretraining context shapes downstream in-context behavior" — the intellectual
  motivation for *all* relatedness-packing including both ICLM and our traversal packing. A more
  foundational framing than ICLM itself for why our packing objective is principled.
- **abbas2023semdedup** — Abbas, Tirumala, Simig, Ganguli, Morcos, "SemDeDup: Data-Efficient
  Learning at Web-Scale through Semantic Deduplication," **arXiv:2303.09540** (believed correct —
  verify). *Why it matters:* ICLM shows semantic dedup is *load-bearing* for relatedness packing
  (near-dups → copying/instability). Our own `related_work_notes.md` explicitly flags SemDeDup as
  "surveyed but BibTeX not retained; uncited" — this closes that gap and directly supports our
  dedup design.
- **zhao2021calibrate** — Zhao, Wallace, Feng, Klein, Singh, "Calibrate Before Use: Improving
  Few-Shot Performance of Language Models," ICML 2021, **arXiv:2102.09690** (believed correct —
  verify). *Why it matters:* ICL few-shot scoring is fragile to ordering/format; if we report any
  ICL-style numbers this backstops the eval-methodology discipline in `related_work_notes.md` §6
  (alongside biderman2024lessons / schaeffer2023mirage).

Lower priority / likely out of scope (listed for completeness, probably skip): Flood (1956) "The
traveling-salesman problem" (our packing is not TSP-formulated — cite only if we discuss their
ordering algorithm in depth); de Vries (2023) "In the long (context) run" (a blog post on
long-context data scarcity, no arXiv id found — do not cite without a stable reference);
Yasunaga et al. (2023) "Retrieval-augmented multimodal language modeling" (multimodal, off-axis).

---
Confirmation: paper method/results verified against arXiv:2310.10638 abstract + v6 HTML; our-side
claims cross-checked against code briefs packing_density.md / traversal.md / masks.md and
related_work_notes.md; missing-citation candidates confirmed absent from refs.bib via grep.
