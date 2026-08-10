# Deep-dive synthesis: cross-cutting findings for the paper

Distilled from 31 per-paper deep dives (`paper/notes/deepdives/*.md`), each cross-referenced
against our implementation. These are the recurring, multiply-corroborated conclusions that
should shape the method, experiments, results, and discussion sections. Bracketed keys are the
comparators that independently support each point.

## 1. The matched-compute concat controls are the make-or-break ablation
Multiple prior works show that *co-locating related documents under ordinary attention* already
buys most of the observed gain, independent of any structural edge:
- LinkBERT's random-link ablation lands exactly on the no-link baseline — only *relevant* links help.
- SPLiCe and In-Context Pretraining get large long-context/ICL gains from related-doc packing alone,
  with standard causal attention and no cross-doc mechanism.
- DRAGON's "concatenate at end" (74.5) and "verbalize graph to text" (74.7) both lose 2–4 pts to the
  fused/explicit-graph model.
- K-BERT with the visible matrix removed (all tokens mutually visible) does *worse than plain BERT*.
- DeepSeek-Coder's Table 7: import-topological ordering alone buys only ~1–2 EM, and only on Java/TS/C#.
**Implication.** The headline number that isolates our contribution is the residual
`cross_doc_link − doc_concat_link` (edge minus matched-FLOP concat), NOT `cross_doc_link − doc_causal`.
K-BERT even predicts `doc_concatenated` may *underperform* `cross_doc_link` — gating is the active
ingredient, not the added FLOPs/adjacency. Report the concat controls as first-class results.
[yasunaga2022linkbert, staniszewski2025structured, shi2024incontext, yasunaga2022dragon, liu2020kbert, guo2024deepseekcoder]

## 2. Ablate per-document / per-edge RoPE position reset — our biggest reviewer exposure
Our masks use global RoPE positions [0,T) with no per-document reset, so a target reached at large
packing distance sits at a large, packing-order-dependent *relative* offset.
- zhao2024analysing resets RoPE per document *on purpose* (and its DistrProp analysis measures 45–52%
  of attention wasted on the irrelevant preceding doc — the harm we must avoid).
- K-BERT's soft-position scheme deliberately overlaps indices to preserve relative offset; its ablation
  shows dropping it hurts — direct evidence a reviewer can cite *against* our no-reset choice.
- DeepSeek-Coder trusts its RoPE-rescaled context only to 16K; we run 32k.
- Lost-in-the-Middle gives the mechanism (U-shaped utilization) by which a distant offset could bite.
- Memorizing Transformers / In-Context Pretraining are the mild counter-evidence (position-free or
  unreset and fine), so the question is genuinely open.
**Implication.** Run an ablation: per-doc (or per-edge) RoPE reset vs. global positions, and log
link-utilization vs. packing distance. This is cheap and pre-empts the sharpest likely referee objection.
[zhao2024analysing, liu2020kbert, guo2024deepseekcoder, liu2024lostmiddle, wu2022memorizing, shi2024incontext]

## 3. Report leakage-stratified Δnll (RETRO's bpb(α) protocol)
Our grant makes verbatim copying from the target *more* direct than retrieval baselines, so raw perplexity
gains risk being re-exposure of memorized/duplicated text.
- RETRO stratifies eval by train-test n-gram overlap α and reports the leakage-filtered curve.
- kNN-LM's "retrieval beats training on the same tokens" is partly the same effect.
- HotpotQA (2017 Wikipedia) and The Stack both overlap our training data; our dedup is weak/sampling-only.
**Implication.** Stratify the cross-doc Δnll by target↔context n-gram overlap; show the effect survives at
low overlap. Expect a steeper leakage slope than RETRO. Strengthen dedup or at least measure the overlap.
[borgeaud2022retro, khandelwal2020knnlm, yang2018hotpotqa, kocetkov2022stack]

## 4. The read/edge must be TRAINED, not bolted on — validates train=inference design
- EMAT: removing the integration pre-training task drops TQA 44.4→24.7; a frozen memory *without*
  pretraining is worse than no memory at all.
- GraphCodeBERT: data-flow-mask alone buys ~half the gain; its two auxiliary structure losses buy the
  other half — we have NO auxiliary loss.
- kNN-LM/Memorizing Transformers/TOME: frozen inference-time memory underperforms trained integration.
**Implication.** Our design (same mask in pretraining and inference) is the right call and should be framed
as such. Concrete new ablation arm suggested by GraphCodeBERT: add an auxiliary link/target-prediction loss
and measure whether it recovers a "structure-objective half" of the gain.
[wu2022emat, guo2021graphcodebert, khandelwal2020knnlm, wu2022memorizing, dejong2022tome]

## 5. Headline benchmarks lack a placebo/counterfactual arm
Our Tier-2 ports have a derangement placebo (right vs. wrong aux), but the headline HotpotQA-cross-doc and
RepoBench-cross-doc numbers report only cross-vs-flat, where the cross arm sees strictly *more tokens*.
- HotpotQA's own single-hop-solvability finding (min2019necessitate) makes "is it the right doc or just more
  context?" acute.
- DraCo's `w/o dataflow` and Repoformer's help/neutral/hurt split are external analogs of the placebo.
**Implication.** Add a placebo (wrong-but-plausible aux) to the headline arms, or caveat prominently. Report
Δnll over the *fired* subset honestly (Repoformer's 20/60/20 help/neutral/hurt split justifies this).
[yang2018hotpotqa, liu2024repobench, cheng2024draco, wu2024repoformer]

## 6. Replace positional max_grants truncation with importance ranking on hub-heavy graphs
Our grant cap (default 64/256) drops excess grants by link *position*, not importance.
- The Stack graph is dominated by auto-generated SDK / re-export `__init__.py` hubs; WikiLinkGraphs and
  HippoRAG both show heavy-tailed hub degree.
- HippoRAG's IDF-weighted node specificity is the natural fix; Lost-in-the-Middle flags positional
  truncation as a self-inflicted bias.
**Implication.** For hub-dense packs, rank grants by an IDF-like specificity before truncating; report how
often the cap binds and how realized grant density compares to nominal node degree.
[kocetkov2022stack, consonni2019wikilinkgraphs, gutierrez2024hipporag, liu2024lostmiddle]

## Recurring quantitative predictions (for the results narrative)
- Effect concentrates on multi-hop / cross-file / rare-entity items; ~null on single-doc MC controls and on
  reasoning/math (RETRO regressed there). [most comparators]
- Effect likely LARGEST at small model scale and shrinks with scale (kNN-LM, RepoCoder, RETRO, Galactica all
  show retrieval/memory helping small models most) — consistent with our own merged-model "flat 3.9B→8B".
- Per-source ordering: strongest on wiki hyperlinks, weaker on code imports (local/predictable), weakest on
  arXiv citations — corroborated mechanistically by unarXive (huge docs, only ~44% of linked pairs co-fit in
  32k) and by our own community-pack Δ (wiki +0.16 ≫ stack +0.08 ≫ arXiv +0.004).
- Advantage should GROW with link-to-target packing distance (Lost-in-the-Middle) and with context length
  (SPLiCe's widening gap) — but concat baselines must be run with RANDOMIZED order too, else targets-first
  packing lets the concat control absorb the effect. [xu2024retrievalmeetslong, staniszewski2025structured]

## Numerical-stability / kernel validations (supporting the systems appendix)
- Our sentinel-LSE NaN guard for fully-masked rows is independently confirmed a real hazard by SCFA (identical
  stranded-query fix) and NSA (forces always-selected initial/local blocks). [pagliardini2023sparseflash, yuan2025nsa]
- FlashMask's column-wise mask provably cannot express our arbitrary A→B grant rectangles — the clean
  justification for our bit-packed grant encoding. [wang2024flashmask]

## Methods-citation fix flagged
Our arXiv graph has ~1.98M nodes, matching a newer "unarXive 2024 extended" release, NOT the cited
saier2023unarxive (2022, 1.88M). Reconcile the Methods citation to the release actually ingested.
