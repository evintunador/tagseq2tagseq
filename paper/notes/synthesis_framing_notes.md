# Synthesis framing notes

Argument-and-framing conclusions distilled from the 31-paper deep dives
(`paper/notes/deepdives/_SYNTHESIS.md`). These are positions the paper should
*state in prose* — they shape how a result is interpreted, not what gets run.
Executable follow-ups from the same synthesis live in `TODOS.md`; the pairing is
noted per item.

## The contribution is native corpus-referencing under maintained causality, not a general-intelligence gain

The concat literature (LinkBERT random-link ablation on baseline; SPLiCe /
In-Context Pretraining; DRAGON concat-vs-fused; K-BERT visible-matrix removal)
converges on: co-locating related documents under ordinary attention buys most of
the *general* long-context/ICL gain. That is a gain we do **not** claim — the
diversity-scaling design explicitly drops the "general intelligence" framing. Our
claim is structural: `cross_doc_link` maintains causality while granting attention
along explicit corpus edges, so the model natively learns to *reference corpus
documents at generation time* — a capability a flat concat baseline cannot have
because it has no notion of which document grants which.

Consequence for the results narrative: the number that isolates our contribution is
the residual `cross_doc_link - doc_concat_link` (edge minus matched-FLOP concat),
not `cross_doc_link - doc_causal`. Report the concat controls as first-class results,
framed as *rigor* (isolating the typed edge from the added FLOPs/adjacency), not as a
survival test. K-BERT even predicts concat may *underperform* cross_doc_link at
matched FLOPs — if that holds it is the strongest possible form of the result.

## RoPE position: a clean per-doc reset is ill-defined once grants cross boundaries

Reviewer exposure: we use global RoPE positions [0,T) with no per-document reset, so
a granted target sits at a packing-order-dependent relative offset. zhao2024analysing
resets RoPE per document — but only because its IntraDoc simultaneously *bans* all
cross-document attention, so each document is self-contained and there is no grant to
place at an offset. Once attention crosses a boundary along a grant (our case), a
clean reset is ill-defined: RoPE rotates each key by its single absolute position
before attention, but a granted target is read by linkers at many packing distances,
so no single re-basing represents all of them. A naive global-index reset only yields
a slightly-closer-but-still-arbitrary offset.

Framing: we therefore assess the concern empirically via the link-utilization-vs-
packing-distance diagnostic (see `TODOS.md`, Model) rather than a reset ablation. A
per-grant additive position embedding is the tractable-but-larger arm if the
diagnostic shows decay; both land in an appendix. State the no-reset choice as
defensible-with-evidence, not as an untested gap.

## Trained same-mask NTP is preferable to a bolted-on structure loss

GraphCodeBERT shows a structure mask *without* a structure objective leaves ~half its
gain on the table, and EMAT drops 44.4->24.7 without integration pretraining — a
reviewer may ask whether an auxiliary link/target-prediction loss would beat our pure
next-token objective. Position (option b): argue in prose that our train=inference
design — the *same* mask in pretraining and inference, learned through the existing
NTP loss — is the right call, and cite EMAT / GraphCodeBERT / kNN-LM / Memorizing
Transformers as evidence that *trained* integration (not a frozen or bolted-on
mechanism) is what matters. No aux-loss arm unless a referee forces it.

## Scale behavior is a hypothesis under test, not a preconclusion

The literature (kNN-LM, RepoCoder, RETRO, Galactica) predicts retrieval/memory gains
are largest at small scale and shrink as models grow. Our scaling runs are designed
precisely to test whether edge-gated cross-doc attention behaves the same way or
bucks the trend; those results are not yet in. State this as an open question the
scaling experiments address — do not preconclude that small-model wins fade.

## Packing-distance limitation

`prefer_targets_first` (Kahn topo sort) is load-bearing: it keeps grants firing under
the backward-only DAG gate, and it is applied identically to `cross_doc_link` and both
concat arms, so the favorable short-offset it produces cancels in the
`cdl - concat` contrast (it is not a confound). The flip side is a limitation to state
plainly: because targets-first keeps packing distances short and correlated, the
"advantage grows with link-to-target packing distance" prediction is not cleanly
testable in our setup.

## Citation reconciliation

Our arXiv graph has ~1.98M nodes, matching a newer "unarXive 2024 extended" release,
not the cited saier2023unarxive (2022, 1.88M). Reconcile the Methods citation to the
release actually ingested.

## Kernel / numerical-stability justifications (systems appendix)

- The sentinel-LSE NaN guard for fully-masked rows is independently confirmed a real
  hazard: SCFA has the identical stranded-query fix, and NSA forces always-selected
  initial/local blocks for the same reason. [pagliardini2023sparseflash, yuan2025nsa]
- FlashMask's column-wise mask provably cannot express our arbitrary A->B grant
  rectangles — the clean justification for the bit-packed grant encoding.
  [wang2024flashmask]

## Bonus diagnostic worth a figure (optional)

zhao2024analysing's DistrProp metric (attention mass a token places on the irrelevant
preceding doc; they measure 45-52% under naive causal packing) computed on
`doc_concatenated` (expect high) vs `cross_doc_link` (expect mass concentrated on the
linked target, near-zero on unlinked docs) would directly show our mask converts
diffuse harmful cross-doc attention into targeted attention. Cheap and publishable;
promote to `TODOS.md` if we decide to run it.
