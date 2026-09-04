# Link-injection causal eval — design

## Goal

Show, at inference time, whether the cross-doc-link training regime lets a model
*exploit* an injected link + auxiliary document more than doc-causal training does.
This is the inference-time causal counterpart to the training-time ablations
(val-loss / diversity scaling) and to the true-link-native benchmarks.

The headline quantity is **not** an absolute benchmark score. It is an
**interaction**: the difference between how much the aux doc helps a cross-doc-trained
model and how much the identical aux doc helps a doc-causal-trained model. Absolute
lift conflates "cross-doc training taught the model to use links" with "relevant
context helps any LM" (generic in-context learning). Only the interaction separates
them.

## What already exists (reuse, do not rebuild)

- `eval/link_annotator.py` — `MarkdownPromptAnnotator` / `ArxivPromptAnnotator`: inject
  a `[display](Title)` / `\cite{Title}` link at the model's highest-probability slot,
  generate the target title, acquire an aux doc. Five `link_retrieval_mode`s:
  `full_skip`, `link_but_skip`, `corpus_only`, `generate_only`, `corpus_then_generate`.
- `eval/nlp_benchmarks.py::run_benchmark_annotated` — scores annotatable benchmarks
  under a flat baseline + link-probability thresholds.
- `eval/scoring.py::score_completion_with_context_docs` — packs aux docs before the
  primary doc and scores completion NLL under `cross_doc_link`.
- `eval/benchmark_harness/tier2.py::placebo_separation` — derangement placebo
  (each example's aux replaced by a wrong-example aux, matched for count/length) with
  a bootstrap-CI gate. Built for the port harness; **not** wired to the injection arm.
- `eval/title_index.py` — `HashNormTitleIndex` / `TrieTitleIndex` (corpus title lookup).

## The mask taxonomy is the lever

The regime is a *masking* choice over shared weights — there are no cross-doc-specific
learned parameters (`model.py::_build_creators`). This makes the whole factorial a set
of mask-type overrides on one packed sequence:

- `doc_causal` — each doc attends only within itself. Aux packed alongside the primary
  is **invisible**. Equivalent to the no-aux baseline; used as a sanity anchor.
- `doc_concatenated` — full causal over the whole pack, no doc boundaries. Aux is
  **visible as ordinary prior context**. This is the "raw-concat" control.
- `cross_doc_link` — `doc_causal` + a grant fired at the link position that lets the
  primary doc attend to the linked aux. This is the system.

`doc_causal` creators are always built. `cross_doc_link` creators are built when the
model is declared `cross_doc_link` (needs a link_detector). `doc_concatenated` is only
built when declared — **extend `_build_creators` to always build it** (needs no
link_detector; reuses the doc-causal varlen kernel), mirroring the doc_causal rationale.
Then a single model construction (`mask_type=cross_doc_link` + `MarkdownLinkDetector` +
always-build `doc_concatenated`) can run every cell.

## The factorial (aux content held constant across all cells)

|                       | grant ON (`cross_doc_link`) | raw-concat (`doc_concatenated`) |
|-----------------------|-----------------------------|---------------------------------|
| cross-doc-trained ckpt | A — the system              | B — mask ablation               |
| doc-causal-trained ckpt| C — mask alone, no training | D — vanilla + context           |

Plus, per weight set, a **no-aux baseline** (`doc_causal` on the primary alone / flat).

- A − baseline = the system's total lift.
- (A − baseline) − (B − baseline) = **mechanism**: what the link grant adds over plain
  concatenation, holding weights + content fixed.
- (A − baseline_xdoc) − (C − baseline_dc) = **training**: whether the model must have
  been *trained* under the regime to exploit the grant, or the inference-time mask alone
  suffices.
- D is the vanilla reference (any LM + more context).

Loading doc-causal weights into a `cross_doc_link`-constructed model is valid because
the mask is non-parametric; assert state-dict keys match on load.

Interpretation guard: cells C/A run the doc-causal weights under a mask their training
never saw (OOD). Cell D (raw-concat, in-distribution for both) is the anchor that
distinguishes "training didn't teach utilization" from "the OOD mask merely breaks
generation."

## Aux-content variants (the relevance gradient)

Scored through cells A and C (and placebo through A):

1. gold / oracle — an aux doc that contains the answer (ceiling).
2. strong-LLM-generated — external-LLM aux (deferred; oracle-ceiling arm).
3. retrieved most-similar — `corpus_only`.
4. trained-model-generated — `generate_only` (the floor the current model can reach).
5. placebo-real — a real but wrong-example aux (derangement; tier2 machinery).
6. placebo-noise — shuffled/corrupted tokens (format-following without content).
7. none — no aux (baseline).

The cross-doc story predicts, for the cross-doc-trained model: a steeper positive slope
across relevance and better rejection (flatter drop) on placebos than the doc-causal
model shows.

## Concentration, not "across the board"

A knowledge-injection mechanism should help *concentrated* on knowledge-gap items, not
diffusely. Stratify by entity popularity (PopQA-style) or by target↔context n-gram
overlap α (leakage): the benefit should concentrate on rare / low-overlap items. Report
this stratification, not just the average.

## Leakage control (independent, pure-eval)

Cross-doc gains risk being re-exposure of memorized/duplicated training text
(Wikipedia, The Stack). On the finished sweep, stratify cross-doc Δnll by target↔context
n-gram overlap α (RETRO bpb(α) protocol) and show the effect survives at low overlap.

## Statistics

Interaction effects are second-order → need more items than a main effect. Use paired
items (same benchmark question across all cells), per-item paired deltas, and bootstrap
CIs on the interaction term specifically (`tunalab.stats_funcs.calculate_bootstrap_ci`,
already used in `run_benchmark_annotated`). Report over the *fired* subset honestly
(a help/neutral/hurt split makes averaging over non-fired items misleading).

## Implementation plan

1. **Annotate once, replay everywhere.** A driver builds `AnnotatedPrompt`s (link slot +
   title + aux) a single time and serializes them, so the identical aux is scored by both
   weight sets and every mask cell. This is what makes the content-controlled comparison
   valid, and it is the hook for swapping in gold / placebo / LLM aux later.
2. **Parametrize scoring by mask.** Add a `mask_type` argument to
   `score_completion_with_context_docs` (default `cross_doc_link`) and a light
   `score_completion_concat` for the `doc_concatenated` / `doc_causal` cells that packs
   aux + primary and skips the link-detection gate.
3. **Extend `_build_creators`** to always build `doc_concatenated` creators.
4. **Weight loading** is already provided: `generate.py::load_inference_model` takes
   `mask_type_override` and `link_detector_override`, so the doc-causal checkpoint loads
   under a `cross_doc_link` mask + markdown detector directly (masking is non-parametric).
   No custom loader needed.
5. **Grid driver** (`eval/link_injection_grid.py`): over the cached annotations, score
   each (checkpoint × mask cell × aux variant) and emit per-item paired deltas + the
   interaction stat with bootstrap CIs.
6. **Placebo**: reuse tier2 derangement over the cached aux docs.

## Concrete run

Matched wiki_merged pair (identical 1024d/24L/8H, 32768 ctx, 3614 steps, seed 42, bfs,
world 2; only `mask_type` differs; byte-identical checkpoint size):

- cross-doc: `runs/20260703_050528/checkpoints/best_model.pt`
- doc-causal: `runs/20260703_051129/checkpoints/best_model.pt`
- annotator corpus: `/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/wiki_merged`

This is a modestly-trained pair (3614 steps) — the interaction must use it because no
matched doc-causal run exists at the 14507-step scale of the most-trained cross-doc runs.
The more-trained cross-doc checkpoints can still serve absolute-capability arms (e.g. how
good a `generate_only` aux the model can write), just not the matched interaction.

Runs live in the main checkout `/fss/evin_t/tagseq2tagseq/`.

## Out of scope

- Link *injection* for code datasets — the repo already rejects this (TODOS L415);
  code uses the real import edge (present/withhold), a separate line of work.

## Implementation notes

- **Grant cells use the detector's coarse mode** (`aux_raw_identifiers=None`): the
  re-detected injected link grants access to every packed aux span. With one link and one
  aux per record this equals the precise grant, and it does not depend on
  `detect_links`' re-extracted `target_str` matching the aux identifier — that match fails
  for titles containing `)` (the detector truncates at the first `)`), which silently
  dropped 10/40 fired sciq items from the grant/placebo cells in the smoke.
- **Gold aux (relevance-gradient ceiling)** is the benchmark's own passage, attached per
  record as `gold_aux_tokens` and scored through the same injected link as
  `grant_gold` / `concat_gold` / `placebo_gold`. Only `sciq` (`support`) qualifies:
  hotpotqa's annotatable context already contains the gold supporting sentences, so an
  injected gold aux would be redundant there. Reported extras: the `_gold` block,
  `relevance_slope[W] = grant − grant_gold` and `relevance_slope_interaction`.
- **Replay** (`--replay-records`): re-score cached annotations without re-annotating or
  loading the corpus — used to add the gold cells to a finished run or to score another
  checkpoint pair on identical injected links + aux. Per-item cell NLLs are written to
  `<benchmark>_cell_scores.json` alongside the report so stratifications (leakage α,
  popularity) can be computed post hoc.
