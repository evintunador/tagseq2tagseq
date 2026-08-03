# Handoff: CompositeLinkDetector (Option A) — BUILT 2026-08-01

## Status (2026-08-01): BUILT + TESTED + SMOKE-VERIFIED on a real merged model.
Files: `model/graph_traversal/composite_link_detector.py` (new),
`link_detector.py` (registered `composite` in `LINK_DETECTOR_NAMES` +
`make_link_detector`), `data/layout.py` (`_DETECTOR_INFERENCE_LAYOUT["composite"]
= "identifier_prefix_eos"`), `generate.py` (`--link-detector` override →
`load_inference_model(link_detector_override=...)`), `tests/test_composite_link_detector.py`
(52 tests). Full suite: 622 pass in the detector/layout/generation/corpus subset, 0
regressions.

**Design realised:** per-document dispatch, ONE sub-detector per doc.
- `index_doc_span(span)` and `detect_links_for_doc(span_tokens, raw_identifier)`
  dispatch by IDENTIFIER first (`_sniff_by_identifier`: extension / `::` / go-host),
  then a content sniff of the span. The identifier sniff reliably covers exactly the
  detectors whose `index_doc_span` transforms the key; ambiguous titles (wiki/arxiv)
  fall through to the identity key, which those detectors return anyway.
- `detect_links(input_ids)` — the generation-loop / whole-sequence path where NO
  identifier is available — dispatches by CONTENT sniff only (`_sniff_by_content`:
  weighted per-language syntax signatures, argmax, TS/JS split on TS markers).
- Cross-firing guard: pick one detector (never run-all-merge); a mis-sniff degrades
  to fewer/again-no links because a wrong `target_str` rarely matches a real span key
  or corpus doc. Verified in tests (a `](`-in-a-python-string does not win markdown).

**Smoke (real merged 3.9B cross_doc ckpt run_20260730_183342_811412, GPU):** identical
Python-flavored generation — stored `markdown` detector = **0 links detected**;
`--link-detector composite` = **6 links detected**. Exactly the gap this closes.

**Known limitation (deferred, Tier 2):** a single generated doc that MIXES syntaxes
(markdown embedding a `\cite{}`) is classified by dominant language and detected with
that one sub-detector. Per-token routing within one doc is not implemented — no current
use case needs it for qualitative single-root generation.

--- ORIGINAL PRE-BUILD BRIEF BELOW (historical) ---

## (pre-build) design settled, NOT on the critical path for the
## current merged-model RESULTS, but REQUIRED for merged-model generation.
Investigation done and design decided. Correcting an earlier (2026-07-31) framing that called
this "not needed yet" — that conflated two different things. The accurate split:

- **The merged-model paper numbers do NOT need the composite.** All three eval surfaces avoid
  cross-source text detection: held_out_perplexity (no links); community_pack_perplexity uses
  Option B graph-edge grants (`eval/scoring.py::link_to_target_from_graph_edges`, done+verified,
  re-run corpus-wide at 32768 on 2026-08-01); and the discriminating benchmark ports are each
  single-language and declare their own `<Lang>ImportDetector` (`detector_for_benchmark()`
  `eval_checkpoints.py:214-238`; harness `PortAdapter` fixes one language per run).

- **Merged-model GENERATION DOES need the composite.** Generation is the one path with no
  graph-edge shortcut: `model/generation_loop.py` text-detects links in the generated stream and
  resolves them against a `PretokCorpus`. A merged `cross_doc_link` model IS the mixed-source
  case by definition (one model, 11 link types); its config `link_detector: markdown` fires only
  on wiki syntax and finds nothing in the other 10 sources. There is no meaningful single-language
  fallback for a merged model's generation. This is the planned "Multi-dataset composition …
  per-dataset link detectors dispatched based on document provenance" capability in README.md.

Build when merged-model generation is wanted. It is not blocking the current eval/RESULTS work.

## Decided design (for whenever it IS built)
- **Dispatch: content sniffing, one sub-detector per doc.** Sniff each doc via its
  `raw_identifier` file extension first (`.py`/`.ts`/`.go`/…), falling back to lightweight
  syntax markers only when the extension is absent. Running exactly one sub-detector per doc
  avoids the TS/JS/markdown cross-firing that "run-all-and-merge" invites (TS/JS
  `\bfrom "..."` / `require(...)` are unanchored and fire on arbitrary text; markdown keys on
  the `](` token 16151 and fires inside code). Optional robustness layer: keep only links
  whose `target_str` resolves to a real co-packed doc, so a mis-sniff yields fewer links
  rather than false grants.
- **Registration: a new `'composite'` entry** in `LINK_DETECTOR_NAMES` +
  `make_link_detector` (`model/graph_traversal/link_detector.py:75,78`), config-selectable via
  `model.link_detector: composite`. Pure wrapper — no mask-creator/training/Option-B changes.
- **Interfaces the wrapper must implement** (verified against `cross_doc_mask.py`): both
  `detect_links_for_doc(span_tokens, raw_identifier)` (per-doc path, selected by
  `hasattr` at `cross_doc_mask.py:946`) AND `index_doc_span(span)`, dispatching BOTH on the
  same per-doc sniff so the target-side match key stays aligned with the source-side
  `target_str` (e.g. Python's bare-path `index_doc_span` override). `detect_links(input_ids)`
  on the whole packed sequence is the fallback interface. All 12 detectors are stateless
  regex/token matchers taking only `decode_fn` — no tree-sitter at runtime — so per-doc
  dispatch is cheap.
- **Not chosen:** a source/language field on `DocSpan` (`data/collate.py:16-38` has none
  today, and it would be unavailable in the raw-text inference case the composite exists for);
  plain run-all-and-merge (cross-firing).

## Original brief (retained for context)
This needed the user's design decisions first (see "Open design questions"). First job was to
investigate the existing code and bring the user a concrete design proposal, NOT to build.

## The problem this solves (background)
A merged model is trained on 11 sources, each with its own link detector (markdown, python,
go, java, typescript, kotlin, rust, javascript, zig, dart, arxiv). At inference/benchmark
time, when links are NOT known ahead of time and must be detected from raw text, the model's
SINGLE configured detector (e.g. markdown) fires on only one source's syntax and finds
nothing in the others. A composite detector that dispatches to the right sub-detector per
document is (probably) the fix.

## What is ALREADY handled (do not rebuild)
- **Training + val/test eval**: use baked/graph-edge links, NOT text detection. That path is
  "Option B", already implemented (`eval/scoring.py::link_to_target_from_graph_edges`). The
  composite detector is ONLY for cases where links are unknown ahead of time (benchmark
  ports, generation).
- The `_BENCHMARK_LINK_DETECTOR` map + `detector_for_benchmark()` in `eval_checkpoints.py`
  already picks a per-benchmark detector for the named cross-doc ports. Understand how that
  interacts with (or is superseded by) a composite detector before proposing anything.

## Where to look first (investigate, report back)
- `model/graph_traversal/link_detector.py` — the `LinkDetector` protocol + `make_link_detector`
  + `LINK_DETECTOR_NAMES`. The composite must satisfy this protocol.
- The per-language detectors: `markdown_link_detector.py`, `python_import_detector.py`, and
  the tree-sitter-based ones (go/java/ts/kotlin/rust/zig/dart/javascript), `arxiv_cite_detector.py`.
- How `CrossDocLinkMaskCreator.__call__` calls the detector: `detect_links(tokens)` vs
  `detect_links_for_doc` + `index_doc_span(span)` (per-doc path). A composite must implement
  BOTH interfaces consistently.
- `eval/benchmark_harness/` — the frozen cross-doc port harness that would consume this.

## Open design questions (bring answers/options to the user — DO NOT decide these yourself)
1. **Dispatch signal**: how does the composite know which sub-detector applies to a given
   document? Options include a per-doc language/source tag threaded through `DocSpan`, or
   content sniffing, or running all detectors and merging. Each has correctness/perf/
   cross-firing tradeoffs. The user must choose — do not assume a source tag exists at
   inference time (it may not).
2. **Cross-firing risk**: if multiple sub-detectors run on one doc, a python detector may
   match something in a JS file, etc. How to prevent false grants?
3. **Scope**: is this needed for BOTH generation and benchmark ports, or benchmarks only
   right now? (Affects whether it must be fast/streaming.)
4. **Registration**: new `'composite'` entry in `LINK_DETECTOR_NAMES`, or a wrapper the
   inference model builds internally? Affects config surface.

## Deliverable (phase 1, before any code)
A 1-page design proposal presenting the options above with tradeoffs and a recommendation,
for the user to approve. Only after approval, implement + test.

## Don't
- Don't touch training, the merged pipeline, or Option B (graph-edge eval).
- Don't disturb live jobs (`ts2ts_*` in squeue) or the yield-watcher.
- Don't assume a per-doc source/language label is available at detect time — verify.
