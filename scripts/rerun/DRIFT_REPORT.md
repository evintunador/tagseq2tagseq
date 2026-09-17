# Eval rerun drift report (ledger-grounded subset)

Rerun of the 13 ledger-grounded training runs with the fixed, decoupled eval
(standalone eval run dirs + provenance re-attach by source_run_id). All 14 jobs
completed rc=0. Re-distilled into `provenance/runs/`. Values below compare the
ledger `expected:` (old, produced under the contamination bug) vs the freshly
re-evaluated value.

## Reproduced EXACTLY (12/16 metrics) — pipeline is faithful
- crossdoc.wiki_hotpotqa.{nll_crossdoc 5.62893, nll_flat 6.92007, n 738}  (incl. the
  max_grants=256 run — reproduced identically, so no regime concern after all)
- crossdoc.repobench_java.{bfs,dfs,rw,random}.{nll_crossdoc,nll_flat}  (all 8, exact)
- crossdoc.repobench_python.{nll_crossdoc 1.69981, nll_flat 1.79249}  (exact)
- singledoc.hellaswag.{cross_doc_link 0.29050, doc_causal 0.28300, doc_concatenated
  0.28450, doc_concat_link 0.28650}  (exact; plus CIs now available)

## CHANGED — compute.repobench_ppl.* (thestack repobench/doceval perplexity)
| claim-key | expected (old) | new (faithful doceval) | new 95% CI |
|---|---|---|---|
| compute.repobench_ppl.cross_doc_link  | 7.248 | 5.928 | [5.58, 6.34] |
| compute.repobench_ppl.doc_concat_link | 8.763 | 5.806 | [5.45, 6.18] |
| compute.repobench_ppl.doc_causal      | 8.941 | 5.901 | [5.55, 6.29] |
| compute.repobench_ppl.doc_concatenated| 10.417| 5.908 | [5.55, 6.30] |

n=500 in both old and new.

### Finding (SETTLED — 2 independent agents + code/history evidence converge)
The ~5.9 collapse is GENUINE, not a bug. Details:
- The flat `repobench` benchmark is **mask-invariant by construction**: its dispatch
  (`eval_checkpoints.py:554-560`) does NOT pass the condition's mask/layout to
  `run_repobench`; scoring hardcodes `mask_type='doc_causal'` on a single isolated DocSpan
  (`eval/scoring.py:585-599`) with cross-file snippets flat-concatenated as text. So under
  ANY condition (baseline/experimental/doceval) all four models are scored under the
  identical plain-causal attention — the mask can't matter. This hardcode predates the old
  numbers (present since 3c8f19a, 2026-07-13, and earlier).
- The forward pass is a real FlexAttention pass on each model's true weights
  (`doc_causal_flex` creator always built, `model/model.py:118-123`) — no degenerate/no-op.
- Corroboration: these 4 models' single-doc `held_ppl` are already tied
  (4.230/4.233/4.279/4.264); the new repobench avg_nll spread (~0.021 nats) matches that
  tightness; new 95% CIs all overlap heavily; the apparent ordering "flip" is within noise.
- => The OLD separated 7.2/8.9/10.4 numbers are the suspect ones. They cannot come from
  `run_repobench` (mask-agnostic); they came from an older/on-completion path and are the
  contamination. The eval-tracking fix does NOT change benchmark math (only output
  location) — independently verified.

### Decision needed (blocks re-grounding these 4 entries)
The compute-control claim "cross_doc_link wins on RepoBench ppl" is grounded on a benchmark
that STRUCTURALLY cannot show a cross-doc mask effect. Re-running repobench under
`experimental`/`baseline` will NOT help — that dispatch still flattens to doc_causal. Options:
- Re-ground the claim on `repobench_cross_doc` (the benchmark that actually fires cross-doc
  links) — but it only runs on cross_doc_link models, so it is NOT a 4-way table; the claim
  would need re-framing (e.g. cross_doc_only vs flat-fallback within the cross_doc_link model).
- Or drop the flat-repobench compute-control comparison from the paper.

The other 12 metrics need no ledger change (expected already matches). Hellaswag CIs are
newly groundable (additive).
