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

### Observations (interpretation UNDER INVESTIGATION — do not treat as settled)
- `doceval` (eval_checkpoints.py:141 comment) = doc_causal + eos layout applied to ALL
  models. Under this common layout the four new perplexities are close (~5.9, CIs overlap)
  and the ordering vs the old numbers appears to flip (cross_doc_link was best-old,
  worst-new).
- The OLD values came from the in-process on-completion eval: its `repobench/doceval`
  entry has ONLY `{perplexity, total_examples}` (no exact_match/nll/CI), i.e. a different
  code path than eval_checkpoints.py's full repobench benchmark.
- Two live hypotheses (independent agents are checking both):
  (a) `doceval`'s common-layout scoring is legitimate and the masks genuinely don't differ
      on this metric — OR it is a BUG that strips each model's learned structure and hides
      a real difference (bug-hunt in progress on the condition/layout override + attention
      backend path).
  (b) The old per-mask separated numbers were the contaminated/wrong ones.
- NOTE: the eval-tracking fix in this PR does NOT change any benchmark math (only output
  location), so the collapse is a property of running repobench under `doceval`, not of the
  fix. (Being independently verified.)

### Decision needed (blocks re-grounding these 4 entries) — after the above is settled
- If the intended comparison is each-model-under-its-own-mask, the ledger should point at
  the `experimental`/`baseline` condition, not `doceval`; re-run under that and re-ground.
- If faithful `doceval` (no effect) is the honest result, drop/re-frame the claim.

The other 12 metrics need no ledger change (expected already matches). Hellaswag CIs are
newly groundable (additive).
