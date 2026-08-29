# STATUS — eval run-dir isolation + rerun

**Read this first to re-orient.** Worktree `/fss/evin_t/tagseq2tagseq-evaltrack`, branch
`eval-run-tracking` (off `provenance-grounding`).

## What this is
The eval script was writing results INTO training run dirs (and merging into training's
own `eval_results.json`), polluting paper grounding. Fix: eval now creates its own
standalone off-repo run dir via ReproducibilityManager (like training does), and the
provenance distiller re-attaches those metrics to the training record by `source_run_id`.
Then re-run the affected evals since the old numbers can't be trusted.

## DONE (validated)
- `eval_checkpoints.py`: every eval writes a standalone dir under `$TS2TS_EVALS_ROOT`
  (default `/fss-data/evin_t/tagseq2tagseq_artifacts/evals/<eval_id>/`) + a `manifest.json`
  linking to source training run(s). Never touches training run dirs. No more merge.
- `scripts/distill_runs.py`: scans the evals root, re-attaches metrics to the training
  record by `source_run_id` (latest-wins, records `eval_provenance`). `ledger.yaml` unchanged.
- Reran all 14 ledger-grounded eval jobs (local, on drained nodes GPU-658/652), rc=0.
- Re-distilled into `provenance/runs/`. Drift computed — see `DRIFT_REPORT.md`.

## Result
- 12/16 ledger metrics reproduced EXACTLY (hotpotqa, all repobench_cross_doc java+python,
  all hellaswag). No ledger change needed for these.
- 4× `compute.repobench_ppl.*` COLLAPSED to ~5.9 (was 7.2/8.9/10.4/8.8) and the mask
  separation vanished. CAUSE UNDER INVESTIGATION — two live hypotheses: (a) `doceval`
  scores every model under a common doc_causal layout, which may be legitimately
  measuring "no effect" OR may be a bug that hides the real mask difference; (b) the old
  per-mask numbers were the contaminated/wrong ones. Do NOT treat either number as truth
  yet. Independent agents are checking the benchmark design + hunting for a bug.

## BLOCKED ON YOU (2 decisions)
1. **repobench_ppl compute-control claim:** pending the investigation into WHY the 4
   numbers collapsed (design vs bug). Likely resolution is to re-run under the
   `experimental`/`baseline` condition (each model under its own trained mask) and
   re-ground — but confirm after the design/bug question is settled.
2. **Quarantine** the old contaminated `runs/*/eval_results.json` + `runs/*/eval/`? Deferred
   because the main checkout's `runs/` is shared with live peer sessions.

## NOT yet done (intentionally)
- Nothing committed. `ledger.yaml` `expected:` values NOT edited. RESULTS*.md NOT edited.
- Old files NOT quarantined.

## To resume
- Detail: `scripts/rerun/DRIFT_REPORT.md`. Rerun matrix: `scripts/rerun/jobs.tsv`.
- Re-run a condition variant: edit conditions in `jobs.tsv`, then `bash scripts/rerun/orchestrate.sh`
  (idempotent — skips jobs with a `.done` marker in `scripts/rerun/logs/`).
- Re-distill: `python scripts/distill_runs.py --roots /fss/evin_t/tagseq2tagseq/runs
  /fss-data/evin_t/tagseq2tagseq_artifacts/runs --evals-roots
  /fss-data/evin_t/tagseq2tagseq_artifacts/evals --run-id <ids>`
