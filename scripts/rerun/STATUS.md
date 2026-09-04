# STATUS — eval run-dir isolation + rerun

**Read this first to re-orient.** Worktree `/fss/evin_t/tagseq2tagseq-evaltrack`, branch
`eval-run-tracking` (off `provenance-grounding`). **Draft PR: #9**
(https://github.com/evintunador/tagseq2tagseq/pull/9), base `provenance-grounding`,
mechanism only (re-grounding held pending the repobench question below).

## repobench finding (why the 4 numbers collapsed)
The flat `repobench` benchmark is **mask-invariant by design**: `run_repobench` →
`score_completions_independent_batched` hardcodes `mask_type='doc_causal'` and applies no
layout (`eval/scoring.py:599`); the eval `condition` only gates whether it RUNS, never how
it scores (agent-confirmed). So under `doceval` all 4 masks score identically flat → ~5.9.
This hardcode dates to 2026-07-13, BEFORE the old numbers (~07-20), so the old separated
7.2/8.9/10.4 came from a DIFFERENT path (main.py on-completion eval, sparse
`{perplexity,total_examples}`), not today's benchmark. The genuine cross-doc probe is
`repobench_cross_doc` — but it only runs on cross_doc_link models, so it can't form a 4-way
table as the ledger's compute-control section assumes. => the compute-control claim's
grounding metric needs rethinking (not just a rerun).

## Re-eval scope (paper vs everything)
- Paper grounding = `ledger.yaml` ONLY (gen_values_tex + check_grounding). ~14 jobs / 10
  runs / 4 metric_paths. This is the subset we reran — DONE (bar the repobench decision).
- Full contaminated universe = ~639 eval_results jobs + sidecars ≈ 800+ job-equivalents,
  ~144 runs, 15+ benchmark families (dominated by held_out_perplexity + community_pack_
  perplexity). ~50× the ledger, but only feeds the RESULTS_*.md analysis docs, NOT the
  paper. Optional.

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
- 4× `compute.repobench_ppl.*` COLLAPSED to ~5.9 (was 7.2/8.9/10.4/8.8). SETTLED (2
  independent agents + code/history): the new ~5.9 is GENUINE and correct — flat `repobench`
  is mask-invariant by construction, so all 4 masks score identically; the OLD separated
  numbers are the contamination (from an older/on-completion path, architecturally
  impossible from run_repobench). NOT a bug in the fix (computation unchanged).

## BLOCKED ON YOU (2 decisions)
1. **repobench_ppl compute-control claim:** the paper's "cross_doc_link wins on RepoBench
   ppl" is grounded on a metric that structurally can't show a cross-doc effect. Re-running
   under experimental/baseline will NOT help (repobench always flattens to doc_causal). Real
   options: re-ground on `repobench_cross_doc` (cross_doc_link-only → not a 4-way table, needs
   re-framing) OR drop the flat-repobench compute-control comparison. Your call.
2. **Quarantine** the old contaminated `runs/*/eval_results.json` + `runs/*/eval/`.
   Answered: nothing in the main checkout's `runs/` has been touched since 2026-08-07 (94
   `eval/` dirs, 141 `eval_results.json`; the 6 live jobs all write under fss-data), so it
   is safe from a liveness standpoint. `scripts/rerun/quarantine_contaminated_evals.py`
   does a reversible MOVE (dry-run default, manifest + `--undo`); current dry run = 239
   paths / 31 MB. ORDERING: run it only after this PR's distiller (evals-root re-attach)
   is on the branch you distill from — before that, main's distiller loses those metrics.
   `--eval-dirs-only` leaves `eval_results.json` in place if you want to keep the training
   runs' own end-of-training eval blended in there.

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
