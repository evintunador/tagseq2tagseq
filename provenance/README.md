# Provenance: keeping the paper's numbers grounded

Every result number in `paper/` must trace to the run — and the exact code + config
state — that produced it. `runs/` is gitignored and ephemeral, so this tracked tree is
the durable record. Numbers reach the paper by *generation*, not by hand-typing, so they
cannot silently drift as prose is edited.

## What's here

- `runs/<run_id>.json` — one distilled, self-contained record per run: git commit + dirty
  flag, config path + CLI overrides, resolved hyperparameters, python/torch/cuda/driver +
  GPU/world-size, timestamps, and a verbatim copy of that run's eval metrics. Survives
  deletion of the ephemeral run dir.
- `patches/<sha256>.patch` — content-addressed store of the uncommitted diff for every
  git-dirty run (deduped by hash). Reproduce a dirty run with:
  `git checkout <commit_hash> && git apply provenance/patches/<hash>.patch`.
- `ledger.yaml` — the curated map from a stable claim-key to a *value source* (a specific
  field of a specific run's record). The single source of truth for paper numbers.

The generated `paper/generated/values.tex` (a committed lockfile) binds each claim-key to
its resolved value as `\valdef{key}{value}`; the paper cites `\val{key}`.

## Tools (`scripts/`)

- `distill_runs.py` — harvest run dirs (both `runs/` and the `/fss-data` runs root) into
  `runs/` + `patches/`. Idempotent; safe to run anytime (cron / pre-push).
- `gen_values_tex.py` — resolve the ledger and (re)write `paper/generated/values.tex`.
  All-or-nothing: any unresolvable entry aborts with a nonzero exit and names the key.
- `check_grounding.py` — the drift gate (wire into pre-push / CI). Verifies the ledger
  resolves and matches `expected`, that `values.tex` is fresh, that every `\val{}` in the
  paper has a ledger key, run-dir liveness, and patch integrity.
- `provenance_lib.py` — shared resolver used by both the generator and the checker (so they
  can never disagree).

## Adding a grounded number (end to end)

1. Run finishes → `python scripts/distill_runs.py` (records + patches refresh; a later
   re-eval is picked up automatically via the metrics hash).
2. Add an entry to `ledger.yaml`: a `claim-key`, its `run_ids`, a `source`
   (`eval` / `hyperparam` / `literal`), optional `expected`, and a `format`.
3. `python scripts/gen_values_tex.py` → regenerate and commit `paper/generated/values.tex`.
4. Cite it in prose: replace the hand-typed number (inside a `\todo{}`) with `\val{claim-key}`.
5. `python scripts/check_grounding.py` must pass before pushing.

## Value sources

- `eval {metric_path, field}` — reads `record.eval.metrics[metric_path][field]`. For a
  confidence interval, use an array field (`accuracy_ci`) or a `<base>_ci` field; the
  resolver also accepts sibling `<base>_ci_low`/`<base>_ci_high` scalars and normalizes
  both to `[lo, hi]` (set `format.kind: interval`).
- `hyperparam {path}` — dotted lookup into `record.hyperparameters`.
- `literal {value}` — escape hatch for a number NOT in any distilled artifact (e.g. read
  from a training log). **Requires a `note`.** These are *grounding debt*: `check_grounding.py`
  lists them in a separate "declared but ungrounded" bucket for reverification at
  camera-ready. Prefer grounding to a real source; a `kind: log` source (parse
  `runs/<id>/logs/`) is the intended future replacement.

## Notes

- The capture side (`tunalab.reproducibility.ReproducibilityManager`, an external editable
  package) writes each run's `reproducibility/` + `hyperparameters.json` +
  `eval_results.json`. The tools here only *read* those — they never modify run dirs or the
  package.
- Records for deleted run dirs are kept, never removed; `distill_runs.py --prune-missing`
  flips their `run_dir_exists` to `false` so the checker stops warning.
- The ledger currently ships `example.*` schema-demonstration entries (they exercise every
  resolver path against real runs). Replace them with real curated claims as the paper's
  `\todo{}` numbers are grounded.
