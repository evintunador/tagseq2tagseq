# STATUS: merged_v2 diversity-scaling ladder

Checkout: `/fss/evin_t/tagseq2tagseq` (branch provenance-grounding). Run dirs:
`/fss-data/evin_t/tagseq2tagseq_artifacts/runs/`. Relaunches are driven by
`scripts/sweep_yield_watcher.sh` (pid on the login node, cwd `tagseq2tagseq-memexp`,
but it launches `$REPO=/fss/evin_t/tagseq2tagseq` code). Ledger:
`pipeline_logs/watcher_state/yielded_jobs.tsv`; log: `pipeline_logs/SWEEP_YIELD_NOTIFY.log`.

## Arm inventory (fixed lineage: within-bucket shuffle seed 42, muon_lr 0.003 / wd 0.1)

| arm | mask | max_steps | packs/8 | state | last ckpt | port_eval |
|---|---|---|---|---|---|---|
| 3.9B | cross_doc | 14790 | 14805 | crashed "exhausted" at the end | 14000 (run_20260821_052234) | yes |
| 3.9B | doc_causal | 14790 | 14805 | INCOMPLETE, abandoned (bad resume path) | 12000 (run_20260821_052244) | no |
| 3.9B div3/5/7/9 | both | 14790 | 14622-14728 | all 8 crashed "exhausted" ~14.6k | 14000 each | div7 cdl only |
| 8B | cross_doc | 30340 | 30359 | complete + eval (repo-local runs/run_20260813_144916) | 30000 | yes |
| 8B | concat, concat_link | 30340 | 30359 | crashed "exhausted" at ~30336 | 30000 | no |
| 16B natural | cross_doc | 60600 | 60618 | complete + eval (repo-local runs/run_20260813_182257) | 60600 | yes |
| 16B natural | doc_causal / concat / concat_link | 60600 | 60618 | lineages dead since Aug 21-26, not resumed | partial | no |
| 16B balanced | cross_doc | 60750 | 60764 | crashed "exhausted" at ~60734 | 60000 (run_20260904_105156) | no |
| 16B balanced | doc_causal | 60750 | 60764 | never launched | - | - |
| 32B balanced | cross_doc | 120888 | 120888 | ORPHANED: killed by root Sep 2, not in watcher ledger | 119000 (run_20260902_065148) | no |
| 32B balanced | doc_causal | 120888 | 120888 | RUNNING job 86513 (GPU-182) | 74000 | no |
| 32B natural | cross_doc | 119901 | 119901 | RUNNING job 86511 (GPU-711) | 82000 | no |
| 32B natural | doc_causal | 119901 | 119901 | RUNNING job 86230 (GPU-602) | 54000 | no |

## Open problems

1. **Step budgets exceed the data.** `BucketedPackDataset` drops up to world_size-1 packs
   per bucket (32 buckets -> up to 224 packs = 28 steps) and then raises
   `RuntimeError: All pre-computed epoch dirs exhausted`. Every merged config sets
   `max_optimizer_steps` at or above `n_packs // 8`, so every arm dies a few steps to a
   few hundred steps before its schedule ends: SLURM state FAILED, no final checkpoint
   (latest.pt sits at the previous 1000-step save), no `run_on_completion` eval. The div
   tiers reuse 14790 although they hold 1.2% fewer packs (exhaust ~14,620-14,730). The
   three running 32B arms will hit the same crash at ~120,860 / ~119,873 unless
   `max_optimizer_steps` is lowered (>= 32 steps below `n_packs // 8`) on their next
   resume, or exhaustion is made a graceful stop that saves + evals.
2. **32B throughput is being destroyed by yield churn.** The watcher cancels
   youngest-first whenever any other job pends on Resources/Priority, relaunches after
   30 idle minutes, and each cycle costs ~25-40 min compile/resume plus up to 1000 lost
   steps (`save_latest_interval: 1000`). It also relaunches within minutes of yielding
   (17:20 yield for an 8-node job, 17:27 relaunch onto other idle nodes). Observed
   progress since Sep 2: natural cross_doc 77000 -> 82000, balanced doc_causal
   67000 -> 74000, i.e. ~10-15% of the ~700-1100 steps/h the nodes deliver.
3. **32B balanced cross_doc is orphaned at 119000** (~1,860 steps from the end). Only
   yields are recorded in the ledger, so a root kill is never auto-resumed.
4. **3.9B doc_causal control incomplete (12000/14790)**, and 16B balanced doc_causal was
   never launched, so neither the 3.9B pair Δ nor the balance ablation has its control.
5. **RESULTS claims not backed by on-disk evals.** `RESULTS_merged_v2_diversity_scaling.md`
   ("compute-matched crossover") says the effect strengthens with scale and typescript
   reaches +0.81 at 16B. The only post-fix port evals on disk give internal_typescript
   use_line Δ = +0.537 (3.9B), +0.539 (8B), +0.454 (16B natural); every port is flat
   across 3.9B/8B/16B. No 16B balanced or div3/5/9 port evals exist.
6. **3.9B and div checkpoints evaluated mid-cooldown.** All 3.9B-family port evals use the
   step-14000 checkpoint of a 14790 schedule (LR ~22% of peak) while specialist baselines
   are fully annealed. Conservative direction for the merge, but a confound.
7. **div7 out-of-distribution ports.** div7 (arxiv, js, python, ts, kotlin, rust, wiki)
   shows internal_dart Δ +1.22 (placebo +0.13) and internal_go +0.82 (placebo +0.37):
   languages it never trained on. Treat OOD-language ports as uninterpretable for the
   diversity-count curve.
