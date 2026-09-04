# STATUS: merged_v2 diversity-scaling ladder

Checkout: `/fss/evin_t/tagseq2tagseq` (branch provenance-grounding). Run dirs:
`/fss-data/evin_t/tagseq2tagseq_artifacts/runs/`. Training stops cleanly at schedule
exhaustion since commit 56ca8b9 (`train_loop.exhaustion_tolerance_frac`, default 2%).
Relaunches are driven by `scripts/sweep_yield_watcher.sh` (pid on the login node, cwd `tagseq2tagseq-memexp`,
but it launches `$REPO=/fss/evin_t/tagseq2tagseq` code). Ledger:
`pipeline_logs/watcher_state/yielded_jobs.tsv`; log: `pipeline_logs/SWEEP_YIELD_NOTIFY.log`.

## Arm inventory (fixed lineage: within-bucket shuffle seed 42, muon_lr 0.003 / wd 0.1)

| arm | mask | max_steps | packs/8 | state | last ckpt | port_eval |
|---|---|---|---|---|---|---|
| 3.9B | cross_doc | 14790 | 14805 | crashed "exhausted" at the end | 14000 (run_20260821_052234) | yes |
| 3.9B | doc_causal | 14790 | 14805 | queued for resume (ledger #2) | 12000 (run_20260821_052244) | no |
| 3.9B div3/5/7/9 | both | 14790 | 14622-14728 | all 8 crashed "exhausted" ~14.6k | 14000 each | div7 cdl only |
| 8B | cross_doc | 30340 | 30359 | complete + eval (repo-local runs/run_20260813_144916) | 30000 | yes |
| 8B | concat, concat_link | 30340 | 30359 | crashed "exhausted" at ~30336 | 30000 | no |
| 16B natural | cross_doc | 60600 | 60618 | complete + eval (repo-local runs/run_20260813_182257) | 60600 | yes |
| 16B natural | doc_causal / concat / concat_link | 60600 | 60618 | lineages dead since Aug 21-26, not resumed | partial | no |
| 16B balanced | cross_doc | 60750 | 60764 | crashed "exhausted" at ~60734 | 60000 (run_20260904_105156) | no |
| 16B balanced | doc_causal | 60750 | 60764 | fresh launch pending, SLURM 86583 | - | - |
| 32B balanced | cross_doc | 120888 | 120888 | queued for resume (ledger #1) | 119000 (run_20260902_065148) | no |
| 32B balanced | doc_causal | 120888 | 120888 | RUNNING job 86513 (GPU-182) | 74000 | no |
| 32B natural | cross_doc | 119901 | 119901 | RUNNING job 86511 (GPU-711) | 82000 | no |
| 32B natural | doc_causal | 119901 | 119901 | RUNNING job 86230 (GPU-602) | 54000 | no |

## Queue (watcher relaunch ledger, FIFO, one entry per node idle >= 30 min with nobody waiting)

1. 32B balanced cross_doc — resume from 119000 (~1,860 steps to a clean stop).
2. 3.9B doc_causal — resume from 12000 (control for the 3.9B pair).
3. 3.9B cross_doc, then div3/5/9/7 cross_doc, div3/5/7/9 doc_causal, 8B concat,
   8B concat_link, 16B balanced cross_doc — each resumes from its step-N000 checkpoint
   and runs the last few hundred steps to a clean stop (final val, final latest.pt,
   completion eval).
4. SLURM job 86583 (`run_20260904_202130_199051`): fresh 16B balanced doc_causal, pending
   on Priority; launched with `--train_loop.exhaustion_tolerance_frac 0.02` so the
   watcher's lineage match cannot fall back to the pre-fix Aug 12 run of the same config.

## Weekend checklist (manual, when nodes free up)

Automatic: the watcher relaunches the 14 ledger entries above, one per node that has
been idle >= 30 min while nobody is pending (up to one per idle node per 2-min poll), and
SLURM places job 86583. Nothing to do for those beyond `squeue -u evin_t`.

Manual, after a clean-stop resume has written its final `latest.pt` (check
`<run>/eval_results.json` exists):

1. Port evals for the cross_doc arms that lack them. Edit the `CK` map in
   `scripts/eval_ports_local.sh` to the FINAL run dir of each arm (the relaunch creates a
   new `run_*` dir; the ledger log names it) and run on a node with 7 free GPUs:
   ```
   SCOPE=use_line GPUS_OVERRIDE="1 2 3 4 5 6 7" scripts/eval_ports_local.sh
   ```
   Arms: 3.9B cross_doc (re-port on annealed weights), div3/5/7/9 cross_doc,
   16B balanced cross_doc, 32B balanced cross_doc, 32B natural cross_doc (when done).
   Use `--checkpoint .../latest.pt` (final, annealed); the doc_causal arms are not
   port-able (no cross-doc mask).
2. Optional 16B natural doc_causal control (lineage dead since Aug 26; 60600 steps, ~3 days):
   ```
   .venv/bin/python launch_slurm.py --nodes 1 --gpus-per-node 8 \
     --config configs/merged_v2_16b_natural_doc_causal.yaml --time 168:00:00 --no-tail \
     --train_loop.exhaustion_tolerance_frac 0.02
   ```
   The override keeps the watcher's lineage matcher from resuming a pre-fix Aug run.
3. Then refresh the tables in `RESULTS_merged_v2_diversity_scaling.md` from the new
   `port_eval/*__use_line.json` files.

## Step-time reference (median s/step, 1024d/24L, 32k ctx, world 8, A100)

| mask | typical | notes |
|---|---|---|
| cross_doc_link (triton_v18) | 2.2-3.0 | 1.8-2.0 on small single-language sets |
| doc_causal (varlen_bim_v2) | 4.2-5.5 | consistently 1.7-2.3x slower than cross_doc on the SAME packs, since at least Aug 3; a kernel-side issue, not the cluster |

## Open problems

1. **Yield churn.** The watcher cancels youngest-first whenever any other job pends on
   Resources/Priority, relaunches after 30 idle minutes, and each cycle costs ~25-40 min
   compile/resume plus up to 1000 lost steps (`save_latest_interval: 1000`). Observed 32B
   progress Sep 2-4: natural cross_doc 77000 -> 82000, balanced doc_causal 67000 -> 74000,
   i.e. ~10-15% of the ~700-1100 steps/h the nodes deliver. Accepted as the cost of
   yielding; nothing is exempt.
2. **Running 32B arms still carry the old exhaustion crash** until their next resume
   (code is imported at launch). A crash at the end leaves latest.pt at the last
   1000-multiple; one more resume then finishes cleanly under the fixed code.
3. **Port evals still missing** for 16B balanced, 32B, div3/5/9, and every doc_causal
   flat-nll baseline. `RESULTS_merged_v2_diversity_scaling.md` now reports only what
   `port_eval/` dirs on disk support (3.9B / 8B / 16B natural cross_doc, div7 cross_doc).
4. **3.9B-family checkpoints evaluated mid-cooldown.** The existing 3.9B and div7 port
   evals used step-14000 weights of a 14790 schedule (LR ~22% of peak) against fully
   annealed specialists. Re-port after the clean-stop resumes land.
5. **div7 out-of-distribution ports.** div7 (arxiv, js, python, ts, kotlin, rust, wiki)
   shows internal_dart Δ +1.22 (placebo +0.13) and internal_go +0.82 (placebo +0.37) on
   languages it never trained on. Treat OOD-language ports as uninterpretable for the
   diversity-count curve.
