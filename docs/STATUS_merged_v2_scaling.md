# STATUS: merged_v2 diversity-scaling ladder

Checkout: `/fss/evin_t/tagseq2tagseq` (branch provenance-grounding). Run dirs:
`/fss-data/evin_t/tagseq2tagseq_artifacts/runs/`. Training stops cleanly at schedule
exhaustion since commit 56ca8b9 (`train_loop.exhaustion_tolerance_frac`, default 2%).
Relaunches are driven by `scripts/sweep_yield_watcher.sh` (pid on the login node, cwd `tagseq2tagseq-memexp`,
but it launches `$REPO=/fss/evin_t/tagseq2tagseq` code). Ledger:
`pipeline_logs/watcher_state/yielded_jobs.tsv`; log: `pipeline_logs/SWEEP_YIELD_NOTIFY.log`.

## Arm inventory (fixed lineage: within-bucket shuffle seed 42, muon_lr 0.003 / wd 0.1)

Final = trained to data exhaustion under the clean-stop code (final val + final latest.pt).

| arm | mask | final step | final run dir | completion eval | port_eval |
|---|---|---|---|---|---|
| 3.9B | cross_doc | 14784 | run_20260905_052305_672857 | NCCL timeout during eval (see below) | job 87045 |
| 3.9B | doc_causal | 14784 | run_20260905_052259_914139 | yes | n/a |
| div3 | cross_doc / doc_causal | 14607 | run_20260905_052310_693329 / run_20260906_035127_955402 | timeout / yes | job 87046 |
| div5 | cross_doc / doc_causal | 14656 | run_20260905_052316_175248 / run_20260906_001456_919330 | timeout / yes | job 87047 |
| div7 | cross_doc / doc_causal | 14688 | run_20260905_055342_251005 / run_20260905_203430_080300 | yes / yes | job 87049 |
| div9 | cross_doc / doc_causal | 14720 | run_20260905_052322_671272 / run_20260906_072353_870455 | timeout / yes | job 87048 |
| 8B | cross_doc | 30000 | repo-local runs/run_20260813_144916_125137 | yes | yes |
| 8B | concat / concat_link | 30335 / 30336 | run_20260905_095922_217348 / run_20260906_110220_009287 | yes / yes | n/a |
| 16B natural | cross_doc | 60600 | repo-local runs/run_20260813_182257_104861 | yes | yes |
| 16B natural | doc_causal / concat / concat_link | — | lineages dead since Aug 26, not relaunched | — | — |
| 16B balanced | cross_doc | 60733 | run_20260905_093303_660287 | yes | job 87050 |
| 16B balanced | doc_causal | 34000 / 60750 | RUNNING job 87028 (GPU-613), lineage run_20260905_062243 | — | n/a |
| 32B balanced | cross_doc | 120864 | run_20260905_052254_667822 | yes | job 87051 |
| 32B balanced | doc_causal | 82000 / 120888 | RUNNING job 87008 (GPU-302) | — | n/a |
| 32B natural | cross_doc | 119877 | run_20260905_063449_204090 | yes | job 87052 |
| 32B natural | doc_causal | 61000 / 119901 | RUNNING job 87018 (GPU-689) | — | n/a |

Port-eval jobs 87045-87052 (submitted 2026-09-07 ~21:45, `scripts/eval_ports_slurm.sh`,
one node each, ~30 min) write `<final run dir>/port_eval/<port>__use_line.json`.

Completion-eval NCCL timeout: on cross_doc arms, rank 0 runs the post-training
benchmarks (repobench_cross_doc downloads from HF) while the other 7 ranks wait at a
collective; when that takes longer than the 10-min NCCL timeout the job aborts AFTER the
final checkpoint is written, so SLURM reports FAILED although training is complete. The
port evals do not depend on that eval; per-source held-out numbers for those arms can be
regenerated offline with `eval_checkpoints.py` if needed.

## Queue

Watcher ledger is empty. Remaining training: the three doc_causal controls above
(32B balanced ~39k steps, 32B natural ~59k steps, 16B balanced ~27k steps, all at
~4.5-5 s/step, all subject to yield churn).

## Next manual steps

1. When jobs 87045-87052 finish, refresh `RESULTS_merged_v2_diversity_scaling.md` from
   the new `port_eval/*__use_line.json` (3.9B on annealed weights, div3/5/7/9, 16B
   balanced, 32B balanced, 32B natural).
2. Port-eval 32B/16B doc_causal arms is not applicable (no cross-doc mask).
3. Optional 16B natural doc_causal control (fresh, ~3 days):
   ```
   .venv/bin/python launch_slurm.py --nodes 1 --gpus-per-node 8 \
     --config configs/merged_v2_16b_natural_doc_causal.yaml --time 168:00:00 --no-tail \
     --train_loop.exhaustion_tolerance_frac 0.02
   ```

## Step-time reference (median s/step, 1024d/24L, 32k ctx, world 8, A100)

| mask | typical | notes |
|---|---|---|
| cross_doc_link (triton_v18) | 2.2-3.0 | 1.8-2.0 on small single-language sets |
| doc_causal (varlen_bim_v2) | 4.2-5.5 | consistently 1.7-2.3x slower than cross_doc on the SAME packs, since at least Aug 3; a kernel-side issue, not the cluster |

## Open problems

1. **Yield churn.** The watcher cancels youngest-first whenever any other job pends on
   Resources/Priority and relaunches after 30 idle minutes; each cycle costs ~25-40 min
   compile/resume plus up to 1000 lost steps (`save_latest_interval: 1000`). Accepted;
   nothing is exempt. The relaunch gate now blocks while any unmet demand exists (patched
   in the running copy, `tagseq2tagseq-memexp/scripts/sweep_yield_watcher.sh`, uncommitted
   there); the main-checkout copy of the script is older and lacks both the lineage
   resume and this gate.
2. **Watcher cwd.** The watcher runs from `tagseq2tagseq-memexp`, so `--config` paths
   resolve there; configs that exist only in the main checkout (the div tiers) were copied
   into `tagseq2tagseq-memexp/configs/` untracked. New configs must be present in both.
3. **Completion eval on cross_doc arms** aborts on NCCL timeout (above); `eval_results.json`
   is missing for 3.9B/div3/div5/div9 cross_doc. Fix is to run post-training eval on
   rank 0 after tearing down the process group, or to raise the NCCL timeout for eval.
4. **div7 out-of-distribution ports.** div7 (arxiv, js, python, ts, kotlin, rust, wiki)
   shows internal_dart Δ +1.22 (placebo +0.13) and internal_go +0.82 (placebo +0.37) on
   languages it never trained on. Treat OOD-language ports as uninterpretable for the
   diversity-count curve; same caveat applies to div3/div5/div9 on their unseen languages.
