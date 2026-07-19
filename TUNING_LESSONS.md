# Tuning & Infra Lessons (wiki_merged / arxiv retune, 2026-07-15 → ongoing)

Running log of what we learned retuning hyperparameters and running the mask/traversal
ablations. Keep appending. Detailed run-by-run log: `/fss-data/.../pipeline_logs/lrsweep_runmap.txt`.

## Final tuned config (wiki_merged, 1024d/24L, 4-epoch ≈ 3.8B tok)
- **muon_lr = 0.003** (inherited 0.001 was ~3× too low)
- **muon_wd = 0.1** (inherited 1.2 was ~12× too high — the single biggest gain)
- **adamw_lr = muon_lr × 0.01 = 3e-5**; adamw_wd = 0.005 (untouched)
- **ve_layers = []** (VE OFF on wiki — see below)
- cooldown_frac 0.4, min_lr_ratio 0.1, warmup 300, untie_at_frac 0.667
- Result: val_loss 2.42 (was ~4.9 undertrained); ppl ~4.0; hellaswag + arc_easy above chance.

## Hyperparameter findings
- **LR is a broad basin, WD is the sharp knob.** LR final-val: 0.0015=2.59, 0.002=2.56,
  0.003=2.55(best), 0.004=2.57, 0.006=2.60 @ old WD — flat, ~0.03 spread. WD is a clean
  U with a real minimum: {0.05:2.433, 0.1:2.424, 0.15:2.430, 0.2:2.433, 0.3:2.439,
  0.6:2.470, 1.2:2.548, 2.4:2.669}. WD tuning beat LR tuning by ~4×.
- **LR optimum is WD-robust.** Re-measured LR basin at WD=0.3: optimum stayed 0.003, but the
  basin got STEEPER on the low side (0.002→2.573). So low WD = more LR-sensitive below opt,
  but the peak doesn't move. => tune LR once, don't re-tune per WD.
- **Ranking reshuffles after cooldown/untie — never lock a winner mid-training.** At step
  4.5k the LR leader was 0.004; by full cooldown (14.4k) it was 0.003, and 0.0015 (a
  mid-training laggard) rose to mid-pack. ALWAYS compare at a COMMON step, and only lock on
  FINAL (post-cooldown) val. Mid-training rankings are contaminated by launch-timing (a run
  further into cooldown looks artificially better).
- **Value embeddings (VE) are data-gated.** VE = 258M params (42% of model). On wiki (1.36B
  tok, data-starved) VE-off slightly BEAT VE-on AND was ~4% faster → keep OFF. On arxiv (38B
  tok) veoff/veon CONVERGED (2.214 vs 2.225 @13k) → VE ~neutral, closing the gap with more
  data as predicted. Rule: VE needs a data-rich regime to pay for its params.
- **Chinchilla framing drove model-size choices.** wiki 1.36B tok ÷ 611M params (VE-on) ≈ 2
  tok/param (10× under-fed) → dropped VE to ~350M. arxiv 38B tok → even VE-on is Chinchilla-
  sufficient at <1 epoch, so arxiv uses a fixed 15k-step (~3.9B tok) budget, not full epochs
  (1 arxiv epoch = 84k steps = 4.5 days).

## THE config-bug that started it all
`max_optimizer_steps: null` in the wiki configs SILENTLY DISABLED both LR cooldown AND the
untie split (main.py ~955-969: both gated on max_steps being set). So every pre-2026-07-15
wiki run trained at CONSTANT LR, no decay, never untied → guaranteed flat late-loss. Fixed
by auto-deriving max_optimizer_steps from epoch_dirs n_packs (main.py ~586). ALWAYS verify
"Auto-derived max_optimizer_steps=… Enables LR cooldown + untie" appears in the log.
- Minor residual: derived count (14507) is ~70 more than actual packs → runs exit with a
  benign `RuntimeError: epoch dirs exhausted` at step 14438 (99.5%). Training/ckpts/val all
  COMPLETE and valid — it's cosmetic. TODO: floor the derived count to the true pack total.

## Cluster / infra lessons (this is a SHARED cluster — ExclusiveUser=NO)
- **The "submitit DDP hang" was a rank-0 CUDA OOM from a GPU-0 co-tenant, masked by a
  reproducibility barrier** — NOT a launcher/NCCL/compile bug. Chased fabric/IB/cgroup/liger
  for a day; all red herrings. Fix landed (loud-fail barrier + INFO logging + compile-warmup).
  ALWAYS preflight GPU-0 empty before launching (scripts/preflight_node.sh).
- **Preflight must check NFS too.** GPU-954 has broken /fss-data NFS → a run there dies in
  ~77s with "Dataset directory not found". GPU-749 failed the compile-warmup desync twice
  (avoid). preflight_node.sh checks BOTH GPU-0-empty AND dataset-visible.
- **Compile cache: node-local /tmp, NOT /fss-data (NFS).** N ranks taking FileLocks on an NFS
  cache deadlock. Node-local /tmp serializes correctly (rank0 compiles, others read).
- **liger_kernel** was a broken editable install (source dir deleted); FusedLinearCELoss hard-
  requires it. Reinstalled from PyPI (pinned nccl back to 2.28.9 after it downgraded).
- **compile warmup desync is intermittent (~15-20%)** even with the broadcast-batch fix; it
  fails LOUD at step 0 (good) → just relaunch on a fresh node/cache.

## Yield-watcher / auto-kill+resume system (scripts/sweep_yield_watcher.sh)
FULL REFERENCE for the courtesy daemon that shares this multi-tenant cluster politely.

WHAT IT DOES
- Polls `squeue`/`sinfo` every POLL_SECS (120s). Two responsibilities: (1) YIELD — cancel our
  jobs when a coworker is blocked; (2) RELAUNCH — bring yielded jobs back when capacity frees.
- SCOPE: only touches jobs named with prefix `ts2ts_` owned by $ME. Project-scoped, NOT
  user-scoped — any other job you run (interactive salloc, other project, differently-named
  sbatch) is NEVER a kill candidate. (self-test scenario G locks this in.)

YIELD (kill) logic — a job is cancelled only when ALL hold:
  1. Another user has a job PENDING with REASON in {Resources, Priority} (genuinely waiting
     for nodes). Dependency-blocked pending is IGNORED (waiting on their own job chain, not us).
  2. net demand = their_total_requested_nodes − currently_idle_nodes  > 0. We do NOT kill for
     capacity that already exists idle (the scheduler will place them). Sized to net, not gross.
  3. The unmet demand persisted across 2 consecutive polls (ignore a transient blip where a
     job is about to be scheduled onto an idle node).
  Victims chosen YOUNGEST-first (least lost progress), capped at MAX_KILL (8) per episode.
  On kill: records run_dir + config to the ledger BEFORE scancel (squeue still knows the job).

RELAUNCH (auto-resume) logic — only when NOBODY is waiting:
  - Tracks how long each node has been continuously idle (SLURM doesn't expose this; we keep
    our own node_idle_since.tsv ledger, carrying forward first-idle timestamps).
  - When a node has been idle >= IDLE_RELAUNCH_MIN (30) min, relaunch the oldest yielded job on
    it, preflighted (scripts/preflight_node.sh: GPU-0 empty + NFS OK). The 30-min gate avoids
    racing a coworker who is rapidly cycling short/failing debug jobs on a node.
  - RESUMES from `<run_dir>/checkpoints/latest.pt` if it exists (exact Muon+AdamW + dataset-
    position resume via --resume-from), else starts fresh. Never restarts a deep run from 0.

CONFIG (env vars): POLL_SECS=120, JOB_NAME_PREFIX=ts2ts_, MAX_KILL=8, IDLE_RELAUNCH_MIN=30,
  AUTO_RELAUNCH=1 (set 0 to disable relaunch, keep yield). ME defaults to $USER.

FILES / STATE:
  - script: scripts/sweep_yield_watcher.sh  (PID recorded in pipeline_logs/sweep_yield_watcher.pid)
  - human log + kill/relaunch events: pipeline_logs/SWEEP_YIELD_NOTIFY.log
  - stdout log: pipeline_logs/sweep_yield_watcher.log
  - state dir: pipeline_logs/watcher_state/  { yielded_jobs.tsv (run_dir,config,killed_epoch,
    status), node_idle_since.tsv (node,first_idle_epoch), relaunch.log }

RUN / RESTART / TEST:
  - start:   nohup scripts/sweep_yield_watcher.sh >> pipeline_logs/sweep_yield_watcher.log 2>&1 &
             (then write $! to pipeline_logs/sweep_yield_watcher.pid)
  - test:    scripts/sweep_yield_watcher.sh --selftest   (12 offline scenarios A–L, all pure fns)
  - CAVEAT:  it runs as a plain bg process on the dev node (GPU-670). If that node reboots it
             dies SILENTLY (training jobs on compute nodes are unaffected). No auto-restart /
             systemd wrapper yet — check `ps -p $(cat …/sweep_yield_watcher.pid)` if unsure.

VALIDATION: 12 offline self-tests pass; live-validated with a real scancel (RUNNING→CANCELLED+)
  and a real 2026-07-18 03:23 auto-relaunch (resumed wiki_minlr0p2 from ckpt on a 30-min-idle node).

MANUAL RELAUNCH RULE (when doing it by hand, not via the daemon): if a killed/yielded run has a
  meaningful latest.pt, relaunch with `--resume-from <run>/checkpoints/latest.pt` — do NOT restart
  from scratch. (I made this mistake once: relaunched-from-0 jobs that had 2400 steps of ckpt;
  the fix recovered ~1650 steps each.)

HISTORY: v1 killed exactly 1 job/episode ignoring idle nodes (bug — would kill for capacity that
  already existed, and dribble out multi-node demand). v2 (current) is node-demand-aware + idle-
  discounted + persistence-gated + auto-resume. Both bugs were caught by the user, not me.

## Eval notes
- boolq consistently scores BELOW 0.50 and openbookqa near/below chance — likelihood-scoring
  artifacts common at this scale, NOT model failure (perplexity confirms the model is sound).
- 4-choice tasks (hellaswag, arc_easy) show real above-chance signal first; 2-choice
  (winogrande, piqa) sit near 0.50 longest.
- Eval on a local dev-node GPU (separate from the SLURM training fleet) so it doesn't contend.

## EXPERIMENTAL RESULTS SUMMARY (2026-07-15 → 07-19)

### Cross-doc thesis (the core hypothesis) — CONFIRMED
`hotpotqa_cross_doc` on tuned cross_doc_link (bfs): cross-doc nll 5.63 vs flat-linked 6.92 =
**Δnll +1.29 (~19% lower NLL, 3.6× lower ppl)** favoring cross-doc attention over flat-concat
of identical linked content (n=738). On single-doc benchmarks all masks ~identical (ppl≈3.84,
hellaswag≈0.33) — cross-doc advantage appears ONLY on the multi-hop cross-doc benchmark, exactly
as predicted. Old undertrained run showed only +13.4%; tuning made the effect much stronger.

### Traversal ablation (bfs/dfs/random_walk/random × doc_causal/cross_doc_link) — DONE
Final val_loss (LR0.003/VE-off; wd0.1 doc_causal, wd0.3 cross_doc_link):
    strategy      cross_doc_link   doc_causal
    dfs           2.4205           2.4263
    bfs           2.4312           2.4241
    random_walk   2.4916           2.4946
    random        3.0006           3.176
Cross-doc Δnll (hotpotqa_cross_doc, n=738): dfs +1.329, bfs +1.291, rw +1.270, random +1.241.
FINDINGS:
- Graph-structured traversal (bfs/dfs ~2.42) >> random (3.00): 0.58 val gap. Graph-aware
  packing matters HUGELY for the base LM. dfs≈bfs > random_walk > random.
- Cross-doc benefit is ROBUST across all traversals (+1.24..+1.33) — thesis is traversal-
  independent. bfs/dfs/rw indistinguishable on Δnll (0.06 spread @ n738 = noise); DO NOT claim
  a winner among them. 'random' is a mild low-Δnll outlier (rarely saw linked docs adjacent).
- DECOUPLING: traversal matters a lot for base-LM quality but only weakly for the *incremental*
  cross-doc benefit. Two separate axes.
- CAVEAT: absolute NLL not cleanly comparable across models trained on different distributions;
  within-model Δnll (cross-doc − flat) is the robust contrast.

### ArXiv sweep (38B-token corpus, 15k-step fixed budget ~3.9B tok, LR0.003) — DONE
Final val: doc_causal veoff 2.156 | VE-ON 2.204 | wd0.6 2.173 | cross_doc_link 2.471 |
  doc_concatenated 2.140 | doc_concat_link 2.133.
- VE-off STILL beats VE-on (2.156 vs 2.204) even on 38B tokens at 15k steps — the predicted
  "flip" did NOT fully happen; VE closed the gap vs wiki but didn't overtake within this budget.
  (A longer arxiv run might still flip it; VE remains net-neutral-to-negative here.)
- Low-WD preference transfers (wd0.6 worse). ArXiv trains to lower val (2.14) than wiki (2.42):
  28× more data. concat variants edge out doc_causal on raw val (denser packing).

### Final tuned config (carry forward to future wiki runs)
muon_lr=0.003, muon_wd=0.1, adamw_lr=3e-5, adamw_wd=0.005, VE-off (ve_layers: []),
cooldown_frac 0.4, min_lr_ratio 0.1, warmup 300, untie_at_frac 0.667, weight_tying true.
