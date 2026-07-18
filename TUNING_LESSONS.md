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

## Yield-watcher (scripts/sweep_yield_watcher.sh) — courtesy daemon
- Kills OUR ts2ts_ jobs (project-scoped, NOT all my jobs) youngest-first when another user is
  PENDING/blocked, sized to net demand = their_demand − idle_nodes (don't kill for capacity
  that already exists), gated on 2-poll persistence (ignore transient/about-to-schedule).
- IGNORES Dependency-blocked pending (their own job chain, not waiting for nodes).
- AUTO-RESUMES yielded jobs from latest.pt when a node has been idle ≥30 min (avoids racing a
  coworker cycling short debug jobs). Records run_dir+config to a ledger on kill.
- 12 offline self-tests + live-validated (real scancel + a real 03:23 auto-relaunch).
- RELAUNCH RULE (also when doing it by hand): if a killed run has a meaningful latest.pt,
  relaunch with `--resume-from <run>/checkpoints/latest.pt` — do NOT restart from scratch.

## Eval notes
- boolq consistently scores BELOW 0.50 and openbookqa near/below chance — likelihood-scoring
  artifacts common at this scale, NOT model failure (perplexity confirms the model is sound).
- 4-choice tasks (hellaswag, arc_easy) show real above-chance signal first; 2-choice
  (winogrande, piqa) sit near 0.50 longest.
- Eval on a local dev-node GPU (separate from the SLURM training fleet) so it doesn't contend.
