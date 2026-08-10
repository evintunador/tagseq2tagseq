# Handoff: merged_v2 diversity-scaling (2026-08-07)

Full detail in memory `[[merged-corpus-build]]` + `RESULTS_merged_v2_diversity_scaling.md`.

## Headline result (STANDS)
8B merged model (11 linked sources, ~727M tok/domain) BEATS per-language specialists
on cross-doc benchmark ports 1.7–11× (use_line Δnll): repobench_python +0.162,
java +0.247, kotlin +0.265, ts +0.564, go +0.426, rust +0.105, js +0.129 — all clean
placebo separation. The cross-doc *mechanism* gain outpaces base-LM gain (merge trails
specialists on raw held-out ppl, the wrong axis). Δ is FLAT 3.9B→8B (saturated, not
scaling). 8B was healthy (held-out stack 2.15).

## 16B is BROKEN — LR too hot at length (NOT resume corruption)
16B held-out worse than 8B AND 3.9B (stack 2.72, wiki 10.7). 4-agent audit + direct
ckpt/log checks cleared all resume paths (data-order, weights, optimizer-step, scheduler
all correct for these runs; "0 skipped" every resume; total_steps=60600 correct).
Cause: WSD holds peak 0.003 flat ~36k steps pre-cooldown = ~2× 8B's peak exposure →
back-half blowup. The 4 broken 16B runs were KILLED.

## Currently running (verify with `squeue -u evin_t`)
- **6-arm LR/WD sweep @16B length** (60600 steps, all fresh-from-scratch): lr∈{0.001,
  0.0015,0.002} + wd∈{0.05,0.2}, base 0.003/0.1. This is the gate. Jobs ~55083/55084/
  56545/56546/56586 (+ maybe base). Rank on final held-out + port Δnll.
- The other agent's sparsity/edge-dropout runs may also be running — NOT mine, don't touch.

## Next steps (gated)
1. Sweep finishes (~2 days) → pick corrected LR (expect ≤0.002).
2. Re-run 16B (both natural+balanced, cdl+dc) at corrected LR → the real 16B scaling point.
3. THEN decide 32B. Do NOT launch 32B before the sweep picks LR.
4. Pending eval: 8B doc_causal-arm is NOT port-able (no cross-doc mask; placebo-sep is
   the control). 16B ports once healthy 16B exists.

## Ops notes
- GPU-495/954 FIXED (drop from --nodelist exclusions).
- Launch onto CONFIRMED-idle nodes (`sinfo -n <n> -o %t`==idle), not the queue — watcher
  yields RUNNING jobs but doesn't reorder PENDING vs coworkers.
- val cadence now val_steps=200/interval=2000 (was 400/1000, ate 2/3 wall-clock).
- Two latent resume bugs being fixed (see TODOS.md "Resume latent bugs"): NULL-path
  cooldown inflation (main.py:1195) + silent optimizer partial-restore (main.py:1100).
</content>
