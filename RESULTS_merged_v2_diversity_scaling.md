# Merged-v2 diversity-scaling experiment — results (in progress, 2026-08-05)

One model trained on **11 linked sources jointly** (wiki + arxiv + 9 code langs;
NO fineweb), vs per-language specialists, at matched-ish compute. Question: does
the cross-document-attention benefit **survive/strengthen** when many link types are
learned together, and how does it scale with tokens? See design in
`docs`/memory `[[merged-corpus-build]]` + TODOS "diversity-scaling experiment".

Rungs (per-domain tokens, ~equal split across 11 sources):
- **3.9B** = 355M tok/domain (compute-match to the small single-source runs)
- **8B** = 727M tok/domain
- **16B** = ~1.45B tok/domain ×2 balance variants (IN PROGRESS)

All rungs: 1024d/24L ~350M, VE-off, muon_lr=0.003/wd=0.1, max_grants=256, 32k ctx.
Each rung is a cross_doc_link vs doc_causal PAIR (within-model Δ). Recipe carried
verbatim from the per-language sweeps — the ONLY variable vs a specialist is corpus
diversity. **LR/WD NOT retuned for the larger rungs yet** (planned before 32B).

---

## ★ HEADLINE: cross-doc benchmark ports — the merge BEATS specialists on the thesis metric

Δnll_real = flat(no aux) − cross-doc(real aux); higher = attending to the linked
doc helps more. `use_line` scope (scored at the first use of an imported symbol),
Tier-2, 8B merge `latest.pt` (run_20260803_145120_344576), placebo-controlled.

| port | **8B merge Δnll** | placebo sep | n | specialist (use_line) | merge/spec |
|---|---|---|---|---|---|
| repobench_python | **+0.162** | +0.167 | 424 | +0.095 | 1.7× |
| repobench_java | **+0.247** | +0.289 | 487 | +0.112 | 2.2× |
| internal_kotlin | **+0.265** | +0.207 | 494 | +0.094 (ase, external) | ~2.8× |
| internal_typescript | **+0.564** | +0.465 | 396 | +0.051 (cceval, external) | ~11× |
| internal_go | **+0.426** | +0.295 | 229 | — (no external port) | — |
| internal_rust | **+0.105** | +0.143 | 420 | — | — |
| internal_javascript | **+0.129** | +0.202 | 181 | — | — |

**Every port strongly positive, clean placebo separation (right aux ≫ wrong aux).
On every comparable benchmark the 8B merge's cross-doc Δ is 1.7–11× LARGER than the
specialist's** — the specialist was trained on ~3.9B of its OWN language; the merge
saw only ~727M of it. This is the diversity-efficiency win: joint training over many
link types produces a *stronger* cross-doc-attention effect than single-domain
specialization, at far less per-domain data.

Native-scope (no tree-sitter) cross-check on the same ckpt agreed: py +0.153, java
+0.227 (vs specialist native ~+0.092/~+0.108). use_line sharpens as expected.

### Scaling 3.9B → 8B: the cross-doc Δ is FLAT (already saturated at 355M tok/dom)
use_line Δnll_real, same ports, 3.9B (355M/dom) vs 8B (727M/dom):

| port | 3.9B | 8B |
|---|---|---|
| repobench_python | +0.172 | +0.162 |
| repobench_java | +0.228 | +0.247 |
| internal_kotlin | +0.256 | +0.265 |
| internal_typescript | +0.620 | +0.564 |
| internal_go | +0.373 | +0.426 |
| internal_rust | +0.097 | +0.105 |
| internal_javascript | +0.121 | +0.129 |

Differences are within noise, no consistent direction — the cross-doc benefit is
**already present at full strength at 355M tok/domain and does NOT grow 3.9B→8B.**
Honest framing: the diversity advantage over specialists is NOT a "keeps scaling"
effect; it's a fixed, large effect present from the smallest rung. (16B point pending
to confirm it stays flat vs eventually moves.) NOTE: rungs are independent samples,
not nested, so small wiggles could be resampling; the flatness is the signal.

### Why this is the interesting result
The naive expectation is that mixing data distributions mainly buys *base-LM* quality
(more varied text → better general modeling). Here the effect on the **cross-doc
mechanism itself** OUTPACES that: the merge trails specialists on raw held-out
perplexity (below) yet exceeds them on the cross-doc benchmark Δ. So the gain is not
"more data → better LM"; it's "learning many link types together makes the
cross-document attention machinery itself more effective."

Caveats: kotlin/ts specialist baselines are EXTERNAL ports (ASE-2025, CrossCodeEval)
vs the merge's INTERNAL self-built ports — same harness/methodology but different
example pools, so those ratios are indicative not exact. doc_causal-arm control ports
(should show ≈0 Δ) + 16B-rung ports still TODO.

---

## Held-out perplexity (base-LM-quality axis — specialists win here)

nll on each source's held-out `val_random`, isolated-doc scoring, identical path for
merge + specialist. This is NOT the thesis metric; shown for completeness.

| source | 4B (355M/dom) | 8B (727M/dom) | specialist (~3.9B/dom) | 8B gap |
|---|---|---|---|---|
| stack (py) | 2.258 | 2.152 | 1.407 | +0.745 |
| go | 1.956 | 1.895 | 1.434 | +0.461 |
| java | 1.563 | 1.542 | 1.212 | +0.330 |
| wiki | 6.635 | 6.794 | — | — |
| arxiv | 3.802 | 3.563 | — | — |
| typescript | 1.715 | 1.630 | — | — |
| others (kotlin/rust/js/zig/dart) | — | 1.2–1.6 | — | — |

Doubling per-domain tokens improves nearly every source (not saturated), but the gap
to specialists closes only ~0.05–0.08 nll/doubling — on raw ppl the merge would need
far more than a specialist's own budget to cross. **This is expected**: fewer domains
= less to model, so specialization wins on raw ppl. The point of the experiment is
that the cross-doc Δ (above) goes the OTHER way. wiki regressed slightly (6.64→6.79),
the lone source to get worse on ppl.

## community_pack cross-doc Δ (held-out linked packs, mask on vs off)
Same-source, within-model. wiki: 4B +0.160 → 8B +0.143 (holds strong at scale).
Full per-source community_pack at 8B partially computed; the discriminating signal is
the ports table above (community_pack is near-noise for code per prior work).

---

## Status / TODO
- Arm-by-arm run state, checkpoints and open problems: `docs/STATUS_merged_v2_scaling.md`.
- Port-evaluated so far (fixed lineage): 3.9B cross_doc, 8B cross_doc, 16B natural
  cross_doc, div7 cross_doc. Still to port: 16B balanced, 32B, div3/5/9 cross_doc arms
  (doc_causal arms are not port-able — no cross-doc mask; placebo separation is the control).
- Controls outstanding: 3.9B doc_causal (stopped at step 12000), 16B balanced doc_causal.
- LR/WD not retuned across rungs (all arms muon_lr 0.003 / wd 0.1).
- Infra notes (fixed this experiment): checkpoint host-OOM barrier, absolute-step
  resume, community_pack 2048-budget, val-loader source-bias/rewind, Option-B
  graph-edge grants for eval, per-pack layout_epoch for multi-epoch, clean stop at
  schedule exhaustion (`train_loop.exhaustion_tolerance_frac`). See `[[merged-corpus-build]]`.

---
## ★ COMPUTE-MATCHED CROSSOVER (post-forgetting-fix, 2026-08-28)
Fixed re-runs (within-bucket shuffle seed=42; wiki no longer forgotten). use_line Δnll.
**3.9B merge (355M tok/domain) vs specialist (3.9B tok/domain) — MATCHED total budget:**
merge WINS 6 / TIE 3 / spec 4 (of 13 ports).
- Merge wins: typescript +0.54 vs +0.28, python +0.30 vs +0.25, kotlin +0.23 vs +0.17,
  zig +0.25 vs +0.14, java +0.15 vs +0.10, repobench_java +0.18 vs +0.08.
- Tie: ase_kotlin, crosscodeeval_ts, repobench_python.
- Spec wins: go (+0.15 vs +0.22), rust (+0.12 vs +0.28), dart (+0.25 vs +0.34), javascript.
HEADLINE: with 1/11th the per-domain tokens, the merge matches-or-beats specialists on
9/13 cross-doc benchmarks (decisively on ts/python/java/kotlin/zig).

### Token-scaling across rungs: cross-doc Δ is FLAT 3.9B → 8B → 16B → 32B
use_line Δnll_real per port from `port_eval/` of the fixed-lineage cross_doc runs, all
evaluated on the FINAL (clean-stop, annealed) `latest.pt`:
3.9B = run_20260905_052305 (step 14784), 8B = repo-local runs/run_20260813_144916 (30000),
16B natural = repo-local runs/run_20260813_182257 (60600), 16B balanced = run_20260905_093303
(60733), 32B balanced = run_20260905_052254 (120864), 32B natural = run_20260905_063449 (119877).

| port | 3.9B | 8B | 16B-nat | 16B-bal | 32B-bal | 32B-nat |
|---|---|---|---|---|---|---|
| repobench_python | +0.113 | +0.120 | +0.092 | +0.101 | +0.060 | +0.079 |
| repobench_java | +0.173 | +0.176 | +0.172 | +0.158 | +0.168 | +0.174 |
| ase_kotlin | +0.108 | +0.108 | +0.112 | +0.103 | +0.117 | +0.113 |
| crosscodeeval_ts | +0.058 | +0.039 | +0.033 | +0.044 | +0.042 | +0.042 |
| internal_python | +0.307 | +0.261 | +0.297 | +0.280 | +0.334 | +0.319 |
| internal_java | +0.140 | +0.114 | +0.135 | +0.136 | +0.155 | +0.121 |
| internal_typescript | +0.528 | +0.539 | +0.454 | +0.480 | +0.401 | +0.516 |
| internal_kotlin | +0.235 | +0.163 | +0.169 | +0.181 | +0.167 | +0.183 |
| internal_go | +0.150 | +0.144 | +0.147 | +0.144 | +0.182 | +0.189 |
| internal_rust | +0.110 | +0.096 | +0.107 | +0.062 | +0.100 | +0.088 |
| internal_javascript | +0.106 | +0.103 | +0.112 | +0.104 | +0.105 | +0.110 |
| internal_zig | +0.282 | +0.315 | +0.255 | +0.218 | +0.301 | +0.431 |
| internal_dart | +0.249 | +0.311 | +0.181 | +0.269 | +0.173 | +0.251 |


Across an 8× range of tokens no port moves by more than noise (typescript +0.53 / +0.54 /
+0.45 / +0.48 / +0.40 / +0.52; python +0.31 / +0.26 / +0.30 / +0.28 / +0.33 / +0.32).
Base-LM ability (mean flat nll, table below) improves monotonically with tokens on every
port while the cross-doc Δ does not: the two axes are DECOUPLED, and the diversity
advantage over specialists is a fixed effect present from the smallest rung, not a
scaling one. Balanced vs natural mixing at 16B and 32B makes no consistent difference
to Δ; natural is slightly better on flat nll for the big sources and worse for
zig/dart, as expected from token share.

### Diversity-count curve at fixed 3.9B: Δ on SEEN languages does not depend on how many domains share the budget
Tiers split the same 3.9B budget over 3/5/7/9/11 sources (build_diversity_tiers.sh; div11
is the full 3.9B merge). Starred cells are ports in a language the tier never trained on:
their Δ is inflated and comes with a large placebo Δ (any context helps an unseen
language), so they are NOT cross-doc evidence and are excluded from the reading.

### Δnll_real (use_line) — diversity-count tiers at fixed 3.9B (placebo Δ in parentheses; * = language NOT in that tier's training mix)

| port | div3 | div5 | div7 | div9 | div11 (=3.9B) |
|---|---|---|---|---|---|
| repobench_python | +0.107 (-0.05) | +0.093 (-0.06) | +0.104 (-0.04) | +0.108 (-0.04) | +0.113 (-0.03) |
| repobench_java | +0.494 (+0.05)* | +0.337 (+0.00)* | +0.353 (+0.02)* | +0.175 (-0.06) | +0.173 (-0.05) |
| ase_kotlin | +0.225 (-0.04)* | +0.124 (-0.02) | +0.123 (-0.03) | +0.110 (-0.04) | +0.108 (-0.04) |
| crosscodeeval_ts | +0.052 (+0.02)* | +0.062 (+0.03) | +0.068 (+0.03) | +0.077 (+0.02) | +0.058 (+0.02) |
| internal_python | — | +0.322 (-0.08) | +0.317 (-0.09) | +0.322 (-0.08) | +0.307 (-0.08) |
| internal_java | +0.297 (+0.06)* | +0.225 (+0.04)* | +0.243 (+0.03)* | +0.131 (-0.02) | +0.140 (-0.02) |
| internal_typescript | +0.703 (+0.13)* | +0.462 (+0.06) | +0.450 (+0.04) | +0.603 (+0.07) | +0.528 (+0.05) |
| internal_kotlin | +0.490 (+0.16)* | +0.287 (+0.03) | +0.162 (+0.02) | +0.230 (+0.00) | +0.235 (+0.03) |
| internal_go | +0.869 (+0.44)* | +0.901 (+0.42)* | +0.792 (+0.36)* | +0.170 (-0.02) | +0.150 (-0.03) |
| internal_rust | +0.543 (+0.19)* | +0.440 (+0.17)* | +0.075 (-0.05) | +0.106 (-0.04) | +0.110 (-0.03) |
| internal_javascript | +0.120 (-0.06) | +0.128 (-0.05) | +0.124 (-0.07) | +0.146 (-0.06) | +0.106 (-0.07) |
| internal_zig | +0.613 (+0.45)* | +0.624 (+0.48)* | +0.426 (+0.39)* | +0.356 (+0.35)* | +0.282 (+0.17) |
| internal_dart | +1.280 (+0.13)* | +1.447 (+0.22)* | +1.173 (+0.10)* | +0.972 (+0.07)* | +0.249 (-0.14) |


On in-distribution ports the Δ is flat from div3 to div11 (repobench_python +0.11 → +0.11,
internal_python +0.32 → +0.31, internal_javascript +0.12 → +0.11, typescript +0.46 to
+0.60 without trend, kotlin +0.29 → +0.24 once kotlin is in the mix). Halving or
quadrupling the tokens a language receives (div3 gives python 3.7× the tokens div11
does) does not move its cross-doc Δ. Consistent with the token-scaling result: the
cross-doc benefit saturates below 355M tokens/domain, and neither more tokens per domain
nor more domains per budget changes it. div3's internal_python cell is pending a re-run
(job 87061; the first audit died on a transient CUDA launch failure).

### mean flat nll (no aux) — base-LM axis

| port | 3.9B | div3 | div5 | div7 | div9 | 8B | 16B-nat | 16B-bal | 32B-bal | 32B-nat |
|---|---|---|---|---|---|---|---|---|---|---|
| repobench_python | 2.06 | 1.98 | 1.97 | 2.05 | 2.05 | 1.99 | 1.77 | 2.02 | 1.70 | 1.71 |
| repobench_java | 1.73 | 3.14 | 2.51 | 2.63 | 1.75 | 1.76 | 1.52 | 1.67 | 1.31 | 1.47 |
| ase_kotlin | 1.34 | 1.93 | 1.28 | 1.35 | 1.34 | 1.28 | 1.12 | 1.27 | 1.05 | 1.09 |
| crosscodeeval_ts | 1.39 | 1.52 | 1.36 | 1.50 | 1.46 | 1.30 | 1.16 | 1.28 | 1.07 | 1.12 |
| internal_python | 2.88 | — | 2.86 | 2.90 | 2.90 | 2.78 | 2.52 | 2.68 | 2.40 | 2.46 |
| internal_java | 1.85 | 2.77 | 2.35 | 2.39 | 1.85 | 1.85 | 1.63 | 1.75 | 1.45 | 1.63 |
| internal_typescript | 2.51 | 3.02 | 2.35 | 2.46 | 2.52 | 2.53 | 2.06 | 2.40 | 1.80 | 1.97 |
| internal_kotlin | 2.24 | 3.23 | 2.27 | 2.36 | 2.25 | 2.24 | 1.96 | 2.05 | 1.95 | 2.01 |
| internal_go | 2.12 | 3.85 | 3.74 | 3.70 | 2.10 | 2.07 | 1.86 | 2.00 | 1.67 | 1.84 |
| internal_rust | 2.12 | 4.57 | 3.85 | 2.13 | 2.11 | 2.05 | 1.88 | 1.99 | 1.74 | 1.85 |
| internal_javascript | 1.82 | 1.92 | 1.81 | 1.88 | 1.91 | 1.85 | 1.69 | 1.76 | 1.59 | 1.58 |
| internal_zig | 2.28 | 3.49 | 3.57 | 3.30 | 3.16 | 2.39 | 2.27 | 2.24 | 1.83 | 2.22 |
| internal_dart | 1.79 | 4.19 | 3.86 | 3.72 | 3.17 | 1.94 | 1.50 | 1.77 | 1.26 | 1.57 |
