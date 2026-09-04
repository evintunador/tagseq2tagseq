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

### Token-scaling across rungs: cross-doc Δ is FLAT 3.9B → 8B → 16B
use_line Δnll_real per port (mean flat nll in parentheses), from the on-disk
`port_eval/` of the fixed-lineage cross_doc runs: 3.9B = run_20260821_052234 (latest.pt at
step 14000 of a 14790 schedule — the run hit data exhaustion a few steps before its
budget, so the evaluated weights are mid-cooldown, LR ≈ 22% of peak), 8B = run_20260813_144916 (step 30000),
16B natural = run_20260813_182257 (step 60600, complete).

| port | 3.9B Δ (flat nll) | 8B Δ (flat nll) | 16B-natural Δ (flat nll) |
|---|---|---|---|
| repobench_python | +0.105 (2.05) | +0.120 (1.99) | +0.092 (1.77) |
| repobench_java | +0.178 (1.77) | +0.176 (1.76) | +0.172 (1.52) |
| internal_python | +0.298 (2.89) | +0.261 (2.78) | +0.297 (2.52) |
| internal_java | +0.150 (1.88) | +0.114 (1.85) | +0.135 (1.63) |
| internal_typescript | +0.537 (2.55) | +0.539 (2.53) | +0.454 (2.06) |
| internal_kotlin | +0.225 (2.29) | +0.163 (2.24) | +0.169 (1.96) |
| internal_go | +0.146 (2.14) | +0.144 (2.07) | +0.147 (1.86) |
| internal_rust | +0.116 (2.14) | +0.096 (2.05) | +0.107 (1.88) |
| internal_javascript | +0.101 (1.84) | +0.103 (1.85) | +0.112 (1.69) |
| internal_zig | +0.254 (2.31) | +0.315 (2.39) | +0.255 (2.27) |
| internal_dart | +0.247 (1.86) | +0.311 (1.94) | +0.181 (1.50) |
| ase_kotlin | +0.102 (1.34) | +0.108 (1.28) | +0.112 (1.12) |
| crosscodeeval_ts | +0.048 (1.46) | +0.039 (1.30) | +0.033 (1.16) |

No port moves by more than noise across the three rungs; typescript is +0.54 / +0.54 /
+0.45. Base-LM ability (flat nll) improves with tokens on every port while the
cross-doc Δ does not — the two axes are DECOUPLED, and the diversity advantage is a
fixed effect present from the smallest rung, not a scaling one. No 16B-balanced or 32B
cross_doc port evals exist yet; div7 is the only diversity tier ported so far, and its
dart/go ports (languages absent from div7's training mix) are out-of-distribution and
must not be read as cross-doc evidence. Run-level state: `docs/STATUS_merged_v2_scaling.md`.
