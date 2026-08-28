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
- 8B cross_doc: ports DONE (above). 3.9B done. 16B ×2 IN PROGRESS (~1.45B/dom point).
- TODO: 8B **doc_causal-arm** ports (control, expect ≈0 Δ); **16B** ports (scaling
  curve — does the merge's cross-doc lead grow further?); formal specialist use_line
  re-pull for the exact paper table; LR/WD sweep before any 32B.
- Infra notes (fixed this experiment): checkpoint host-OOM barrier, absolute-step
  resume, community_pack 2048-budget, val-loader source-bias/rewind, Option-B
  graph-edge grants for eval, per-pack layout_epoch for multi-epoch. See
  `[[merged-corpus-build]]`.

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
9/13 cross-doc benchmarks (decisively on ts/python/java/kotlin/zig). Effect STRENGTHENS
with scale — at 8B/16B the merge win-or-tie count rises (typescript reaches +0.81 @16B).
Base-LM ability (flat_nll) improves ~uniformly with tokens (~-0.23/doubling); cross-doc
Δnll grows heterogeneously (ts/kotlin climb, python/rust/java saturate) — the two axes
are DECOUPLED. Ladder completion + diversity-count curve (div3/5/7/9) still filling in.
