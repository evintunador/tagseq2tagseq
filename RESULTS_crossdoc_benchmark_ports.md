# Cross-doc benchmark ports + target-scope ablation — results

External RepoBench-analogous cross-file benchmarks ported into the frozen
verification harness (`eval/benchmark_harness/`, see
`docs/crossdoc_benchmark_port_harness_DESIGN.md`), plus the target-scope
ablation that re-anchors scoring at import-USE sites.

## Metrics recap
Per example we score the same completion tokens under three conditions and
report paired deltas (bootstrap 95% CI, 10k resamples):
- **Δnll_real** = flat(no aux) − cross-doc(real aux). Positive ⇒ the imported
  context helps predict the completion.
- **placebo separation** = placebo(wrong aux) − cross-doc(real aux), per example.
  Positive ⇒ the RIGHT aux beats a wrong-but-plausible one. This is the
  base-difficulty-free signal and the most trustworthy legitimacy test.

## Target scopes (the key methodological contribution)
The import *line* is uninteresting; the cross-doc signal lives at later *uses*
of imported symbols. Every port with a reconstructable full file is scored at
four scopes (`--scope`), holding CONTEXT identical and varying only the scored
target:
- **native** — the port's own target (arbitrary FIM span / next line).
- **use_line** — the single logical statement at the first line USING a symbol
  declared in a granted aux doc.
- **use_block** — that use site → end of enclosing syntactic block (tree-sitter).
- **rest_of_doc** — that use site → EOF.

"Uses an imported symbol" is decided by tree-sitter over the aux docs (their
top-level declared names), so `import *` resolves for free and no import-syntax
parse is needed. **We report all three use-scopes in the paper** — they measure
where the dependence concentrates, not just whether it exists.

## Calibration references (python/java RepoBench) — the legitimacy band
500 examples, cross_doc_link ckpts. Both PASS all gates.

| port | T1 prec | T2 fire | Δnll_real (CI) | placebo sep (CI) |
|---|---|---|---|---|
| repobench_python | 1.000 | 0.870 | +0.094 (0.072..0.117) | +0.127 (0.105..0.150) |
| repobench_java   | 1.000 | 0.940 | +0.072 (0.053..0.091) | +0.086 (0.068..0.107) |

Placebo separation > Δnll_real on both: wrong aux actively hurts vs no aux —
the benchmarks reward the RIGHT cross-file context, not extra tokens.

RepoBench (python AND java) has no post-hole file body (`all_code` is just a
license header / the next_line is the last known token), so its use-scopes
collapse to use_line — a built-in check that re-anchoring reproduces native.

### Python scopes (ckpt run_20260720_063128_690228, bfs)
| scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|
| native | +0.0936 (0.072..0.117) | +0.127 (0.105..0.150) | 500 |
| use_line = use_block = rest_of_doc | +0.0975 (0.074..0.123) | +0.135 (0.111..0.160) | 424 |

### Java scopes (ckpt run_20260722_191916_590119)
| scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|
| native | +0.072 (0.053..0.091) | +0.087 (0.068..0.107) | 500 |
| use_line | +0.085 (0.066..0.106) | +0.099 (0.079..0.119) | 487 |
| use_block = rest_of_doc | +0.080 (0.062..0.100) | +0.096 (0.077..0.116) | 487 |

Both languages: restricting to genuine cross-file-use lines SHARPENS the signal
(py +0.0975>+0.0936; java +0.085>+0.072). Re-anchoring is sound and Java PASSES
every scope with n≈487.

## Kotlin — ASE-2025 (JetBrains/Mistral), the target-scope proof
Native scoring (arbitrary FIM `middle`) FAILS; use-site anchoring recovers the
signal and it strengthens toward the use line. This is the clearest evidence
that target definition — not just checkpoint strength — governs whether a
cross-doc benchmark discriminates.

### random-traversal ckpt (run_20260724_095209_785799), 500-cap
| scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|
| native | −0.056 (−0.134..+0.011) | −0.033 (−0.099..+0.019) | 242 |
| rest_of_doc | −0.010 (−0.023..+0.004) | +0.051 (+0.034..+0.070) | 135 |
| use_block | +0.004 (−0.016..+0.024) | +0.058 (+0.040..+0.077) | 138 |
| use_line | +0.011 (−0.030..+0.055) | +0.106 (+0.069..+0.148) | 138 |

placebo separation: NEGATIVE at native → POSITIVE and monotonically stronger as
the span narrows to the use line (0.051→0.058→0.106, all CIs exclude 0).

### bfs-traversal ckpt (run_20260722_181228_995658), full 242 pool
| scope | Δnll_real (CI) | placebo sep (CI) | n | CIs exclude 0 |
|---|---|---|---|---|
| native | +0.012 (−0.033..+0.061) | +0.041 (−0.004..+0.090) | 242 | ✗ both |
| rest_of_doc | +0.024 (+0.014..+0.036) | +0.048 (+0.036..+0.063) | 135 | ✓ both |
| use_block | +0.047 (+0.028..+0.068) | +0.069 (+0.049..+0.090) | 138 | ✓ both |
| use_line | **+0.094 (+0.052..+0.140)** | +0.123 (+0.083..+0.166) | 138 | ✓ both |

**Both axes matter, and they compound.** vs the random-traversal ckpt above,
BFS lifts Δnll_real at every scope, and both Δnll_real AND placebo separation
climb monotonically native→rest_of_doc→use_block→use_line. Only the use-scopes
get CIs off zero — even the strong BFS ckpt cannot rescue the arbitrary-span
`native` target (both CIs still cross 0). BFS + use_line = Δnll +0.094, dead in
the python (+0.094) / java (+0.072) reference band. The ONLY remaining gate
failure is n<200 (the ASE 242-pool ceiling), not signal.

**Checkpoint answer ("was Kotlin weaker?"): YES, and it was the traversal, not
epochs.** The first ckpt used the **random** traversal — weakest for cross-doc
signal in BOTH the Python sweep and Java RepoBench ablation (random +0.031 vs
bfs/dfs/rw +0.065..0.079; RESULTS_code_crossdoc.md). All Kotlin ckpts trained 2
epochs / ~14k steps, so epoch count was NOT the difference. Swapping to the BFS
ckpt (above) roughly 8× the use_line Δnll (+0.011 → +0.094).

**n ceiling:** the ASE-2025 public pool is 242 examples (practice+public; the
private split has no public ground truth). GPU time cannot raise n past 242 for
this benchmark — power comes from a stronger ckpt (bigger effect) and use-scope
selection. Unlimited n lives only in the self-built test_community path (TODOS).

## TypeScript — CrossCodeEval (AWS)
Aux are retrieval CHUNKS (not whole import-resolved files), so signal is
diluted; still shows a positive placebo separation.

### all scopes (ckpt run_20260722_003634_268441, 500-cap)
| scope | Δnll_real (CI) | placebo sep (CI) | n_fired | n scored |
|---|---|---|---|---|
| native | +0.013 (−0.003..+0.031) | **+0.023 (+0.007..+0.040)** | 200 | 197 |
| use_line | +0.018 (−0.009..+0.045) | **+0.030 (+0.010..+0.055)** | — | 25 |
| use_block | +0.010 (−0.006..+0.027) | +0.002 (−0.008..+0.012) | — | 25 |
| rest_of_doc | −0.003 (−0.008..+0.004) | +0.001 (−0.006..+0.008) | — | 25 |

Unlike Kotlin/Java, TS does NOT show a clean use-site gradient — and that is
itself diagnostic of the CHUNK-based aux. Positive placebo separation (CI
excludes 0) appears at native and use_line but collapses to ~0 at use_block/
rest_of_doc, because a retrieved 10-line chunk may carry the symbol's immediate
use context but not the wider block's dependencies. Only 72/500 examples resolve
a use site (25 fire) since chunks frequently lack the declared symbol entirely.
Fire-rate 0.40 (static; runtime relative-import resolution reaches ~0.67).
**v2 is required for TS to be a clean use-scope benchmark**: re-clone repos from
`metadata.repository` for WHOLE-FILE aux (not chunks), which should both lift
fire-rate and restore the use_block/rest_of_doc signal. Report TS at native +
use_line only until v2, and flag the chunk limitation.

## Go: no external cross-doc benchmark
No usable upstream external cross-file benchmark exists for Go (RepoBench has no
Go variant). Go cross-doc eval must come from the self-built test_community
path (see TODOS.md); it has no external port here.

## Cross-benchmark takeaways
1. **Target scope matters as much as checkpoint quality.** An arbitrary-span
   benchmark can show ZERO/negative cross-doc signal that use-site anchoring
   recovers (Kotlin native −0.033 placebo → use_line +0.106..+0.123). And a weak
   traversal (random) buries signal a strong one (bfs) shows (Kotlin use_line
   +0.011→+0.094). Both compound.
2. **Restricting to genuine use lines SHARPENS the signal even on already-good
   benchmarks** (python +0.0936→+0.0975; java +0.072→+0.085).
3. **placebo separation is the robust primary metric**; Δnll_real needs power (n
   + a link-exploiting ckpt). Report both.
4. **Report use_line / use_block / rest_of_doc together** — the gradient is the
   finding (signal concentrates at the use site).
5. **Aux granularity gates the use-scope gradient.** Whole-file/skeleton aux
   (Kotlin ASE, RepoBench) give a clean monotone gradient; retrieval-CHUNK aux
   (CCEval TS) only helps the immediate use_line and collapses at wider scopes —
   evidence for the TS v2 whole-file upgrade.

## Summary table (use_line scope, the headline) — all working ports
| port | ckpt | Δnll_real (CI) | placebo sep (CI) | n | pass |
|---|---|---|---|---|---|
| repobench_python | bfs cdl | +0.098 (0.074..0.123) | +0.135 (0.111..0.160) | 424 | ✓ (n>200) |
| repobench_java | dfs cdl | +0.085 (0.066..0.106) | +0.099 (0.079..0.119) | 487 | ✓ |
| ase_kotlin | bfs cdl | +0.094 (0.052..0.140) | +0.123 (0.083..0.166) | 138 | signal ✓, n<200 |
| crosscodeeval_ts | ts cdl | +0.018 (−0.009..0.045) | +0.030 (0.010..0.055) | 25 | chunk-limited, needs v2 |
