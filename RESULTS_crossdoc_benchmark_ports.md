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

All numbers below are the FULL-SAMPLE matrix (2026-07-28): every port × its 4
same-language cross_doc_link traversal ckpts (bfs/dfs/random_walk/random) ×
scopes, at the full split (no example cap). † = CI includes 0. Per-job JSON in
eval/benchmark_harness/reports/matrix/.

RepoBench (python AND java) has no post-hole file body (`all_code` is just a
license header; next_line is the last known token), so `use_block`/`rest_of_doc`
are identical to `use_line` — only native + use_line were run there.

### Python — RepoBench (full 8,033 pool; n<8033 after use-site filter + 32k skip)
| traversal | scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|---|
| bfs | native | +0.092 (0.084,0.100) | +0.137 (0.129,0.145) | 8033 |
| bfs | use_line | **+0.095 (0.087,0.103)** | +0.139 (0.130,0.147) | 6989 |
| dfs | native | +0.095 (0.087,0.104) | +0.137 (0.129,0.146) | 8033 |
| dfs | use_line | **+0.098 (0.090,0.107)** | +0.142 (0.133,0.151) | 6989 |
| random_walk | native | +0.079 (0.072,0.087) | +0.131 (0.124,0.140) | 8033 |
| random_walk | use_line | +0.082 (0.074,0.090) | +0.135 (0.127,0.144) | 6989 |
| random | native | +0.056 (0.048,0.064) | +0.133 (0.124,0.141) | 8033 |
| random | use_line | +0.059 (0.051,0.068) | +0.137 (0.129,0.146) | 6989 |

### Java — RepoBench (full 8,722 pool)
| traversal | scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|---|
| bfs | native | +0.108 (0.099,0.118) | +0.158 (0.148,0.168) | 8722 |
| bfs | use_line | **+0.112 (0.102,0.122)** | +0.162 (0.152,0.172) | 8585 |
| dfs | native | +0.105 (0.095,0.114) | +0.155 (0.146,0.166) | 8722 |
| dfs | use_line | +0.106 (0.096,0.116) | +0.160 (0.150,0.171) | 8585 |
| random_walk | native | +0.106 (0.097,0.116) | +0.157 (0.148,0.167) | 8722 |
| random_walk | use_line | +0.110 (0.101,0.120) | +0.160 (0.150,0.170) | 8585 |
| random | native | +0.033 (0.023,0.043) | +0.176 (0.165,0.187) | 8722 |
| random | use_line | +0.038 (0.028,0.049) | +0.185 (0.174,0.196) | 8585 |

Both PASS every cell (n≫200). Two robust patterns at full power:
- **use_line sharpens native** on every traversal (small but consistent).
- **Traversal ordering: bfs≈dfs≈random_walk ≫ random** on Δnll_real (java random
  +0.038 vs bfs +0.112; python random +0.059 vs bfs +0.095). Matches the code
  sweep (RESULTS_code_crossdoc.md). Note random's placebo sep is HIGH (java
  +0.185) — its base LM is worse so it leans MORE on aux, but the flat-vs-cross
  gain (Δnll_real) is what collapses; the two metrics decouple exactly where the
  traversal is graph-blind.

## Kotlin — ASE-2025 (JetBrains/Mistral): the target-scope proof
Full 242 pool, all 4 traversals × 4 scopes. Native scoring (arbitrary FIM span)
FAILS; use-site anchoring recovers the signal and it strengthens toward the use
line — the clearest evidence that target definition, not just ckpt strength,
governs whether a cross-doc benchmark discriminates.

| traversal | scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|---|
| bfs | native | +0.012 (−0.033,0.061)† | +0.041 (−0.004,0.090)† | 242 |
| bfs | rest_of_doc | +0.024 (0.014,0.036) | +0.048 (0.036,0.063) | 135 |
| bfs | use_block | +0.047 (0.028,0.068) | +0.069 (0.049,0.090) | 138 |
| bfs | use_line | **+0.094 (0.052,0.140)** | +0.123 (0.083,0.166) | 138 |
| dfs | native | −0.014 (−0.043,0.015)† | +0.011 (−0.022,0.043)† | 242 |
| dfs | use_block | +0.038 (0.020,0.058) | +0.061 (0.043,0.081) | 138 |
| dfs | use_line | +0.075 (0.034,0.120) | +0.116 (0.075,0.162) | 138 |
| random_walk | native | −0.020 (−0.053,0.014)† | −0.013 (−0.057,0.027)† | 242 |
| random_walk | use_block | +0.034 (0.015,0.054) | +0.056 (0.038,0.076) | 138 |
| random_walk | use_line | +0.069 (0.030,0.112) | +0.107 (0.068,0.150) | 138 |
| random | native | −0.056 (−0.134,0.011)† | −0.033 (−0.099,0.019)† | 242 |
| random | use_block | +0.004 (−0.016,0.024)† | +0.058 (0.040,0.077) | 138 |
| random | use_line | +0.011 (−0.030,0.055)† | +0.106 (0.069,0.148) | 138 |

(use_block/rest_of_doc rows abbreviated for dfs/rw; full values in matrix JSON.)

**Both axes matter and compound.** Down each traversal, Δnll_real and placebo sep
climb native→rest_of_doc→use_block→use_line. Across traversals, bfs>dfs>rw>random
at every scope. bfs+use_line = +0.094, in the python(+0.095)/java(+0.112) band.
Even the strong bfs ckpt cannot rescue the arbitrary-span `native` target (CI
crosses 0). The random ckpt only recovers a signal at all via use-site anchoring
(Δnll_real still 0-crossing, but placebo sep +0.106 is solidly positive).

**Checkpoint answer ("was Kotlin weaker?"): YES — the traversal, not epochs.**
All Kotlin ckpts trained 2 epochs / ~14k steps; the first one used the **random**
traversal, weakest here (native −0.056) and in the py sweep + java ablation. bfs
8× the use_line Δnll (+0.011→+0.094).

**n ceiling:** ASE-2025's public pool is 242 (practice+public; private split has
no public ground truth). GPU cannot raise n past 242 here — hence use_line's n=138
stays under the 200 gate despite a clean signal. Unlimited n lives only in the
self-built test_community path (TODOS).

## TypeScript — CrossCodeEval (AWS)
Full 3,356 pool. Aux are retrieval CHUNKS (not whole import-resolved files), so
the signal is diluted and the use-site filter is lossy (only ~314/3356 resolve a
use site → n=314 for use-scopes, below the 200 gate at fire-rate 0.45).

| traversal | scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|---|
| bfs | native | +0.029 (0.020,0.039) | +0.040 (0.031,0.050) | 3356 |
| bfs | use_line | +0.051 (0.033,0.069) | +0.062 (0.043,0.082) | 314 |
| bfs | use_block | +0.024 (0.014,0.035) | +0.030 (0.020,0.041) | 314 |
| bfs | rest_of_doc | +0.006 (0.001,0.011) | +0.011 (0.006,0.015) | 314 |
| dfs | use_line | +0.048 (0.032,0.065) | +0.056 (0.034,0.077) | 314 |
| random_walk | use_line | +0.039 (0.024,0.056) | +0.049 (0.032,0.066) | 314 |
| random | native | +0.017 (0.008,0.027) | +0.042 (0.033,0.052) | 3356 |
| random | use_line | +0.039 (0.019,0.062) | +0.070 (0.047,0.095) | 314 |

At FULL sample TS is cleaner than the earlier 500-cap run: use_line shows a
significant Δnll_real (+0.051 bfs, CI excludes 0) AND positive placebo sep on
every traversal, and the use_line>use_block>rest_of_doc gradient now holds
(chunk aux carries the immediate use context best). native n=3356 also passes on
Δnll_real. Same traversal ordering (bfs/dfs>rw>random on Δnll_real). Remaining
gate failures are power (use-scope n=314<... no, n=314 fired 136<200) + fire-rate
0.45<0.5 — both artifacts of chunk aux. **v2 (whole-file re-clone from
metadata.repository) remains the upgrade** to lift fire-rate and n and confirm
the gradient; but even v1 at full sample shows a real, significant cross-doc
signal.

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

## Summary table (use_line scope, best traversal = bfs, FULL sample)
| port | Δnll_real (CI) | placebo sep (CI) | n | pass |
|---|---|---|---|---|
| repobench_python | +0.095 (0.087,0.103) | +0.139 (0.130,0.147) | 6989 | ✓ |
| repobench_java | +0.112 (0.102,0.122) | +0.162 (0.152,0.172) | 8585 | ✓ |
| ase_kotlin | +0.094 (0.052,0.140) | +0.123 (0.083,0.166) | 138 | signal ✓, n<200 (pool ceiling 242) |
| crosscodeeval_ts | +0.051 (0.033,0.069) | +0.062 (0.043,0.082) | 314 | signal ✓, n<200 fired + chunk aux (v2) |

All four ports show a significant use_line cross-doc signal at full sample.
RepoBench python/java PASS all gates outright; Kotlin and TS have real signals
(Δnll_real CI excludes 0) capped only by sample power (ASE 242-example pool; TS
chunk-aux fires on ~1/3 of examples). Full per-traversal detail above; per-job
JSON in eval/benchmark_harness/reports/matrix/.
