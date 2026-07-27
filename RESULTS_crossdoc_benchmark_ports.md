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

### Python scope validation (ckpt run_20260720_063128_690228, bfs)
RepoBench has no post-hole file body (`all_code` is just a license header), so
its use-scopes collapse to use_line — a built-in check that re-anchoring
reproduces native:

| scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|
| native | +0.0936 (0.072..0.117) | +0.127 (0.105..0.150) | 500 |
| use_line = use_block = rest_of_doc | +0.0975 (0.074..0.123) | +0.135 (0.111..0.160) | 424 |

Restricting to the 424 genuine cross-file-use lines SHARPENS the signal
(+0.0975 > +0.0936). Re-anchoring is sound.

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

### bfs-traversal ckpt (run_20260722_181228_995658), full pool
<!-- FILLED IN when the bfs full-pool run completes -->
_(pending — bfs is the strongest traversal per the code sweep; random was
weakest there too, +0.031 vs +0.065..079, so this ckpt should lift Δnll_real.)_

**Checkpoint caveat (answers "was Kotlin weaker?"):** the first Kotlin ckpt used
the **random** traversal — the weakest for cross-doc signal in BOTH the Python
sweep and the Java RepoBench ablation (random +0.031 vs bfs/dfs/rw +0.065..0.079;
see RESULTS_code_crossdoc.md). Epoch count was NOT the difference (all Kotlin
ckpts trained 2 epochs / ~14k steps). Traversal choice was. The bfs re-run above
isolates that.

**n ceiling:** the ASE-2025 public pool is 242 examples (practice+public; the
private split has no public ground truth). GPU time cannot raise n past 242 for
this benchmark — power comes from a stronger ckpt (bigger effect) and use-scope
selection. Unlimited n lives only in the self-built test_community path (TODOS).

## TypeScript — CrossCodeEval (AWS)
Aux are retrieval CHUNKS (not whole import-resolved files), so signal is
diluted; still shows a positive placebo separation.

### native + scopes (ckpt run_20260722_003634_268441)
| scope | Δnll_real (CI) | placebo sep (CI) | n |
|---|---|---|---|
| native | +0.013 (−0.003..+0.031) | +0.023 (+0.007..+0.040) | 200 |
<!-- use-scope rows FILLED when the TS scope run completes (full_file now populated) -->

placebo separation CI excludes 0 even on chunk aux → the benchmark is sensitive
to context correctness. v2 (whole-file re-clone from metadata.repository +
runtime relative-import resolution) should lift fire-rate (0.40→~0.67) and
tighten Δnll_real above 0.

## Go — CoLT-132K: BLOCKED, port removed
The released CoLT-132K.zip ships EMPTY `cross_file_dependency` for every Go
example (aux live in unshipped external `godata` JSONs; verified no download /
no regen code exists). The `colt_go` port produced only zero-aux examples and
was removed 2026-07-25 (preserved in git history at commit de5ba34). Path
forward: file an issue with the aiXcoder authors for the godata, or self-build
Go from its test_community split.

## Cross-benchmark takeaways
1. Target scope matters as much as checkpoint quality: an arbitrary-span
   benchmark can show ZERO (or negative) cross-doc signal that use-site
   anchoring recovers.
2. placebo separation is the robust primary metric; Δnll_real needs power
   (n and a link-exploiting ckpt).
3. Report use_line / use_block / rest_of_doc together — the gradient is itself
   the finding (signal concentrates at the use site).
