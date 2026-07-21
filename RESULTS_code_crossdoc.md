# Code cross-doc generalization sweep — results (2026-07-21)

Tests whether wiki's cross-doc-link gains generalize to code. All runs: 1024d/24L
~350M model, VE-off, muon_lr=0.003/wd=0.1, 15k steps (~3.9B tok chinchilla), 8×A100.
18 runs, all completed 15k steps + full eval. Headline cross-doc metric for code =
`repobench_cross_doc` (Python only; same-token NLL with cross-doc attention on vs off).
Go/Java lack a cross-doc code benchmark → `community_pack_perplexity [experimental]`
paired delta is the language-agnostic proxy.

## 4-condition sweep (per language)

| condition | held_ppl | repobench_ppl | humaneval_acc | commPack_exp_Δ | repobench_cross_doc Δnll |
|-----------|---------:|--------------:|--------------:|---------------:|-------------------------:|
| PY doc_causal        | 4.230 |  8.941 | 0.659 |   —    |   —   |
| PY cross_doc_link    | 4.233 |  **7.248** | 0.640 | 0.0018 | **+0.135** |
| PY doc_concat        | 4.279 | 10.417 | 0.640 | 0.0321 |   —   |
| PY doc_concat_link   | 4.264 |  8.763 | 0.628 | 0.0095 |   —   |
| GO doc_causal        | 3.773 |   —    | 0.665 |   —    |   —   |
| GO cross_doc_link    | 3.758 |   —    | 0.665 | 0.0012 |   —   |
| GO doc_concat        | 3.789 |   —    | 0.671 | 0.0036 |   —   |
| GO doc_concat_link   | 3.786 |   —    | 0.665 | 0.0033 |   —   |
| JAVA doc_causal      | 3.206 |   —    | 0.622 |   —    |   —   |
| JAVA cross_doc_link  | 3.169 |   —    | 0.628 | 0.0002 |   —   |
| JAVA doc_concat      | 3.237 |   —    | 0.628 | 0.0016 |   —   |
| JAVA doc_concat_link | 3.181 |   —    | 0.659 | 0.0007 |   —   |

## Traversal ablation (Python; mirrors wiki bfs/dfs/rw/random × dc/cdl)

| traversal × mask | held_ppl | repobench_ppl | humaneval_acc | repobench_cross_doc Δnll |
|------------------|---------:|--------------:|--------------:|-------------------------:|
| bfs    doc_causal     | 4.230 | 8.941 | 0.659 |   —    |
| bfs    cross_doc_link | 4.233 | 7.248 | 0.640 | +0.135 |
| dfs    doc_causal     | 4.327 | 8.563 | 0.610 |   —    |
| dfs    cross_doc_link | 4.425 | 7.346 | 0.665 | +0.100 |
| rw     doc_causal     | 4.269 | 9.238 | 0.598 |   —    |
| rw     cross_doc_link | 4.324 | 8.858 | 0.604 | +0.069 |
| random doc_causal     | 4.178 | 9.683 | 0.628 |   —    |
| random cross_doc_link | 4.161 | 7.797 | 0.622 | +0.151 |

## Findings

1. **Cross-doc thesis generalizes to code — direction confirmed, magnitude ~10× smaller than wiki.**
   On the discriminating benchmark (`repobench_cross_doc`), cross_doc_link beats
   flat-concat on every traversal (Δnll +0.07 to +0.15). Wiki's hotpotqa Δnll was
   +1.29. So attending across import edges helps predict imported code, but the
   effect is modest — code cross-file dependency is more local/predictable than
   Wikipedia bridge reasoning.

2. **`repobench_ppl`: cross_doc_link consistently beats doc_causal** (PY 7.25 vs 8.94;
   holds across all traversals). This is the clearest signal — the cross-doc model
   is a materially better next-line predictor on RepoBench's cross-file completions.

3. **community_pack_perplexity is near-noise for code** (deltas 0.0002–0.032), unlike
   wiki. Code import-graph neighborhoods are dense/predictable, so toggling the mask
   on held-out packs barely moves NLL. Java (sparsest graph, out-deg 0.72) ≈ 0.
   → For code, use `repobench_cross_doc`, not community_pack, as the cross-doc metric.

4. **doc_concat_link/doc_concat FLOP controls do NOT beat cross_doc_link.**
   On repobench_ppl, doc_concat is WORSE than doc_causal (PY 10.4 vs 8.9) — naive
   whole-file concatenation hurts. cross_doc_link's gated link attention is the best
   code predictor at equal-ish FLOPs. (Curiously doc_concat has the largest
   community_pack delta, but that metric is unreliable for code per #3.)

5. **Traversal ablation mirrors wiki's decoupling.** Cross-doc Δnll is robust across
   bfs/dfs/rw/random (all positive; bfs 0.135, random 0.151 — no clean ordering,
   spread is within noise at n≈430). Traversal affects base-LM quality more than the
   incremental cross-doc benefit — same two-axes story as wiki.

6. **Single-doc metrics tied across masks** (held_ppl, humaneval within CI) — expected;
   cross-doc structure only helps cross-doc-structured benchmarks, matching wiki.

## Caveats
- Java/Go have no cross-doc code benchmark, so their cross-doc claim rests only on the
  weak community_pack signal — inconclusive for those languages. A language-split
  RepoBench adapter (tianyang/repobench_java_v1.1 exists) would close this.
- 15k-step chinchilla subset; larger budgets may widen or shrink the gap.
- Run map: runs/CODE_SWEEP_RUNMAP.txt. Per-run detail: runs/<id>/eval_results.json.
