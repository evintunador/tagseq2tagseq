# Handoff: arxiv community_pack sparsity (n=5) — RESOLVED

## TL;DR (2026-07-31)
**Not a data-sparsity artifact and not arxiv-specific — it was an eval-infra bug.**
`run_community_pack_perplexity` resolved the pack token budget through a chain that
never matched a `TS2TSModel`, so it silently fell through to a hardcoded **2048**.
Every community_pack eval packed at 2048 tokens instead of the trained 32768. arxiv,
whose docs are huge (median 14.7k tok/doc), collapsed to **n=5** scoreable packs;
every other source was under-packed too (their n was just high enough to look fine).

Fixed in `eval/perplexity.py` (one budget-resolution block) and **the full per-source
community_pack pass has been re-run at 32768** (2026-08-01). arxiv is now n=392, Δ=+0.0039
(95% CI [0.0023, 0.0059], strictly positive). See "Re-run results" below.

## Re-run results (2026-08-01, corrected 32768 budget)
All 11 sources re-run for `community_pack_perplexity` only (held_out was unaffected).
EXPERIMENTAL condition (cross_doc_link vs doc_causal baseline), n = scoreable packs:

| src        | n (2048, buggy) | n (32k, fixed) | Δ (2048) | Δ (32k) | Δ 95% CI (32k) |
|------------|----------------:|---------------:|---------:|--------:|:---------------|
| wiki       | 471 | 497 | 0.1193 | **0.1595** | [0.1523, 0.1666] |
| stack      | 276 | 490 | 0.0341 | **0.0761** | [0.0701, 0.0827] |
| arxiv      |   5 | 392 | 0.0000 | **0.0039** | [0.0023, 0.0059] |
| go         | 200 | 432 | 0.0039 | **0.0101** | [0.0085, 0.0119] |
| java       | 212 | 484 | 0.0010 | **0.0044** | [0.0033, 0.0055] |
| typescript | 361 | 496 | 0.0260 | **0.0568** | [0.0519, 0.0620] |
| kotlin     | 217 | 459 | 0.0056 | **0.0212** | [0.0188, 0.0235] |
| rust       | 191 | 473 | 0.0163 | **0.0295** | [0.0256, 0.0339] |
| javascript | 318 | 496 | 0.0431 | **0.0586** | [0.0487, 0.0707] |
| zig        | 220 | 356 | 0.0041 | **0.0052** | [0.0042, 0.0064] |
| dart       | 231 | 447 | 0.0085 | **0.0308** | [0.0281, 0.0337] |

Every source's n roughly doubled and its Δ grew; all Δ CIs are now strictly positive.
The buggy 2048 JSONs are preserved in `eval_by_source/prefix2048_buggy_backup/`.

Operational note: dart initially crashed on the `ReproducibilityManager` run-dir
collision (two evals starting in the same wall-clock second share `eval/<ts>/`, per the
CLAUDE.md "never launch simultaneously" gotcha) and was re-run solo. Any future re-run
of `eval_merged_v2_run.sh` should keep the 6s inter-launch stagger it already has, or
give each eval a unique `--output`-derived run dir.

## Root cause (verified from source + runtime + exact reproduction)
`eval/perplexity.py::run_community_pack_perplexity` had:
```python
token_budget = getattr(model, "max_seq_len", None)          # TS2TSModel has NO such attr → None
if token_budget is None:
    try:
        token_budget = model.backbone.config.max_position_embeddings  # backbone has no .config → AttributeError
    except AttributeError:
        token_budget = 2048                                  # ← always lands here
```
- `TS2TSModel` stores no top-level `.max_seq_len` (grep of `model/model.py`: only a
  `from_config` param, never `self.max_seq_len =`).
- `TS2TSBackbone` exposes `self.max_seq_len` **directly** (`model/modules/backbone.py:64`),
  NOT an HF-style `.config.max_position_embeddings` (`hasattr(backbone,'config') == False`,
  confirmed at runtime).
- So the budget was **always 2048**, regardless of the 32768 in `hyperparameters.json`.
  `eval_checkpoints.py` passes no `--max_seq_len` override, so nothing corrected it.

Note the other four scoring paths already resolve it correctly via
`getattr(getattr(model, "backbone", None), "max_seq_len", None)`
(`eval/scoring.py:120,220,532,924`) — `run_community_pack_perplexity` was the lone copy
that used the broken chain. Introduced in `bbb2fe4` (2026-04-23).

## Evidence
1. **arxiv val_community has abundant graph structure** — NOT sparse:
   55,026 nodes, 302,003 in-split outgoing edges, 46,629 nodes with ≥1 in-split
   outgoing edge. (thestack for comparison: 89,040 nodes / 244,016 edges.)
2. **Co-pack feasibility is high at 32k**: 44.4% of arxiv's 302k linked pairs have
   `src_tok + tgt_tok ≤ 32768` (thestack 98.6%). Plenty of scoreable pairs exist.
3. **Exact reproduction** — replaying the real `PackBatchSampler` + `PackedSequenceDataset`
   on arxiv val_community (BFS, seed=42, EOS layout, `prefer_targets_first`), sweeping the
   token budget:

   | budget | packs | scoreable | skipped |
   |-------:|------:|----------:|--------:|
   | 32768  |  500  |    392    |   108   |
   | 16384  |  500  |    251    |   249   |
   |  8192  |  500  |    106    |   394   |
   |  4096  |  500  |     14    |   486   |
   | **2048** | **500** | **5** | **495** |

   The eval's recorded `n=5, Skipped 495` is an **exact match to budget=2048** — and
   what the trained model should have used (32768) yields **n=392**.
4. **`mean_delta == 0.0` exactly** (cross == baseline bit-for-bit) and the whole arxiv
   community_pack finished in ~6s (wiki took ~44s): both consistent with tiny 2048-token
   packs where each doc is head-truncated to ~1k tokens, leaving no room for the cross-doc
   grant region to differ from the doc_causal baseline.

## The fix (applied)
`eval/perplexity.py::run_community_pack_perplexity` — resolve the budget from the backbone
first, mirroring `score_doc`:
```python
token_budget = getattr(getattr(model, "backbone", None), "max_seq_len", None)
if token_budget is None:
    token_budget = getattr(model, "max_seq_len", None)
if token_budget is None:
    try:
        token_budget = model.backbone.config.max_position_embeddings
    except AttributeError:
        token_budget = 2048
```
Backbone `.max_seq_len` returns 32768 for this checkpoint (runtime-confirmed). Scope of the
change: `eval/perplexity.py` only. It does **not** touch `split_graph.py`, the arxiv dataset,
training, or the merged pipeline (all of which the handoff protected).

## Corpus-wide impact (important)
This was **not** arxiv-only. EVERY source in `eval_merged_v2_run.sh`'s community_pack pass
ran at 2048 instead of 32768, so all recorded `n`/`delta` values in
`runs/run_20260730_183342_811412/eval_by_source/*__community_pack_perplexity.json` are from
under-packed 2048-token packs. The directional signal (wiki > js > stack > … > arxiv) is
probably still valid, but the magnitudes and especially the low-n sources (arxiv, and any
other long-doc source) should be regenerated. arxiv specifically will go from n=5 (Δ=0) to
~n=392 with a real positive Δ once re-run at 32k.

## To regenerate
Re-run `scripts/eval_merged_v2_run.sh run_20260730_183342_811412 cdl` (now that the budget
resolves to 32768). Expensive at 32k context; if you only care about arxiv, run the single
`community_pack_perplexity` invocation for arxiv from that script. Held-out perplexity is
unaffected (it uses `score_docs_batched`, which already resolved the budget correctly).

## Residual note
arxiv community_pack remains a MINOR signal; arxiv's real cross-doc benchmark is `\cite`
resolution (separate). But the n=5 was a bug, not "arxiv is just sparse" — worth fixing so
the number isn't misread as evidence that cross-doc doesn't help arxiv.
