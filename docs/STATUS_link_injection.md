# STATUS — link-injection causal eval

**Read this first to re-orient.** Worktree `/fss/evin_t/tagseq2tagseq-linkeval`, branch
`link-injection-eval` (rebased onto `main`; only its own commits). **Draft PR #11 → main.**
Design: `docs/link_injection_causal_eval_DESIGN.md`. Driver: `eval/link_injection_grid.py`.

## What this is
Inference-time causal test of whether cross-doc-link TRAINING lets a model exploit an
injected link + aux doc more than doc-causal training does. Same injected link + aux
(annotated once with the cross-doc checkpoint) replayed to both checkpoints of the
matched 07-03 wiki_merged pair under baseline / grant / concat / invisible / placebo.
Headline = paired bootstrap-CI interaction (aux lift cross-doc − aux lift doc-causal).
Optional gold-aux gradient (sciq `support`) adds grant_gold / concat_gold / placebo_gold and
`relevance_slope = grant − grant_gold`.

## DONE
- Harness + 15 CPU tests (+ concat scoring tests); 448 tests pass across `tests/eval`,
  `tests/model`.
- Mask semantics validated: sciq smoke n=40, 40/40 fired, `invisible_check ≈ 0` on both
  checkpoints (`results/link_grid_smoke_sciq/`).
- Grant/placebo cells use the detector's coarse mode (precise re-detection dropped 10/40
  fired items: `detect_links` truncates titles at the first `)`).
- Gold-aux gradient (sciq only — hotpotqa's context already holds the gold sentences),
  `--replay-records` re-scoring, per-item `<bench>_cell_scores.json` dump.

## RESULTS — sciq, n=1000 (999 fired; 887 carry a gold passage), matched 07-03 pair
`results/link_grid_sciq_full/` (retrieved arm) and `results/link_grid_sciq_full_gold/`
(same records replayed with gold cells; per-item NLLs in `sciq_cell_scores.json`).
All deltas are gold-completion NLL, "+" = the aux HELPED; mean [95% CI] (median).

| effect | cross-doc ckpt | doc-causal ckpt | interaction (cross − dc) |
|---|---|---|---|
| invisible_check (must be ~0) | −0.003 [−0.011, 0.004] | +0.005 [−0.002, 0.013] | — |
| aux lift, grant, model-retrieved aux | −1.74 [−2.34, −1.16] (−0.00) | −0.39 [−0.68, −0.12] (+0.01) | **−1.35 [−1.89, −0.84]** (−0.12) |
| aux lift, grant, GOLD aux | +5.29 [4.79, 5.80] (+2.55) | +5.02 [4.56, 5.51] (+2.35) | **+0.27 [0.08, 0.46]** (+0.10) |
| aux lift, concat, GOLD aux | +7.92 [7.33, 8.54] (+5.03) | +7.86 [7.27, 8.47] (+5.15) | — |
| mechanism (grant − concat), GOLD | −2.65 [−3.10, −2.23] | −2.85 [−3.31, −2.42] | — |
| mechanism (grant − concat), retrieved | +0.43 [−0.01, 0.85] | +0.83 [0.34, 1.33] | — |
| placebo sep (grant − placebo), GOLD | +5.35 [4.87, 5.86] | +4.87 [4.42, 5.36] | — |
| relevance slope (gold − retrieved lift) | +6.95 [6.21, 7.71] (+2.67) | +5.41 [4.88, 5.96] (+2.26) | **+1.54 [0.97, 2.12]** (+0.20) |

**Reading.**
- Mask semantics hold at n=1000 (`invisible_check ≈ 0`, CIs within ±0.01 nats).
- The "retrieved" arm is a FLOOR, not retrieval: the 3614-step model opens a link with
  median prob 0.002 and the greedy (beam_width=1) trie walk lands on junk titles — 302/999
  are literally `?`, 506/999 are ≤3 chars, 242 aux docs are <100 tokens. That aux HURTS both
  models and is barely distinguishable from a wrong-item placebo (placebo_sep ≈ +0.25/+0.37).
- The cross-doc-trained model is MORE sensitive to aux content in both directions: junk
  hurts it ~1.35 nats more (negative retrieved interaction), gold helps it 0.27 nats more
  (positive gold interaction), and the relevance slope is 1.5 nats steeper. So cross-doc
  training does change how the model treats a linked doc — but the extra utilization of a
  *good* aux over generic in-context learning is small (+0.27 on a +5 main effect; medians
  +0.10 on +2.4): at this scale, "relevant context helps any LM" carries most of the lift.
- The link GRANT under-performs raw concatenation on gold aux by ~2.7 nats for BOTH models
  (grant only exposes the aux from the injected link position onward, and the injected
  slot is often late in the question; concat exposes it from token 0). Grant is the
  *safer* mechanism for junk aux (less harmed than concat), the *weaker* one for gold aux.
- Means are heavy-tailed (gold grant mean +5.3 vs median +2.5): a gold passage that
  literally contains the answer collapses NLL on a subset of items. Report medians too.

## NEXT
1. A real "retrieved" rung: the model-generated title is a floor. Needs a proper
   retriever (BM25 over article text, or entity match with a stopword filter — naive
   longest-n-gram title match picks "that is"/"found in"). Also try re-annotating with
   `--beam-width 5` (length-normalized beam avoids the `?` degenerate path).
2. Stratify the per-item deltas in `sciq_cell_scores.json` by aux length and by
   question↔gold n-gram overlap (leakage α) — the heavy tails suggest the effect
   concentrates on answer-containing passages.
3. Other knowledge benchmarks (wiki_qa, openbookqa, arc_*) for the retrieved/placebo arms
   (no gold passage there); the Claude-generated-aux ceiling arm.
4. The matched pair is weak (3614 steps). A stronger matched doc-causal run would be
   needed to say whether the +0.27 gold interaction grows with training.

## Commands
```bash
cd /fss/evin_t/tagseq2tagseq-linkeval
PY=/fss/evin_t/tagseq2tagseq/.venv/bin/python   # worktree has no venv; don't `uv run` here
# add the gold-aux cells to the finished sciq run without re-annotating:
CUDA_VISIBLE_DEVICES=<free> $PY -m eval.link_injection_grid \
  --cross-ckpt /fss/evin_t/tagseq2tagseq/runs/20260703_050528/checkpoints/best_model.pt \
  --doc-causal-ckpt /fss/evin_t/tagseq2tagseq/runs/20260703_051129/checkpoints/best_model.pt \
  --replay-records results/link_grid_sciq_full/sciq_records.jsonl --gold-aux \
  --out-dir results/link_grid_sciq_full_gold
```
Then: leakage-α / popularity stratification over `_cell_scores.json`; other knowledge
benchmarks (wiki_qa, openbookqa, arc_*); Claude-generated-aux ceiling arm.

## Interpretation guard
Effect sizes from n≤40 smokes are noise. The matched pair is weakly trained (3614 steps);
absolute lifts are expected to be small — the interaction and the relevance slope are the
quantities of interest, and `invisible_check ≈ 0` must hold in every run.
