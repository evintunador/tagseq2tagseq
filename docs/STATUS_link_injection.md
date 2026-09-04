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

## IN FLIGHT
- Full sciq run (n=1000, retrieved aux, coarse grant) → `results/link_grid_sciq_full/`.
  Launched 2026-09-04 on GPU-670 GPU 1 with the main checkout's venv.

## NEXT (commands)
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
