# Roadmap notes — incomplete work

Loose PM-level notes on what still needs planning + implementation.
Not a spec; just enough to orient a planning session for each item.

---

## Eval pipeline — extensions

### Cross-doc contrastive perplexity — DONE
The core research claim is that cross_doc_link attention helps the model leverage
linked documents. `run_pack_contrastive_perplexity` scores pre-computed training
packs under both cross_doc_link and doc_causal masks, reporting mean NLL delta
per traversal strategy.

Implementation uses pack-based topology-aware scoring: within each pack, only body
tokens of docs with incoming cross-doc edges (target docs) are scored. Context-only
docs (no incoming edges in the pack) produce identical NLL under both conditions
and are excluded to keep the signal clean. `score_doc_with_context` in
`eval/scoring.py` is the primitive; the full forward pass runs over the entire
packed sequence so cross-doc grants can fire normally.

### Batched MC scoring — DONE
`score_completions_batched` in `eval/scoring.py` packs K (context + choice) sequences
as K DocSpans in one forward pass (~K× faster than K individual calls).

### Split annotations — DONE
`data/split_graph.py` writes five self-contained dataset subdirs under
`dataset_dir/splits/{train,val_community,val_random,test_community,test_random}/`.
Each subdir is a complete `GraphIndex`-compatible directory (filtered
`tokenized_graph.jsonl` + `metadata.json` with absolute shard paths). Cross-split
edges are dropped so each split has no knowledge of the others.

Split design: 2.5% each for val/test community + random (~90% train).
- `val_community` / `test_community` — BFS-identified subgraphs; internal link
  structure intact. Used for `community_pack_perplexity` and periodic val loss.
- `val_random` / `test_random` — uniform random scattered nodes. Used for
  `held_out_perplexity`. Neighbors remain in train, so this is softer held-out.

All four production datasets have been split:
- `simplewiki`: 246k train / 7.5k val_community / 6.9k val_random / …
- `stack_10m`: 2.14M train / 59.5k val_community / 59.5k val_random / …
- `stack_100m`: 3.2M train / 89k val_community / 89k val_random / …

Training pipeline integration:
- `main.py` reads `data.train_dir`, `data.val_dirs`, `data.test_dirs` from config
  (explicit, no auto-detect). Falls back to full `dataset_dir` if unset.
- `data.val_epoch_dirs` — optional precomputed val packs (same format as
  `epoch_dirs`); key must match a corresponding `val_dirs` entry.
- `multi_val.py` atomic feature: evaluates each named val loader independently
  every `val_interval` steps, saves checkpoint on best mean val loss.
- Post-training eval automatically runs `community_pack_perplexity` on all
  `val_dirs`/`test_dirs` community entries and `held_out_perplexity` on random
  entries, under doceval (doc_causal) or baseline+experimental (cross_doc_link).

Config matrix (all fully self-contained with split dirs):
- `configs/simplewiki_doc_causal.yaml`
- `configs/simplewiki_cross_doc.yaml`
- `configs/stack_100m_doc_causal.yaml`
- `configs/stack_100m_cross_doc.yaml`

Caveat for existing checkpoints (20260308_*): trained before splits existed, so
all val/test nodes were in training data. Numbers are diagnostic baselines only.
Future checkpoints using the new configs will have genuine held-out eval.

### NL benchmarks — DONE
All NLP benchmarks active in `eval/nlp_benchmarks.py`:
HellaSwag, WikiQA, LAMBADA, WinoGrande, ARC (easy/challenge), PIQA, BoolQ,
CommonsenseQA, COPA, OpenBookQA, SciQ — all via tunalab NLP catalog.

### STEM / math benchmarks — DONE
Added to `eval/nlp_benchmarks.py` (HF-direct, no tunalab adapter needed):
- `run_mmlu(subject)` — 4-way MC, 13 STEM subjects available (college_mathematics,
  high_school_physics, machine_learning, etc.). Same structure as ARC.
- `run_mathqa()` — 5-way MC, 2985 test items, math word problems.
- `run_math(subject)` — LaTeX fill-in-blank, competition math (Hendrycks MATH dataset),
  7 subjects. Perplexity over full solution tokens. Relevant for future ArXiv models;
  useful now as a diagnostic for mathematical LaTeX fluency.

### Code benchmarks — DONE
Added to `eval/nlp_benchmarks.py` (HF-direct):
- `run_codexglue_code_to_text()` — Python function → docstring, 14918 test items.
  Cleaner signal than line completion (semantically rich targets).
- `run_repobench(split)` — cross-file next-line prediction with explicit repo context.
  3 splits: cross_file_first (most interesting for cross_doc_link), cross_file_random,
  in_file (control). The cross_file_first split is a natural controlled experiment for
  whether cross-doc attention benefits code with import dependencies.
- `run_humaneval_buggy(language)` — canonical vs buggy 2-way MC, 164 items per language
  (python/cpp/go/java/js/rust). No execution needed; unique contrastive angle.

All wired into `eval_checkpoints.py` dispatcher and CLI with flags:
  --mmlu-subject, --math-subject, --repobench-split, --humaneval-language

### Wikipedia cross-doc benchmark — TODO
Wikipedia-trained models have no equivalent of `repobench_cross_doc`. Candidates:
- HotpotQA / 2WikiMultihopQA: multi-hop QA where the answer requires two linked
  Wikipedia articles. Pack both supporting docs as aux DocSpans (analogous to
  repobench_cross_doc), score the answer under cross_doc_link vs doc_causal mask.
  This would be the clearest "smoking gun" for Wikipedia cross-doc attention.
- Strategy: implement `run_hotpotqa_cross_doc` in `eval/nlp_benchmarks.py` on
  the same pattern as `run_repobench_cross_doc` — fetch both gold supporting
  passages from the corpus by their Wikipedia page name, pack as aux DocSpans,
  score answer NLL under both conditions.

### RepoBench cross-doc-link mode — DONE
`run_repobench_cross_doc` in `eval/nlp_benchmarks.py` packs each example's cross-file
snippets as proper aux DocSpans and scores with cross_doc_link attention. Uses precise
per-import matching: each snippet carries `raw_identifier="repo:path/to/file.py"` so
PythonImportDetector can match each import statement to its specific snippet.

HF dataset: `tianyang/repobench_python_v1.1` (moved from `Leolty/repobench-python-v1.1`).
Schema: `context` is list of `{identifier, path, snippet}` dicts; `cropped_code` is the
file body; `import_statement` is prepended so the detector finds the relevant imports.

New scoring primitive: `score_completion_with_context_docs` in `eval/scoring.py`.
Accepts `source_file_path` and resolves relative imports (`from . import X`) at eval
time using the file's known path — brings effective cross-file match rate from ~67% → ~97%.
Reports both `perplexity_cross_doc_only` (samples with detected imports) and
`perplexity_with_fallback` (all samples; no-link cases fall back to flat scoring).

Only runs on `cross_doc_link` models with a `PythonImportDetector`. For future languages,
split by language and match each to its `<Language>ImportDetector`.

Validated result on stack_10m: cross_doc ppl=31 vs flat doc_causal ppl=62 (~2× improvement).

### PythonImportDetector: relative imports in training — TODO (discuss separately)
At eval time, relative imports are now resolved via `source_file_path` in
`score_completion_with_context_docs`. The training pipeline question is separate:
the epoch precompute fast-path uses pre-built `link_to_target` from graph edges (which
ARE built with relative imports resolved by `data/github_graph_extractor/extract.py:80`),
so training may already handle this correctly. Needs verification before any changes to
`PythonImportDetector.detect_links` or the training collation path.

### Link injection eval for other external benchmarks — TODO
For all external benchmarks OTHER than RepoBench, the path to cross-doc-link
evaluation is via prompt preprocessing (link injection), not direct structural
reformatting. This requires `eval/link_annotator.py`:
  `annotate_prompt_with_links(model, prompt_tokens, threshold)`:
  - Single forward pass over prompt
  - Find positions where the link-opener token (e.g. ` [` for markdown) has
    logit probability above threshold
  - Insert link + generate target + fetch/generate aux doc
  - Return augmented prompt tokens
Then the eval comparison: score benchmark items with bare prompt vs
link-annotated prompt, report delta.

### Parallelized eval in main.py — TODO
Currently the post-training eval runs serially. If multiple benchmarks are
configured, they could run in parallel threads/processes after training.

---

## Data & datasets

### Dataset plan (settled)
Three datasets, trained independently:
1. **Wikipedia** — combine simplewiki + enwikisource + full English Wikipedia into
   one pretokenized dataset. Markdown link detector.
2. **Stack 100M** — the 100M-node split only (stack_10m retired). Python import
   detector. `stack_100m_32k.yaml` config already exists.
3. **ArXiv** (future) — LaTeX citation link detector. Dataset not yet built.

Aggregate model combining all three with a composite link detector (runs all three
individual detectors, doesn't need to be fast) is a future milestone.

No FineWeb or other flat data planned — the point is structured graph data.

---

## Model

### Scale-up
36L/1280D is placeholder. For the paper, need at least one run at a meaningfully
larger scale. Needs decisions on: target param count, num_layers/model_dim
tradeoff, whether to use the same BFS+cross_doc_link config as current best runs.

### Layout policies — DONE (BOS removed)
All layout policies now use EOS-only suffix (no BOS prefix). Current names:
- `eos` — EOS suffix, no prefix (`EOSLayoutPolicy`)
- `identifier_prefix_eos` — "# {id}\n\n" prefix + EOS suffix
- `stochastic_identifier_prefix` — stochastic 50/50 prefix + always EOS suffix
- `identifier_prefix` — prefix only, no EOS (external benchmark use)
- `null` — no decoration

Training always uses `stochastic_identifier_prefix`. Inference uses either
`identifier_prefix_eos` (prefix on) or `eos` (prefix off) — both stored on
`TS2TSModel` as `training_layout_policy` / `inference_layout_policy`. `BucketedPackDataset`
calls `set_epoch()` on epoch advance; `PackedSequenceDataset` + `multi_epoch` path
has no epoch hook (deferred until BucketedPackDataset is sole production path).
Science question remains open: does stochastic training meaningfully help on
external benchmarks vs. fixed eos layout?

---

## Generation feature — Stage 3

`GENERATION_WORK_BREAKDOWN.md` defines Stage 3 as:
- `find_evicted` / `restore_evicted` re-eviction logic in `DocumentContext` — **DONE**
  (both fully implemented; no stubs remain)
- `link_retrieval_mode` in `GenerationConfig` — **DONE**
  (`corpus_only`, `generate_only`, `corpus_then_generate`, `link_but_skip`, `full_skip`)
- `process_prompt_links` (scan completed prompt for links before generation) — TODO
- `GenerationTrace` completion — TODO

---

## Infrastructure / tech debt

### Multi-mode model config — DONE
`TS2TSModel` now exposes four explicit axes: mask_type, link_detector,
training/inference backends, dual layout policies. `_creators` dict keyed by
`'{mask_type}_{backend}'`; `forward_inference(mask_type=, backend=)` accepts
per-call overrides. `to_inference_model()` takes the new axes directly; callers
no longer build mask creators. `load_inference_model()` in `generate.py` reads
`data.inference_layout_policy` from hyperparameters.json.

### Eval conditions dispatch — DONE
`eval_checkpoints.py` runs named conditions (experimental / baseline) per benchmark.
`baseline` overrides `mask_type='doc_causal'` + eos layout; skipped automatically
for doc_causal models. Conditions configurable via `eval.conditions` in YAML.

### TS2TSAttention single-class refactor — DONE
`BIMv12Attention` / `VarlenBIMv1Attention` / `FlexSelfAttention` collapsed into
`TS2TSAttention(backend='triton'|'flex')`. Inference switch is `layer.attn.backend='flex'`
not `__class__` reassignment. Flex path calls `flex_attention` directly, not via
tunalab's FlexSelfAttention.forward.

### `smart_train` compiled loop for checkpoint saving — DONE
`tunalab/smart_train.py` now validates `__atomic_features__` on cache load and
recompiles if there's a mismatch. The stale `device-grad_accum-logging-multi_epoch-tqdm`
file has been deleted; the correct loop (with `checkpoint_best_model`) will be
compiled on the next training run.
