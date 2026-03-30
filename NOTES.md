# Roadmap notes — incomplete work

Loose PM-level notes on what still needs planning + implementation.
Not a spec; just enough to orient a planning session for each item.

---

## Eval pipeline — extensions

### Split annotations
The existing datasets have no `split` field in `tokenized_graph.jsonl`, so
`split="all"` just random-samples from the whole corpus. For real held-out
eval, the pretokenization scripts need to assign `"train"` / `"val_community"` /
`"val_random"` labels at graph-construction time. Needs design: what fraction
held out, community split vs random split semantics, whether existing checkpoints
can be retroactively evaluated against a newly annotated graph.

### NL benchmarks (HellaSwag, WikiQA, LAMBADA, WinoGrande, ARC)
`eval/hellaswag.py` has the full commented implementation ready to activate.
Blocked on Wikipedia/fineweb data being online. Once data exists:
- Activate HellaSwag (uncomment implementation)
- Add WikiQA adapter (already in tunalab)
- Add LAMBADA adapter (fill-in-the-blank last-word prediction)
- Consider ARC-Easy/Challenge and WinoGrande

### Code benchmarks (no execution infra)
For stack-trained models, are there code-specific MC or fill-in-the-blank
benchmarks that don't require running code? Worth a literature search.
HumanEval/MBPP are out until execution infra exists.

### Link injection eval
The `prompt_preprocessor` hook in `score_completion` is the slot for this.
Needs `eval/link_annotator.py` implementing `annotate_prompt_with_links(model,
prompt_tokens, threshold)`:
- Single forward pass over prompt
- Find positions where the link-opener token (e.g. ` [` for markdown) has
  logit probability above threshold
- Insert link + generate target + fetch/generate aux doc
- Return augmented prompt tokens
Then the eval comparison: score benchmark items with bare prompt vs
link-annotated prompt, report delta. Interesting science but needs a model
smart enough for meaningful link placement — probably defer until larger scale.

### Parallelized eval in main.py
Currently the post-training eval runs serially. If multiple benchmarks are
configured, they could run in parallel threads/processes after training.
Low priority but worth doing before the pipeline gets heavy.

---

## Data & datasets

### Wikipedia / ArXiv / FineWeb pipeline
For NL benchmarks to matter, we need NL training data. Open questions:
- Bring back simplewiki/enwikisource? Or go straight to full English Wikipedia?
- ArXiv: good graph structure (citations), high-quality text, relevant to paper
- FineWeb: flat (no graph), but volume; how does it mix with TAG data?
- Mixing strategy: what ratio TAG-structured vs flat data? Separate streams or
  interleaved packs?
Needs a proper data planning session before any implementation.

### Stack 100M training
The `stack_100m_32k.yaml` config exists. A full 100M node run is the natural
next training milestone after the current 10M checkpoint experiments.

---

## Model

### Scale-up
36L/1280D is placeholder. For the paper, need at least one run at a meaningfully
larger scale. Needs decisions on: target param count, num_layers/model_dim
tradeoff, whether to use the same BFS+cross_doc_link config as current best runs.

### Stochastic layout policy — DONE
`StochasticIdentifierPrefixLayoutPolicy` is implemented in `data/layout.py`.
Use `layout_policy: stochastic_identifier_prefix` in config for training.
Set `inference_layout_policy: identifier_prefix` (or `identifier_prefix_bos_eos`)
so inference always uses a deterministic prefix. `BucketedPackDataset` calls
`set_epoch()` on epoch advance; `PackedSequenceDataset` + `multi_epoch` path
has no epoch hook (deferred until BucketedPackDataset is sole production path).
Science question remains open: does stochastic training meaningfully help on
external benchmarks vs. fixed null layout?

---

## Generation feature — Stage 3

`GENERATION_WORK_BREAKDOWN.md` defines Stage 3 as:
- `find_evicted` / `restore_evicted` re-eviction logic in `DocumentContext` — **DONE**
  (both fully implemented; no stubs remain)
- `link_retrieval_mode` in `GenerationConfig` — **DONE**
  (`corpus_only`, `generate_only`, `corpus_then_generate`, `link_but_skip`, `full_skip`)
- `process_prompt_links` (scan completed prompt for links before generation) — still deferred
- `GenerationTrace` completion — still deferred

---

## Infrastructure / tech debt

### `smart_train` compiled loop for checkpoint saving
`tunalab/smart_train.py` now validates `__atomic_features__` on cache load and
recompiles if there's a mismatch. The stale `device-grad_accum-logging-multi_epoch-tqdm`
file has been deleted; the correct loop (with `checkpoint_best_model`) will be
compiled on the next training run.
