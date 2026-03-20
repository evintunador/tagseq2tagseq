# TS2TS Generation — Work Breakdown

Dependency-ordered stages for implementing the generation system. Each stage has a clear deliverable and can be planned in detail independently. See `GENERATION_TECHNICAL_SPEC.md` for implementation specifics.

---

## Stage 0 — Foundations ✅ COMPLETE
*Prerequisite. Blocks all other stages.*

### 0.1 — Model stores its training settings ✅
Add `tokenizer`, `link_detector`, `layout_policy` as first-class attributes of `TS2TSModel` (inference model only — the training module does not need them). Update `TS2TSModel.__init__` and `from_config()` to accept them. Update `to_inference_model()` to take them as explicit arguments (not read from `self` on the training module). Update `main.py` to construct them explicitly: create the `LinkDetector`, pass it to both `CrossDocLinkMaskCreator` and `to_inference_model()` so the mask creator and generation loop share the same instance. Uncomment `TS2TSModel.eval()` / `train()`.

Also in this stage:
- Move `cross_doc_mask.py` (root) → `model/graph_traversal/cross_doc_mask.py`, replacing the old monolithic version. Fix the `seq_len = tokens.shape[-1] - 1` bug in `__call__` as part of the move. ✅
- Move `python_import_detector.py` (root) → `model/graph_traversal/python_import_detector.py`. ✅
- Extract `LinkDetector` protocol and `LinkInfo` into `model/graph_traversal/link_detector.py`; `MarkdownLinkDetector` into `model/graph_traversal/markdown_link_detector.py` (beyond original plan, done for symmetry with `PythonImportDetector`). ✅
- Add `max_recent_link_tokens: int = 200` to `GenerationConfig`. ✅
- Remove hardcoded `MarkdownLinkDetector` from the mask registry; add `make_mask_creator_callable_from(creator)` to `block_mask_creator.py`. ✅
- `main.py` dispatches `cross_doc_link` mask type by reading `model.link_detector` config key (`markdown` or `python`). ✅

**Touches**: `model/model.py`, `main.py`, `model/graph_traversal/cross_doc_mask.py`, `model/graph_traversal/link_detector.py` (new), `model/graph_traversal/markdown_link_detector.py` (new), `model/graph_traversal/python_import_detector.py`, `model/graph_traversal/block_mask_creator.py`, `model/generation_config.py`

**Deliverable**: `TS2TSModel` has explicit `tokenizer`, `link_detector`, and `layout_policy` attributes. `CrossDocLinkMaskCreator` is bug-free, takes a pluggable `LinkDetector`, and is wired into `main.py` via `model.link_detector` config key with no hardcoded detector. `main.py` constructs all non-weight components explicitly from config. The `link_detector` instance shared between mask creator and generation loop is wired at Stage 4.1 (end-of-training demo) when `main.py` calls `to_inference_model()`.

---

## Stage 1 — Single-Document Baseline ✅ COMPLETE
*Depends on Stage 0. No cross-doc features — just autoregressive text generation.*

### 1.1 — `forward_inference()` ✅
### 1.2 — Minimal `DocumentContext` (single doc) ✅
### 1.3 — Basic generation loop (no link handling) ✅
### 1.4 — `generate()` wired end-to-end ✅

**Files created/modified**: `model/model.py`, `model/document_context.py` (new), `model/generation_loop.py` (new), `model/modules/training_module.py`, `model/__init__.py`

**Tests**: `tests/model/test_document_context.py` (22 tests), `tests/model/test_generation.py` (14 CPU + 5 CUDA = 19 tests). 279 total tests pass.

**Smoke test**: `smoke_test_generation.py` — loads trained checkpoint, verifies shape/NaN/entropy.

**Deliverable**: `model.generate("some prompt")` returns coherent text from a trained checkpoint. Single document, no links. Validated against both random weights (terminates) and trained checkpoint (entropy_frac ~0.2–0.4 vs ~1.0 for random).

---

## Stage 2 — Multi-Document MVP ✅ COMPLETE
*Depends on Stage 1. The core cross-doc generation feature.*

### 2.1 — Full `DocumentContext` ✅
`add_corpus_doc`, `add_generated_doc`, `can_add_document`, `make_room`, `evict_oldest_aux`,
`has_identifier`. `_DocEntry` now splits `prefix_tokens` / `tokens` (body) / `suffix_tokens`
so layout policies are applied correctly at construction and at `mark_done`. `add_corpus_doc`
applies both prefix and suffix (matching the training distribution). `get_all_documents()`
returns root first, then active aux in topological order, then evicted.

**Touches**: `model/document_context.py`

### 2.2 — Link detection in generation loop ✅
Per-token link detection in `_generate_doc()`. Scans last `max_recent_link_tokens` of the
active doc each step; calls `_handle_link()` when a complete link closes on the just-appended
token. `link_detector=None` is handled gracefully (single-doc baseline still works).

**Touches**: `model/generation_loop.py`

### 2.3 — `_handle_link()` — corpus and generation branches ✅
Full decision tree: skip empty / duplicate, re-eviction stub (returns None), corpus fetch with
layout-aware token-count estimation for `make_room`, recursive generation fallback. Includes
`_process_existing_doc_links()` for scanning corpus docs at depth+1. Bug fixed in
`cross_doc_mask.py`: `link_end_pos` is exclusive, containment check corrected to
`span.start < link_pos <= span.end`.

**Touches**: `model/generation_loop.py`, `model/graph_traversal/cross_doc_mask.py`

### 2.4 — `GenerationResult` fully populated ✅
All aux documents appear with correct `source`, `parent_raw_identifier`, `depth`, `truncated`,
decoded `text`. `GenerationConfig` gains `repetition_penalty` (default 1.0 in config; 1.3
recommended in `generate.py`) applied per-document before sampling to prevent mode collapse.

**Touches**: `model/generation_loop.py`, `model/generation_config.py`

**Deliverable**: `model.generate("See [Python](Python) for details.")` with simplewiki corpus
produces a `GenerationResult` containing the root doc plus a fetched/generated `Python` aux doc.
Recursive links at depth 2 work. Eviction handles context overflow without crashing. Validated
end-to-end with `generate.py` against trained checkpoints.

---

## Stage 3 — Full Feature Set
*Depends on Stage 2.*

### 3.1 — Re-eviction (restore previously evicted docs) ✅ COMPLETE
`find_evicted` scans `_evicted` list by `raw_identifier`; `restore_evicted` pops from `_evicted` and re-inserts before `before_entry` in `_docs`. `_handle_link` already had the call sites — they were stubs that now work. Also fixed a token-count bug in the evicted-doc room estimate (was omitting `prefix_tokens`). Topological order relative to root is preserved; aux-aux ordering is not tracked (out of scope).

**Touches**: `model/document_context.py`, `model/generation_loop.py`

### 3.2 — Prompt link pre-processing ✅ COMPLETE
After `add_root` and before `_generate_doc`, `run_generation` calls `_process_existing_doc_links` on the root entry at depth=0 when `config.process_prompt_links=True` and a link detector is present. Reuses the exact same machinery as recursive corpus-doc link scanning.

**Note**: Python import detector emits relative paths (e.g. `Phaedra/Notebook.py`) but multi-repo corpus identifiers are repo-qualified (`000alen/Phaedra:Phaedra/Notebook.py`). Corpus hits will never fire for Python imports against a full multi-repo dataset. Fix: either build a single-repo corpus, or make the import detector emit repo-qualified identifiers when context is available.

**Touches**: `model/generation_loop.py`

### 3.3 — Metrics / generation trace
Add a `GenerationTrace` dataclass (global counters: total_forward_passes, total_tokens_generated, links_detected, corpus_fetches, docs_generated, docs_evicted, max_depth_reached) stored as `GenerationResult.trace`. Also emit `logging.DEBUG` per-event and `logging.INFO` run summary. Controlled by `GenerationConfig.record_trace: bool = True`. See technical spec for full field list.

**Touches**: `model/generation_result.py`, `model/generation_loop.py`, `model/generation_config.py`

### 3.4 — Edge case hardening & config validation
- `max_auxiliary_documents` enforcement throughout
- Depth-0 root doc still generates correctly with no links
- `max_link_depth=0` disables all aux doc insertion (corpus and generation)
- Very large corpus docs that alone exceed `max_context_length` — handled gracefully (skip, not crash)
- Validation in `GenerationConfig.__post_init__` for nonsensical combos

**Touches**: `model/generation_config.py`, `model/generation_loop.py`, `model/document_context.py`

**Deliverable**: Full feature set working reliably. All verification cases from the technical spec pass.

---

## Stage 4 — Infrastructure & Tooling
*Depends on Stage 2+ for the core loop, Stage 3 metrics for richer output.*

### 4.1 — End-of-training generation demo in `main.py`
After the training loop, under `if is_main_process():`: load best checkpoint, call `model.generate()` with 2–3 hardcoded dataset-appropriate prompts, print root + aux docs with clear labeling. Gives a qualitative sanity check at the end of every training run.

**Touches**: `main.py`

### 4.2 — Standalone `generate.py` ✅ COMPLETE
CLI script at project root. Accepts `--checkpoint`, `--dataset`, `--prompt`, `--max-link-depth`,
`--max-new-tokens`, `--temperature`, `--top-k`, `--repetition-penalty`, `--max-display-tokens`,
`--allow-generation-fallback`, `--no-color`. Outputs formatted result with ANSI link highlighting,
per-doc truncation with full link list, and per-generated-doc quality metrics (entropy_frac,
bits/tok). `--allow-generation-fallback` defaults to off when `--dataset` is provided.

FlexAttention is compiled by patching `flex_attention` in the tunalab module namespace
(`dynamic=True` for variable-length inference contexts). Compiled kernels cached at
`.torch_compile_cache/` (stable across runs; overridable via `TORCHINDUCTOR_CACHE_DIR`).

Shared `load_inference_model(checkpoint_path)` helper reconstructs architecture, weights,
tokenizer, link detector, and layout policy from the run's `hyperparameters.json`.
`PretokCorpus` wraps `GraphIndex` + `PretokShardedBackend` for corpus lookup.

**Touches**: `generate.py` (new), `configs/large_32k.yaml` (new training config)

**Deliverable**: `python generate.py --checkpoint runs/.../best_model.pt --prompt "Python is"` produces readable output with root + aux docs, link highlights, and quality metrics.

---

## Stage Dependencies at a Glance

```
Stage 0 (foundations)
    └─ Stage 1 (single-doc baseline)
            └─ Stage 2 (multi-doc MVP)
                    ├─ Stage 3 (full feature set)
                    └─ Stage 4 (tooling — can start after Stage 2)
```

Stage 3 and Stage 4 can proceed in parallel once Stage 2 is complete.
