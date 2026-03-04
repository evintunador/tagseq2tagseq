# TS2TS Generation — Work Breakdown

Dependency-ordered stages for implementing the generation system. Each stage has a clear deliverable and can be planned in detail independently. See `GENERATION_TECHNICAL_SPEC.md` for implementation specifics.

---

## Stage 0 — Foundations
*Prerequisite. Blocks all other stages.*

### 0.1 — Model stores its training settings
Add `tokenizer`, `link_detector`, `layout_policy` as first-class attributes of `TS2TSModel` (inference model only — the training module does not need them). Update `TS2TSModel.__init__` and `from_config()` to accept them. Update `to_inference_model()` to take them as explicit arguments (not read from `self` on the training module). Update `main.py` to construct them explicitly: create the `LinkDetector`, pass it to both `CrossDocLinkMaskCreator` and `to_inference_model()` so the mask creator and generation loop share the same instance. Uncomment `TS2TSModel.eval()` / `train()`.

Also in this stage:
- Move `cross_doc_mask.py` (root) → `model/graph_traversal/cross_doc_mask.py`, replacing the old monolithic version. Fix the `seq_len = tokens.shape[-1] - 1` bug in `__call__` as part of the move.
- Move `python_import_detector.py` (root) → `model/graph_traversal/python_import_detector.py`.
- Add `max_recent_link_tokens: int = 200` to `GenerationConfig`.

**Touches**: `model/model.py`, `main.py`, `cross_doc_mask.py` (→ `model/graph_traversal/`), `python_import_detector.py` (→ `model/graph_traversal/`), `model/graph_traversal/block_mask_creator.py`, `model/generation_config.py`

**Deliverable**: `TS2TSModel` is fully self-describing. All inference can be driven purely from the model object with no extra configuration. The single canonical `CrossDocLinkMaskCreator` is bug-free and wired into training.

---

## Stage 1 — Single-Document Baseline
*Depends on Stage 0. No cross-doc features — just autoregressive text generation.*

### 1.1 — `forward_inference()`
Implement the stubbed method: block mask → embed → backbone → norm → project → logits. Straightforward given the Stage 0 refactor.

**Touches**: `model/model.py`

### 1.2 — Minimal `DocumentContext` (single doc)
The context management class, scoped to the single-document case: no `insert_before`, no eviction, no re-eviction. Just: initialize with root, append tokens, build packed sequence + DocSpans for forward pass.

**Touches**: `model/document_context.py` (new)

### 1.3 — Basic generation loop (no link handling)
`run_generation()` and `_generate_doc()` without any link detection or aux doc logic. Token sampling, stopping conditions (EOS / max tokens), `GenerationResult` returned with just the root doc.

**Touches**: `model/generation_loop.py` (new)

### 1.4 — `generate()` wired end-to-end
Implement the `generate(prompt: str, corpus=None, config=GenerationConfig())` stub. Tokenize prompt, delegate to `run_generation()`, return result.

**Touches**: `model/model.py`

**Deliverable**: `model.generate("some prompt")` returns coherent-ish text from a trained checkpoint. Single document, no links. Validate against both random weights (should terminate) and trained checkpoint (should produce non-random output).

---

## Stage 2 — Multi-Document MVP
*Depends on Stage 1. The core cross-doc generation feature.*

### 2.1 — Full `DocumentContext`
Extend with `add_corpus_doc` and `add_generated_doc` factory methods (handle `doc_id` assignment, layout prefix seeding, `_DocEntry` construction, and topological insertion — all in one call), safe eviction loop (`make_room` — loops until enough space freed or gives up), and all DocSpan offset bookkeeping. Re-eviction (`find_evicted`, `restore_evicted`) deferred to Stage 3.

**Touches**: `model/document_context.py`

### 2.2 — Link detection in generation loop
Add per-token link detection to `_generate_doc()`. Scan last `max_recent_link_tokens` of active doc on every step. Call `_handle_link()` when a complete link is detected at the current token.

**Touches**: `model/generation_loop.py`

### 2.3 — `_handle_link()` — corpus and generation branches
Implement the full link-handling decision tree:
- Skip empty targets and duplicates already in context
- Corpus fetch: insert, then recursively process corpus doc's own links at depth+1
- Generate: insert empty entry, recurse into `_generate_doc()` at depth+1
- Respect `max_link_depth`, `allow_generation_fallback`, eviction policy

Includes `_process_existing_doc_links()` for scanning corpus docs and restored-evicted docs.

**Touches**: `model/generation_loop.py`

### 2.4 — `GenerationResult` fully populated
Ensure all auxiliary documents (corpus + generated) appear in the result with correct metadata: `source`, `parent_raw_identifier`, `depth`, `truncated`, decoded `text` if tokenizer available.

**Touches**: `model/generation_loop.py`, `model/generation_result.py` (minor)

**Deliverable**: `model.generate("See [Python](Python) for details.")` with simplewiki corpus produces a `GenerationResult` containing the root doc plus a fetched/generated `Python` aux doc. Recursive links at depth 2 work. Eviction handles context overflow without crashing.

---

## Stage 3 — Full Feature Set
*Depends on Stage 2.*

### 3.1 — Re-eviction (restore previously evicted docs)
Add `find_evicted` / `restore_evicted` to `DocumentContext`. Update `_handle_link()` to check the evicted list before attempting corpus fetch or generation. A restored doc re-enters the attention window; its links are scanned at depth+1 as if freshly inserted.

**Touches**: `model/document_context.py`, `model/generation_loop.py`

### 3.2 — Prompt link pre-processing
Before generation starts, scan the initial prompt tokens for any already-present links (`process_prompt_links` flag in `GenerationConfig`). Handle each the same way as a link detected mid-generation — corpus fetch or queue for generation at depth 0.

**Touches**: `model/generation_loop.py`

### 3.3 — Metrics / generation trace
Record what happened during a run: links detected, corpus fetches, docs generated, docs evicted, max depth reached, total forward passes, per-doc token counts.

*Open question*: Should this be (a) a structured `GenerationTrace` object embedded in `GenerationResult`, (b) live-logged via Python `logging`, or (c) both? Recommendation is both — structured trace for programmatic use, logging for interactive runs — but needs sign-off on exact fields and format before implementation.

**Touches**: `model/generation_result.py`, `model/generation_loop.py`

### 3.4 — Edge case hardening & config validation
- `max_auxiliary_documents` enforcement throughout
- Depth-0 root doc still generates correctly with no links
- `allow_recursive_links=False` still allows depth-1 links from root
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

### 4.2 — Standalone `generate.py`
CLI script at project root. Accepts `--checkpoint`, `--dataset` (corpus path), `--prompt`, and key `GenerationConfig` overrides. Outputs formatted result: root doc + each aux doc with identifier, source, parent, depth. For post-training exploration without modifying training code.

**Touches**: `generate.py` (new)

**Deliverable**: `python generate.py --checkpoint runs/best/checkpoints/best_model.pt --prompt "Python is"` produces readable output.

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
