# TS2TS Generation — Feature Spec

## Overview

This document describes the intended behavior of TS2TS generation: how the model should produce text while respecting document structure, graph links, and the multi-document context window used during training.

This is a **behavioral spec**, not an implementation guide. It describes *what* the generation system should do, not how to decompose it into files or classes. See `GENERATION_TECHNICAL_SPEC.md` for implementation details.

---

## What Already Exists

The following foundation pieces are implemented and tested:

- **DocumentCorpus** (`model/document_corpus.py`): Wraps `GraphIndex` + `PretokShardedBackend` for retrieving pretokenized documents by identifier (handles normalized+hashed lookup).
- **Identifier utilities** (`model/identifier_utils.py`): `normalize_identifier`, `generate_identifier_hash`, `create_normed_identifier`, `strip_hash`, `verify_identifier_hash`. (`model/title_utils.py` is a backward-compat shim re-exporting these under the old names.)
- **GenerationResult / GeneratedDocument** (`model/generation_result.py`): Result data structures for generation output. Tracks root + auxiliary documents with fields `raw_identifier`, `normed_identifier`, `source`, `parent_raw_identifier`, `depth`, `truncated`, `tokens`, `text`, `is_root`.
- **GenerationConfig** (`model/generation_config.py`): Configuration dataclass covering sampling params, document structure limits, eviction policy, link handling, and stopping conditions.
- **Token sampling** (`model/sampling.py`): `greedy_sample` and `sample_token` (temperature, top-k, top-p).
- **Link detection** (`cross_doc_mask.py`, `python_import_detector.py`): `LinkDetector` protocol with `MarkdownLinkDetector` (for Wikipedia) and `PythonImportDetector` (for TheStack) implementations. Produces `LinkInfo(link_end_pos, target_str)`. More detectors (e.g., LaTeX citations) to be added later. Both files will be moved to `model/graph_traversal/` in Stage 0 (replacing the older monolithic `model/graph_traversal/cross_doc_mask.py`).
- **TS2TSModel** (`model/model.py`): Inference wrapper with stubbed `forward_inference` and `generate` methods (raise `NotImplementedError`).
- **DocSpan** (`data/collate.py`): Dataclass tracking document boundaries in a packed sequence (`doc_id`, `title`, `start`, `end`, `truncated`, `outgoing_titles`, `clean_title`).
- **Block mask creators** (`model/graph_traversal/block_mask_creator.py`): Callables that produce FlexAttention `BlockMask` from tokens + doc_spans. Supports `doc_causal`, `causal`, `full`, `doc_bidirectional`, and `cross_doc_link` strategies.

---

## What Needs to Be Built

Three capabilities remain unimplemented:

1. **Context management** — maintaining a growing multi-document packed sequence during generation
2. **The generation loop** — autoregressive token prediction with link detection and recursive auxiliary document handling
3. **Model integration** — wiring `TS2TSModel.forward_inference()` and `TS2TSModel.generate()`

---

## 1. Context Management

During training, the model sees packed sequences of multiple documents with `DocSpan` metadata and FlexAttention block masks. Generation must reproduce this structure dynamically as new tokens are generated and new documents are added to the context.

### What context management must do:

- **Track a packed token sequence** that grows as tokens are generated, and can have whole documents inserted (from corpus or as new generation targets).
- **Maintain `DocSpan` metadata** for every document in the context, with correct `start`/`end` indices that update as the sequence grows.
- **Support multiple documents simultaneously**: a root document being generated, plus auxiliary documents (from corpus or also being generated) that the model can attend to.
- **Enforce limits** from `GenerationConfig`: `max_context_length`, `max_tokens_per_document`, `max_auxiliary_documents`.
- **Handle eviction** when the context is full:
  - `drop_oldest`: remove the oldest auxiliary document to make room, adjusting all `DocSpan` indices.
  - `stop_new`: refuse to add new documents once full.
- **Support re-eviction**: if a previously evicted document is linked to again, it can be re-inserted into the context window (potentially evicting another document to make room under `drop_oldest`).
- **Track which document is currently being generated** (the "active" generation target), and support pausing the current doc to recursively generate a linked doc, then resuming.
- **Provide the model's expected inputs**: a `[1, T]` token tensor and a `List[DocSpan]` suitable for passing to the block mask creator.
- **Apply layout policy**: each document's token list is seeded with layout prefix tokens and closed with suffix tokens on completion, mirroring the training layout policy (currently `NullLayoutPolicy` — adds nothing).

### Key invariants:

- `DocSpan` indices must always be consistent with the actual token positions in the packed sequence.
- The root document is never evicted.
- Evicted documents are retained in a side list for the final `GenerationResult` and for re-eviction lookups.

---

## 2. The Generation Loop

The generation loop produces tokens autoregressively. When the model generates a link to another document, that document is either retrieved from a corpus or recursively generated, then inserted into the context so the model can attend to it.

The loop is dataset-agnostic: it uses a pluggable `LinkDetector` (e.g., `MarkdownLinkDetector` for Wikipedia, `PythonImportDetector` for TheStack). Link detection runs after every generated token.

### High-level algorithm:

```
generate(prompt, corpus, config, link_detector, layout_policy):
    1. Initialize context with the prompt as the root document
       (prefixed with layout_policy.prefix_tokens if any).
    2. If config.process_prompt_links: scan prompt for existing links,
       resolve them (corpus lookup or queue for generation).
    3. Begin autoregressive generation of the root document:
        a. Run forward_inference on the current packed sequence + doc_spans.
        b. Sample the next token from the last position's logits.
        c. Append token to the root document in context.
        d. Run link_detector on the last max_recent_link_tokens of the active doc.
           If a complete link ends at the current token:
                i.   Extract the target document identifier.
                ii.  If already in the active window, skip (cross-doc mask handles it).
                iii. If previously evicted, re-insert it (evicting another if needed).
                iv.  If in the corpus, insert corpus doc into context before the active doc.
                v.   Else if depth < max_link_depth and allow_generation_fallback:
                     pause generation, recursively generate the aux doc
                     (inserted before the active doc), then resume.
        e. Stop if: EOS token generated (if layout policy uses EOS), max_new_tokens
           reached, or max_tokens_per_document reached.
    4. Return a GenerationResult with root + all auxiliary documents.
```

### Recursive auxiliary generation:

- The auxiliary document is inserted **before the active document as a whole** in the packed sequence (topological order: deps first). Never in the middle of its tokens.
- The aux doc is seeded with layout prefix tokens (e.g., its identifier, if identifier-in-prefix is enabled in training).
- The model generates aux doc tokens autoregressively; it can see prior aux docs but not the root's tokens (causal attention).
- When complete, generation returns to the parent document.

### Document ordering in the packed sequence:

Training uses `prefer_targets_first` ordering: dependency documents appear before the documents that link to them. Generation replicates this physically by inserting aux docs before the linking document. Concretely:

```
[aux2_tokens][aux1_tokens][root_tokens_so_far]
```

Where root → aux1 → aux2. Every token added to aux2 shifts aux1's and root's `DocSpan` offsets.

### EOS and document boundaries:

Generation should mirror the training layout policy. Currently `NullLayoutPolicy` is active (no BOS/EOS tokens), so stopping on EOS is configurable — it may be appropriate to stop on the EOS token regardless, since the model will likely still emit it as a natural end-of-document signal. This is a configurable parameter.

---

## 3. Model Integration

### forward_inference

`TS2TSModel.forward_inference(tokens, doc_spans)` takes the current packed token sequence and doc span metadata, runs a full forward pass, and returns logits over the vocabulary for every position.

### generate

`TS2TSModel.generate()` is the user-facing entry point. Accepts:
- `prompt: str` — tokenized internally using `self.tokenizer`
- Optional corpus
- `GenerationConfig`

The model supplies its own `LinkDetector` and `DocLayoutPolicy` (stored at training time via Stage 0). The caller provides only what varies per-call: the prompt, an optional corpus, and generation config overrides.

Returns a `GenerationResult`.

---

## Design Decisions (Resolved)

| Question | Decision |
|---|---|
| Mask type for generation | `cross_doc_link`; `doc_causal` only for no-link baseline |
| BOS/EOS handling | Mirrors training layout policy (configurable); EOS token always terminates a document when sampled regardless |
| Identifier collision | Reuse the first occurrence; skip if already in context window |
| Re-eviction | If a previously evicted doc is linked to again, re-insert it (potentially evicting another). Always re-process its links at depth+1, regardless of whether it was originally corpus or generated. |
| Empty link target `[text]()` | Skip |
| Link detection frequency | Every token; scan only last `max_recent_link_tokens` of active doc for efficiency |
| Dataset-specific link syntax | Fully pluggable via `LinkDetector` protocol; no hardcoded assumptions about link delimiters |
| Per-document token budget | No hard limit from training (`doc_budget: null`); controlled by `GenerationConfig.max_tokens_per_document` at inference |
| Multi-GPU | Single device for now; inference runs under `is_main_process()` when called from training loop |
| Test data | Use existing training datasets (`data/pretokenized_datasets/`) |
| Test checkpoints | Available in `runs/`; sanity-check that output looks like early-training LM, not random noise |
| `LinkDetector` ownership | Model stores `self.link_detector`; same instance is passed to `CrossDocLinkMaskCreator` at construction. Two consumers (mask creator + generation loop), one object. |
| `max_new_tokens` vs `max_tokens_per_document` | Both are kept. `max_new_tokens` caps tokens generated per `_generate_doc` call and does **not** set `truncated=True`. `max_tokens_per_document` caps total document length (including prefix/prompt) and **does** set `truncated=True`. |

---

## Risks and Performance Considerations

- **FlexAttention recompilation**: `create_block_mask` is called on every token step. This is correct and expected; FlexAttention compilation is fast enough for MVP.
- **Context management index bookkeeping**: Inserting an aux doc before a later doc shifts all subsequent `DocSpan` offsets. Eviction and re-eviction require the same. Error-prone; needs thorough testing.
- **Memory**: Large contexts with many documents could consume significant GPU memory. `max_context_length` and `max_auxiliary_documents` in `GenerationConfig` are the safety valves.
- **KV caching**: Not implemented — every forward pass recomputes attention over the full sequence. Post-MVP optimization.
