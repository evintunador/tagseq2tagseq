# TS2TS Generation — Technical Spec

Implementation guide for the generation system. See `GENERATION_FEATURE_SPEC.md` for the behavioral spec and design rationale.

---

## Training Context (Key Facts)

- No BOS/EOS during current training (`use_bos_eos: false`, `NullLayoutPolicy` active) — configurable, will change
- No per-document token budget (`doc_budget: null`)
- Block mask: `cross_doc_link` for linked generation, `doc_causal` for no-link baseline
- `DocSpan.start` inclusive, `DocSpan.end` exclusive; `DocSpan.clean_title` holds the raw identifier matched by `LinkDetector.index_doc_span()`
- Checkpoints: `runs/20260224_*/checkpoints/best_model.pt` (4 × 1.1 GB)
- Training datasets: `data/pretokenized_datasets/simplewiki/` and `stack_10m/`

**Terminology note**: The dataset code uses `raw_identifier` (human-readable) and `normed_identifier` (normalized). `DocSpan` uses `clean_title` / `title` for the same concepts — this is a pre-existing naming inconsistency that doesn't need to be fixed as part of this work, but new generation code uses `raw_identifier` / `normed_identifier` throughout.

---

## Stage 0: Prerequisites (Blocks Everything)

### 0.1 — Training Settings Stored in Model Hierarchy

Settings that are fixed at training time (tokenizer, link detector, layout policy) should live in the model so inference cannot silently use wrong settings.

**New attributes on `TS2TSTrainingModule`:**
- `tokenizer` — object with `.encode(str) -> List[int]` and `.decode(List[int]) -> str`
- `link_detector` — `LinkDetector` instance (e.g. `MarkdownLinkDetector` for simplewiki, `PythonImportDetector` for stack)
- `layout_policy` — `DocLayoutPolicy` instance

**`TS2TSTrainingModule.from_config()` updated** to accept these as required parameters. `main.py` constructs them explicitly and passes them in. Concretely, `main.py` does:

```python
link_detector = MarkdownLinkDetector(decode_fn=tokenizer.decode)   # or PythonImportDetector
mask_creator  = CrossDocLinkMaskCreator(link_detector=link_detector)
block_mask_creator = make_mask_creator_callable_from(mask_creator)
model = TS2TSTrainingModule.from_config(...,
    block_mask_creator=block_mask_creator,
    link_detector=link_detector,   # stored on model; same object as inside mask_creator
    ...)
```

The `link_detector` stored on the model is the **same instance** held by `CrossDocLinkMaskCreator`. There is no second copy. The generation loop accesses it via `model.link_detector`; the mask creator accesses it via its own reference — both point to the same object.

**`cross_doc_mask.py` and `python_import_detector.py` are moved from the project root into `model/graph_traversal/`** as part of this stage. The old `model/graph_traversal/cross_doc_mask.py` (monolithic version) is deleted and replaced by the moved file. Final layout:

```
model/graph_traversal/
    block_mask_creator.py        # unchanged; already imports from .cross_doc_mask
    cross_doc_mask.py            # moved from root; canonical LinkDetector protocol +
                                 #   MarkdownLinkDetector + CrossDocLinkMaskCreator
    python_import_detector.py    # moved from root; PythonImportDetector
```

The `seq_len = tokens.shape[-1] - 1` bug in `cross_doc_mask.py.__call__` must be fixed to `seq_len = tokens.shape[-1]` as part of the move.

**`to_inference_model()` updated** to pass `tokenizer`, `link_detector`, `layout_policy` through to `TS2TSModel`.

**New attributes on `TS2TSModel`:** same three, plus anything else derivable at training time that inference needs (e.g. `eos_token_id` from the tokenizer or config).

**Impact on `generate()`:** signature simplifies to:
```python
def generate(self, prompt: str, corpus=None, config=GenerationConfig()) -> GenerationResult:
```
The model tokenizes the prompt with `self.tokenizer.encode`, uses `self.link_detector` for link detection, and applies `self.layout_policy` for document prefix/suffix tokens. Nothing training-specific is re-specified by the caller.

---

## Document Ordering

Packed sequences are topologically ordered: dependencies before dependents. Root is last:

```
[aux2_tokens][aux1_tokens][root_tokens]
```

When a link is detected in the active document, the aux doc is inserted **before the active document as a whole**. All `DocSpan` offsets for subsequent documents shift. The full packed sequence is rebuilt before every forward pass.

---

## Tokenizer Flow

With training settings stored in the model, the tokenizer touches generation at:

1. **Prompt encoding**: `self.tokenizer.encode(prompt)` → `List[int]`, once at the start of `generate()`.
2. **Document prefix encoding**: If `self.layout_policy` adds identifier text to prefixes, it calls `self.tokenizer.encode` internally when constructing prefix tokens. This is the layout policy's responsibility.
3. **Output text decoding** (optional): `self.tokenizer.decode(tokens)` to populate `GeneratedDocument.text`.

`LinkDetector` handles its own decoding internally — it is constructed with a `decode_fn` that matches the training tokenizer, and stored in the model. The generation loop never decodes tokens directly.

---

## Layout Policy in Generation

`DocLayoutPolicy` (from `data/layout.py`) controls prefix/suffix tokens per document. Currently `NullLayoutPolicy` (adds nothing). Stored in the model; applied consistently at generation time.

When a document is initialized (root or aux), its token list is seeded with `self.layout_policy.prefix_tokens(doc_id)`. When marked done, `self.layout_policy.suffix_tokens(doc_id)` is appended. When training later adds identifier-in-prefix support, the stored policy will handle it automatically.

---

## Files to Create / Modify

### 1. `model/model.py` — implement stubs

**`forward_inference(tokens: Tensor, doc_spans: Optional[List[Any]] = None, **kwargs) -> Tensor`**

```
block_mask = self.block_mask_creator(tokens=tokens, doc_spans=doc_spans or [], **kwargs)
x = F.embedding(tokens, self.embedding_weight)    # [1, T, D]
x = self.backbone(x, block_mask=block_mask)       # [1, T, D]
x = self.norm(x)
logits = F.linear(x, self.lm_head_weight)         # [1, T, V]
return logits
```

Wrap in `torch.no_grad()`.

**`generate(prompt: str, corpus=None, config=GenerationConfig()) -> GenerationResult`**

```python
prompt_tokens = self.tokenizer.encode(prompt)
return run_generation(
    model=self,
    prompt_tokens=prompt_tokens,
    corpus=corpus,
    config=config,
    link_detector=self.link_detector,
    tokenizer_decode=self.tokenizer.decode,
    layout_policy=self.layout_policy,
)
```

---

### 2. `model/document_context.py` — NEW

Manages the growing packed sequence. Uses `raw_identifier` / `normed_identifier` terminology.

**Internal dataclass `_DocEntry`** (not exported):
```
normed_identifier: str        # normalized form (for DocSpan.title)
raw_identifier: str           # human-readable form as decoded from link (DocSpan.clean_title)
tokens: list[int]             # accumulated token IDs (seeded with layout prefix)
done: bool
truncated: bool
doc_id: int                   # sequential counter, not tied to corpus
source: Literal["generated", "corpus"]
outgoing_identifiers: list[str]
parent_raw_identifier: Optional[str]
depth: int                    # recursion depth at which this doc was created
```

**`DocumentContext(max_context_length, max_auxiliary_documents, eviction_policy, device)`**

Internal state:
- `_docs: list[_DocEntry]` — topological order (deps first, root last)
- `_root: _DocEntry`
- `_evicted: list[_DocEntry]` — removed from window but kept for `GenerationResult` and re-eviction
- `_next_doc_id: int`

Methods:

**`add_root(raw_identifier, prompt_tokens, layout_policy) -> _DocEntry`**
- Assigns `doc_id = _next_doc_id` (always 0), increments `_next_doc_id`
- `tokens = list(layout_policy.prefix_tokens(doc_id)) + list(prompt_tokens)`
- Appends to `_docs`; sets `_root`

**`add_corpus_doc(raw_identifier, corpus_tokens, layout_policy, parent_raw_identifier, depth, before_entry) -> _DocEntry`**
- Assigns `doc_id = _next_doc_id`, increments `_next_doc_id`
- `tokens = list(layout_policy.prefix_tokens(doc_id)) + list(corpus_tokens)`
- Constructs `_DocEntry(done=True, source="corpus", ...)`; inserts before `before_entry` in `_docs`
- Returns the new entry (caller may pass it to `_process_existing_doc_links`)

**`add_generated_doc(raw_identifier, layout_policy, parent_raw_identifier, depth, before_entry) -> _DocEntry`**
- Assigns `doc_id = _next_doc_id`, increments `_next_doc_id`
- `tokens = list(layout_policy.prefix_tokens(doc_id))`  — body tokens accumulated later via `append_token`
- Constructs `_DocEntry(done=False, source="generated", ...)`; inserts before `before_entry` in `_docs`
- Returns the new entry (caller drives generation via `_generate_doc`)

**`append_token(entry, token_id) -> None`**

**`mark_done(entry, layout_policy) -> None`**
- Appends `layout_policy.suffix_tokens(entry.doc_id)` to `entry.tokens`
- Sets `entry.done = True`

**`build_sequence() -> (Tensor[1,T], List[DocSpan])`**
- Recomputes all DocSpan offsets from scratch (O(total_tokens))
- `DocSpan.title = entry.normed_identifier`, `DocSpan.clean_title = entry.raw_identifier`
- Returns `(LongTensor[1, T], doc_spans)` on `self.device`

**`total_tokens` property** → `sum(len(e.tokens) for e in _docs)`

**`num_aux_docs` property** → `len(_docs) - 1`

**`make_room(num_tokens_needed) -> bool`**
- Evict oldest aux docs in a loop until `can_add_document(num_tokens_needed)` is True or no more aux docs remain
- Returns `True` if room was made, `False` if impossible (e.g. only root remains)
- Only call if `eviction_policy == "drop_oldest"`; caller checks policy

**`can_add_document(num_new_tokens) -> bool`**
- `total_tokens + num_new_tokens <= max_context_length AND num_aux_docs < max_auxiliary_documents`

**`evict_oldest_aux() -> _DocEntry`**
- Remove and return leftmost non-root entry; append to `_evicted`

**`find_evicted(raw_identifier) -> Optional[_DocEntry]`**

**`restore_evicted(entry, before_entry) -> None`**
- Remove from `_evicted`; insert before `before_entry` in `_docs`
- Caller is responsible for ensuring space exists before calling (via `make_room`)

**`has_identifier(raw_identifier) -> bool`**
- True if found in `_docs` (active window only; evicted docs do not count)

**`get_all_documents() -> List[GeneratedDocument]`**
- Returns `_docs + _evicted`; root doc first regardless of eviction order

---

### 3. `model/generation_loop.py` — NEW

**`run_generation(model, prompt_tokens, corpus, config, link_detector, tokenizer_decode, layout_policy) -> GenerationResult`**

1. Create `DocumentContext(config.max_context_length, config.max_auxiliary_documents, config.eviction_policy, config.device)`.
2. `root_entry = context.add_root(raw_identifier="root", prompt_tokens=prompt_tokens, layout_policy=layout_policy)`.
3. If `config.process_prompt_links`: scan `root_entry.tokens` with `link_detector.detect_links(tensor(root_entry.tokens))`; call `_handle_link` for each (no generation at depth -1 — corpus fetch or insert only, same depth rules apply as depth=0).
4. Call `_generate_doc(root_entry, context, model, link_detector, corpus, config, layout_policy, depth=0)`.
5. Populate `GeneratedDocument.text = tokenizer_decode(entry.tokens)` for each doc if `tokenizer_decode` is provided.
6. Return `GenerationResult` from `context.get_all_documents()`.

**`_generate_doc(entry, context, model, link_detector, corpus, config, layout_policy, depth) -> None`**

```
tokens_generated = 0

while not entry.done:
    tokens_tensor, doc_spans = context.build_sequence()
    logits = model.forward_inference(tokens_tensor, doc_spans)   # [1, T, V]
    next_token = sample_token(logits[0, -1, :], config.temperature, config.top_k, config.top_p)
    context.append_token(entry, next_token)
    tokens_generated += 1

    # Link detection — run every token, scan last max_recent_link_tokens of active doc only
    recent = torch.tensor(entry.tokens[-config.max_recent_link_tokens:], dtype=torch.long)
    links = link_detector.detect_links(recent)
    for link in links:
        if link.link_end_pos == len(recent):   # link_end_pos is exclusive; link closed at last token
            _handle_link(link, entry, context, model, link_detector,
                         corpus, config, layout_policy, depth)
            break  # at most one new doc triggered per token step

    # Stopping conditions (checked after link handling)
    if next_token == config.eos_token_id:
        context.mark_done(entry, layout_policy)
    elif tokens_generated >= config.max_new_tokens:
        # max_new_tokens: caps tokens generated in this _generate_doc call.
        # Does NOT set truncated=True — this is a generation budget, not a hard structural limit.
        context.mark_done(entry, layout_policy)
    elif len(entry.tokens) >= config.max_tokens_per_document:
        # max_tokens_per_document: caps total document length (prefix + body).
        # Sets truncated=True — this is a hard structural limit.
        entry.truncated = True
        context.mark_done(entry, layout_policy)
```

**`_handle_link(link, active_entry, context, model, link_detector, corpus, config, layout_policy, depth) -> None`**

```
target = link.target_str
if not target:
    return  # skip [text]()

if context.has_identifier(target):
    return  # already in active window; cross-doc mask handles repeated links

# Restore a previously evicted doc if possible
evicted = context.find_evicted(target)
if evicted is not None:
    if config.eviction_policy == "drop_oldest":
        if not context.make_room(len(evicted.tokens)):
            return  # can't fit even after evicting everything except root
    elif not context.can_add_document(len(evicted.tokens)):
        return
    context.restore_evicted(evicted, before_entry=active_entry)
    # A restored corpus doc may still have unprocessed links at depth+1 — fall through
    if evicted.source == "corpus" and depth + 1 <= config.max_link_depth:
        _process_existing_doc_links(evicted, context, model, link_detector,
                                    corpus, config, layout_policy, depth + 1)
    return

# Corpus fetch
if corpus is not None and corpus.has_document(target):
    corpus_tokens = list(corpus.get_document(target))
    if config.eviction_policy == "drop_oldest":
        if not context.make_room(len(corpus_tokens)):
            return
    elif not context.can_add_document(len(corpus_tokens)):
        return
    new_entry = context.add_corpus_doc(
        raw_identifier=target,
        corpus_tokens=corpus_tokens,
        layout_policy=layout_policy,
        parent_raw_identifier=active_entry.raw_identifier,
        depth=depth + 1,
        before_entry=active_entry,
    )
    # Corpus doc's own links are game at depth+1
    if depth + 1 <= config.max_link_depth:
        _process_existing_doc_links(new_entry, context, model, link_detector,
                                    corpus, config, layout_policy, depth + 1)
    return

# Generate the aux doc
if not config.allow_corpus_fallback:
    return
if depth >= config.max_link_depth:
    return

if config.eviction_policy == "drop_oldest":
    if not context.make_room(0):  # just need a slot
        return
elif not context.can_add_document(0):
    return

new_entry = context.add_generated_doc(
    raw_identifier=target,
    layout_policy=layout_policy,
    parent_raw_identifier=active_entry.raw_identifier,
    depth=depth + 1,
    before_entry=active_entry,
)

if config.allow_recursive_links or depth == 0:
    _generate_doc(new_entry, context, model, link_detector,
                  corpus, config, layout_policy, depth + 1)
else:
    context.mark_done(new_entry, layout_policy)
```

**`_process_existing_doc_links(entry, context, model, link_detector, corpus, config, layout_policy, depth) -> None`**

Scans an already-complete document (corpus or restored-evicted) for links and handles each one, at the given depth. This is how corpus docs participate in the full recursive link graph:

```
all_links = link_detector.detect_links(torch.tensor(entry.tokens, dtype=torch.long))
for link in all_links:
    _handle_link(link, entry, context, model, link_detector,
                 corpus, config, layout_policy, depth)
```

Note: `normalize_identifier` from `model/title_utils.py`.

---

### 4. `model/generation_config.py` — minor additions

Add `max_recent_link_tokens: int = 200`. This is used in `_generate_doc` to limit how many tokens of the active document are scanned for links on each step (efficiency — avoids re-scanning the full document every token).

`eos_token_id` already exists (default 50256). Generation always stops on it.

---

## Metrics / Generation Trace

The generation system should record what happened during a run. Exact scope TBD (see open question below), but at minimum useful to track:

- Per-document: token count, source (generated/corpus/restored-evicted), recursion depth, parent identifier, whether truncated
- Global: total tokens generated, total forward passes, number of links detected, number of corpus fetches, number of docs generated, number of evictions, max depth reached

**Open question — format**: Should this be (a) live-logged via Python `logging` during generation, (b) a structured `GenerationTrace` object stored in `GenerationResult`, or (c) both? Option (b) enables programmatic analysis post-run. Option (a) is useful for long-running interactive use. Recommendation: both, with (b) as the primary record and (a) as optional debug output.

---

## Dependency Map

```
Stage 0:
  cross_doc_mask.py → model/graph_traversal/cross_doc_mask.py (move + fix seq_len bug)
  python_import_detector.py → model/graph_traversal/python_import_detector.py (move)
  model/graph_traversal/cross_doc_mask.py (old monolithic version — deleted)
  model/modules/training_module.py (stores tokenizer + link_detector + layout_policy)
  model/model.py (receives them via to_inference_model())

Stage 1 (depends on Stage 0):
  model/model.py → forward_inference()
  model/document_context.py (single-doc subset)
  model/generation_loop.py → run_generation(), _generate_doc() (no link handling)
  model/model.py → generate()

Stage 2 (depends on Stage 1):
  model/document_context.py (full: add_corpus_doc, add_generated_doc, make_room, eviction)
  model/generation_loop.py → _handle_link(), _process_existing_doc_links()

Stage 3 (depends on Stage 2):
  model/document_context.py (find_evicted, restore_evicted)
  model/generation_loop.py (process_prompt_links, metrics)
  GenerationTrace in model/generation_result.py

Stage 4 (depends on Stage 2+):
  main.py (end-of-training demo, under is_main_process())
  generate.py (standalone CLI)
```

---

## Inference at End of Training & Standalone Script

**`main.py`**: After the training loop, under `if is_main_process():`, load the best checkpoint, run `model.generate()` with 2–3 hardcoded seed prompts appropriate to the dataset. Print root doc and all aux docs with clear labeling (identifier, source, parent). This is a qualitative sanity check, not a benchmark.

**`generate.py`** (new standalone script at project root): CLI accepting `--checkpoint`, `--dataset` (corpus path), `--prompt`, and key `GenerationConfig` overrides. Outputs root + aux docs with metadata. Intended for post-training exploration.

---

## Verification

**1. Checkpoint loading**: `torch.load("runs/20260224_212158/checkpoints/best_model.pt", map_location="cpu")`. Identify format, reconstruct `TS2TSTrainingModule`, load weights, call `.to_inference_model()`.

**2. `forward_inference` shape**: Two-doc packed sequence (~50 tokens). Assert output `[1, T, 50257]`, no NaN/Inf.

**3. End-to-end with random weights**: `model.generate("Python is a")`. Assert returns `GenerationResult`, terminates within token limit.

**4. Trained checkpoint sanity**: `max_new_tokens=100`. Output looks like early-LM text, not noise. Token entropy higher than random-weight output.

**5. Link + corpus round-trip**: Prompt with a link to an identifier in simplewiki corpus. Assert aux doc `source="corpus"`, tokens match `corpus.get_document(target)`. Assert aux doc's own links are also processed.

**6. Recursive generation**: Link not in corpus, `max_link_depth=2`. Assert aux doc `source="generated"`, depth ≤ 2 enforced.

**7. Eviction safety**: Fill context to capacity, trigger more links. Assert `make_room` loops correctly; no crash when only root can't be evicted.
