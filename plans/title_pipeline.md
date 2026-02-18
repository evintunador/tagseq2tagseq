# Title Pipeline: Original Titles in Token Stream

## Overview
This document describes the prerequisite work required to ensure the LLM trains on **original, human-readable titles** in link targets, not normalized filesystem identifiers. This is a prerequisite for the generation plan (`plans/generation.md`).

**Core Principle**: Normalization and hashing exist solely for filesystem storage and corpus lookup. The model should never see `python_programming_a7f8c3` — it should see `Python Programming`. Normalization happens only at the boundary between "what the model sees" and "where things are stored on disk."

**Status**: Not started. Must be completed before generation plan Phase 2+ can proceed.

**Backwards Compatibility**: Not a concern. No trained models or checkpoints exist.

---

## Problem Statement

The current pipeline has normalization leaking into the token stream at multiple points:

1. **Content files (Wikipedia)**: The dump extractor writes markdown files with normalized link targets: `[Python](python_programming_a7f8c3)`. The pretokenizer strips the hash suffix but leaves the normalized form: `[Python](python_programming)`.

2. **Collate `clean_title`**: `DocSpan.clean_title` is computed by stripping the hash from the normalized title, yielding e.g. `python_programming` instead of `Python Programming`.

3. **Graph JSONL**: Stores the normalized title as the primary key. The original title is stored in `source_identifier` (now reliably, after the `graph_builder.py` fix).

4. **Document content**: Link targets within the actual text are normalized forms, not originals.

The result: the model learns to produce normalized link targets, but during generation we need it to produce human-readable titles that can be detected, decoded, and then normalized for corpus lookup.

---

## Architecture

### Title Flow (Current — Broken)
```
Original title: "Python Programming"
    ↓ GraphBuilder normalizes
Normalized: "python_programming_a7f8c3"
    ↓ Written to graph.jsonl as 'title' key
    ↓ Content files have [Python](python_programming_a7f8c3)
    ↓ Pretokenizer strips hash
Token stream: [Python](python_programming)  ← model trains on this
    ↓ DocSpan.clean_title = "python_programming"
```

### Title Flow (Target — Fixed)
```
Original title: "Python Programming"
    ↓ GraphBuilder stores both
graph.jsonl: title="python_programming_a7f8c3", source_identifier="Python Programming"
    ↓ Content files still have [Python](python_programming_a7f8c3)
    ↓ Pretokenizer replaces with original title
Token stream: [Python](Python Programming)  ← model trains on this
    ↓ DocSpan.clean_title = "Python Programming"
    ↓ At generation time, link detector finds "Python Programming"
    ↓ Normalizer converts to "python_programming_a7f8c3" for corpus lookup
```

---

## Phase T.1: Pretokenizer — Replace Normalized Targets with Originals (MEDIUM)
**File**: `data/pretokenize.py`

### Problem
The `tokenize_worker` currently does a regex to strip hashes from link targets:
```python
hash_pattern = rf'(\]\(.*?)_[0-9a-f]{{{hash_length}}}(\))'
content = re.sub(hash_pattern, r'\1\2', content)
```
This leaves the normalized form (e.g., `python_programming`). We need to replace the **entire normalized+hashed target** with the **original title**.

### Solution

**New mapping required**: Build a `normalized_title_to_original` mapping from the graph data. The graph JSONL has both `title` (normalized+hashed) and `source_identifier` (original). But link targets in markdown content are also normalized+hashed (they reference other documents by their filesystem name). So we need:

```python
# In run_preprocessing, before tokenization:
normalized_to_original = {}
for title, node_data in graph_data.items():
    source_id = node_data.get('source_identifier')
    if source_id:
        normalized_to_original[title] = source_id
```

**Updated `tokenize_worker`**:
```python
def tokenize_worker(
    doc_tuple: tuple,
    queue: mp.Queue,
    encode_fn: Callable[[str], List[int]],
    dtype: np.dtype,
    source_id_to_title: dict,
    normalized_to_original: dict,  # NEW: normalized+hashed title -> original title
):
    source_id, content = doc_tuple

    try:
        title = source_id_to_title.get(source_id)
        if title is None:
            return

        # Replace normalized+hashed link targets with original titles.
        # Pattern: ](normalized_title_hash) -> ](Original Title)
        # We iterate over known normalized titles and replace them in link targets.
        def replace_link_target(match):
            prefix = match.group(1)   # ](
            target = match.group(2)   # the link target text
            suffix = match.group(3)   # )

            # Look up the original title for this normalized target
            original = normalized_to_original.get(target)
            if original is not None:
                return f'{prefix}{original}{suffix}'

            # If not found, try stripping hash and leaving as-is
            # (fallback for targets not in our graph)
            stripped = re.sub(r'_[0-9a-f]{6}$', '', target)
            return f'{prefix}{stripped}{suffix}'

        # Match markdown link targets: ](target)
        content = re.sub(r'(\]\()([^)]+)(\))', replace_link_target, content)

        tokens = encode_fn(content)
        tokens_np = np.asarray(tokens, dtype=dtype)
        queue.put((title, tokens_np))
    except Exception as e:
        logger.error(f"Could not process document '{source_id}': {e}")
```

**Key subtlety**: The link targets in the markdown content are the **full normalized+hashed** form (e.g., `python_programming_a7f8c3`), matching the `title` field in graph.jsonl. So the `normalized_to_original` dict is keyed by the full normalized+hashed title, which is exactly what appears as the link target in content files.

**For non-markdown sources (TheStack)**: Python import statements already contain the original module name (`import numpy`). No replacement is needed. The `replace_link_target` regex only matches markdown `](target)` patterns, so it's a no-op for Python files.

### Testing
- Test that `[Python](python_programming_a7f8c3)` becomes `[Python](Python Programming)` after processing
- Test that unknown targets fall back to hash-stripped form
- Test that Python source files are not affected
- Test roundtrip: original title → normalize → look up in graph → matches

---

## Phase T.2: Collate — Original Title in `clean_title` (EASY)
**File**: `data/collate.py`

### Problem
`clean_title` is currently computed by stripping the hash from the normalized title:
```python
clean_title = formatter.strip_hash(title)  # "python_programming"
```

We need it to be the original human-readable title: `"Python Programming"`.

### Solution

The `build_packed_batch` function receives a `GraphIndex` which provides the title for a doc_id. We need to also look up the original title (source_identifier) from the graph.

**Option A: GraphIndex provides original title**
Add a `get_source_identifier(doc_id)` method to `GraphIndex` that returns the `source_identifier` field from the graph JSONL. This is the cleanest approach.

**Option B: Pass a title mapping**
Pass a `normalized_to_original: Dict[str, str]` to `build_packed_batch`. More explicit but adds a parameter.

**Recommended: Option A.** `GraphIndex` already loads the full JSONL; adding accessor for `source_identifier` is trivial.

```python
# In build_packed_batch:
# Instead of:
clean_title = formatter.strip_hash(title)
# Do:
clean_title = graph.get_source_identifier(p.doc_id) or formatter.strip_hash(title)
```

The fallback (`formatter.strip_hash(title)`) handles graph entries that might not have `source_identifier` (shouldn't happen after the graph_builder fix, but defensive).

### Changes to GraphIndex
```python
class GraphIndex:
    def get_source_identifier(self, doc_id: int) -> Optional[str]:
        """Get the original (unnormalized) title for a document."""
        # Look up from the loaded JSONL data
        ...
```

### Testing
- Verify `DocSpan.clean_title` is the original title for all docs in a batch
- Verify link matching (cross_doc_mask uses `clean_title`) still works with original titles
- Verify fallback works when `source_identifier` is missing

---

## Phase T.3: Cross-Doc Mask Compatibility (EASY)
**File**: `model/graph_traversal/cross_doc_mask.py`

### Problem
`_match_links_to_docs` matches decoded link target text against `span.clean_title`. After Phase T.1, the token stream contains original titles (e.g., `Python Programming`). After Phase T.2, `clean_title` is also the original title. So the matching should work without changes — **but we need to verify**.

### Verification Required
- The `MarkdownLinkDetector` decodes target tokens. After T.1, those tokens encode `Python Programming` (the original title).
- `_match_links_to_docs` compares decoded text to `span.clean_title`. After T.2, `clean_title = "Python Programming"`.
- Match should succeed.

**Potential issue**: Whitespace/encoding differences. The decoded token text might have leading/trailing whitespace or different Unicode normalization than the stored `source_identifier`. The `.strip()` call in the mask creator handles leading/trailing whitespace.

### Testing
- End-to-end test: pretokenize with original titles → build batch → create cross-doc mask → verify links matched correctly

---

## Phase T.4: Update `outgoing_titles` in Collate (MEDIUM)
**File**: `data/collate.py`, potentially `data/dataset.py`

### Problem
`DocSpan.outgoing_titles` is populated from `graph.get_outgoing_links(title)`, which returns **normalized+hashed** titles (the keys in the graph JSONL). The `PythonImportDetector` (which uses `outgoing_titles` via `uses_outgoing_titles = True`) matches these against `span.clean_title` of other docs. After Phase T.2, `clean_title` is the original title, but `outgoing_titles` are still normalized. **These won't match.**

### Solution
Two options:

**Option A**: Convert `outgoing_titles` to original titles at collate time using the same `get_source_identifier` lookup. This keeps the `PythonImportDetector` working as-is.

**Option B**: This becomes moot if `PythonImportDetector` is refactored to not use `outgoing_titles` (Phase 0.3 in the generation plan). After refactoring, `PythonImportDetector` will decode module paths from tokens and set `uses_outgoing_titles = False`.

**Recommendation**: Do Option A now for correctness, and it becomes redundant (but harmless) after Phase 0.3.

```python
# In build_packed_batch, after getting outgoing_titles:
outgoing_titles_raw = graph.get_outgoing_links(title)
# Convert to original titles for matching against clean_title
outgoing_titles = []
for ot in outgoing_titles_raw:
    original = graph.get_source_identifier_by_title(ot)
    outgoing_titles.append(original if original else formatter.strip_hash(ot))
```

### Testing
- Verify `PythonImportDetector` cross-doc mask creation works with original titles in both `outgoing_titles` and `clean_title`

---

## Phase T.5: Document Corpus Lookup Chain (EASY)
**File**: `model/document_corpus.py`

### Problem
After the title pipeline changes, generation will produce original titles (e.g., `Python Programming`). The corpus needs to look these up. Currently `DocumentCorpus` uses `title_utils.create_filename()` which produces a **different hash** than `FilesafeNormalizer` (title_utils hashes the raw original; FilesafeNormalizer hashes the canonical form).

### Solution
`DocumentCorpus` should accept a `DatasetConfig` or normalizer and use it for the lookup chain:

```python
class DocumentCorpus:
    def __init__(self, dataset_path: Path, normalizer: Optional[LinkNormalizer] = None):
        self.dataset_path = Path(dataset_path)
        self.index = GraphIndex(self.dataset_path)
        self.backend = PretokShardedBackend(self.index)

        # Use provided normalizer or load from dataset config
        if normalizer is None:
            from data.dataset_config import load_config_from_pretokenized_dir
            config = load_config_from_pretokenized_dir(self.dataset_path)
            self.normalizer = config.get_normalizer()
        else:
            self.normalizer = normalizer

    def get_document(self, title: str) -> Optional[np.ndarray]:
        # Try direct lookup (already normalized+hashed)
        if title in self.index:
            return self.backend.get_tokens(title)

        # Normalize and try again
        normalized = self.normalizer.normalize(title)
        if normalized in self.index:
            return self.backend.get_tokens(normalized)

        return None
```

This also deletes the dependency on `title_utils.py`, completing Phase 0.4 from the generation plan.

### Testing
- Test lookup by original title (normalizes internally)
- Test lookup by already-normalized title (direct hit)
- Test with different normalizer types (Wikipedia, TheStack)

---

## Phase T.6: Delete `model/title_utils.py` (EASY)
**File**: `model/title_utils.py`, `tests/model/test_title_utils.py`

### Prerequisite
Phase T.5 must be complete (DocumentCorpus no longer imports from title_utils).

### Changes
1. Delete `model/title_utils.py`
2. Delete `tests/model/test_title_utils.py`
3. `model/__init__.py` already updated (title_utils exports removed)
4. `model/generation_result.py` already updated (import removed)
5. Verify no remaining imports via `grep -r "title_utils" .`

---

## Dependency Graph

```
T.1 Pretokenizer (original titles in content)
    ↓
T.2 Collate clean_title ← depends on GraphIndex change
    ↓
T.3 Cross-doc mask verification ← depends on T.1 + T.2
    ↓
T.4 outgoing_titles conversion ← depends on T.2

T.5 DocumentCorpus normalizer ← independent (but logically part of this effort)
    ↓
T.6 Delete title_utils ← depends on T.5
```

**T.1 and T.5 can proceed in parallel.** T.2 depends on a minor GraphIndex addition. T.3 is verification. T.4 is a small follow-up. T.6 is cleanup.

---

## Relationship to Generation Plan

This plan **must be completed before** the generation plan's Phase 2 (DocumentContext) can be implemented. Specifically:

- **Phase 0.2** in `generation.md` is replaced by this document (Phases T.1–T.4).
- **Phase 0.4** in `generation.md` (consolidate normalization / delete title_utils) is replaced by T.5–T.6.
- **Phase 0.5** in `generation.md` (fix DocSpan for generation) is covered by T.2.
- After this plan is complete, `generation.md` Phase 0 reduces to: 0.1 (mask refactor), 0.3 (Python import detection), and 0.6 (GenerationConfig fields).

---

## Implementation Status

- [ ] T.1 Pretokenizer: original titles in link targets — MEDIUM
- [ ] T.2 Collate: original title in `clean_title` — EASY
- [ ] T.3 Cross-doc mask verification — EASY
- [ ] T.4 `outgoing_titles` conversion — MEDIUM
- [ ] T.5 DocumentCorpus normalizer — EASY
- [ ] T.6 Delete `title_utils.py` — EASY
