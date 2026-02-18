# TS2TS Generation Implementation Plan

## Overview
This document provides a detailed, dependency-ordered plan for implementing the generation functionality for the TS2TS (TagSeq2TagSeq) model. The generation system enables autoregressive text generation with graph-aware attention patterns, supporting recursive document generation through link detection.

**Status**: Updated 2026-02-17. Revised after codebase audit, design review, and code review feedback. Title pipeline work extracted to `plans/title_pipeline.md` (prerequisite). Fixed topological ordering to proper Kahn-style DAG sort. Added padding guard for inference masks. Clarified coordinate systems, link detection, and DocLayoutPolicy reuse for aux doc seeding.

**Difficulty Levels:**
- **EASY**: Sub-agent can implement with one-time prompt (minimal context, clear spec)
- **MEDIUM**: Primary agent with optional clarifying questions (moderate complexity, some design decisions)
- **HARD**: Primary agent after extensive conversation (complex, significant design decisions, integration challenges)

**Backwards Compatibility**: Not a concern. Training has not yet started; no trained models or checkpoints exist.

---

## Architecture Overview

**Key Principles:**

1. **Topological Ordering (Critical)**: Documents in the packed sequence are ALWAYS arranged so that linked-to documents appear BEFORE the documents that link to them. This matches training (which uses `prefer_targets_first` ordering in the pack sampler) and is required by the cross-doc attention mask's DAG property check. During generation this means auxiliary documents are prepended before the root, and the root is always the LAST document in the packed sequence. See "Topological Ordering During Generation" below.

2. **Causality**: The underlying document graph may be bidirectional or cyclic, but any single generation context can only represent a DAG (directed acyclic subgraph). This is fundamental to autoregressive LLMs: we can only attend to things that come before us in the sequence. The topological ordering of the DAG ensures linked-to documents are attendable.

3. **Continuous Link Detection**: After EVERY generated token, run the link detector on the current document. When a new link is detected, trigger corpus lookup or auxiliary document generation.

4. **Immediate Pause on Link Detection (Core Innovation)**: When the root document generates a token that completes a link (e.g., the `)` in `[Python](Python Programming)`), generation of the root **IMMEDIATELY PAUSES** — it does NOT generate any further root tokens. The linked auxiliary document is then generated (or fetched from corpus), prepended to the sequence, and ONLY THEN does root generation resume. This means every root token generated after the link was generated with the aux doc already in context. There is never a situation where root tokens exist that were generated without access to an aux doc that should have been available.

5. **FlexAttention with B=1**: All inference uses batch size 1. Must pad token sequences to multiples of 128 for FlexAttention compatibility.

6. **Original Titles in Token Stream**: The LLM trains on and generates original, human-readable titles (e.g., "Python Programming"), NOT normalized filesystem identifiers (e.g., "python_programming_a7f8c3"). Normalization and hashing are purely for filesystem/corpus lookup. See "Title Pipeline" below.

7. **Context Management**: Track multiple documents (root + auxiliary), their tokens, DocSpans, link relationships, and current generation state. For MVP, stop generation when context is full (no eviction logic; aux doc dropping is a post-MVP enhancement).

### Topological Ordering During Generation

During training, `PackBatchSampler` with `order_mode="prefer_targets_first"` places linked-to documents before linkers via Kahn-style topological sort (with cycle-breaking via insertion order). The `CrossDocLinkMaskCreator` enforces this via a DAG check (`cross_doc_mask.py:137-139`): cross-doc attention is only granted when `target_start_pos < link_end_pos`.

During generation, we maintain this invariant. The packed sequence is always:

```
[deepest_aux_docs] [depth-1_aux_docs] ... [depth-1_aux_docs] [root_doc]
```

**Example**: Root generates link to "Python Programming" at token position P (the `)` closing the link). Generation immediately pauses. The sequence transforms:

```
Before link: [root_tokens_0..P]
After aux:   [python_doc_tokens] [root_tokens_0..P..resuming_here]
                                  ^-- root shifted right
```

Since there's NO KV cache (MVP), we recompute the full forward pass every step anyway. Rebuilding the packed sequence with new ordering costs nothing extra.

**When root resumes after aux generation**: The very first new root token (at position P + 1 in root-local coordinates, shifted to P + 1 + len(python_doc) in the packed sequence) is generated with the full aux doc already in context. The mask grants cross-doc attention from positions after `link_end_pos` in the source doc to all positions in the target doc. Since the aux doc starts at position 0 and all root positions are > 0, the DAG check passes naturally.

**Nested recursion**: If the Python aux doc itself generates a link to "Guido van Rossum" (depth 2), the sequence becomes:

```
[guido_doc] [python_doc_partial...resuming] [root_partial...paused]
```

The topological sort is maintained at every step.

### Mask Architecture (Core/Training/Inference Split)

**Problem identified**: All existing mask creators (`cross_doc_mask.py`, `block_mask_creator.py`) compute `seq_len = tokens.shape[-1] - 1` because training passes `tokens[:, :-1]` as input and `tokens[:, 1:]` as target. This T-1 assumption is training-specific and breaks inference.

**Solution**: Refactor ALL mask creators into three layers:

1. **Core mask logic** (shared): Takes `seq_len` explicitly, builds mask. No assumptions about token shifting.
2. **Training callable**: Computes `seq_len = tokens.shape[-1] - 1`, delegates to core.
3. **Inference callable**: Uses `real_seq_len` from batch dict (excludes padding), delegates to core.

The `make_mask_creator_callable` function accepts a `mode='training'|'inference'` parameter and returns the appropriate callable. Both callables have the same `(**batch) -> BlockMask` signature so `forward_inference` can use `self.block_mask_creator(**batch_dict)` directly, same pattern as training.

This is a prerequisite refactor (Phase 0) before generation can work. ALL mask types (`doc_causal`, `causal`, `full`, `doc_bidirectional`, `cross_doc_link`) must support both modes.

### Padding Token Attention Guard (Inference Only)

Inference masks must include a guard preventing padding tokens (with `document_ids == -1`) from attending to each other or being attended to. Training does not need this since training uses packing (no padding). The inference mask callable composes the existing mask logic with a padding guard via FlexAttention's boolean AND:

```python
# Inference mask = existing_mask_logic AND padding_guard
valid_q = document_ids[q_idx] != -1
valid_kv = document_ids[kv_idx] != -1
return valid_q & valid_kv & existing_mask_result
```

This is implemented as a separate composable mask function that wraps the existing mask logic, ensuring the training-mode masks remain unchanged. FlexAttention's `mask_mod` functions are simple boolean expressions, so composing them is straightforward.

### Title Pipeline

**Extracted to separate plan**: See `plans/title_pipeline.md` for the full prerequisite implementation plan. This is a prerequisite that must be completed before Phase 2+ of this plan.

**Summary**: The `data/` normalization infrastructure (`data/extractors/normalization.py`) is the source of truth. `model/title_utils.py` is deleted. The LLM trains on original human-readable titles (e.g., `Python Programming`). Normalization and hashing happen only at filesystem/corpus boundaries. Corpus lookup during generation uses `DatasetConfig.get_normalizer()` to convert original titles to corpus keys.

### Token Budget Architecture

Two distinct token limits govern generation:

1. **`max_new_tokens`**: Maximum new tokens for the ROOT document only. This is the primary generation length control.
2. **`max_total_new_tokens`**: Global maximum across ALL newly generated tokens (root + all aux docs). Prevents unbounded token generation from recursive link following. When this budget is exhausted, no more tokens can be generated for any document.
3. **`max_tokens_per_document`**: Per-document cap for individual auxiliary documents.
4. **`max_context_length`**: Hard limit on total packed sequence size (corpus docs + generated docs). When full, generation stops.

The global budget (`max_total_new_tokens`) is tracked via a shared mutable counter passed through recursive calls. When an aux doc consumes tokens from this budget, the remaining budget for ALL subsequent generation (parent docs included) decreases accordingly.

---

## Phase 0: Prerequisites (Codebase Changes Before Generation)

These are modifications to EXISTING code required before generation components can work.

### 0.1 Refactor Mask Creators: Remove Training-Specific T-1 Shift (MEDIUM)
**Files**: `model/graph_traversal/cross_doc_mask.py`, `model/graph_traversal/block_mask_creator.py`

**Problem**: All mask creators compute `seq_len = tokens.shape[-1] - 1` because training splits tokens into input/target. This means inference can't use them directly.

**Solution**: Extract core mask logic that takes an explicit `seq_len` parameter. Create training and inference callables via `make_mask_creator_callable(mask_type, mode)`.

**Changes to `cross_doc_mask.py`**:
```python
class CrossDocLinkMaskCreator:
    def _create_mask_core(
        self,
        input_ids: torch.Tensor,  # 1D tensor of token IDs [seq_len]
        seq_len: int,
        doc_spans: List[Any],
        device: torch.device
    ) -> BlockMask:
        """Core mask creation logic. No assumptions about token shifting."""
        # Detect links, match to docs, build cross-doc mask, create BlockMask
        # ... (existing logic, but using explicit seq_len)
        # Include padding guard: valid_q & valid_kv & causal & (same_doc | cross_doc_link)

    def create_training_mask(self, tokens, doc_spans, **kwargs):
        """Training entry point: applies T-1 shift."""
        seq_len = tokens.shape[-1] - 1
        input_ids = tokens[0, :-1]
        return self._create_mask_core(input_ids, seq_len, doc_spans, tokens.device)

    def create_inference_mask(self, tokens, doc_spans, real_seq_len, **kwargs):
        """Inference entry point: uses provided real_seq_len (excludes padding)."""
        input_ids = tokens[0, :real_seq_len]
        return self._create_mask_core(input_ids, real_seq_len, doc_spans, tokens.device)
```

**Changes to `block_mask_creator.py`**: Same pattern for ALL mask functions (`create_doc_causal_block_mask`, `create_causal_block_mask`, `create_full_attention_block_mask`, `create_doc_bidirectional_block_mask`). Each gets a `_core` version taking explicit `seq_len`, plus training/inference wrappers. Inference wrappers include the padding guard (`document_ids != -1`); training wrappers do not (training uses packing).

**Remove module-level singleton**: The `_cross_doc_link_creator` global at `block_mask_creator.py:259` must be removed. Each `make_mask_creator_callable` call should create (or accept) a properly-configured `CrossDocLinkMaskCreator` instance with the correct tokenizer for the model being used.

`make_mask_creator_callable` accepts `mode='training'|'inference'`:

```python
def make_mask_creator_callable(mask_type: str, mode: str = 'training'):
    """
    Create a callable with signature (**batch) -> BlockMask.

    mode='training': extracts tokens, doc_spans from batch, applies T-1 shift.
    mode='inference': extracts tokens, doc_spans, real_seq_len from batch, no shift.
    """
    if mode == 'training':
        # Returns callable that does: mask_fn.create_training_mask(**batch)
        ...
    elif mode == 'inference':
        # Returns callable that does: mask_fn.create_inference_mask(**batch)
        ...
```

Both return callables with `(**batch) -> BlockMask` signature. The inference callable expects `real_seq_len` in the batch dict.

**Testing**: Verify that the refactored training path produces identical masks to the current implementation. Verify inference path produces correct masks without the -1 shift. Test all five mask types in both modes.

### 0.2 Title Pipeline (EXTRACTED)
**Moved to**: `plans/title_pipeline.md`

This phase has been extracted into its own plan document due to scope and complexity. It covers: pretokenizer changes, collate `clean_title` fix, `outgoing_titles` conversion, DocumentCorpus normalizer update, and `title_utils.py` deletion. **Must be completed before Phase 2.**

### 0.3 Refactor Python Import Detection for Generation (HARD)
**Files**: `model/graph_traversal/link_detectors.py`

**Problem**: `PythonImportDetector` sets `uses_outgoing_titles = True` and only finds import POSITIONS (where `import`/`from` keywords are), relying on precomputed `doc_spans.outgoing_titles` from the graph for target resolution. During generation, generated documents don't have precomputed `outgoing_titles` — so the detector can say "there's an import at position 47" but cannot say "it imports numpy." This makes Python/Stack support impossible at inference time.

**Solution**: Make `PythonImportDetector` actually parse import statements to extract module paths, similar to how `MarkdownLinkDetector` decodes link targets from tokens. The detector should:
1. Find `from X import Y` and `import X` patterns in tokens
2. Decode the module path from the token sequence (the `X` part)
3. Return `LinkInfo` with `target_start`/`target_end` pointing to the module path tokens
4. Set `uses_outgoing_titles = False` so the mask creator decodes targets from tokens

This makes Python imports work like markdown links: the detector finds the target text in the token stream, and the mask creator/generation loop decodes it and matches to documents.

**Module path to document title resolution**: The generation loop needs a mapping function (configurable per dataset) that converts a Python module path (e.g., `"numpy.array"`) to a corpus document title (e.g., the file path `numpy/array.py`). This is dataset-specific logic that should be pluggable via `DatasetConfig`.

**This is NOT deferred** — Python/Stack support is desired for the initial release.

### 0.4 Consolidate Normalization (EXTRACTED)
**Moved to**: `plans/title_pipeline.md` (Phases T.5 and T.6)

### 0.5 Fix DocSpan for Generation (EXTRACTED)
**Moved to**: `plans/title_pipeline.md` (Phase T.2)

### 0.6 Fix GenerationConfig (EASY)
**File**: `model/generation_config.py`

**Changes**:
- Remove `eviction_policy` field (defer to post-MVP)
- Rename `allow_corpus_fallback` to `generate_missing_docs` (clearer: "if a linked doc isn't in corpus, generate it")
- `allow_recursive_links: bool = True` already exists — keep as-is
- Add `link_format: str = 'markdown'` field
- Add `max_total_new_tokens: int = 4096` field (global budget across all generated docs)
- Add `layout_policy_name: Optional[str] = None` field (name of DocLayoutPolicy for aux doc seeding; None = NullLayoutPolicy)

---

## Phase 1: Foundation Components (PARTIALLY COMPLETE)

Several Phase 1 components exist but have issues identified during the codebase audit.

### 1.1 DocumentCorpus Class (NEEDS UPDATE)
**File**: `model/document_corpus.py`
**Status**: Exists but uses hardcoded `title_utils.create_filename()` for normalization.
**Required fix**: Covered by `plans/title_pipeline.md` Phase T.5. Accept normalizer; use dataset-appropriate normalization for lookups.

### 1.2 Title Utilities (DELETE)
**File**: `model/title_utils.py`
**Status**: To be deleted. Covered by `plans/title_pipeline.md` Phase T.6.

### 1.3 GenerationResult Data Structures (COMPLETE)
**File**: `model/generation_result.py`
**Status**: Complete, no changes needed.

### 1.4 Link Detectors (NEEDS UPDATE)
**File**: `model/graph_traversal/link_detectors.py`
**Status**: `MarkdownLinkDetector` works. `PythonImportDetector` needs refactor to actually parse imports (Phase 0.3 — NOT deferred).
**API note**: `detect_links()` takes `(input_ids: torch.Tensor, tokenizer_decode_fn: Callable)`. Generation code must pass torch tensors and the decode function, not numpy arrays.

### 1.5 Token Sampling Utilities (COMPLETE)
**File**: `model/sampling.py`
**Status**: Complete, no changes needed.

### 1.6 GenerationConfig (NEEDS UPDATE)
**File**: `model/generation_config.py`
**Status**: Needs fixes listed in Phase 0.6.

---

## Phase 2: Core Generation Context Management

### 2.1 DocumentContext Class (HARD)
**File**: `model/document_context.py` (NEW)

**Dependencies**:
- Phase 0.1 (mask refactor)
- Title pipeline (`plans/title_pipeline.md`) — complete before this phase
- GenerationResult structures (1.3)
- LinkDetectors (1.4)
- TokenizerConfig
- DatasetConfig (from `data/dataset_config.py`)
- DocLayoutPolicy (from `data/layout.py`) — for aux doc seeding

**Description**: Manages generation state for multiple documents during recursive generation. Maintains topological ordering where linked-to documents always precede linkers in the packed sequence. This is the core state manager.

**Key Responsibilities**:
1. **Topological Token Packing**: Maintain the packed token sequence with aux docs BEFORE the docs that link to them. Root is always last.
2. **Dynamic Reordering**: When a new aux doc is added (from corpus or generation), rebuild the packed sequence to maintain topological order.
3. **DocSpan Tracking**: Build `List[DocSpan]` matching the current packed sequence layout. `clean_title` is always the original human-readable title.
4. **Generation Stack**: Track which document is currently being generated (pause/resume via stack).
5. **Link Tracking**: Remember which links have been detected and processed.
6. **Context Window**: Enforce max_context_length, pad to multiples of 128.
7. **Real Sequence Length Tracking**: Track actual token count (excluding padding) for correct logits extraction.
8. **Global Token Budget**: Track total new tokens generated across all documents via a shared counter.

**MVP Simplifications**:
- NO eviction logic (stop when context full; aux doc dropping is post-MVP)
- NO KV caching (full recomputation each step)
- Topological sort can be simple since the graph is built incrementally (new docs are always leaves)

**Data Structures**:
```python
@dataclass
class DocumentState:
    """State of a single document during generation."""
    original_title: str       # Human-readable title (what LLM sees and generates in links)
    normalized_title: str     # Filesystem-safe identifier for corpus lookup
    tokens: np.ndarray        # Generated/corpus tokens (int32 or int64 numpy array)
    is_complete: bool         # Whether generation finished
    parent_title: Optional[str]  # Which doc linked to this (original_title)
    source: Literal["generated", "corpus"]
    depth: int                # Link depth (root=0, root's links=1, etc.)
    detected_links: Set[str]  # Original titles of links found in this doc
    doc_id: int               # Sequential integer ID for mask creation
    linked_by: Set[str]       # Set of doc titles that link TO this doc


@dataclass
class GlobalTokenBudget:
    """Shared mutable counter for total new tokens across all generated docs."""
    max_total: int
    consumed: int = 0

    @property
    def remaining(self) -> int:
        return self.max_total - self.consumed

    def consume(self, n: int = 1) -> bool:
        """Consume n tokens. Returns False if budget exhausted."""
        if self.consumed + n > self.max_total:
            return False
        self.consumed += n
        return True
```

**Key Methods**:
```python
class DocumentContext:
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        tokenizer_encode: Callable[[str], List[int]],
        tokenizer_decode: Callable[[List[int]], str],
        link_detector: TokenizedLinkDetector,
        normalizer: LinkNormalizer,       # From DatasetConfig
        layout_policy: DocLayoutPolicy,   # For aux doc seeding (reuses training prefix)
        max_context_length: int,
        max_tokens_per_doc: int,
        max_auxiliary_documents: int,
        global_token_budget: GlobalTokenBudget,
        device: torch.device
    ):
        self.documents: Dict[str, DocumentState] = {}  # keyed by original_title
        self.generation_stack: List[str] = []  # Stack of paused doc titles
        self.root_title: Optional[str] = None
        self.current_doc_title: Optional[str] = None
        self.next_doc_id: int = 0
        self._real_seq_len: int = 0  # Actual tokens (excluding padding)
        self.global_token_budget = global_token_budget
        self.layout_policy = layout_policy

    def initialize_root(self, prompt_tokens: List[int], title: str):
        """Set up root document. Root is always last in topological order."""

    def add_corpus_document(
        self, original_title: str, tokens: np.ndarray,
        linked_from_title: str, depth: int
    ) -> bool:
        """Add complete doc from corpus. Inserts BEFORE linker in topo order."""

    def start_auxiliary_generation(
        self, original_title: str, linked_from_title: str, depth: int,
        title_seed_tokens: Optional[List[int]] = None
    ) -> bool:
        """
        Begin generating a new auxiliary document.
        If title_seed_tokens is None, uses default seeding (e.g., "# {title}\n").
        Inserts BEFORE linker in topo order.
        Pushes current doc onto generation_stack.
        """

    def append_token_to_current(self, token_id: int) -> bool:
        """
        Append token to currently-generating document.
        Consumes from global_token_budget.
        Returns False if any limit hit (per-doc, context, or global budget).
        """

    def check_for_new_links(self) -> List[str]:
        """
        Run link detector on current doc's tokens.
        Returns list of newly detected original titles not yet processed.
        Uses torch.Tensor and passes tokenizer_decode_fn to detector.
        """

    def complete_current_document(self):
        """Mark current doc as complete. Pops generation_stack to resume parent."""

    def get_tokens_for_model(self) -> Tuple[torch.Tensor, int]:
        """
        Return (packed_tokens, real_seq_len).
        packed_tokens: shape [1, T] padded to multiple of 128.
        real_seq_len: actual number of tokens before padding.
        Documents are in topological order (aux before root).
        """

    def get_doc_spans(self) -> List[DocSpan]:
        """
        Return DocSpan objects matching current topological layout.
        Each span has original_title in clean_title for link matching.
        """

    def get_real_seq_len(self) -> int:
        """Actual token count excluding padding. For correct logits extraction."""

    def to_generation_result(
        self, tokenizer_decode: Callable[[List[int]], str],
        config: GenerationConfig
    ) -> GenerationResult:
        """
        Build a GenerationResult from current context state.
        Decodes tokens to text for each document. Root is identified,
        all others are auxiliary. Topological ordering preserved.
        """
```

**Topological Ordering Implementation**:

The document graph during generation is a full DAG (not just a tree) — the same document may be linked by multiple parents (e.g., root links to "Python" and aux doc "NumPy" also links to "Python"). This matches training, which uses Kahn-style topological sort with cycle-breaking via insertion order. We use the same approach:

```python
def _compute_topological_order(self) -> List[str]:
    """
    Kahn-style topological sort of documents.
    Linked-to docs come BEFORE the docs that link to them.

    Edges: for each doc D that has a link to doc T, there is an edge T -> D
    (T must come before D). This matches training's prefer_targets_first ordering.

    Ties broken by doc_id (insertion order) for stability.
    Cycles (shouldn't happen in generation, but defensive) broken by
    falling back to insertion order for remaining nodes.
    """
    # Build adjacency: edges[T] = set of docs that link TO T (i.e., T must come before them)
    # Equivalently: for each doc D with detected_links containing T, add edge T -> D
    in_degree = {t: 0 for t in self.documents}
    adjacency = {t: [] for t in self.documents}

    for title, doc in self.documents.items():
        for linked_title in doc.detected_links:
            if linked_title in self.documents:
                # linked_title must come before title
                adjacency[linked_title].append(title)
                in_degree[title] += 1

    # Kahn's algorithm with insertion-order tie-breaking
    # Use a list sorted by doc_id as the "queue" for determinism
    queue = sorted(
        [t for t, deg in in_degree.items() if deg == 0],
        key=lambda t: self.documents[t].doc_id
    )
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in adjacency[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                # Insert in doc_id order for stability
                queue.append(neighbor)
                queue.sort(key=lambda t: self.documents[t].doc_id)

    # Any remaining nodes (cycles) appended in doc_id order
    if len(result) < len(self.documents):
        remaining = sorted(
            [t for t in self.documents if t not in set(result)],
            key=lambda t: self.documents[t].doc_id
        )
        result.extend(remaining)

    return result
```

This handles the general DAG case (diamond dependencies, multiple parents) correctly, matching training behavior.

**get_tokens_for_model Implementation**:
```python
def get_tokens_for_model(self) -> Tuple[torch.Tensor, int]:
    ordered_titles = self._compute_topological_order()

    all_tokens = []
    for title in ordered_titles:
        all_tokens.extend(self.documents[title].tokens)

    real_len = len(all_tokens)

    # Pad to multiple of 128
    if real_len % 128 != 0:
        pad_to = ((real_len // 128) + 1) * 128
        all_tokens.extend([0] * (pad_to - real_len))

    tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
    return tensor, real_len
```

**check_for_new_links Implementation**:

Link detection always operates on a single document's **full token sequence**. The detector receives all tokens for the current document and returns link positions in **document-local coordinates** (0-based offsets within that document's tokens). These local positions are used only for decoding the link target text. The cross-doc mask is built separately by the mask creator operating on the **full packed sequence**, which re-runs link detection in packed-sequence coordinates. This two-stage approach avoids coordinate translation bugs.

```python
def check_for_new_links(self) -> List[str]:
    """
    Run link detector on the current document's tokens (local coordinates).
    Returns list of newly detected original titles not yet processed.

    Note: This uses document-local coordinates for target extraction only.
    The cross-doc mask creator independently detects links in the full packed
    sequence (global coordinates) when building the attention mask.
    """
    current_doc = self.documents[self.current_doc_title]

    # Detector receives the FULL token sequence of this document
    tokens_tensor = torch.tensor(current_doc.tokens, dtype=torch.long)

    all_links = self.link_detector.detect_links(tokens_tensor, self.tokenizer_decode)

    new_titles = []
    for link in all_links:
        # Decode target tokens to get original title (local coordinates)
        target_tokens = tokens_tensor[link.target_start:link.target_end].tolist()
        title = self.tokenizer_decode(target_tokens).strip()

        if title and title not in current_doc.detected_links:
            current_doc.detected_links.add(title)
            new_titles.append(title)

    return new_titles
```

**Auxiliary Document Seeding via DocLayoutPolicy**:

During training, `DocLayoutPolicy` (`data/layout.py`) defines per-doc prefix/suffix tokens (e.g., `BOSEOSLayoutPolicy` adds BOS/EOS tokens). Generation should reuse the same policy to ensure consistency between training prefixes and generation seeds.

`DocumentContext` accepts a `DocLayoutPolicy` instance. When starting auxiliary generation, it calls `layout_policy.prefix_tokens(doc_id)` to get the seed tokens. This guarantees the generated doc starts with exactly the same prefix the model was trained with.

```python
class DocumentContext:
    def __init__(self, ..., layout_policy: DocLayoutPolicy = None):
        self.layout_policy = layout_policy or NullLayoutPolicy()

    def start_auxiliary_generation(self, original_title, ...):
        seed_tokens = self.layout_policy.prefix_tokens(self.next_doc_id)
        # If layout provides no prefix, fall back to title-based seed
        if not seed_tokens:
            seed_tokens = self.tokenizer_encode(f"# {original_title}\n")
        ...
```

For custom seeding beyond what the layout policy provides, the caller can pass `title_seed_tokens` directly to override.

**Testing**:
- Test topological ordering with various depth configurations
- Test that root is always last in packed sequence
- Test adding documents shifts existing doc positions correctly
- Test real_seq_len tracking through padding
- Test generation stack (pause/resume)
- Test link detection with correct API (torch tensor + decode fn)
- Test context full stopping
- Test global token budget exhaustion stops all generation

---

## Phase 3: Generation Loop Implementation

### 3.1 Single Document Generator (MEDIUM)
**File**: `model/generation/single_doc_generator.py` (NEW)

**Dependencies**:
- DocumentContext (2.1)
- GenerationConfig (1.6, updated)
- Sampling utilities (1.5)
- DS2DSModel.forward_inference (4.1)

**Description**: Generate a single document autoregressively WITHOUT handling links. Baseline generator and building block for the full system.

**Key Algorithm**:
```python
class SingleDocumentGenerator:
    def __init__(self, model: 'DS2DSModel'):
        self.model = model

    def generate(self, context: DocumentContext, config: GenerationConfig,
                 max_tokens: Optional[int] = None) -> int:
        tokens_generated = 0
        effective_max = max_tokens or config.max_tokens_per_document

        while tokens_generated < effective_max:
            if context.global_token_budget.remaining <= 0:
                break

            tokens_tensor, real_seq_len = context.get_tokens_for_model()
            doc_spans = context.get_doc_spans()

            with torch.no_grad():
                logits = self.model.forward_inference(
                    tokens=tokens_tensor,
                    doc_spans=doc_spans,
                    real_seq_len=real_seq_len
                )  # [1, T, V]

            # Extract logits at LAST REAL TOKEN position (not last padded)
            last_real_pos = real_seq_len - 1
            next_logits = logits[0, last_real_pos, :]  # [V]

            # Sample
            if config.temperature == 0:
                next_token = greedy_sample(next_logits)
            else:
                next_token = sample_token(
                    next_logits, config.temperature, config.top_k, config.top_p
                )

            if not context.append_token_to_current(next_token):
                break  # Per-doc, context, or global budget limit hit

            tokens_generated += 1

            if next_token == config.eos_token_id:
                break

        return tokens_generated
```

**Testing**:
- Test with simple prompt, verify tokens generated
- Test EOS stopping, length limit stopping
- Test that logits are extracted at correct (non-padded) position
- Test global budget exhaustion

---

### 3.2 Prompt Link Processor (MEDIUM)
**File**: `model/generation/prompt_processor.py` (NEW)

**Dependencies**:
- LinkDetector (1.4)
- DocumentContext (2.1)
- DocumentCorpus (1.1, updated)

**Description**: Process links in the initial prompt BEFORE generation starts.

**Implementation**:
```python
class PromptLinkProcessor:
    def __init__(self, link_detector: TokenizedLinkDetector,
                 tokenizer_decode: Callable[[List[int]], str]):
        self.link_detector = link_detector
        self.tokenizer_decode = tokenizer_decode

    def process_prompt(
        self, prompt_tokens: List[int], context: DocumentContext,
        corpus: Optional[DocumentCorpus], config: GenerationConfig
    ) -> List[str]:
        """
        Process links in initial prompt. Returns titles needing generation.
        Deduplicates: if the same title appears multiple times in the prompt,
        only the first occurrence is processed.
        """
        tokens_tensor = torch.tensor(prompt_tokens, dtype=torch.long)
        detected_links = self.link_detector.detect_links(tokens_tensor, self.tokenizer_decode)

        titles_to_generate = []
        seen_titles: Set[str] = set()

        for link in detected_links:
            target_tokens = tokens_tensor[link.target_start:link.target_end].tolist()
            title = self.tokenizer_decode(target_tokens).strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            # Try corpus lookup (normalizer handles title -> corpus key)
            if corpus is not None and corpus.has_document(title):
                doc_tokens = corpus.get_document(title)
                if len(doc_tokens) > config.max_tokens_per_document:
                    doc_tokens = doc_tokens[:config.max_tokens_per_document]

                added = context.add_corpus_document(
                    original_title=title, tokens=doc_tokens,
                    linked_from_title=context.root_title, depth=1
                )
                if not added:
                    break  # Context full
            elif config.generate_missing_docs:
                titles_to_generate.append(title)

        return titles_to_generate
```

---

### 3.3 Linked Document Generator (HARD)
**File**: `model/generation/linked_doc_generator.py` (NEW)

**Dependencies**: All of Phase 1-2, SingleDocumentGenerator (3.1), PromptLinkProcessor (3.2)

**Description**: Main generation orchestrator with recursive document generation and link detection.

**Key Algorithm**:
```
generate_with_links(prompt, corpus, config):
    1. Initialize DocumentContext with topological ordering and global token budget
    2. Initialize root document with prompt tokens (root = last in order)
    3. Process any links in prompt (PromptLinkProcessor)
       - Corpus docs added BEFORE root in topological order
       - Non-corpus titles queued for generation
    4. Generate queued prompt-link docs (if generate_missing_docs)
    5. Generate root document with link handling:
       a. Generate one token at a time
       b. After each token, check for new links
       c. If new link found AND depth < max_link_depth:
          - Check corpus first
          - If in corpus: add to context (before linker in topo order)
          - If not in corpus AND generate_missing_docs AND allow_recursive_links:
            * IMMEDIATELY PAUSE current doc (push onto generation_stack)
            * DO NOT generate any more tokens for current doc
            * Start aux doc generation (inserts before current in topo order)
            * RECURSIVELY generate aux doc (may itself find links)
            * Aux doc completes -> pop stack, RESUME current doc
            * Next token generated for current doc now has aux doc in context
       d. Continue until root complete or stopping condition
    6. If max_link_depth > 0, also process links in corpus docs added to context
    7. Build GenerationResult from context
```

**Implementation**:
```python
class LinkedDocumentGenerator:
    def __init__(
        self, model: 'DS2DSModel',
        tokenizer_encode: Callable[[str], List[int]],
        tokenizer_decode: Callable[[List[int]], str],
        link_detector: TokenizedLinkDetector,
        tokenizer_config: TokenizerConfig,
        normalizer: LinkNormalizer,
        layout_policy: DocLayoutPolicy,
    ):
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.link_detector = link_detector
        self.tokenizer_config = tokenizer_config
        self.normalizer = normalizer
        self.layout_policy = layout_policy
        self.single_gen = SingleDocumentGenerator(model)
        self.prompt_processor = PromptLinkProcessor(link_detector, tokenizer_decode)

    def generate(
        self, prompt: Optional[str] = None,
        prompt_tokens: Optional[List[int]] = None,
        corpus: Optional['DocumentCorpus'] = None,
        config: GenerationConfig = None,
    ) -> 'GenerationResult':
        if prompt_tokens is None:
            if prompt is None:
                raise ValueError("Must provide prompt or prompt_tokens")
            prompt_tokens = self.tokenizer_encode(prompt)
        elif isinstance(prompt_tokens, np.ndarray):
            prompt_tokens = prompt_tokens.tolist()

        global_budget = GlobalTokenBudget(max_total=config.max_total_new_tokens)

        context = DocumentContext(
            tokenizer_config=self.tokenizer_config,
            tokenizer_encode=self.tokenizer_encode,
            tokenizer_decode=self.tokenizer_decode,
            link_detector=self.link_detector,
            normalizer=self.normalizer,
            layout_policy=self.layout_policy,
            max_context_length=config.max_context_length,
            max_tokens_per_doc=config.max_tokens_per_document,
            max_auxiliary_documents=config.max_auxiliary_documents,
            global_token_budget=global_budget,
            device=torch.device(config.device),
        )

        context.initialize_root(prompt_tokens, title="Root Document")

        # Process prompt links
        titles_to_generate = self.prompt_processor.process_prompt(
            prompt_tokens, context, corpus, config
        )

        # Generate prompt-link docs that weren't in corpus
        if config.generate_missing_docs:
            for title in titles_to_generate:
                if not context.can_add_document(1)[0]:
                    break
                if global_budget.remaining <= 0:
                    break
                started = context.start_auxiliary_generation(
                    original_title=title, linked_from_title=context.root_title, depth=1
                )
                if started:
                    self._generate_document_recursive(
                        context, corpus, config, current_depth=1
                    )
                    context.complete_current_document()

        # Generate root with link handling
        context.switch_to_document(context.root_title)
        self._generate_document_recursive(
            context, corpus, config, current_depth=0,
            max_for_this_doc=config.max_new_tokens
        )

        return context.to_generation_result(self.tokenizer_decode, config)

    def _generate_document_recursive(
        self, context: DocumentContext,
        corpus: Optional['DocumentCorpus'],
        config: GenerationConfig,
        current_depth: int,
        max_for_this_doc: Optional[int] = None,
    ) -> int:
        """
        Generate current doc, recursively handling links.
        Returns tokens generated for THIS document (not counting recursive aux docs).

        Global token budget is enforced via context.global_token_budget (shared).
        Per-document limit is min(max_for_this_doc, config.max_tokens_per_document).
        """
        tokens_generated = 0
        doc_limit = min(
            max_for_this_doc or config.max_tokens_per_document,
            config.max_tokens_per_document
        )

        while tokens_generated < doc_limit:
            if context.global_token_budget.remaining <= 0:
                break

            # Forward pass
            tokens_tensor, real_seq_len = context.get_tokens_for_model()
            doc_spans = context.get_doc_spans()

            with torch.no_grad():
                logits = self.model.forward_inference(
                    tokens=tokens_tensor, doc_spans=doc_spans,
                    real_seq_len=real_seq_len
                )

            # Sample from last real position
            next_logits = logits[0, real_seq_len - 1, :]
            if config.temperature == 0:
                next_token = greedy_sample(next_logits)
            else:
                next_token = sample_token(
                    next_logits, config.temperature, config.top_k, config.top_p
                )

            if not context.append_token_to_current(next_token):
                break  # Per-doc, context, or global budget limit hit
            tokens_generated += 1

            if next_token == config.eos_token_id:
                break

            # Check for new links — THE KEY STEP
            # This happens IMMEDIATELY after each token. If a link is found,
            # we pause HERE and handle it BEFORE generating the next token.
            new_link_titles = context.check_for_new_links()

            for link_title in new_link_titles:
                if current_depth >= config.max_link_depth:
                    continue  # Skip, depth limit

                # Try corpus first
                if corpus is not None and corpus.has_document(link_title):
                    doc_tokens = corpus.get_document(link_title)
                    if len(doc_tokens) > config.max_tokens_per_document:
                        doc_tokens = doc_tokens[:config.max_tokens_per_document]

                    added = context.add_corpus_document(
                        original_title=link_title, tokens=doc_tokens,
                        linked_from_title=context.current_doc_title,
                        depth=current_depth + 1
                    )
                    if not added:
                        return tokens_generated  # Context full

                    # Process links in corpus doc too (if depth allows)
                    if current_depth + 1 < config.max_link_depth:
                        self._process_corpus_doc_links(
                            context, corpus, config, link_title, current_depth + 1
                        )

                elif config.generate_missing_docs and config.allow_recursive_links:
                    # IMMEDIATELY PAUSE current doc, generate aux
                    started = context.start_auxiliary_generation(
                        original_title=link_title,
                        linked_from_title=context.current_doc_title,
                        depth=current_depth + 1
                    )
                    if not started:
                        return tokens_generated  # Can't add more docs

                    # RECURSIVE generation of aux doc
                    # Global budget is shared — aux tokens reduce budget for everyone
                    self._generate_document_recursive(
                        context, corpus, config,
                        current_depth=current_depth + 1
                    )

                    # complete_current_document pops stack, resuming parent
                    context.complete_current_document()

        return tokens_generated

    def _process_corpus_doc_links(
        self, context, corpus, config, doc_title, depth
    ):
        """Process links found in a corpus document (non-recursive, just adds docs)."""
        doc = context.documents[doc_title]
        tokens_tensor = torch.tensor(doc.tokens, dtype=torch.long)
        links = self.link_detector.detect_links(tokens_tensor, self.tokenizer_decode)

        for link in links:
            target_tokens = tokens_tensor[link.target_start:link.target_end].tolist()
            title = self.tokenizer_decode(target_tokens).strip()
            if not title or title in doc.detected_links:
                continue
            doc.detected_links.add(title)

            if depth >= config.max_link_depth:
                continue

            if corpus is not None and corpus.has_document(title):
                doc_tokens = corpus.get_document(title)
                if len(doc_tokens) > config.max_tokens_per_document:
                    doc_tokens = doc_tokens[:config.max_tokens_per_document]
                context.add_corpus_document(
                    original_title=title, tokens=doc_tokens,
                    linked_from_title=doc_title, depth=depth + 1
                )
```

**Testing**:
- Test with max_link_depth=0 (should behave like single doc generation)
- Test with max_link_depth=1 (root links processed, aux links ignored)
- Test with max_link_depth=2 (nested recursion)
- Test corpus doc link processing
- Test topological order is maintained throughout
- Test context full stopping at various recursion depths
- Test generation stack correctness (proper pause/resume)
- Test global token budget properly shared across recursive calls
- Test that no tokens are generated for a doc after a link is detected and before the aux doc is processed

---

## Phase 4: Model Integration

### 4.1 Implement forward_inference in DS2DSModel (MEDIUM)
**File**: `model/model.py` (MODIFY existing)

**Dependencies**: Phase 0.1 (mask refactor)

**Description**: Implement the `forward_inference` method. Uses the same `block_mask_creator` callable pattern as training, but the callable is created with `mode='inference'` so it reads `real_seq_len` from the batch dict instead of applying T-1 shift.

**Implementation**:

Unlike training (which uses a `(**batch) -> loss` abstraction for compatibility with tunalab's training loop), inference uses explicit parameters since there's no analogous inference framework abstraction to conform to.

```python
def forward_inference(
    self, tokens: Tensor, doc_spans: Optional[List[Any]] = None,
    real_seq_len: Optional[int] = None,
) -> Tensor:
    """
    Forward pass for inference: tokens in, logits out.

    Args:
        tokens: Token IDs, shape [B, T]. T may include padding to multiple of 128.
        doc_spans: List of DocSpan objects for document-aware masking.
        real_seq_len: Actual sequence length before padding. If None, uses T.

    Returns:
        logits: Shape [B, T, V] where V is vocab size.
                Caller should extract logits at position real_seq_len - 1.
    """
    B, T = tokens.shape

    if T % 128 != 0:
        raise ValueError(f"Sequence length {T} is not a multiple of 128.")

    effective_len = real_seq_len if real_seq_len is not None else T

    # Build the inference mask directly (not via the training callable)
    block_mask = self.inference_mask_creator(
        tokens=tokens, doc_spans=doc_spans or [], real_seq_len=effective_len
    )

    # Embed -> Backbone -> Norm -> LM Head
    x = torch.nn.functional.embedding(tokens, self.embedding_weight)
    x = self.backbone(x, block_mask=block_mask)
    x = self.norm(x)
    logits = torch.nn.functional.linear(x, self.lm_head_weight)

    return logits  # [B, T, V]
```

**Note on mask creators**: `DS2DSModel` stores two mask creators:
- `self.block_mask_creator`: Training-mode callable with `(**batch) -> BlockMask` signature (used by `to_training_module()`).
- `self.inference_mask_creator`: Inference-mode callable with explicit `(tokens, doc_spans, real_seq_len) -> BlockMask` signature (used by `forward_inference()`).

Both are created from the same underlying mask logic via `make_mask_creator_callable(mask_type, mode)`. The inference callable includes the padding guard; the training callable does not (training uses packing, not padding).

**Backbone API verified**: `DS2DSBackbone.forward(x: Tensor, block_mask: BlockMask) -> Tensor` — confirmed at `model/modules/backbone.py:67`.

---

### 4.2 Implement generate in DS2DSModel (EASY)
**File**: `model/model.py` (MODIFY existing)

**Dependencies**: Phase 3, Phase 0

**Implementation**:
```python
def generate(
    self, prompt: Optional[str] = None,
    prompt_tokens: Optional[Tensor] = None,
    corpus: Optional['DocumentCorpus'] = None,
    tokenizer=None,  # tiktoken Encoding or compatible
    tokenizer_config: Optional['TokenizerConfig'] = None,
    dataset_config: Optional['DatasetConfig'] = None,
    config: Optional['GenerationConfig'] = None,
    **kwargs
) -> 'GenerationResult':
    """
    Generate text using graph-aware attention patterns.

    Args:
        prompt: Text prompt to start generation
        prompt_tokens: Pre-tokenized prompt (alternative to prompt)
        corpus: Optional DocumentCorpus for link resolution
        tokenizer: Tokenizer with encode()/decode() methods
        tokenizer_config: Token ID configuration for link detection
        dataset_config: Dataset configuration for normalization
        config: Generation configuration (sampling, limits, etc.)
        **kwargs: Override individual GenerationConfig fields
    """
    from .generation.linked_doc_generator import LinkedDocumentGenerator
    from .generation_config import GenerationConfig
    from .graph_traversal.link_detectors import MarkdownLinkDetector, PythonImportDetector
    from .tokenizer_config import TokenizerConfig
    from data.extractors.normalization import FilesafeNormalizer

    # Tokenizer setup
    if tokenizer is None:
        import tiktoken
        tokenizer = tiktoken.get_encoding("gpt2")

    if tokenizer_config is None:
        tokenizer_config = TokenizerConfig.from_tokenizer(tokenizer)

    # Normalizer from dataset config, or default
    if dataset_config is not None:
        normalizer = dataset_config.get_normalizer()
    else:
        normalizer = FilesafeNormalizer()

    # Config
    if config is None:
        config = GenerationConfig(
            max_context_length=self.backbone.max_seq_len,
            eos_token_id=getattr(tokenizer, 'eot_token', 50256),
            device=str(next(self.backbone.parameters()).device),
            **kwargs
        )

    # Link detector
    link_format = config.link_format
    if link_format == 'markdown':
        detector = MarkdownLinkDetector(tokenizer_config)
    elif link_format == 'python_import':
        detector = PythonImportDetector(tokenizer_config)
    else:
        raise ValueError(f"Unknown link_format: {link_format}")

    # Layout policy for aux doc seeding (matches training prefix)
    from data.layout import NullLayoutPolicy
    layout_policy = NullLayoutPolicy()  # TODO: load from config/checkpoint

    generator = LinkedDocumentGenerator(
        model=self, tokenizer_encode=tokenizer.encode,
        tokenizer_decode=tokenizer.decode, link_detector=detector,
        tokenizer_config=tokenizer_config, normalizer=normalizer,
        layout_policy=layout_policy,
    )

    # Normalize prompt_tokens to list (may be Tensor or np.ndarray)
    if prompt_tokens is not None:
        if isinstance(prompt_tokens, Tensor):
            prompt_tokens = prompt_tokens.cpu().numpy().tolist()
        elif isinstance(prompt_tokens, np.ndarray):
            prompt_tokens = prompt_tokens.tolist()

    return generator.generate(prompt=prompt, prompt_tokens=prompt_tokens,
                              corpus=corpus, config=config)
```

---

## Phase 5: Testing & Validation

### 5.1 Unit Tests (EASY per module)
**Files**: `tests/model/test_*.py`

**Test Modules**:
- `test_document_context.py`: Topological ordering, padding, real_seq_len, generation stack, link detection API, global token budget
- `test_single_doc_generator.py`: Basic generation, logits extraction at correct position, budget exhaustion
- `test_prompt_processor.py`: Prompt link processing with torch tensor API
- `test_linked_doc_generator.py`: Recursive generation, depth limits, corpus doc link processing, immediate-pause semantics
- `test_forward_inference.py`: Inference mask path, padding handling, all mask types

### 5.2 Integration Tests (MEDIUM)
**File**: `tests/model/test_generation_integration.py`

**Scenarios**:
- max_link_depth=0: Reduces to standard autoregressive generation
- max_link_depth=1: Root links processed, single level of aux docs
- max_link_depth=2: Nested recursion (aux doc generates link to sub-aux)
- Corpus-only: All links resolved from corpus, no generation fallback
- Mixed: Some links from corpus, some generated
- Context full: Generation stops gracefully when max_context_length reached
- Global budget: Verify max_total_new_tokens limits total generation across all docs
- Topological ordering: Verify packed sequence always has aux before root
- Tokenizer consistency: Verify encode/decode roundtrip for link targets
- Immediate-pause: Verify no root tokens exist between link detection and aux doc availability
- Python imports: Test PythonImportDetector with decoded module paths (once Phase 0.3 complete)

### 5.3 Example Script (EASY)
**File**: `examples/generate_example.py`
- CLI accepting model path, corpus path, prompt, generation params
- Displays root document and auxiliary documents
- Shows topological ordering of generated context

---

## Dependency Graph

```
PREREQUISITE (separate plan):
└── Title Pipeline (plans/title_pipeline.md) ← must complete before Phase 2

Phase 0 (Prerequisites — this plan):
├── 0.1 Refactor Mask Creators (MEDIUM)     ← no deps (ALL mask types, both modes, padding guard for inference)
├── 0.3 Python Import Detection (HARD)       ← no deps (NOT deferred)
└── 0.6 Fix GenerationConfig (EASY)          ← no deps

Phase 1 (Foundation) PARTIALLY COMPLETE:
├── DocumentCorpus (needs title_pipeline T.5)
├── GenerationResult ✅
├── Link Detectors (needs 0.3 for Python)
├── Sampling ✅
└── GenerationConfig (needs 0.6)

Phase 2 (Context Management):
└── DocumentContext (HARD) ← depends on Phase 0.1, 0.6, title_pipeline

Phase 3 (Generation Logic):
├── SingleDocumentGenerator (MEDIUM) ← depends on 2.1, 4.1
├── PromptLinkProcessor (MEDIUM) ← depends on 2.1
└── LinkedDocumentGenerator (HARD) ← depends on 3.1, 3.2, 2.1

Phase 4 (Model Integration):
├── forward_inference (MEDIUM) ← depends on 0.1
└── generate method (EASY) ← depends on Phase 3

Phase 5 (Testing):
├── Unit Tests ← depends on respective modules
├── Integration Tests ← depends on all
└── Example Script ← depends on all
```

**Critical Path**: title_pipeline → 0.1 (mask refactor) → 2.1 (DocumentContext) → 4.1 (forward_inference) → 3.1 (SingleDocGen) → 3.3 (LinkedDocGen) → 4.2 (generate) → 5.2 (integration tests)

**Parallel work**: 0.3 (Python imports), 0.6 (GenerationConfig), and title_pipeline can all proceed in parallel with 0.1.

---

## Resolved Design Decisions

### 1. Document Ordering
**Decision**: Kahn-style topological sort with linked-to docs first, root last. Handles full DAG structure (diamond dependencies, multiple parents). Matches training's `prefer_targets_first` ordering in `PackBatchSampler._order_placements` and the mask's DAG property check. Ties broken by doc_id (insertion order) for stability.

### 2. Immediate Pause on Link Detection
When the generation loop detects a new link (after appending the completing token), it **immediately pauses** the current document. No further tokens are generated for that document until the linked auxiliary document has been generated/fetched and prepended to the context. This ensures every token after a link was generated with full knowledge of the linked document. This is enforced by the generation loop structure: link checking happens inside the token-by-token loop, and any detected link triggers recursive generation before the loop continues.

### 3. Cross-Document Attention Scope
The mask grants cross-doc attention from positions after `link_end_pos` in the source doc to all positions in the target doc. Because of the immediate-pause semantics, every position after `link_end_pos` was generated with the aux doc already in context. The `link_end_pos + 1` grant start (not `link_end_pos`) maintains causality: the token that completes the link itself cannot attend to the target, but the very next token can.

### 4. max_link_depth Semantics
- `0`: Standard autoregressive generation (no link processing)
- `1`: Root's links processed (corpus lookup or generation). Aux doc links ignored.
- `2`: Root's links AND first-level aux doc links processed.
- `N`: N levels of recursive link processing.

### 5. Corpus Document Link Processing
When a corpus document is added to context, its links are also processed (if depth allows). This means adding a Wikipedia article about "Python" may also pull in "Guido van Rossum" if depth permits.

### 6. Padding and Logits Extraction
Tokens are padded to multiples of 128 for FlexAttention. `get_tokens_for_model()` returns both the padded tensor AND the real sequence length. Logits are extracted at position `real_seq_len - 1`, not at the tensor's last position. Padding positions have `document_ids == -1` and are excluded from attention via explicit guard.

### 7. Mask Callable Pattern
Training uses `(**batch) -> BlockMask` callables (to fit tunalab's "batch in, loss out" abstraction). Inference uses explicit `(tokens, doc_spans, real_seq_len) -> BlockMask` callables (no framework abstraction to conform to). Both are created from the same underlying core mask logic via `make_mask_creator_callable(mask_type, mode)`. The training callable computes `seq_len = tokens.shape[-1] - 1`. The inference callable uses the provided `real_seq_len` and includes a padding guard. `DS2DSModel` stores both (`self.block_mask_creator` for training, `self.inference_mask_creator` for inference).

### 8. Title Deduplication
If the same title is linked multiple times (within a single document or across documents), only the first occurrence triggers processing. For generated documents, tracked via `DocumentState.detected_links` set. For prompt links, tracked via a `seen_titles` set in `PromptLinkProcessor`. For cross-document deduplication, `DocumentContext.add_corpus_document` and `start_auxiliary_generation` check whether the title already exists in `self.documents` and skip if so (the existing doc is used, and any new linker just gets cross-doc attention to it via the mask).

### 9. Auxiliary Document Seeding
Reuses the training `DocLayoutPolicy` to ensure consistency between training prefixes and generation seeds. `DocumentContext` accepts a `DocLayoutPolicy` instance and calls `layout_policy.prefix_tokens(doc_id)` when starting auxiliary generation. If the policy provides no prefix (e.g., `NullLayoutPolicy`), falls back to `"# {title}\n"`. This ensures the model sees the same document framing at inference as during training.

### 10. Empty Link Targets
If model generates `[text]()`, the link detector finds it but the decoded target is empty. Skip processing.

### 11. Link Detection Frequency and Scope
Check after every token for MVP. Link detection always receives the **full token sequence of the current document** (document-local coordinates). The detector's internal logic determines when a link is "complete" (e.g., finding a matched `]()` for markdown, or a newline after an import statement for Python). The generation loop uses detected targets only for title extraction and corpus lookup. The cross-doc mask creator independently re-detects links in the full packed sequence (global coordinates) when building the attention mask. No coordinate translation between these two stages is needed.

### 12. Tokenizer Consistency
Generation must use the same tokenizer the model was trained with. The tokenizer should be loaded from training config or provided explicitly. `TokenizerConfig` should be saved alongside the trained model checkpoint.

### 13. Python Import Detection
Python import detector is refactored (Phase 0.3) to actually parse module paths from tokens, setting `uses_outgoing_titles = False`. This makes it work identically to markdown link detection for both training and inference. Module path → document title resolution is pluggable via `DatasetConfig`.

### 14. Token Budget Architecture
Two-level budget: `max_new_tokens` caps the root document, `max_total_new_tokens` caps total new tokens across all generated documents. The global budget is tracked via a shared `GlobalTokenBudget` object passed through `DocumentContext`. When exhausted, all generation stops regardless of which document is being generated.

### 15. Original Titles
The model trains on and generates original human-readable titles in link targets (e.g., `[Python](Python Programming)`). Corpus lookup normalizes these via `DatasetConfig.get_normalizer()` to find the matching entry. `DocSpan.clean_title` always contains the original title for link-to-doc matching.

---

## Implementation Status

- [ ] **Prerequisite**: Title Pipeline (`plans/title_pipeline.md`) — must complete first
  - [ ] T.1 Pretokenizer: original titles in link targets - MEDIUM
  - [ ] T.2 Collate: original title in clean_title - EASY
  - [ ] T.3 Cross-doc mask verification - EASY
  - [ ] T.4 outgoing_titles conversion - MEDIUM
  - [ ] T.5 DocumentCorpus normalizer - EASY
  - [ ] T.6 Delete title_utils.py - EASY
- [x] graph_builder.py: source_identifier now reliably written to JSONL
- [x] generation_result.py: title_utils import removed
- [x] model/__init__.py: title_utils exports removed
- [ ] Phase 0: Prerequisites (0%)
  - [ ] 0.1 Refactor Mask Creators (ALL types, both modes, padding guard for inference) - MEDIUM
  - [ ] 0.3 Python Import Detection refactor - HARD (NOT deferred)
  - [ ] 0.6 Fix GenerationConfig - EASY
- [ ] Phase 1: Foundation Components (PARTIAL)
  - [x] GenerationResult (1.3) - complete
  - [x] Sampling (1.5) - complete
  - [ ] DocumentCorpus (1.1) - needs title_pipeline T.5
  - [ ] Link Detectors (1.4) - markdown works, Python needs 0.3
  - [ ] GenerationConfig (1.6) - needs 0.6
- [ ] Phase 2: Context Management (0%)
  - [ ] DocumentContext (2.1) - HARD
- [ ] Phase 3: Generation Logic (0%)
  - [ ] SingleDocumentGenerator (3.1) - MEDIUM
  - [ ] PromptLinkProcessor (3.2) - MEDIUM
  - [ ] LinkedDocumentGenerator (3.3) - HARD
- [ ] Phase 4: Model Integration (0%)
  - [ ] forward_inference (4.1) - MEDIUM
  - [ ] generate method (4.2) - EASY
- [ ] Phase 5: Testing (0%)
  - [ ] Unit tests - EASY
  - [ ] Integration tests - MEDIUM
  - [ ] Example script - EASY

**Next Step**: Begin title pipeline (`plans/title_pipeline.md`) and Phase 0.1 (mask refactor) in parallel. Phase 0.3 (Python imports) and 0.6 (GenerationConfig) can also proceed in parallel. These unblock Phase 2.
