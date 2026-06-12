"""
Block Mask Creator for FlexAttention

This module provides a registry of different attention mask strategies for use with
PyTorch's FlexAttention API. It supports both standalone visualization and integration
with the TS2TSTrainingModule.

Available Mask Types:
--------------------
- 'doc_causal': Causal attention with document isolation (default)
    Each position can only attend to previous positions in the same document.

- 'causal': Standard causal attention without document boundaries
    Each position can attend to all previous positions, across documents.

- 'full': Full bidirectional attention
    Each position can attend to all positions in the sequence.

- 'doc_bidirectional': Bidirectional attention within documents
    Each position can attend to all positions in its document, but not across boundaries.

Usage Examples:
--------------

1. As a standalone script (visualization):
   $ python block_mask_creator.py /path/to/dataset --mask-type doc_causal --seed 42

2. Importing in training code:
   >>> from block_mask_creator import make_mask_creator_callable
   >>> block_mask_creator = make_mask_creator_callable('doc_causal')
   >>> model = TS2TSTrainingModule(
   ...     block_mask_creator=block_mask_creator,
   ...     vocab_size=50257,
   ...     ...
   ... )

3. Direct function access:
   >>> from block_mask_creator import get_mask_creator
   >>> mask_fn = get_mask_creator('causal')
   >>> mask = mask_fn(tokens, doc_spans)

4. Listing available masks:
   >>> from block_mask_creator import list_mask_creators
   >>> print(list_mask_creators())
   ['doc_causal', 'causal', 'full', 'doc_bidirectional']

Adding New Mask Types:
---------------------
1. Define a function with signature: (tokens, doc_spans, **kwargs) -> BlockMask
2. Add it to the MASK_CREATORS dictionary
3. Optionally add visualization logic in the __main__ section
"""

from typing import Any, List
import sys
import argparse
from pathlib import Path
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
try:
    import tiktoken
except ImportError:
    tiktoken = None

from data.pack_sampler import PackBatchSampler
from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import NullLayoutPolicy, make_layout_policy
from data.collate import build_packed_batch
from data.traversal import (
    BFSStrategy,
    DFSStrategy,
    RandomWalkStrategy,
    RandomSelectionStrategy,
    CompositeTraversalStrategy
)
from .cross_doc_mask import CrossDocLinkMaskCreator, DocCausalTritonMaskInputs

# =============================================================================
# 1. Mask Logic
# =============================================================================

def create_doc_causal_block_mask(tokens: torch.Tensor, doc_spans: List[Any], **kwargs) -> BlockMask:
    """
    Creates a FlexAttention block mask that enforces:
    1. Causal attention (can't attend to future).
    2. Document isolation (can't attend to other documents).

    Args:
        tokens: Tensor of shape [B, T] — the token sequence to build the mask for.
        doc_spans: List of DocSpan objects with start, end, doc_id attributes.
        kwargs: Extra args from batch.

    Returns:
        BlockMask
    """
    device = tokens.device
    seq_len = tokens.shape[-1]

    # Construct a tensor mapping each position to its doc_id
    # Initialize with -1 (or unique negative values) to represent "no document" / padding / layout
    # We use int32 for the document IDs
    document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)

    for span in doc_spans:
        # Clip start/end to valid range of input_ids
        # span.start and span.end are indices into the FULL tokens tensor
        # We are masking for tokens[:-1]
        start = max(0, span.start)
        end = min(seq_len, span.end)

        if start < end:
            document_ids[start:end] = span.doc_id

    # Define the score_mod / mask_mod function
    # This function is captured by create_block_mask and compiled
    def doc_causal_mod(b, h, q_idx, kv_idx):
        # Causal mask: query can attend to key if q_idx >= kv_idx
        causal = q_idx >= kv_idx

        # Document mask: query and key must belong to the same document
        # Note: we access the captured document_ids tensor
        same_doc = document_ids[q_idx] == document_ids[kv_idx]

        return causal & same_doc

    # Create the block mask
    block_mask = create_block_mask(
        doc_causal_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device
    )

    return block_mask


def create_doc_causal_triton_mask(
    tokens: torch.Tensor, doc_spans: List[Any], **kwargs
) -> DocCausalTritonMaskInputs:
    """Doc-causal mask for varlen_bim_v1 Triton kernel.

    Returns a DocCausalTritonMaskInputs instead of a FlexAttention BlockMask.
    Use with attention_backend="varlen_bim_v1".
    """
    device = tokens.device
    seq_len = tokens.shape[-1]
    document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    for span in doc_spans:
        start = max(0, span.start)
        end = min(seq_len, span.end)
        if start < end:
            document_ids[start:end] = span.doc_id
    return DocCausalTritonMaskInputs(document_ids=document_ids)


def _build_component_document_ids(
    tokens: torch.Tensor, doc_spans: List[Any]
) -> torch.Tensor:
    """Build a [T] int32 document_ids tensor keyed by ``component_id``.

    Documents sharing a ``component_id`` (a connected sub-graph of the pack) are
    assigned the SAME id, so the doc-causal kernel — which merges positions that
    lie in the same contiguous run of equal ids — treats the whole component as
    one causally-concatenated super-document.

    This relies on each component occupying a single contiguous run in the pack
    (guaranteed by ``PackBatchSampler._order_placements``). We assert that
    invariant here and fail loud rather than silently building a malformed mask
    if a future ordering change interleaves components.
    """
    device = tokens.device
    seq_len = tokens.shape[-1]
    document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)

    for span in doc_spans:
        start = max(0, span.start)
        end = min(seq_len, span.end)
        if start < end:
            # Fall back to doc_id when a span carries no component assignment
            # (component_id == -1), making it its own singleton component.
            comp = getattr(span, "component_id", -1)
            document_ids[start:end] = comp if comp >= 0 else span.doc_id

    # Contiguity check: every distinct component id (ignoring -1 gaps) must form
    # exactly one contiguous run. Count runs per id and assert at most one.
    ids_cpu = document_ids.detach().to("cpu")
    if seq_len > 0:
        ids_list = ids_cpu.tolist()
        runs_seen: dict = {}
        prev = None
        for cid in ids_list:
            if cid != prev:
                if cid != -1:
                    runs_seen[cid] = runs_seen.get(cid, 0) + 1
                prev = cid
        offenders = {cid: n for cid, n in runs_seen.items() if n > 1}
        if offenders:
            raise AssertionError(
                "doc_concatenated requires each component to be one contiguous "
                f"run, but these component ids are split across multiple runs: "
                f"{offenders}. doc_spans order: "
                f"{[(s.doc_id, getattr(s, 'component_id', -1), s.start, s.end) for s in doc_spans]}"
            )

    return document_ids


def create_doc_concat_triton_mask(
    tokens: torch.Tensor, doc_spans: List[Any], **kwargs
) -> DocCausalTritonMaskInputs:
    """doc_concatenated mask for the varlen_bim_v2 Triton kernel.

    Identical to ``create_doc_causal_triton_mask`` except positions are labelled
    by ``component_id`` instead of ``doc_id``, so a whole connected component is
    merged into one causally-concatenated super-document.
    """
    document_ids = _build_component_document_ids(tokens, doc_spans)
    return DocCausalTritonMaskInputs(document_ids=document_ids)


def create_doc_concat_block_mask(
    tokens: torch.Tensor, doc_spans: List[Any], **kwargs
) -> BlockMask:
    """doc_concatenated mask as a FlexAttention BlockMask (inference/eval/viz).

    Causal + component isolation: a position attends to earlier positions in the
    same connected component (the flex twin of ``create_doc_concat_triton_mask``).
    """
    device = tokens.device
    seq_len = tokens.shape[-1]
    document_ids = _build_component_document_ids(tokens, doc_spans)

    def doc_concat_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_component = document_ids[q_idx] == document_ids[kv_idx]
        return causal & same_component

    return create_block_mask(
        doc_concat_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )


def create_causal_block_mask(tokens: torch.Tensor, doc_spans: List[Any], **kwargs) -> BlockMask:
    """
    Creates a standard causal block mask (no document isolation).
    Each position can attend to all previous positions, regardless of document boundaries.

    Args:
        tokens: Tensor of shape [B, T] — the token sequence to build the mask for.
        doc_spans: List of DocSpan objects (unused, but kept for interface consistency).
        kwargs: Extra args from batch.

    Returns:
        BlockMask
    """
    device = tokens.device
    seq_len = tokens.shape[-1]

    def causal_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    block_mask = create_block_mask(
        causal_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device
    )

    return block_mask


def create_full_attention_block_mask(tokens: torch.Tensor, doc_spans: List[Any], **kwargs) -> BlockMask:
    """
    Creates a full bidirectional attention mask (no restrictions).
    Each position can attend to all positions in the sequence.
    Useful for debugging or prefix-LM style training.

    Args:
        tokens: Tensor of shape [B, T] — the token sequence to build the mask for.
        doc_spans: List of DocSpan objects (unused, but kept for interface consistency).
        kwargs: Extra args from batch.

    Returns:
        BlockMask
    """
    device = tokens.device
    seq_len = tokens.shape[-1]

    def full_mod(b, h, q_idx, kv_idx):
        return True

    block_mask = create_block_mask(
        full_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device
    )

    return block_mask


def create_doc_bidirectional_block_mask(tokens: torch.Tensor, doc_spans: List[Any], **kwargs) -> BlockMask:
    """
    Creates a block mask with bidirectional attention within documents.
    Each position can attend to all positions in the same document (past and future),
    but cannot attend across document boundaries.

    Args:
        tokens: Tensor of shape [B, T] — the token sequence to build the mask for.
        doc_spans: List of DocSpan objects with start, end, doc_id attributes.
        kwargs: Extra args from batch.

    Returns:
        BlockMask
    """
    device = tokens.device
    seq_len = tokens.shape[-1]

    document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)

    for span in doc_spans:
        start = max(0, span.start)
        end = min(seq_len, span.end)
        if start < end:
            document_ids[start:end] = span.doc_id

    def doc_bidirectional_mod(b, h, q_idx, kv_idx):
        # Can attend within the same document
        return document_ids[q_idx] == document_ids[kv_idx]

    block_mask = create_block_mask(
        doc_bidirectional_mod,
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device
    )

    return block_mask


# =============================================================================
# Registry System
# =============================================================================

MASK_CREATORS = {
    'doc_causal': create_doc_causal_block_mask,
    'doc_concatenated': create_doc_concat_block_mask,
    'causal': create_causal_block_mask,
    'full': create_full_attention_block_mask,
    'doc_bidirectional': create_doc_bidirectional_block_mask,
    # 'cross_doc_link' is intentionally absent: it requires a dataset-specific
    # LinkDetector and cannot be constructed from a name alone. Use
    # make_mask_creator_callable_from(CrossDocLinkMaskCreator(link_detector=...))
    # to build the callable, then pass it directly to TS2TSTrainingModule.
}


def get_mask_creator(name: str):
    """
    Retrieve a mask creator function by name.

    Args:
        name: Name of the mask creator. Must be one of the keys in MASK_CREATORS.

    Returns:
        Callable that takes (tokens, doc_spans, **kwargs) and returns a BlockMask.

    Raises:
        ValueError: If the name is not found in the registry.
    """
    if name not in MASK_CREATORS:
        available = ', '.join(MASK_CREATORS.keys())
        raise ValueError(f"Unknown mask creator '{name}'. Available options: {available}")
    return MASK_CREATORS[name]


def list_mask_creators() -> List[str]:
    """Return a list of all available mask creator names."""
    return list(MASK_CREATORS.keys())


def make_mask_creator_callable_from(creator):
    """
    Wrap any mask creator callable into the **batch interface for TS2TSTrainingModule.

    Use this when you need to control construction — for example, to pick which
    LinkDetector a CrossDocLinkMaskCreator uses:

        detector = MarkdownLinkDetector(decode_fn=tokenizer.decode)
        creator  = CrossDocLinkMaskCreator(link_detector=detector)
        block_mask_creator = make_mask_creator_callable_from(creator)
        training_module = TS2TSTrainingModule(..., block_mask_creator=block_mask_creator)

    Args:
        creator: Any callable with signature (tokens, doc_spans, **kwargs) -> BlockMask.

    Returns:
        A callable with signature (**batch) -> BlockMask.
    """
    # Disable dynamo tracing so create_block_mask always runs eagerly and
    # returns a real BlockMask (with a compiled .graph), not a traced proxy.
    @torch._dynamo.disable
    def callable_wrapper(**batch):
        tokens = batch.get('tokens')
        doc_spans = batch.get('doc_spans', [])
        if tokens is None:
            raise ValueError("Batch must contain 'tokens' key")
        extra = {k: v for k, v in batch.items() if k not in ('tokens', 'doc_spans')}
        return creator(tokens, doc_spans, **extra)

    return callable_wrapper


def make_mask_creator_callable(mask_type: str):
    """
    Create a **batch callable for a named mask type (e.g. 'doc_causal').

    For cross_doc_link, construct CrossDocLinkMaskCreator with the appropriate
    LinkDetector yourself and use make_mask_creator_callable_from() instead.

    Args:
        mask_type: One of the keys in MASK_CREATORS ('doc_causal', 'causal',
                   'full', 'doc_bidirectional').

    Returns:
        A callable with signature (**batch) -> BlockMask.
    """
    return make_mask_creator_callable_from(get_mask_creator(mask_type))


# =============================================================================
# 2. Visualization & Testing System
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize FlexAttention mask for a real batch.")
    parser.add_argument("dataset_dir", type=Path,
                        help="Path to pretokenized dataset directory (REQUIRED)")
    _viz_mask_types = list_mask_creators() + ['cross_doc_link', 'doc_concat_link']
    parser.add_argument("--mask-type", type=str, default="doc_causal",
                        choices=_viz_mask_types,
                        help=f"Type of attention mask to create. Available: {', '.join(_viz_mask_types)}")
    parser.add_argument("--strategy", type=str, default="bfs",
                        choices=['bfs', 'dfs', 'random_walk', 'random'],
                        help="Graph traversal strategy (bfs=breadth-first, dfs=depth-first, random_walk=Markov walk, random=uniform random)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for batch selection")
    parser.add_argument("--token-budget", type=int, default=16_384, help="Max tokens per batch")
    parser.add_argument("--doc-budget", type=int, default=4096, help="Max tokens per document")
    parser.add_argument("--link-detector", type=str, default="markdown",
                        choices=["markdown", "python"],
                        help="Link detector for cross_doc_link: 'markdown' (Wikipedia) or 'python' (TheStack)")
    parser.add_argument("--layout-policy", type=str, default="stochastic_identifier_prefix",
                        choices=["null", "eos", "identifier_prefix",
                                 "identifier_prefix_eos", "stochastic_identifier_prefix"],
                        help="Layout policy for packing. Defaults to the training policy "
                             "'stochastic_identifier_prefix' so link targets resolve to "
                             "co-packed docs (cross_doc_link / doc_concat_link grants only "
                             "fire when the identifier prefix is emitted into the tokens).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not args.dataset_dir.exists():
        logger.error(f"Dataset directory {args.dataset_dir} does not exist. Please run pretokenization.")
        sys.exit(1)

    # 1. Init Data Components
    logger.info(f"Loading dataset from {args.dataset_dir}...")
    graph_index = GraphIndex(args.dataset_dir)
    backend = PretokShardedBackend(graph_index)
    
    # 2. Setup layout policy. The identifier-prefix policies emit each doc's
    # identifier into its tokens, which is what lets the link detector resolve a
    # link's target to a co-packed doc — without it cross_doc_link /
    # doc_concat_link grants almost never fire. Defaults to the training policy.
    _enc = tiktoken.get_encoding('gpt2') if tiktoken is not None else None
    layout_policy = make_layout_policy(
        args.layout_policy,
        encode_fn=(_enc.encode_ordinary if _enc is not None else None),
    )

    # 3. Setup Sampler
    logger.info(f"Initializing sampler with seed {args.seed} and strategy {args.strategy}...")

    # Map strategy name to factory function
    strategy_factories = {
        'bfs': lambda: BFSStrategy(edge_mode="outgoing"),
        'dfs': lambda: DFSStrategy(edge_mode="outgoing"),
        'random_walk': lambda: RandomWalkStrategy(restart_prob=0.15, edge_mode="outgoing"),
        'random': lambda: RandomSelectionStrategy(),
    }

    pack_sampler = PackBatchSampler(
        graph=graph_index,
        strategy_factory=strategy_factories[args.strategy],
        token_budget=args.token_budget,
        doc_budget=args.doc_budget,
        seed=args.seed,
        overflow_policy="truncate",
        order_mode="prefer_targets_first",
        layout_policy=layout_policy,
    )

    # 4. Fetch Batch
    logger.info("Fetching batch...")
    try:
        placements = next(iter(pack_sampler))
    except StopIteration:
        logger.error("Sampler yielded no packs. Check budget or dataset.")
        sys.exit(1)

    batch = build_packed_batch(
        graph=graph_index,
        backend=backend,
        layout=layout_policy,
        placements=placements,
        as_2d=True
    )
    
    raw_tokens = batch['tokens']   # [B, T+1] from the collator
    doc_spans = batch['doc_spans']
    tokens = raw_tokens[:, :-1]    # [B, T] — what the model actually sees

    logger.info(f"Batch generated. Tokens shape: {tokens.shape}")
    doc_identifiers = [s.normed_identifier for s in doc_spans]
    logger.info(f"Docs in batch ({len(doc_identifiers)}): {doc_identifiers}")

    # 5. Create Mask
    cross_doc_creator = None
    if args.mask_type in ('cross_doc_link', 'doc_concat_link'):
        if tiktoken is None:
            raise ImportError(f"tiktoken is required for {args.mask_type} visualization. Install with: pip install tiktoken")
        enc = tiktoken.get_encoding('gpt2')
        if args.link_detector == 'python':
            from model.graph_traversal.python_import_detector import PythonImportDetector
            detector = PythonImportDetector(decode_fn=enc.decode)
        else:
            from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
            detector = MarkdownLinkDetector(decode_fn=enc.decode)
        # doc_concat_link: whole-doc grants (full concatenation, no link gate).
        cross_doc_creator = CrossDocLinkMaskCreator(
            link_detector=detector,
            whole_doc_grant=(args.mask_type == 'doc_concat_link'),
        )
        block_mask = cross_doc_creator(tokens, doc_spans)
    else:
        block_mask = get_mask_creator(args.mask_type)(tokens, doc_spans)
    logger.info(f"Block mask created using '{args.mask_type}' strategy.")

    # 6. Visualization
    input_len = tokens.shape[-1]

    # Reconstruct the dense mask by re-applying the mask logic
    # This is generic and works for any mask type
    doc_map = torch.full((input_len,), -1, dtype=torch.int32)
    comp_map = torch.full((input_len,), -1, dtype=torch.int32)
    for span in doc_spans:
        s, e = max(0, span.start), min(input_len, span.end)
        if s < e:
            doc_map[s:e] = span.doc_id
            comp = getattr(span, "component_id", -1)
            comp_map[s:e] = comp if comp >= 0 else span.doc_id

    # Generate dense mask based on the selected mask type
    # We reconstruct the logic here for visualization purposes
    q_indices = torch.arange(input_len).unsqueeze(1)  # [T, 1]
    k_indices = torch.arange(input_len).unsqueeze(0)  # [1, T]

    if args.mask_type == 'doc_causal':
        # Causal + document isolation
        causal_mask = q_indices >= k_indices
        same_doc_mask = doc_map.unsqueeze(1) == doc_map.unsqueeze(0)
        dense_mask = causal_mask & same_doc_mask
    elif args.mask_type == 'doc_concatenated':
        # Causal + component isolation (connected docs merged into a super-doc)
        causal_mask = q_indices >= k_indices
        same_component = comp_map.unsqueeze(1) == comp_map.unsqueeze(0)
        dense_mask = causal_mask & same_component
    elif args.mask_type == 'causal':
        # Just causal
        dense_mask = q_indices >= k_indices
    elif args.mask_type == 'full':
        # Full attention
        dense_mask = torch.ones((input_len, input_len), dtype=torch.bool)
    elif args.mask_type == 'doc_bidirectional':
        # Same document only (bidirectional within docs)
        dense_mask = doc_map.unsqueeze(1) == doc_map.unsqueeze(0)
    elif args.mask_type in ('cross_doc_link', 'doc_concat_link'):
        dense_mask = cross_doc_creator.build_dense_mask_for_visualization(
            tokens, doc_spans, device=torch.device('cpu')
        )
    else:
        # Fallback: try to reconstruct generically (might not match all custom masks)
        logger.warning(f"No specific visualization logic for mask type '{args.mask_type}'. Using full attention as fallback.")
        dense_mask = torch.ones((input_len, input_len), dtype=torch.bool)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(dense_mask.numpy(), cmap='Greys', interpolation='nearest', origin='upper')
    
    # Draw boundaries
    boundaries = []
    for span in doc_spans:
        boundaries.append(span.start)
        boundaries.append(span.end)
        # Label
        mid = (max(0, span.start) + min(input_len, span.end)) / 2
        if 0 <= mid < input_len:
            plt.text(mid, -1, span.normed_identifier[:15], ha='center', rotation=45, color='red', fontsize=8)
            plt.text(-1, mid, span.normed_identifier[:15], va='center', color='red', fontsize=8)

    valid_bounds = sorted(list(set([b for b in boundaries if 0 <= b <= input_len])))
    for b in valid_bounds:
        plt.axhline(y=b - 0.5, color='blue', linestyle='--', linewidth=0.5)
        plt.axvline(x=b - 0.5, color='blue', linestyle='--', linewidth=0.5)

    plt.title(f"FlexAttention Mask: {args.mask_type} (Seed={args.seed})")
    plt.tight_layout()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(this_dir, 'artifacts')
    output_img = os.path.join(artifacts_dir, f"mask_viz_{args.mask_type}_seed{args.seed}.png")
    plt.savefig(output_img)
    logger.info(f"Saved visualization to {output_img}")

    # 7. Dump Batch Info
    # Initialize decoder
    enc = None
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("gpt2")
        except:
            pass

    output_txt = os.path.join(artifacts_dir, f"batch_info_{args.mask_type}_seed{args.seed}.txt")
    with open(output_txt, "w") as f:
        f.write(f"Mask Type: {args.mask_type}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Batch Tokens Shape: {tokens.shape}\n")
        f.write(f"Number of Docs: {len(doc_spans)}\n\n")
        
        for i, span in enumerate(doc_spans):
            f.write(f"--- Document {i}: {span.normed_identifier} (ID: {span.doc_id}) ---\n")
            f.write(f"Span: [{span.start}, {span.end})\n")
            f.write(f"Length: {span.end - span.start}\n")
            f.write(f"Truncated: {span.truncated}\n")
            f.write(f"Outgoing Links: {span.outgoing_identifiers}\n")
            
            # Decode text
            if enc:
                doc_tokens_list = tokens[0, span.start:span.end].tolist()
                try:
                    text = enc.decode(doc_tokens_list)
                    preview = text[:100].replace('\n', '\\n')
                    f.write(f"Text Preview: {preview}...\n")
                except Exception as e:
                    f.write(f"Decoding failed: {e}\n")
            f.write("\n")
            
        f.write("\nFull Token Sequence:\n")
        f.write(str(tokens.tolist()))

    logger.info(f"Saved batch info to {output_txt}")
    
    # Cleanup
    backend.close()
