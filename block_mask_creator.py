"""
Block Mask Creator for FlexAttention

This module provides a registry of different attention mask strategies for use with
PyTorch's FlexAttention API. It supports both standalone visualization and integration
with the DS2DSTrainingModule.

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
   >>> model = DS2DSTrainingModule(
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

import torch
import matplotlib.pyplot as plt
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from typing import Any, Dict, List, Optional
import sys
import argparse
from pathlib import Path
import logging
import numpy as np

# Adjust path to allow imports from local experiment modules
sys.path.append(str(Path(__file__).parent))

try:
    import tiktoken
except ImportError:
    tiktoken = None

from data.pack_sampler import PackBatchSampler
from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import NullLayoutPolicy
from data.collate import build_packed_batch, DocSpan
from data.traversal import BFSStrategy

# =============================================================================
# 1. Mask Logic
# =============================================================================

def create_doc_causal_block_mask(tokens: torch.Tensor, doc_spans: List[Any], **kwargs) -> BlockMask:
    """
    Creates a FlexAttention block mask that enforces:
    1. Causal attention (can't attend to future).
    2. Document isolation (can't attend to other documents).

    Args:
        tokens: Tensor of shape [B, T] or [1, T]. The full sequence of tokens.
                NOTE: The training module splits this into input/target.
                input_ids = tokens[:, :-1]
                So the mask must correspond to length T-1.
        doc_spans: List of DocSpan objects with start, end, doc_id attributes.
        kwargs: Extra args from batch.

    Returns:
        BlockMask
    """
    device = tokens.device

    # tokens is [B, T], usually B=1 for packed sequences
    # We need the length of the inputs to the model, which is T-1
    seq_len = tokens.shape[-1] - 1

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


def create_causal_block_mask(tokens: torch.Tensor, doc_spans: List[Any], **kwargs) -> BlockMask:
    """
    Creates a standard causal block mask (no document isolation).
    Each position can attend to all previous positions, regardless of document boundaries.

    Args:
        tokens: Tensor of shape [B, T] or [1, T].
        doc_spans: List of DocSpan objects (unused, but kept for interface consistency).
        kwargs: Extra args from batch.

    Returns:
        BlockMask
    """
    device = tokens.device
    seq_len = tokens.shape[-1] - 1

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
        tokens: Tensor of shape [B, T] or [1, T].
        doc_spans: List of DocSpan objects (unused, but kept for interface consistency).
        kwargs: Extra args from batch.

    Returns:
        BlockMask
    """
    device = tokens.device
    seq_len = tokens.shape[-1] - 1

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
        tokens: Tensor of shape [B, T] or [1, T].
        doc_spans: List of DocSpan objects with start, end, doc_id attributes.
        kwargs: Extra args from batch.

    Returns:
        BlockMask
    """
    device = tokens.device
    seq_len = tokens.shape[-1] - 1

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
    'causal': create_causal_block_mask,
    'full': create_full_attention_block_mask,
    'doc_bidirectional': create_doc_bidirectional_block_mask,
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


def make_mask_creator_callable(mask_type: str):
    """
    Create a callable that can be passed to DS2DSTrainingModule.

    This is a convenience function that wraps get_mask_creator to provide
    a cleaner interface when initializing the training module.

    Args:
        mask_type: Name of the mask creator (e.g., 'doc_causal', 'causal', 'full').

    Returns:
        A callable that takes (**batch) and returns a BlockMask.

    Example:
        >>> from block_mask_creator import make_mask_creator_callable
        >>> block_mask_creator = make_mask_creator_callable('doc_causal')
        >>> model = DS2DSTrainingModule(
        ...     block_mask_creator=block_mask_creator,
        ...     vocab_size=50257,
        ...     num_layers=12,
        ...     ...
        ... )
    """
    mask_fn = get_mask_creator(mask_type)

    def callable_wrapper(**batch):
        # Extract the required arguments from batch
        tokens = batch.get('tokens')
        doc_spans = batch.get('doc_spans', [])

        if tokens is None:
            raise ValueError("Batch must contain 'tokens' key")

        return mask_fn(tokens, doc_spans, **batch)

    return callable_wrapper


# =============================================================================
# 2. Visualization & Testing System
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize FlexAttention mask for a real batch.")
    parser.add_argument("dataset_dir", type=Path,
                        help="Path to pretokenized dataset directory (REQUIRED)")
    parser.add_argument("--mask-type", type=str, default="doc_causal",
                        choices=list_mask_creators(),
                        help=f"Type of attention mask to create. Available: {', '.join(list_mask_creators())}")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for batch selection")
    parser.add_argument("--token-budget", type=int, default=16_384, help="Max tokens per batch")
    parser.add_argument("--doc-budget", type=int, default=4096, help="Max tokens per document")
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
    
    # 2. Setup Sampler
    logger.info(f"Initializing sampler with seed {args.seed}...")
    pack_sampler = PackBatchSampler(
        graph=graph_index,
        strategy_factory=lambda: BFSStrategy(edge_mode="outgoing"),
        token_budget=args.token_budget,
        doc_budget=args.doc_budget,
        seed=args.seed,
        overflow_policy="truncate",
        order_mode="prefer_targets_first"
    )
    
    # 3. Fetch Batch
    logger.info("Fetching batch...")
    try:
        placements = next(iter(pack_sampler))
    except StopIteration:
        logger.error("Sampler yielded no packs. Check budget or dataset.")
        sys.exit(1)
        
    batch = build_packed_batch(
        graph=graph_index,
        backend=backend,
        layout=NullLayoutPolicy(),
        placements=placements,
        as_2d=True
    )
    
    if 'tokens' not in batch and 'input_ids' in batch:
        batch['tokens'] = batch['input_ids']
        
    tokens = batch['tokens']
    doc_spans = batch['doc_spans']
    
    logger.info(f"Batch generated. Tokens shape: {tokens.shape}")
    doc_titles = [s.title for s in doc_spans]
    logger.info(f"Docs in batch ({len(doc_titles)}): {doc_titles}")

    # 4. Create Mask
    mask_creator_fn = get_mask_creator(args.mask_type)
    block_mask = mask_creator_fn(tokens, doc_spans)
    logger.info(f"Block mask created using '{args.mask_type}' strategy.")

    # 5. Visualization
    input_len = tokens.shape[-1] - 1

    # Reconstruct the dense mask by re-applying the mask logic
    # This is generic and works for any mask type
    doc_map = torch.full((input_len,), -1, dtype=torch.int32)
    for span in doc_spans:
        s, e = max(0, span.start), min(input_len, span.end)
        if s < e:
            doc_map[s:e] = span.doc_id

    # Generate dense mask based on the selected mask type
    # We reconstruct the logic here for visualization purposes
    q_indices = torch.arange(input_len).unsqueeze(1)  # [T, 1]
    k_indices = torch.arange(input_len).unsqueeze(0)  # [1, T]

    if args.mask_type == 'doc_causal':
        # Causal + document isolation
        causal_mask = q_indices >= k_indices
        same_doc_mask = doc_map.unsqueeze(1) == doc_map.unsqueeze(0)
        dense_mask = causal_mask & same_doc_mask
    elif args.mask_type == 'causal':
        # Just causal
        dense_mask = q_indices >= k_indices
    elif args.mask_type == 'full':
        # Full attention
        dense_mask = torch.ones((input_len, input_len), dtype=torch.bool)
    elif args.mask_type == 'doc_bidirectional':
        # Same document only (bidirectional within docs)
        dense_mask = doc_map.unsqueeze(1) == doc_map.unsqueeze(0)
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
            plt.text(mid, -1, span.title[:15], ha='center', rotation=45, color='red', fontsize=8)
            plt.text(-1, mid, span.title[:15], va='center', color='red', fontsize=8)

    valid_bounds = sorted(list(set([b for b in boundaries if 0 <= b <= input_len])))
    for b in valid_bounds:
        plt.axhline(y=b - 0.5, color='blue', linestyle='--', linewidth=0.5)
        plt.axvline(x=b - 0.5, color='blue', linestyle='--', linewidth=0.5)

    plt.title(f"FlexAttention Mask: {args.mask_type} (Seed={args.seed})")
    plt.tight_layout()
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(this_dir, 'artifacts')
    output_img = os.path.join(artifacts_dir, f"mask_viz_{args.mask_type}_seed{args.seed}.png")
    plt.savefig(output_img)
    logger.info(f"Saved visualization to {output_img}")

    # 6. Dump Batch Info
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
            f.write(f"--- Document {i}: {span.title} (ID: {span.doc_id}) ---\n")
            f.write(f"Span: [{span.start}, {span.end})\n")
            f.write(f"Length: {span.end - span.start}\n")
            f.write(f"Truncated: {span.truncated}\n")
            f.write(f"Outgoing Links: {span.outgoing_titles}\n")
            
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
