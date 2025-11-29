
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


# =============================================================================
# 2. Visualization & Testing System
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize FlexAttention mask for a real batch.")
    parser.add_argument("dataset_dir", type=Path, 
                        help="Path to pretokenized dataset directory (REQUIRED)")
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
    block_mask = create_doc_causal_block_mask(tokens, doc_spans)
    logger.info("Block mask created.")

    # 5. Visualization
    input_len = tokens.shape[-1] - 1
    dense_mask = torch.zeros((input_len, input_len), dtype=torch.bool)
    
    # Helper to reconstruct doc_ids map for visualization
    doc_map = torch.full((input_len,), -1, dtype=torch.int32)
    for span in doc_spans:
        s, e = max(0, span.start), min(input_len, span.end)
        if s < e:
            doc_map[s:e] = span.doc_id
            
    # Vectorized mask creation to avoid slow loops
    # Indices
    q_indices = torch.arange(input_len).unsqueeze(1)  # [T, 1]
    k_indices = torch.arange(input_len).unsqueeze(0)  # [1, T]
    
    # Causal mask
    causal_mask = q_indices >= k_indices
    
    # Doc isolation mask
    # doc_map is [T]
    same_doc_mask = doc_map.unsqueeze(1) == doc_map.unsqueeze(0)
    
    dense_mask = causal_mask & same_doc_mask

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
        
    plt.title(f"FlexAttention Mask (Real Batch, Seed={args.seed})")
    plt.tight_layout()
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(this_dir, 'artifacts')
    output_img = os.path.join(artifacts_dir, f"mask_viz_seed{args.seed}.png")
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

    output_txt = os.path.join(artifacts_dir, f"batch_info_seed{args.seed}.txt")
    with open(output_txt, "w") as f:
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
