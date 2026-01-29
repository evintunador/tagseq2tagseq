"""
Token sampling utilities for DS2DS generation.

Provides functions for sampling next tokens from model logits using various
sampling strategies (greedy, temperature, top-k, nucleus/top-p).
"""
from typing import Optional

import torch
import torch.nn.functional as F


def greedy_sample(logits: torch.Tensor) -> int:
    """
    Sample token greedily by taking the argmax.
    
    Args:
        logits: Logits tensor of shape [vocab_size] or [1, vocab_size]
        
    Returns:
        Token ID as an integer
    """
    # Handle both [vocab_size] and [1, vocab_size] shapes
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    
    token_id = torch.argmax(logits, dim=-1).item()
    return int(token_id)


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> int:
    """
    Sample a token from logits using temperature scaling and optional filtering.
    
    Applies the following transformations in order:
    1. Temperature scaling
    2. Top-k filtering (if specified)
    3. Nucleus (top-p) sampling (if specified)
    4. Multinomial sampling from resulting distribution
    
    Args:
        logits: Logits tensor of shape [vocab_size] or [1, vocab_size]
        temperature: Temperature for scaling logits. Use 0.0 for greedy sampling.
        top_k: If specified, only sample from top k tokens
        top_p: If specified, sample from smallest set of tokens whose cumulative
               probability mass exceeds p (nucleus sampling)
               
    Returns:
        Token ID as an integer
        
    Examples:
        >>> logits = torch.randn(50257)  # GPT-2 vocab size
        >>> token = sample_token(logits, temperature=0.8, top_k=50)
    """
    # Handle both [vocab_size] and [1, vocab_size] shapes
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    
    # Temperature = 0 means greedy sampling
    if temperature == 0.0:
        return greedy_sample(logits)
    
    # Apply temperature scaling
    logits = logits / temperature
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        # Get top-k logits and their indices
        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
        
        # Create a mask of -inf for all positions not in top-k
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits[top_k_indices] = top_k_logits
        logits = filtered_logits
    
    # Apply top-p (nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Compute cumulative probabilities
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find the cutoff: first position where cumsum > p
        # We want to keep tokens up to and including this position
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Keep at least one token (the first one)
        sorted_indices_to_remove[0] = False
        
        # Shift right by one to keep the first token that exceeds p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        # Create mask in original order
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Handle edge case: all probs are 0 (or all logits were -inf)
    if torch.all(probs == 0) or torch.any(torch.isnan(probs)):
        # Fall back to uniform sampling over non-inf logits
        valid_mask = ~torch.isinf(logits)
        if torch.any(valid_mask):
            uniform_probs = valid_mask.float()
            uniform_probs = uniform_probs / uniform_probs.sum()
            token_id = torch.multinomial(uniform_probs, num_samples=1).item()
        else:
            # All logits are -inf, return 0 as fallback
            token_id = 0
    else:
        # Sample from the distribution
        token_id = torch.multinomial(probs, num_samples=1).item()
    
    return int(token_id)
