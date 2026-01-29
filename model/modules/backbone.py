from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .layer import Layer


class DS2DSBackbone(nn.Module):
    """
    The "spine" of the DS2DS transformer architecture.
    
    This module contains only the core transformer layers with skip connections,
    intentionally excluding the embedding and output head. This design follows
    the "backbone as spine" metaphor, facilitating:
    - Separation of concerns between architecture and I/O
    - Easier multimodal extensions with different input/output layers
    - Cleaner weight sharing between training and inference modes
    
    The backbone expects pre-embedded inputs and returns hidden states that
    can be processed by a normalization layer and output head.
    
    Architecture details:
    - Stack of transformer layers with RMS normalization and GLU channel mixing
    - Skip connections from first half to second half of layers
    - FlexAttention with custom block masks for graph-aware attention patterns
    
    Args:
        num_layers: Number of transformer layers in the stack
        model_dim: Hidden dimension size (d_model)
        num_heads: Number of attention heads per layer
        max_seq_len: Maximum sequence length supported
        dropout: Dropout probability for channel mixing
        drop_path_rate: Stochastic depth probability for regularization
        fp8: Whether to use FP8 precision for linear projections
        **kwargs: Additional arguments (ignored with optional warning)
    """
    def __init__(
        self,
        num_layers: int, 
        model_dim: int, 
        num_heads: int,
        max_seq_len: int, 
        dropout: float,
        drop_path_rate: float,
        fp8: bool = False,
        **kwargs
    ):
        # if kwargs:
        #     print(f"[{self.__class__.__name__}] Unused kwargs passed to constructor: {kwargs}")
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        
        self.layers = nn.ModuleList([Layer(
            n_embd=model_dim, 
            n_head=num_heads, 
            max_seq_len=max_seq_len,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
        ) for _ in range(num_layers)])
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))

    def forward(self, x: Tensor, block_mask: Any) -> Tensor:
        """
        Forward pass through the transformer backbone.
        
        This method applies the full stack of transformer layers with skip connections
        but does NOT handle embedding, normalization, or output projection. Those are
        the responsibility of the surrounding TrainingModule or Model.
        
        Args:
            x: Pre-embedded input tensor of shape (B, T, C) where:
               - B is batch size (typically 1 for packed sequences)
               - T is sequence length
               - C is model_dim
            block_mask: FlexAttention BlockMask for custom attention patterns.
                       Must be created externally with proper document boundaries
                       and graph structure.
        
        Returns:
            Hidden states of shape (B, T, C) after all transformer layers.
        """
        skip_connections = []
        n_skip = len(self.skip_weights)
        
        for i, layer in enumerate(self.layers):
            if i >= n_skip:
                 # Pop skip connection
                 if skip_connections:
                    x = x + self.skip_weights[i - n_skip] * skip_connections.pop()
            
            x = layer(x, block_mask)
            
            if i < n_skip:
                skip_connections.append(x)
                
        return x
