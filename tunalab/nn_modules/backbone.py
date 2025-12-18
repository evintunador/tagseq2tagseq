from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from tunalab.modules.channel_mixing.fp8_linear import is_hopper_available
from .layer import Layer


class DS2DSBackbone(nn.Module):
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

    def forward(self, x: Tensor, block_mask: Any):
        """
        Args:
            x: Input embeddings of shape (B, T, C) or (Total_Tokens, C)
            block_mask: The attention block mask (passed in, not created here)
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
