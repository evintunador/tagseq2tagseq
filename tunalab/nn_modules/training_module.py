from typing import Dict, Any, Type, Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from tunalab.modules.norms.rms_norm import RMSNorm
from tunalab.modules.losses.fused_cross_entropy import FusedLinearCELoss
from .backbone import DS2DSBackbone


class DS2DSTrainingModule(nn.Module):
    def __init__(
        self,
        block_mask_creator: Callable[[], BlockMask],
        vocab_size: int,
        num_layers: int, 
        model_dim: int, 
        num_heads: int,
        max_seq_len: int, 
        dropout: float,
        drop_path_rate: float,
        fp8: bool = False,
        weight_tying: bool = True,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.weight_tying = weight_tying

        self.block_mask_creator = block_mask_creator
        
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.backbone = DS2DSBackbone(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
        )
        self.norm = RMSNorm(model_dim)
        loss_weight = self.embedding.weight if weight_tying else None
        self.loss_fn = FusedLinearCELoss(
            D=model_dim,
            V=vocab_size,
            dtype=dtype,
            ignore_index=ignore_index,
            weight=loss_weight
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Type[Tensor] | Any]:
        tokens = batch['tokens']
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]

        block_mask = self.block_mask_creator(**batch)
        
        x = self.embedding(input_ids)
        x = self.backbone(x, block_mask=block_mask)
        x = self.norm(x)
        
        loss = self.loss_fn(x, target_ids)
        
        return {'loss': loss, 'ce_loss': loss}
