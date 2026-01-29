from typing import Dict, Any, Type, Callable, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from tunalab.modules.norms.rms_norm import RMSNorm
from tunalab.modules.losses.fused_cross_entropy import FusedLinearCELoss
from .backbone import DS2DSBackbone


class DS2DSTrainingModule(nn.Module):
    """
    Training wrapper for DS2DS model that handles loss computation.
    
    This module follows the "batch in, loss out" abstraction for training loops.
    It combines the backbone architecture with embeddings, normalization, and
    a fused cross-entropy loss function optimized with Liger kernels.
    
    The module can be constructed either directly via __init__ (for tests/benchmarks)
    or via the from_config classmethod (for standard training). After training,
    use to_inference_model() to extract an inference-ready DS2DSModel.
    
    Architecture:
        Input Batch → Embedding → Backbone → Norm → Fused Linear + CE Loss
    
    Attributes:
        backbone: The transformer layer stack (DS2DSBackbone)
        embedding: Token embedding layer
        norm: Final RMS normalization layer
        loss_fn: Fused linear projection + cross-entropy loss
        block_mask_creator: Callable that creates attention masks from batch
    """
    
    def __init__(
        self,
        backbone: DS2DSBackbone,
        embedding: nn.Embedding,
        norm: RMSNorm,
        loss_fn: FusedLinearCELoss,
        block_mask_creator: Callable,
        vocab_size: int,
        ignore_index: int = -100,
    ):
        """
        Initialize the training module with pre-constructed components.
        
        Args:
            backbone: Pre-constructed DS2DSBackbone instance
            embedding: Token embedding layer (nn.Embedding)
            norm: RMS normalization layer for final hidden states
            loss_fn: Fused linear + cross-entropy loss (contains lm_head weight)
            block_mask_creator: Callable that takes **batch and returns BlockMask
            vocab_size: Vocabulary size (stored for inference model creation)
            ignore_index: Index to ignore in loss computation (e.g., padding)
        """
        super().__init__()
        self.backbone = backbone
        self.embedding = embedding
        self.norm = norm
        self.loss_fn = loss_fn
        self.block_mask_creator = block_mask_creator
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
    
    @classmethod
    def from_config(
        cls,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float,
        drop_path_rate: float,
        block_mask_creator: Callable,
        fp8: bool = False,
        weight_tying: bool = True,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.bfloat16,
    ) -> 'DS2DSTrainingModule':
        """
        Factory method to construct a training module from configuration parameters.
        
        This is the recommended way to create a training module for standard training.
        It handles constructing all components with proper initialization and weight tying.
        
        Args:
            vocab_size: Size of the vocabulary
            num_layers: Number of transformer layers
            model_dim: Hidden dimension size (d_model)
            num_heads: Number of attention heads per layer
            max_seq_len: Maximum sequence length
            dropout: Dropout probability for channel mixing
            drop_path_rate: Stochastic depth probability
            block_mask_creator: Callable that creates attention masks from batch
            fp8: Whether to use FP8 precision for linear layers
            weight_tying: Whether to tie embedding and output head weights
            ignore_index: Index to ignore in loss computation
            dtype: Data type for loss computation
        
        Returns:
            Configured DS2DSTrainingModule ready for training
        """
        # Construct backbone
        backbone = DS2DSBackbone(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
        )
        
        # Construct embedding
        embedding = nn.Embedding(vocab_size, model_dim)
        
        # Construct normalization
        norm = RMSNorm(model_dim)
        
        # Construct loss function with optional weight tying
        loss_weight = embedding.weight if weight_tying else None
        loss_fn = FusedLinearCELoss(
            D=model_dim,
            V=vocab_size,
            dtype=dtype,
            ignore_index=ignore_index,
            weight=loss_weight
        )
        
        return cls(
            backbone=backbone,
            embedding=embedding,
            norm=norm,
            loss_fn=loss_fn,
            block_mask_creator=block_mask_creator,
            vocab_size=vocab_size,
            ignore_index=ignore_index,
        )
    
    def to_inference_model(self):
        """
        Convert this training module to an inference-ready DS2DSModel.
        
        This extracts the trained weights and passes them to a DS2DSModel instance
        which provides generation and evaluation capabilities. The weights are
        passed as tensor references (not Parameters) to avoid unnecessary copying
        while maintaining gradient-free inference.
        
        Returns:
            DS2DSModel instance ready for inference/evaluation
        """
        from .model import DS2DSModel
        
        return DS2DSModel(
            backbone=self.backbone,
            embedding_weight=self.embedding.weight,
            lm_head_weight=self.loss_fn.weight,
            norm=self.norm,
            block_mask_creator=self.block_mask_creator,
            vocab_size=self.vocab_size,
            ignore_index=self.ignore_index,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Type[Tensor] | Any]:
        """
        Forward pass: batch in, loss out.
        
        This method implements the complete training forward pass including
        token shifting for autoregressive prediction, mask creation, embedding,
        backbone processing, normalization, and loss computation.
        
        Args:
            batch: Dictionary containing at minimum:
                - 'tokens': Tensor of shape (B, T+1) with token IDs
                - Additional keys may be used by block_mask_creator
                  (e.g., 'doc_spans' for document-aware masking)
        
        Returns:
            Dictionary containing:
                - 'loss': Scalar tensor with the training loss
                - 'ce_loss': Cross-entropy loss (same as 'loss' in this implementation)
        """
        tokens = batch['tokens']
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]

        block_mask = self.block_mask_creator(**batch)
        
        x = self.embedding(input_ids)
        x = self.backbone(x, block_mask=block_mask)
        x = self.norm(x)
        
        loss = self.loss_fn(x, target_ids)
        
        return {'loss': loss, 'ce_loss': loss}
