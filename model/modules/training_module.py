from typing import Dict, Any, Type, Callable, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from tunalab.modules.norms.rms_norm import RMSNorm
from tunalab.modules.losses.fused_cross_entropy import FusedLinearCELoss
from .backbone import TS2TSBackbone


class TS2TSTrainingModule(nn.Module):
    """
    Training wrapper for TS2TS model that handles loss computation.
    
    This module follows the "batch in, loss out" abstraction for training loops.
    It combines the backbone architecture with embeddings, normalization, and
    a fused cross-entropy loss function optimized with Liger kernels.
    
    The module can be constructed either directly via __init__ (for tests/benchmarks)
    or via the from_config classmethod (for standard training). After training,
    use to_inference_model() to extract an inference-ready TS2TSModel.
    
    Architecture:
        Input Batch → Embedding → Backbone → Norm → Fused Linear + CE Loss
    
    Attributes:
        backbone: The transformer layer stack (TS2TSBackbone)
        embedding: Token embedding layer
        norm: Final RMS normalization layer
        loss_fn: Fused linear projection + cross-entropy loss
        block_mask_creator: Callable that creates attention masks from batch
    """
    
    def __init__(
        self,
        backbone: TS2TSBackbone,
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
            backbone: Pre-constructed TS2TSBackbone instance
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
        activation_checkpointing: bool = False,
        attention_backend: str = "flex",
    ) -> 'TS2TSTrainingModule':
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
            attention_backend: "flex" (default) or "triton_v12"

        Returns:
            Configured TS2TSTrainingModule ready for training
        """
        # Construct backbone
        backbone = TS2TSBackbone(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
            activation_checkpointing=activation_checkpointing,
            attention_backend=attention_backend,
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
    
    def to_inference_model(
        self,
        tokenizer,
        mask_type: str = 'doc_causal',
        link_detector=None,
        training_backend: str = 'triton',
        inference_backend: str = 'flex',
        training_layout_policy=None,
        inference_layout_policy=None,
    ):
        """
        Convert this training module to an inference-ready TS2TSModel.

        Weights are passed as tensor references (not Parameters) to avoid
        unnecessary copying. The resulting TS2TSModel builds its own _creators
        dict from mask_type, link_detector, and the backend args — no creator
        objects need to be constructed at the call site.

        When inference_backend != training_backend, the backbone's attention
        layer classes are swapped zero-copy so the uncompiled backbone uses
        FlexSelfAttention for inference (faster for single-doc paths with
        torch.compile) without duplicating any weights.

        Args:
            tokenizer:               Required for generate().
            mask_type:               'doc_causal' | 'cross_doc_link'.
            link_detector:           LinkDetector instance; None for doc_causal.
            training_backend:        'triton' | 'flex'. Used when model._is_training.
            inference_backend:       'flex' | 'triton'. Used otherwise.
            training_layout_policy:  DocLayoutPolicy used during training.
            inference_layout_policy: DocLayoutPolicy used during inference/eval.
                                     Defaults to training_layout_policy if None.

        Returns:
            TS2TSModel instance ready for inference/evaluation.
        """
        from model.model import TS2TSModel

        backbone = self.backbone

        # Switch attention backend from triton → flex for inference.
        # TS2TSAttention holds its kernel choice in self.backend, so flipping
        # that attribute is all that's needed — no __class__ reassignment,
        # no weight copying.
        if inference_backend == 'flex' and training_backend == 'triton':
            raw_backbone = getattr(backbone, '_orig_mod', backbone)
            for layer in raw_backbone.layers:
                layer.attn.backend = 'flex'
            backbone = raw_backbone

        return TS2TSModel(
            backbone=backbone,
            embedding_weight=self.embedding.weight,
            lm_head_weight=self.loss_fn.weight,
            norm=self.norm,
            vocab_size=self.vocab_size,
            mask_type=mask_type,
            link_detector=link_detector,
            training_backend=training_backend,
            inference_backend=inference_backend,
            training_layout_policy=training_layout_policy,
            inference_layout_policy=inference_layout_policy,
            ignore_index=self.ignore_index,
            tokenizer=tokenizer,
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
                The mask creator receives 'tokens' as (B, T) — the input slice only.
        
        Returns:
            Scalar loss tensor (cross-entropy loss over the sequence).
        """
        tokens = batch['tokens']
        input_ids = tokens[:, :-1].contiguous()
        target_ids = tokens[:, 1:].contiguous()

        mask_batch = {**batch, 'tokens': input_ids}
        block_mask = self.block_mask_creator(**mask_batch)
        
        x = self.embedding(input_ids)
        x = self.backbone(x, block_mask=block_mask)
        x = self.norm(x)
        
        # Disable dynamo tracing for the Liger loss: its AOT-compiled forward has
        # a buffer overread bug (inductor sizes intermediate buffers incorrectly
        # for the chunked accumulation loop), causing CUDA illegal memory access.
        loss = torch._dynamo.disable(self.loss_fn)(x, target_ids)

        return loss
