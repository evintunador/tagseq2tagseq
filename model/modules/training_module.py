from typing import Dict, Any, Type, Callable, List, Tuple, Optional

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

    Multi-Token Prediction (MTP):
        When mtp_extra_weights is non-empty, forward() additionally predicts
        tokens at offsets +2, +3, … using the same lm_head, weighted by the
        corresponding entry in mtp_extra_weights.  Weights decay linearly to 0
        over mtp_decay_micro_steps training forward-passes.  MTP is skipped
        during eval so that the reported val loss is always standard next-token
        CE, comparable across runs.

    Attributes:
        backbone: The transformer layer stack (TS2TSBackbone)
        embedding: Token embedding layer
        norm: Final RMS normalization layer
        loss_fn: Fused linear projection + cross-entropy loss
        block_mask_creator: Callable that creates attention masks from batch
        mtp_extra_weights: Initial weights for MTP offsets 1, 2, …
        mtp_decay_micro_steps: Training forward-passes over which to decay to 0
        _mtp_step: Persistent counter of training forward-passes (saved in ckpt)
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
        mtp_extra_weights: Optional[List[float]] = None,
        mtp_decay_micro_steps: int = 0,
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
            mtp_extra_weights: Initial weights for MTP offsets 1, 2, … beyond
                standard next-token prediction.  E.g. [0.3, 0.1] adds offset-2
                with weight 0.3 and offset-3 with weight 0.1.  Empty = disabled.
            mtp_decay_micro_steps: Number of training forward-passes over which
                to linearly decay mtp_extra_weights to 0.  0 = constant weights.
        """
        super().__init__()
        self.backbone = backbone
        self.embedding = embedding
        self.norm = norm
        self.loss_fn = loss_fn
        self.block_mask_creator = block_mask_creator
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.mtp_extra_weights: List[float] = list(mtp_extra_weights or [])
        self.mtp_decay_micro_steps: int = mtp_decay_micro_steps
        # Persistent counter of training-mode forward passes (survives checkpoints).
        self.register_buffer('_mtp_step', torch.tensor(0, dtype=torch.long))
    
    @classmethod
    def from_config(
        cls,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        max_seq_len: int,
        drop_path_rate: float,
        block_mask_creator: Callable,
        fp8: bool = False,
        weight_tying: bool = True,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.bfloat16,
        activation_checkpointing: bool = False,
        attention_backend: str = "flex",
        logit_softcap: float = None,
        mtp_extra_weights: Optional[List[float]] = None,
        mtp_decay_micro_steps: int = 0,
        ve_layers: Optional[List[int]] = None,
        shared_ve_bank: bool = False,
        use_bigram: bool = False,
        bigram_vocab_size: int = 50304 * 5,
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
            drop_path_rate: Stochastic depth probability
            block_mask_creator: Callable that creates attention masks from batch
            fp8: Whether to use FP8 precision for linear layers
            weight_tying: Whether to tie embedding and output head weights
            ignore_index: Index to ignore in loss computation
            dtype: Data type for loss computation
            attention_backend: "flex" (default), "triton_v12", or "triton_v17"

        Returns:
            Configured TS2TSTrainingModule ready for training
        """
        # Construct backbone
        backbone = TS2TSBackbone(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
            activation_checkpointing=activation_checkpointing,
            attention_backend=attention_backend,
            ve_layers=ve_layers or [],
            shared_ve_bank=shared_ve_bank,
            vocab_size=vocab_size,
            use_bigram=use_bigram,
            bigram_vocab_size=bigram_vocab_size,
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
            weight=loss_weight,
            softcap=logit_softcap,
        )
        
        return cls(
            backbone=backbone,
            embedding=embedding,
            norm=norm,
            loss_fn=loss_fn,
            block_mask_creator=block_mask_creator,
            vocab_size=vocab_size,
            ignore_index=ignore_index,
            mtp_extra_weights=mtp_extra_weights,
            mtp_decay_micro_steps=mtp_decay_micro_steps,
        )
    
    def to_inference_model(self, tokenizer, link_detector=None, layout_policy=None):
        """
        Convert this training module to an inference-ready TS2TSModel.

        Weights are passed as tensor references (not Parameters) to avoid
        unnecessary copying while maintaining gradient-free inference.

        Args:
            tokenizer: Tokenizer for encoding prompts and decoding output text.
                Required for generate(). Must match the tokenizer used during
                data pre-tokenization.
            link_detector: LinkDetector for cross-doc link detection (Stage 2+).
            layout_policy: DocLayoutPolicy for document prefix/suffix tokens (Stage 2+).

        Returns:
            TS2TSModel instance ready for inference/evaluation.
        """
        from model.model import TS2TSModel

        return TS2TSModel(
            backbone=self.backbone,
            embedding_weight=self.embedding.weight,
            lm_head_weight=self.loss_fn.weight,
            norm=self.norm,
            block_mask_creator=self.block_mask_creator,
            vocab_size=self.vocab_size,
            ignore_index=self.ignore_index,
            tokenizer=tokenizer,
            link_detector=link_detector,
            layout_policy=layout_policy,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Type[Tensor] | Any]:
        """
        Forward pass: batch in, loss out.

        Implements the complete training forward pass including token shifting,
        mask creation, embedding, backbone, normalization, and loss.

        Multi-Token Prediction (MTP): when mtp_extra_weights is non-empty and
        the module is in training mode, additional offset losses are accumulated:

            loss = CE(x, t+1) + w1*CE(x[:,:-1], t+2) + w2*CE(x[:,:-2], t+3) + …

        where wK decays linearly from mtp_extra_weights[K-1] to 0 over
        mtp_decay_micro_steps training forward-passes.  During eval the standard
        single-offset loss is returned so val metrics are always comparable.

        Args:
            batch: Dictionary containing at minimum:
                - 'tokens': Tensor of shape (B, T+1) with token IDs
                - Additional keys may be used by block_mask_creator
                  (e.g., 'doc_spans' for document-aware masking)
                The mask creator receives 'tokens' as (B, T) — the input slice only.

        Returns:
            Scalar loss tensor.
        """
        tokens = batch['tokens']
        input_ids = tokens[:, :-1].contiguous()
        target_ids = tokens[:, 1:].contiguous()

        mask_batch = {**batch, 'tokens': input_ids}
        block_mask = self.block_mask_creator(**mask_batch)

        x = self.embedding(input_ids)
        ve_map = self.backbone.prepare_ve(input_ids)
        bigram = self.backbone.prepare_bigram(input_ids)
        x = self.backbone(x, block_mask=block_mask, ve_map=ve_map, bigram=bigram)
        x = self.norm(x)

        # Disable dynamo tracing for the Liger loss: its AOT-compiled forward has
        # a buffer overread bug (inductor sizes intermediate buffers incorrectly
        # for the chunked accumulation loop), causing CUDA illegal memory access.
        loss = torch._dynamo.disable(self.loss_fn)(x, target_ids)

        # MTP: additional offset predictions (training mode only).
        if self.training and self.mtp_extra_weights:
            step = int(self._mtp_step.item())
            for k, w_init in enumerate(self.mtp_extra_weights, start=1):
                if self.mtp_decay_micro_steps > 0:
                    w = w_init * max(0.0, 1.0 - step / self.mtp_decay_micro_steps)
                else:
                    w = w_init
                if w <= 0.0:
                    continue
                # Position i predicts token i+(k+1): use x[:,:-k,:] vs tokens[:,k+1:]
                loss_k = torch._dynamo.disable(self.loss_fn)(x[:, :-k, :], tokens[:, k + 1:])
                loss = loss + w * loss_k
            self._mtp_step.add_(1)

        return loss
