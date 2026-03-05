from typing import List, Any, Dict, Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

from tunalab.evaluation import register_handler
from tunalab.modules.norms.rms_norm import RMSNorm
from tunalab.modules.losses.fused_cross_entropy import FusedLinearCELoss
from .modules import TS2TSBackbone
from .generation_config import GenerationConfig
from .generation_loop import run_generation
from .generation_result import GenerationResult


class TS2TSModel:
    """
    Inference and evaluation wrapper for TS2TS models.

    This class does NOT inherit from nn.Module, providing a cleaner interface
    for inference and evaluation without the nn.Module ceremony. It holds
    references to the trained components (backbone, weights, norms) and provides
    methods for generation and benchmark evaluation.

    Attributes:
        backbone: The transformer layer stack (nn.Module)
        embedding_weight: Token embedding matrix (Tensor reference)
        lm_head_weight: Output projection matrix (Tensor reference)
        norm: RMS normalization layer (nn.Module)
        block_mask_creator: Callable for creating attention masks
        vocab_size: Vocabulary size
        ignore_index: Index to ignore in loss/eval computations
        tokenizer: Tokenizer for prompt encoding / output decoding (required for generate())
        link_detector: LinkDetector for cross-doc link detection (Stage 2+)
        layout_policy: DocLayoutPolicy for document prefix/suffix tokens (Stage 2+)
    """

    def __init__(
        self,
        backbone: TS2TSBackbone,
        embedding_weight: Tensor,
        lm_head_weight: Tensor,
        norm: nn.Module,
        block_mask_creator: Callable,
        vocab_size: int,
        ignore_index: int = -100,
        tokenizer=None,
        link_detector=None,
        layout_policy=None,
    ):
        self.backbone = backbone
        self.embedding_weight = embedding_weight
        self.lm_head_weight = lm_head_weight
        self.norm = norm
        self.block_mask_creator = block_mask_creator
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
        self.link_detector = link_detector
        self.layout_policy = layout_policy
    
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
        tokenizer=None,
        link_detector=None,
        layout_policy=None,
    ) -> 'TS2TSModel':
        # Construct backbone
        backbone = TS2TSBackbone(
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
        
        # Construct lm_head weight
        if weight_tying:
            lm_head_weight = embedding.weight
        else:
            lm_head_weight = nn.Parameter(torch.empty(vocab_size, model_dim))
            nn.init.normal_(lm_head_weight, mean=0.0, std=0.02)
        
        return cls(
            backbone=backbone,
            embedding_weight=embedding.weight,
            lm_head_weight=lm_head_weight,
            norm=norm,
            block_mask_creator=block_mask_creator,
            vocab_size=vocab_size,
            ignore_index=ignore_index,
            tokenizer=tokenizer,
            link_detector=link_detector,
            layout_policy=layout_policy,
        )

    def to_training_module(
        self,
        dtype: torch.dtype = torch.bfloat16,
    ) -> 'TS2TSTrainingModule':
        """
        Convert this inference model to a training-ready TS2TSTrainingModule.
        
        This wraps the model's components in a training module that provides
        the "batch in, loss out" interface required for training loops. The
        weights are shared (not copied), so training will update this model's
        weights in-place.
        
        Args:
            dtype: Data type for loss computation
        
        Returns:
            TS2TSTrainingModule ready for training, sharing weights with this model
        """
        from .modules.training_module import TS2TSTrainingModule
        
        # Create embedding layer from weight reference
        embedding = nn.Embedding(self.vocab_size, self.embedding_weight.shape[1])
        embedding.weight = nn.Parameter(self.embedding_weight)
        
        # Create loss function with weight reference
        loss_fn = FusedLinearCELoss(
            D=self.embedding_weight.shape[1],
            V=self.vocab_size,
            dtype=dtype,
            ignore_index=self.ignore_index,
            weight=self.lm_head_weight
        )
        
        return TS2TSTrainingModule(
            backbone=self.backbone,
            embedding=embedding,
            norm=self.norm,
            loss_fn=loss_fn,
            block_mask_creator=self.block_mask_creator,
            vocab_size=self.vocab_size,
            ignore_index=self.ignore_index,
        )
    
    def update_from_training_module(self, training_module: 'TS2TSTrainingModule') -> 'TS2TSModel':
        """
        Update this model's weights and components from a trained training module.
        
        This method updates the model in-place with weights from a trained module,
        useful for updating an inference model after training or fine-tuning.
        
        Args:
            training_module: Trained TS2TSTrainingModule to extract weights from
        
        Returns:
            self for method chaining
        """
        self.backbone = training_module.backbone
        self.embedding_weight = training_module.embedding.weight
        self.lm_head_weight = training_module.loss_fn.weight
        self.norm = training_module.norm
        self.block_mask_creator = training_module.block_mask_creator
        self.vocab_size = training_module.vocab_size
        self.ignore_index = training_module.ignore_index
        return self
    
    def eval(self):
        """Set the backbone to evaluation mode (disables dropout)."""
        self.backbone.eval()
        return self

    def train(self, mode: bool = True):
        """Set the backbone to training mode."""
        self.backbone.train(mode)
        return self
    
    def to(self, device, dtype=None):
        """
        Move all components to the specified device (and optionally dtype).

        Args:
            device: Target device or torch.dtype (passed through to nn.Module.to).
            dtype: Optional dtype (e.g. torch.bfloat16).

        Returns:
            self for method chaining
        """
        self.backbone.to(device, dtype)
        self.norm.to(device, dtype)
        # embedding_weight and lm_head_weight may not be owned by any nn.Module
        # stored on this object (e.g. when constructed via from_config), so we
        # must move them explicitly.  Handle the tied case (same object) carefully.
        tied = self.lm_head_weight is self.embedding_weight
        self.embedding_weight = self.embedding_weight.to(device=device, dtype=dtype)
        self.lm_head_weight = self.embedding_weight if tied else self.lm_head_weight.to(device=device, dtype=dtype)
        return self
    
    @torch.no_grad()
    def forward_inference(
        self,
        tokens: Tensor,
        doc_spans: Optional[List[Any]] = None,
        **kwargs
    ) -> Tensor:
        """
        Forward pass for inference: tokens in, logits out.

        Args:
            tokens: Input token IDs of shape [1, T]
            doc_spans: List of DocSpan objects for document-aware masking
            **kwargs: Additional arguments forwarded to block_mask_creator

        Returns:
            Logits tensor of shape [1, T, vocab_size]
        """
        block_mask = self.block_mask_creator(tokens=tokens, doc_spans=doc_spans or [], **kwargs)
        x = F.embedding(tokens, self.embedding_weight)   # [1, T, D]
        x = self.backbone(x, block_mask=block_mask)      # [1, T, D]
        x = self.norm(x)
        logits = F.linear(x, self.lm_head_weight)        # [1, T, V]
        return logits

    def generate(
        self,
        prompt: str,
        corpus=None,
        config=None,
    ) -> GenerationResult:
        """
        Generate text autoregressively, returning a structured GenerationResult.

        Args:
            prompt: Text prompt to condition on. Encoded using self.tokenizer.
            corpus: Optional DocumentCorpus for cross-doc link resolution (Stage 2+).
            config: GenerationConfig. Defaults to GenerationConfig() if None.

        Returns:
            GenerationResult with the root document (and aux docs in Stage 2+).

        Raises:
            RuntimeError: If self.tokenizer is not set.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "tokenizer must be set on TS2TSModel before calling generate(). "
                "Pass tokenizer= to to_inference_model() or TS2TSModel.__init__()."
            )
        if config is None:
            config = GenerationConfig()

        self.eval()
        prompt_tokens = list(self.tokenizer.encode(prompt))
        return run_generation(
            model=self,
            prompt_tokens=prompt_tokens,
            corpus=corpus,
            config=config,
            link_detector=self.link_detector,
            tokenizer_decode=self.tokenizer.decode,
            layout_policy=self.layout_policy,
        )
    
    @register_handler("perplexity")
    def compute_perplexity(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute perplexity for a batch of examples.
        
        TODO: Implement this evaluation handler:
        1. Convert eval batch format to your packed sequence format
        2. Create doc_spans metadata
        3. Generate appropriate block masks
        4. Run forward_inference to get logits
        5. Compute per-token cross-entropy losses
        6. Return perplexity and related metrics
        
        Args:
            batch: List of evaluation examples, each a dict with:
                - 'text': str or 'tokens': Tensor
                - Additional context fields as needed
        
        Returns:
            Dictionary containing:
                - 'perplexity': float
                - 'avg_loss': float
                - 'num_tokens': int
                Additional metrics as desired
        
        Raises:
            NotImplementedError: This is a stub for user implementation
        """
        raise NotImplementedError(
            "compute_perplexity evaluation handler must be implemented. "
            "This requires converting standard eval format to your packed "
            "sequence format with doc_spans."
        )
    
    @register_handler("next_token_prediction")
    def predict_next_token(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate next token prediction accuracy.
        
        TODO: Implement this evaluation handler:
        1. Process batch into packed sequences
        2. Run forward_inference
        3. Compare predicted tokens with ground truth
        4. Compute accuracy metrics (overall, per-document, cross-document)
        
        Args:
            batch: List of examples with context and target tokens
        
        Returns:
            Dictionary containing:
                - 'accuracy': float
                - 'top5_accuracy': float
                Additional metrics as desired
        
        Raises:
            NotImplementedError: This is a stub for user implementation
        """
        raise NotImplementedError(
            "predict_next_token evaluation handler must be implemented. "
            "Consider metrics that highlight your model's unique graph-aware "
            "capabilities, such as cross-document prediction accuracy."
        )

