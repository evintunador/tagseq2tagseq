from typing import List, Any, Dict, Callable, Optional

import torch
from torch import Tensor
import torch.nn as nn

from tunalab.evaluation import register_handler
from tunalab.modules.norms.rms_norm import RMSNorm
from tunalab.modules.losses.fused_cross_entropy import FusedLinearCELoss
from .modules import DS2DSBackbone


class DS2DSModel:
    """
    Inference and evaluation wrapper for DS2DS models.
    
    This class does NOT inherit from nn.Module, providing a cleaner interface
    for inference and evaluation without the nn.Module ceremony. It holds
    references to the trained components (backbone, weights, norms) and provides
    methods for generation and benchmark evaluation.
    
    The model is designed for graph-aware sequence generation with custom
    attention patterns over packed document sequences. Generation and evaluation
    methods are left as stubs for implementation of the unique graph-traversal
    and document-aware logic.
    
    Attributes:
        backbone: The transformer layer stack (nn.Module)
        embedding_weight: Token embedding matrix (Tensor reference)
        lm_head_weight: Output projection matrix (Tensor reference)
        norm: RMS normalization layer (nn.Module)
        block_mask_creator: Callable for creating attention masks
        vocab_size: Vocabulary size
        ignore_index: Index to ignore in loss/eval computations
    """
    
    def __init__(
        self,
        backbone: DS2DSBackbone,
        embedding_weight: Tensor,
        lm_head_weight: Tensor,
        norm: nn.Module,
        block_mask_creator: Callable,
        vocab_size: int,
        ignore_index: int = -100,
    ):
        """
        Initialize the inference model with references to trained components.
        
        Args:
            backbone: Pre-trained DS2DSBackbone instance
            embedding_weight: Token embedding tensor (reference, not Parameter)
            lm_head_weight: Output head weight tensor (reference, not Parameter)
            norm: RMS normalization layer
            block_mask_creator: Callable that creates attention masks from inputs
            vocab_size: Size of the vocabulary
            ignore_index: Index to ignore in computations (e.g., padding token)
        """
        self.backbone = backbone
        self.embedding_weight = embedding_weight
        self.lm_head_weight = lm_head_weight
        self.norm = norm
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
    ) -> 'DS2DSModel':
        """
        Factory method to construct an inference model from configuration parameters.
        
        This creates a fresh DS2DSModel with randomly initialized weights. For loading
        trained weights, use update_from_training_module() after creation or construct
        directly from a trained DS2DSTrainingModule using its to_inference_model() method.
        
        Args:
            vocab_size: Size of the vocabulary
            num_layers: Number of transformer layers
            model_dim: Hidden dimension size (d_model)
            num_heads: Number of attention heads per layer
            max_seq_len: Maximum sequence length
            dropout: Dropout probability for channel mixing
            drop_path_rate: Stochastic depth probability
            block_mask_creator: Callable that creates attention masks from inputs
            fp8: Whether to use FP8 precision for linear layers
            weight_tying: Whether to tie embedding and output head weights
            ignore_index: Index to ignore in computations
        
        Returns:
            Configured DS2DSModel with fresh weights
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
        )
    
    def to_training_module(
        self,
        dtype: torch.dtype = torch.bfloat16,
    ) -> 'DS2DSTrainingModule':
        """
        Convert this inference model to a training-ready DS2DSTrainingModule.
        
        This wraps the model's components in a training module that provides
        the "batch in, loss out" interface required for training loops. The
        weights are shared (not copied), so training will update this model's
        weights in-place.
        
        Args:
            dtype: Data type for loss computation
        
        Returns:
            DS2DSTrainingModule ready for training, sharing weights with this model
        """
        from .modules.training_module import DS2DSTrainingModule
        
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
        
        return DS2DSTrainingModule(
            backbone=self.backbone,
            embedding=embedding,
            norm=self.norm,
            loss_fn=loss_fn,
            block_mask_creator=self.block_mask_creator,
            vocab_size=self.vocab_size,
            ignore_index=self.ignore_index,
        )
    
    def update_from_training_module(self, training_module: 'DS2DSTrainingModule') -> 'DS2DSModel':
        """
        Update this model's weights and components from a trained training module.
        
        This method updates the model in-place with weights from a trained module,
        useful for updating an inference model after training or fine-tuning.
        
        Args:
            training_module: Trained DS2DSTrainingModule to extract weights from
        
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
    
    # def eval(self):
    #     """Set the backbone to evaluation mode."""
    #     self.backbone.eval()
    #     return self
    
    # def train(self, mode: bool = True):
    #     """Set the backbone to training mode."""
    #     self.backbone.train(mode)
    #     return self
    
    def to(self, device: torch.device):
        """
        Move all components to the specified device.
        
        Args:
            device: Target device (e.g., torch.device('cuda'))
        
        Returns:
            self for method chaining
        """
        self.backbone.to(device)
        # Weights are references to Parameters in backbone/embedding,
        # so they move automatically when their parent modules move
        self.norm.to(device)
        return self
    
    def forward_inference(
        self,
        tokens: Tensor,
        doc_spans: Optional[List[Any]] = None,
        **kwargs
    ) -> Tensor:
        """
        Forward pass for inference: tokens in, logits out.
        
        TODO: Implement this method with the following logic:
        1. Create block_mask from tokens and doc_spans using self.block_mask_creator
        2. Embed tokens using self.embedding_weight
        3. Pass through self.backbone with block_mask
        4. Apply self.norm
        5. Project to vocabulary using self.lm_head_weight
        6. Return logits of shape (B, T, V)
        
        Args:
            tokens: Input token IDs of shape (B, T)
            doc_spans: Optional list of DocSpan objects for document-aware masking
            **kwargs: Additional arguments for block_mask_creator
        
        Returns:
            Logits tensor of shape (B, T, vocab_size)
        
        Raises:
            NotImplementedError: This is a stub for user implementation
        """
        raise NotImplementedError(
            "forward_inference must be implemented with graph-aware masking logic. "
            "See docstring for implementation guidance."
        )
    
    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_tokens: Optional[Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using graph-aware attention patterns.
        
        TODO: Implement this method with your unique generation strategy:
        - Option A: Standard autoregressive generation within a single document
        - Option B: Generate across a pack of documents (matches training)
        - Option C: Generate one document while attending to context documents
        
        The implementation should handle:
        1. Graph traversal to select context documents
        2. Creating packed sequences with doc_spans
        3. Generating block masks for each step
        4. Autoregressive token prediction
        5. Sampling with temperature/top-k/top-p
        
        Args:
            prompt: Optional text prompt (requires tokenizer)
            prompt_tokens: Optional pre-tokenized prompt of shape (1, T)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling threshold
            **kwargs: Additional arguments (e.g., graph_context, traversal_strategy)
        
        Returns:
            Generated text as a string
        
        Raises:
            NotImplementedError: This is a stub for user implementation
        """
        raise NotImplementedError(
            "generate must be implemented with graph-aware generation logic. "
            "See docstring for implementation guidance. Consider the unique "
            "challenges of generating with packed document sequences and "
            "custom attention patterns."
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

