from typing import List, Any, Dict, Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

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
    for inference and evaluation without the nn.Module ceremony.

    The model tracks four independent axes:

      mask_type        — 'doc_causal' | 'cross_doc_link'  (model identity)
      link_detector    — dataset-specific link extractor; None for doc_causal
      training_backend — 'triton' | 'flex'  (default for self._is_training=True)
      inference_backend— 'flex' | 'triton'  (default for self._is_training=False)

    These are combined into a ``_creators`` dict keyed by
    ``'{mask_type}_{backend}'`` (e.g. ``'doc_causal_flex'``).
    ``forward_inference`` accepts optional ``mask_type=`` and ``backend=``
    overrides at call time, enabling eval under different conditions without
    reconstructing the model.

    Layout policies are stored separately for training and inference:
      training_layout_policy  — used when self._is_training is True
      inference_layout_policy — used otherwise
    ``active_layout_policy`` switches between them automatically.

    Attributes:
        backbone:                  The transformer layer stack (nn.Module)
        embedding_weight:          Token embedding matrix (Tensor reference)
        lm_head_weight:            Output projection matrix (Tensor reference)
        norm:                      RMS normalization layer (nn.Module)
        mask_type:                 'doc_causal' | 'cross_doc_link'
        link_detector:             LinkDetector or None
        training_backend:          'triton' | 'flex'
        inference_backend:         'flex' | 'triton'
        training_layout_policy:    DocLayoutPolicy used during training
        inference_layout_policy:   DocLayoutPolicy used during inference/eval
        vocab_size:                Vocabulary size
        ignore_index:              Index ignored in loss/eval computations
        tokenizer:                 Required for generate()
        _creators:                 Dict[str, Callable] — mask creator per key
        _is_training:              Bool tracking current mode (set by train/eval)
    """

    def __init__(
        self,
        backbone: TS2TSBackbone,
        embedding_weight: Tensor,
        lm_head_weight: Tensor,
        norm: nn.Module,
        vocab_size: int,
        mask_type: str = 'doc_causal',
        link_detector=None,
        training_backend: str = 'triton',
        inference_backend: str = 'flex',
        training_layout_policy=None,
        inference_layout_policy=None,
        ignore_index: int = -100,
        tokenizer=None,
        logit_softcap: Optional[float] = None,
    ):
        from data.layout import NullLayoutPolicy

        self.backbone = backbone
        self.embedding_weight = embedding_weight
        self.lm_head_weight = lm_head_weight
        self.norm = norm
        self.vocab_size = vocab_size
        self.mask_type = mask_type
        self.link_detector = link_detector
        self.training_backend = training_backend
        self.inference_backend = inference_backend
        self.training_layout_policy = training_layout_policy or NullLayoutPolicy()
        self.inference_layout_policy = inference_layout_policy or self.training_layout_policy
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer
        self.logit_softcap = logit_softcap
        self._is_training = False  # default to eval mode
        self._creators = self._build_creators()

    def _build_creators(self) -> Dict[str, Callable]:
        """Build the _creators dict from mask_type, link_detector, and backends.

        doc_causal creators are always present (needed for eval overrides on
        cross_doc_link models). cross_doc_link creators are only built when
        mask_type == 'cross_doc_link'.

        Keys follow the pattern '{mask_type}_{backend}', e.g.:
            'doc_causal_flex', 'doc_causal_triton',
            'cross_doc_link_flex', 'cross_doc_link_triton'
        """
        from model.graph_traversal.block_mask_creator import (
            make_mask_creator_callable,
            make_mask_creator_callable_from,
            create_doc_causal_triton_mask,
            create_doc_concat_triton_mask,
        )
        from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator

        creators: Dict[str, Callable] = {}

        # doc_causal always available — needed for eval overrides on every model
        # (the apples-to-apples isolated-attention eval condition).
        creators['doc_causal_flex'] = make_mask_creator_callable('doc_causal')
        creators['doc_causal_triton'] = make_mask_creator_callable_from(
            create_doc_causal_triton_mask
        )

        if self.mask_type == 'doc_concatenated':
            # Merge each traversal component into one causally-concatenated
            # super-doc; reuses the doc-causal varlen kernel via component ids.
            creators['doc_concatenated_flex'] = make_mask_creator_callable(
                'doc_concatenated'
            )
            creators['doc_concatenated_triton'] = make_mask_creator_callable_from(
                create_doc_concat_triton_mask
            )

        if self.mask_type in ('cross_doc_link', 'doc_concat_link'):
            if self.link_detector is None:
                raise ValueError(
                    f"link_detector must be set when mask_type='{self.mask_type}'"
                )
            # doc_concat_link grants whole-doc attention (full concatenation, no
            # link-position gate); cross_doc_link gates from the link position.
            whole_doc = self.mask_type == 'doc_concat_link'
            # Triton creator: no warmup state (warmup lives in the training module)
            creators[f'{self.mask_type}_triton'] = make_mask_creator_callable_from(
                CrossDocLinkMaskCreator(
                    link_detector=self.link_detector,
                    backend='triton_v12',
                    whole_doc_grant=whole_doc,
                )
            )
            # Flex creator: used for inference and eval
            creators[f'{self.mask_type}_flex'] = make_mask_creator_callable_from(
                CrossDocLinkMaskCreator(
                    link_detector=self.link_detector,
                    backend='flex',
                    whole_doc_grant=whole_doc,
                )
            )

        return creators

    @property
    def active_layout_policy(self):
        """Return training or inference layout policy based on current mode."""
        return self.training_layout_policy if self._is_training else self.inference_layout_policy

    @classmethod
    def from_config(
        cls,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        max_seq_len: int,
        drop_path_rate: float,
        mask_type: str = 'doc_causal',
        fp8: bool = False,
        weight_tying: bool = True,
        ignore_index: int = -100,
        tokenizer=None,
        link_detector=None,
        training_layout_policy=None,
        inference_layout_policy=None,
        training_backend: str = 'triton',
        inference_backend: str = 'flex',
    ) -> 'TS2TSModel':
        """Construct a TS2TSModel from configuration parameters.

        Primarily used in tests and benchmarks. For production use, prefer
        constructing via TS2TSTrainingModule.to_inference_model() or
        load_inference_model() in generate.py.
        """
        backbone = TS2TSBackbone(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
        )

        embedding = nn.Embedding(vocab_size, model_dim)
        norm = RMSNorm(model_dim)

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
            vocab_size=vocab_size,
            mask_type=mask_type,
            link_detector=link_detector,
            training_backend=training_backend,
            inference_backend=inference_backend,
            training_layout_policy=training_layout_policy,
            inference_layout_policy=inference_layout_policy,
            ignore_index=ignore_index,
            tokenizer=tokenizer,
        )

    def to_training_module(
        self,
        dtype: torch.dtype = torch.bfloat16,
    ) -> 'TS2TSTrainingModule':
        """Convert this inference model to a training-ready TS2TSTrainingModule.

        Weights are shared (not copied), so training will update this model's
        weights in-place. The training module receives the triton creator so
        it uses the optimal training kernel.
        """
        from .modules.training_module import TS2TSTrainingModule

        embedding = nn.Embedding(self.vocab_size, self.embedding_weight.shape[1])
        embedding.weight = nn.Parameter(self.embedding_weight)

        loss_fn = FusedLinearCELoss(
            D=self.embedding_weight.shape[1],
            V=self.vocab_size,
            dtype=dtype,
            ignore_index=self.ignore_index,
            weight=self.lm_head_weight
        )

        training_key = f'{self.mask_type}_{self.training_backend}'
        training_creator = self._creators[training_key]

        return TS2TSTrainingModule(
            backbone=self.backbone,
            embedding=embedding,
            norm=self.norm,
            loss_fn=loss_fn,
            block_mask_creator=training_creator,
            vocab_size=self.vocab_size,
            ignore_index=self.ignore_index,
        )

    def update_from_training_module(self, training_module: 'TS2TSTrainingModule') -> 'TS2TSModel':
        """Update this model's weights from a trained training module.

        Updates backbone, embedding, lm_head, and norm in-place.
        mask_type, link_detector, and _creators are identity properties and
        are not updated (they depend on architecture, not weights).
        """
        self.backbone = training_module.backbone
        self.embedding_weight = training_module.embedding.weight
        self.lm_head_weight = training_module.loss_fn.weight
        self.norm = training_module.norm
        self.vocab_size = training_module.vocab_size
        self.ignore_index = training_module.ignore_index
        return self

    def eval(self):
        """Set to evaluation mode (disables dropout, selects inference defaults)."""
        self.backbone.eval()
        self._is_training = False
        return self

    def train(self, mode: bool = True):
        """Set to training mode."""
        self.backbone.train(mode)
        self._is_training = mode
        return self

    def to(self, device, dtype=None):
        """Move all components to the specified device (and optionally dtype).

        Returns self for method chaining.
        """
        self.backbone.to(device, dtype)
        self.norm.to(device, dtype)
        # embedding_weight and lm_head_weight may not be owned by any nn.Module
        # stored on this object, so move them explicitly. Handle the tied case.
        tied = self.lm_head_weight is self.embedding_weight
        self.embedding_weight = self.embedding_weight.to(device=device, dtype=dtype)
        self.lm_head_weight = (
            self.embedding_weight if tied
            else self.lm_head_weight.to(device=device, dtype=dtype)
        )
        return self

    @torch.no_grad()
    def forward_inference(
        self,
        tokens: Tensor,
        doc_spans: Optional[List[Any]] = None,
        mask_type: Optional[str] = None,
        backend: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass for inference: tokens in, logits out.

        Args:
            tokens:    Input token IDs of shape [1, T]
            doc_spans: List of DocSpan objects for document-aware masking.
            mask_type: Override the model's default mask type for this call.
                       'doc_causal' | 'cross_doc_link'. Useful for running the
                       same model under different eval conditions (e.g. contrastive
                       perplexity: once with cross_doc_link, once with doc_causal).
            backend:   Override the backend for this call. 'flex' | 'triton'.
                       Defaults to training_backend when self._is_training, else
                       inference_backend.
            **kwargs:  Additional arguments forwarded to the mask creator.

        Returns:
            Logits tensor of shape [1, T, vocab_size]

        Raises:
            KeyError: If the resolved (mask_type, backend) combination has no
                      creator (e.g. requesting 'cross_doc_link' on a doc_causal
                      model that has no link_detector).
        """
        effective_mask = mask_type or self.mask_type
        effective_backend = backend or (
            self.training_backend if self._is_training else self.inference_backend
        )
        key = f'{effective_mask}_{effective_backend}'
        if key not in self._creators:
            raise KeyError(
                f"No creator for {key!r}. Available: {sorted(self._creators)}. "
                f"Cross-doc-link creators only exist on cross_doc_link models "
                f"(this model has mask_type={self.mask_type!r})."
            )
        block_mask = self._creators[key](tokens=tokens, doc_spans=doc_spans or [], **kwargs)
        x = F.embedding(tokens, self.embedding_weight)                        # [1, T, D]
        ve_map = self.backbone.prepare_ve(tokens)
        bigram = self.backbone.prepare_bigram(tokens)
        x = self.backbone(x, block_mask=block_mask, ve_map=ve_map, bigram=bigram)  # [1, T, D]
        x = self.norm(x)
        logits = F.linear(x, self.lm_head_weight)        # [1, T, V]
        if self.logit_softcap:
            cap = self.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return logits

    def generate(
        self,
        prompt: str,
        corpus=None,
        config=None,
        root_identifier: str = "",
    ) -> GenerationResult:
        """Generate text autoregressively, returning a structured GenerationResult.

        Args:
            prompt:          Text prompt to condition on. Encoded using self.tokenizer.
            corpus:          Optional PretokCorpus for cross-doc link resolution.
            config:          GenerationConfig. Defaults to GenerationConfig() if None.
            root_identifier: Filename / identifier prefix for the root document.

        Returns:
            GenerationResult with the root document (and aux docs if links resolved).

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
            layout_policy=self.inference_layout_policy,
            root_identifier=root_identifier,
        )
