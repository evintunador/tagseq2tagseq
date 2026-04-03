from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor

from .layer import Layer


class TS2TSBackbone(nn.Module):
    """
    The "spine" of the TS2TS transformer architecture.

    Contains only the core transformer layers with skip connections,
    intentionally excluding the embedding and output head.

    Architecture details:
    - Stack of transformer layers with RMS normalization and GLU channel mixing
    - Skip connections from first half to second half of layers
    - Per-layer x0 injection (embedding residual highway, zero-init)
    - Optional value embeddings (ve_layers / shared_ve_bank config)
    - FlexAttention with custom block masks for graph-aware attention patterns

    Args:
        num_layers: Number of transformer layers
        model_dim: Hidden dimension (d_model)
        num_heads: Number of attention heads per layer
        max_seq_len: Maximum sequence length supported
        drop_path_rate: Stochastic depth probability
        fp8: Whether to use FP8 precision for linear projections
        activation_checkpointing: Enable gradient checkpointing per layer
        attention_backend: Kernel selection string
        ve_layers: Layer indices that receive value-embedding injection.
            Empty list (default) disables the feature entirely.
        shared_ve_bank: If True, all ve_layers share one bank + gate (cheap
            test mode, ~1/N memory).  If False (default), each layer gets
            its own bank + gate (reference behaviour).
        vocab_size: Vocabulary size — required when ve_layers is non-empty.
    """
    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        max_seq_len: int,
        drop_path_rate: float,
        fp8: bool = False,
        activation_checkpointing: bool = False,
        attention_backend: str = "flex",
        ve_layers: Optional[List[int]] = None,
        shared_ve_bank: bool = False,
        vocab_size: int = 0,
        **kwargs
    ):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList([Layer(
            n_embd=model_dim,
            n_head=num_heads,
            max_seq_len=max_seq_len,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
            attention_backend=attention_backend,
        ) for _ in range(num_layers)])
        self.skip_weights = nn.Parameter(torch.ones(num_layers // 2))
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.activation_checkpointing = activation_checkpointing

        # Value embeddings (optional).
        # ve_layers is stored as a list to preserve bank-assignment order.
        self.ve_layers: List[int] = list(ve_layers) if ve_layers else []
        self.shared_ve_bank: bool = shared_ve_bank

        if self.ve_layers:
            if vocab_size <= 0:
                raise ValueError("vocab_size must be > 0 when ve_layers is non-empty")
            num_banks = 1 if shared_ve_bank else len(self.ve_layers)
            # BF16 to match the reference; ~98 MB per bank at vocab=50304, D=1024.
            self.value_embeds = nn.Parameter(
                0.01 * torch.randn(num_banks, vocab_size, model_dim, dtype=torch.bfloat16)
            )
            # Gate weights: (num_banks, num_heads, 12), zero-init → no-op at start.
            self.ve_gate_bank = nn.Parameter(torch.zeros(num_banks, num_heads, 12))

    def prepare_ve(self, input_ids: Tensor) -> Dict[int, Tuple[Tensor, Tensor]]:
        """Look up value embeddings for a sequence and return a per-layer map.

        Call this before forward() when value embeddings are enabled.  Keeping
        the vocab lookup here (rather than inside forward) preserves the
        backbone's contract of operating purely on continuous representations.

        Args:
            input_ids: Token IDs (B, T) with B=1 for packed sequences.

        Returns:
            Dict mapping layer_idx → (ve, ve_gate_w) where ve is (T, D) BF16
            and ve_gate_w is (H, 12).  Empty dict if ve_layers is empty.
        """
        if not self.ve_layers:
            return {}
        ids = input_ids.squeeze(0)  # (T,)
        ve_map: Dict[int, Tuple[Tensor, Tensor]] = {}
        for bank_idx, layer_idx in enumerate(self.ve_layers):
            actual_bank = 0 if self.shared_ve_bank else bank_idx
            ve_map[layer_idx] = (
                self.value_embeds[actual_bank, ids, :],  # (T, D) BF16
                self.ve_gate_bank[actual_bank],           # (H, 12)
            )
        return ve_map

    def forward(self, x: Tensor, block_mask: Any,
                ve_map: Optional[Dict[int, Tuple[Tensor, Tensor]]] = None) -> Tensor:
        """
        Forward pass through the transformer backbone.

        Args:
            x: Pre-embedded input (B, T, C).
            block_mask: FlexAttention BlockMask or TritonMaskInputs.
            ve_map: Per-layer value-embedding map from prepare_ve().
                    None or empty dict disables value embeddings.

        Returns:
            Hidden states (B, T, C) after all transformer layers.
        """
        skip_connections = []
        n_skip = len(self.skip_weights)
        x0 = x  # embedding residual highway: saved before any transformer layers

        if ve_map is None:
            ve_map = {}

        for i, layer in enumerate(self.layers):
            if i >= n_skip:
                if skip_connections:
                    x = x + self.skip_weights[i - n_skip] * skip_connections.pop()

            ve_i, ve_gate_i = ve_map.get(i, (None, None))

            if self.activation_checkpointing:
                if ve_i is not None:
                    # ve_i and ve_gate_i are tensors whose gradients must flow,
                    # so they must be explicit args (not closure) for checkpointing.
                    x = torch.utils.checkpoint.checkpoint(
                        lambda x_, ve_, vg_: layer(x_, block_mask, ve=ve_, ve_gate_w=vg_),
                        x, ve_i, ve_gate_i,
                        use_reentrant=False,
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        layer, x, block_mask, use_reentrant=False
                    )
            else:
                x = layer(x, block_mask, ve=ve_i, ve_gate_w=ve_gate_i)

            x = x + self.x0_lambdas[i] * x0

            if i < n_skip:
                skip_connections.append(x)

        return x
