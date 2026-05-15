from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor

from .layer import Layer

_BIGRAM_VOCAB_SIZE_DEFAULT = 50304 * 5  # 251520 — matches modded-nanogpt reference
_BIGRAM_RAND_A = 36313
_BIGRAM_RAND_B = 27191


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
    - Optional bigram hash embedding (use_bigram config)
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
        shared_ve_bank: If True, all ve_layers share one bank + gate.
        vocab_size: Vocabulary size — required when ve_layers is non-empty.
        use_bigram: Enable bigram hash embedding injection (default False).
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
        use_bigram: bool = False,
        bigram_vocab_size: int = _BIGRAM_VOCAB_SIZE_DEFAULT,
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
        self.ve_layers: List[int] = list(ve_layers) if ve_layers else []
        self.shared_ve_bank: bool = shared_ve_bank

        if self.ve_layers:
            if vocab_size <= 0:
                raise ValueError("vocab_size must be > 0 when ve_layers is non-empty")
            num_banks = 1 if shared_ve_bank else len(self.ve_layers)
            # BF16; ~98 MB per bank at vocab=50304, D=1024.
            self.value_embeds = nn.Parameter(
                0.01 * torch.randn(num_banks, vocab_size, model_dim, dtype=torch.bfloat16)
            )
            # Gate weights: (num_banks, num_heads, 12), zero-init → no-op at start.
            self.ve_gate_bank = nn.Parameter(torch.zeros(num_banks, num_heads, 12))

        # Bigram hash embedding (optional).
        # bigram_lambdas: 0.05-init (non-zero, but embed weight=0 → still no-op at start).
        if use_bigram:
            self.bigram_embed = nn.Embedding(bigram_vocab_size, model_dim)
            nn.init.zeros_(self.bigram_embed.weight)
            self.bigram_lambdas = nn.Parameter(0.05 * torch.ones(num_layers))

    # ------------------------------------------------------------------
    # Preparation helpers (called by training_module / inference model
    # before forward() to keep the backbone's forward interface clean)
    # ------------------------------------------------------------------

    def prepare_ve(self, input_ids: Tensor) -> Dict[int, Tuple[Tensor, Tensor]]:
        """Look up value embeddings for a sequence and return a per-layer map.

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

    def prepare_bigram(self, input_ids: Tensor) -> Optional[Tensor]:
        """Compute bigram hash indices and look up embeddings.

        Hash: position 0 → reserved index (bigram_vocab_size - 1).
              position i≥1 → (A * curr_token XOR B * prev_token) % (vocab-1).
        Matches modded-nanogpt get_bigram_hash exactly.

        Args:
            input_ids: Token IDs (B, T) with B=1 for packed sequences.

        Returns:
            Bigram embedding tensor (T, D), or None if use_bigram is False.
        """
        if not hasattr(self, 'bigram_embed'):
            return None
        ids = input_ids.squeeze(0).to(torch.int32)  # (T,)
        mod = self.bigram_embed.num_embeddings - 1
        bigram_ids = ids.clone()
        bigram_ids[0] = mod
        # bigram_ids[:-1] at this point: [mod, ids[1], ..., ids[T-2]]
        bigram_ids[1:] = torch.bitwise_xor(
            _BIGRAM_RAND_A * ids[1:],
            _BIGRAM_RAND_B * bigram_ids[:-1],
        ) % mod
        return self.bigram_embed(bigram_ids.long())  # (T, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        block_mask: Any,
        ve_map: Optional[Dict[int, Tuple[Tensor, Tensor]]] = None,
        bigram: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through the transformer backbone.

        Args:
            x: Pre-embedded input (B, T, C).
            block_mask: FlexAttention BlockMask or TritonMaskInputs.
            ve_map: Per-layer value-embedding map from prepare_ve().
            bigram: Pre-looked-up bigram embeddings (T, D) from prepare_bigram().

        Returns:
            Hidden states (B, T, C) after all transformer layers.
        """
        skip_connections = []
        n_skip = len(self.skip_weights)
        x0 = x  # embedding residual highway: saved before any injections

        if ve_map is None:
            ve_map = {}

        # Pre-loop bigram injection (layer-0 contribution).
        if bigram is not None:
            x = x + self.bigram_lambdas[0] * bigram.unsqueeze(0)

        for i, layer in enumerate(self.layers):
            if i >= n_skip:
                if skip_connections:
                    x = x + self.skip_weights[i - n_skip] * skip_connections.pop()

            ve_i, ve_gate_i = ve_map.get(i, (None, None))

            if self.activation_checkpointing:
                if ve_i is not None:
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

            # x0 injection (all layers) + bigram injection (layers 1+).
            x = x + self.x0_lambdas[i] * x0
            if bigram is not None and i >= 1:
                x = x + self.bigram_lambdas[i] * bigram.unsqueeze(0)

            if i < n_skip:
                skip_connections.append(x)

        return x
