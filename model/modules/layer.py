import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any

from tunalab.modules.regularization.drop_path import DropPath
from tunalab.modules.norms.rms_norm import RMSNorm
from kernels.fused_relu_sq_mlp import FusedReLUSquaredMLP

from model.modules.attention import TS2TSAttention


class Layer(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        max_seq_len: int,
        fp8: bool,
        drop_path_rate: float,
        attention_backend: str = "flex",
    ):
        super().__init__()
        self.drop_path = DropPath(drop_path_rate)
        self.ln_1 = RMSNorm(n_embd)

        # Map all backend names to the TS2TSAttention unified class.
        # 'triton_v12' → 'triton', 'varlen_bim_v1' → 'triton' (kernel selected at runtime
        # from block_mask type). v17/v18/varlen_bim_v2 pass through directly.
        if attention_backend in ('triton_v12',):
            backend = 'triton'
        elif attention_backend in ('varlen_bim_v1',):
            backend = 'triton'
        elif attention_backend in ('triton_v17', 'triton_v18', 'varlen_bim_v2', 'flex', 'triton'):
            backend = attention_backend
        else:
            backend = 'flex'
        self.attn = TS2TSAttention(
            dim=n_embd, num_heads=n_head, max_seq_len=max_seq_len,
            fp8_out_proj=fp8, backend=backend,
        )

        self.ln_2 = RMSNorm(n_embd)
        self.mlp = FusedReLUSquaredMLP(model_dim=n_embd)

        # Per-sublayer learnable residual scaling (from modded-nanogpt).
        # resid_lambdas[0/1]: scale applied to the residual stream before adding
        #   the sublayer output. Initialized to sqrt(1.1) so that, at init,
        #   each sublayer slightly amplifies the residual stream.
        # post_lambdas[0/1]: scale applied to the sublayer output itself.
        #   Initialized to 1.0 to match the original residual connection at init.
        # Index 0 = attention sublayer, index 1 = MLP sublayer.
        self.resid_lambdas = nn.Parameter(torch.full((2,), math.sqrt(1.1)))
        self.post_lambdas = nn.Parameter(torch.ones(2))

    def forward(self, x: Tensor, block_mask: Any,
                ve: Optional[Tensor] = None, ve_gate_w: Optional[Tensor] = None):
        rl = self.resid_lambdas
        pl = self.post_lambdas
        x = rl[0] * x + pl[0] * self.drop_path(
            self.attn(self.ln_1(x), block_mask=block_mask, ve=ve, ve_gate_w=ve_gate_w)
        )
        x = rl[1] * x + pl[1] * self.drop_path(self.mlp(self.ln_2(x)))
        return x
