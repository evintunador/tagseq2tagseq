import math

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any

from tunalab.modules.sequence_mixing.flex_self_attention import FlexSelfAttention
from tunalab.modules.regularization.drop_path import DropPath
from tunalab.modules.norms.rms_norm import RMSNorm
from kernels.fused_relu_sq_mlp import FusedReLUSquaredMLP


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

        if attention_backend == "triton_v12":
            from model.modules.attention import BIMv12Attention
            self.attn = BIMv12Attention(
                dim=n_embd, num_heads=n_head, max_seq_len=max_seq_len, fp8_out_proj=fp8,
            )
        elif attention_backend in ("triton_v17", "triton_v18"):
            from model.modules.attention import BIMv17Attention, BIMv18Attention
            _cls = BIMv18Attention if attention_backend == "triton_v18" else BIMv17Attention
            self.attn = _cls(
                dim=n_embd, num_heads=n_head, max_seq_len=max_seq_len, fp8_out_proj=fp8,
            )
        elif attention_backend in ("varlen_bim_v1", "varlen_bim_v2"):
            from model.modules.attention import VarlenBIMv1Attention, VarlenBIMv2Attention
            _cls = VarlenBIMv2Attention if attention_backend == "varlen_bim_v2" else VarlenBIMv1Attention
            self.attn = _cls(
                dim=n_embd, num_heads=n_head, max_seq_len=max_seq_len, fp8_out_proj=fp8,
            )
        else:
            self.attn = FlexSelfAttention(
                dim=n_embd, num_heads=n_head, max_seq_len=max_seq_len, fp8_out_proj=fp8,
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

    def forward(self, x: Tensor, block_mask: Any):
        rl = self.resid_lambdas
        pl = self.post_lambdas
        x = rl[0] * x + pl[0] * self.drop_path(self.attn(self.ln_1(x), block_mask=block_mask))
        x = rl[1] * x + pl[1] * self.drop_path(self.mlp(self.ln_2(x)))
        return x
