import torch.nn as nn
from torch import Tensor
from typing import Any

from tunalab.modules.sequence_mixing.flex_self_attention import FlexSelfAttention
from tunalab.modules.channel_mixing.glu import GLU
from tunalab.modules.regularization.drop_path import DropPath
from tunalab.modules.norms.rms_norm import RMSNorm


class Layer(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
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
        elif attention_backend == "varlen_bim_v1":
            from model.modules.attention import VarlenBIMv1Attention
            self.attn = VarlenBIMv1Attention(
                dim=n_embd, num_heads=n_head, max_seq_len=max_seq_len, fp8_out_proj=fp8,
            )
        else:
            self.attn = FlexSelfAttention(
                dim=n_embd, num_heads=n_head, max_seq_len=max_seq_len, fp8_out_proj=fp8,
            )

        self.ln_2 = RMSNorm(n_embd)
        self.mlp = GLU(
            in_dim=n_embd, out_dim=n_embd, hidden_dim=int(8/3*n_embd),
            activation="silu", dropout=dropout, fp8=fp8,
        )

    def forward(self, x: Tensor, block_mask: Any):
        x = x + self.drop_path(self.attn(self.ln_1(x), block_mask=block_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x
