import torch.nn as nn
from torch import Tensor
from typing import Any

from tunalab.modules.channel_mixing.glu import GLU
from tunalab.modules.regularization.drop_path import DropPath
from tunalab.modules.norms.rms_norm import RMSNorm

from model.modules.attention import TS2TSAttention


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

        # Normalize to 'flex' or 'triton'. Legacy specific-kernel names
        # ('triton_v12', 'varlen_bim_v1') map to 'triton'; TS2TSAttention
        # selects the right kernel from the block_mask type at runtime.
        backend = 'flex' if attention_backend == 'flex' else 'triton'
        self.attn = TS2TSAttention(
            dim=n_embd, num_heads=n_head, max_seq_len=max_seq_len,
            fp8_out_proj=fp8, backend=backend,
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
