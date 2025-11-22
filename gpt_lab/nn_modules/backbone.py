import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
import liger_kernel.transformers.LigerFusedLinearCrossEntropyLoss as FusedCELoss

from gpt_lab.nn_modules.norms.rms_norm import RMSNorm
from gpt_lab.nn_modules.catalog_utils import next_multiple, ignore_if_no_cuda

# Check for CUDA availability before importing CUDA-specific modules
ignore_if_no_cuda()

from gpt_lab.nn_modules.layer import Layer
from gpt_lab.nn_modules.channel_mixing.fp8_linear import FP8Linear


class DS2DS(nn.Module):
    def __init__(
        vocab_size: int,
        num_layers: int, 
        layer_repeat: int,
        model_dim: int, 
        num_heads: int,
        max_seq_len: int, 
        dropout: float,
        drop_path_rate: float,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.layers = nn.ModuleList([Layer(
            n_embd=model_dim, 
            n_head=num_heads, 
            max_seq_len=max_seq_len,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        ) for _ in range(num_layers)])
        # Repeat the sequence of layers layer_repeat times (allowing for recurrence)
        self.layers_repeated = [self.layers[i % num_layers] for i in range(num_layers * layer_repeat)]
