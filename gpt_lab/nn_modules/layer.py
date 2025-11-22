from typing import Tuple, Any, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from gpt_lab.nn_modules.sequence_mixing.flex_self_attention import FlexSelfAttention
from gpt_lab.nn_modules.channel_mixing.glu import GLU
from gpt_lab.nn_modules.regularization.drop_path import DropPath
from gpt_lab.nn_modules.norms.rms_norm import RMSNorm
from gpt_lab.nn_modules.catalog_utils import ModuleTestConfig, BenchmarkConfig, Competitor, ignore_if_no_cuda
from gpt_lab.nn_modules.channel_mixing.fp8_linear import is_hopper_available


ignore_if_no_cuda()


class Layer(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, max_seq_len: int, fp8: bool, drop_path_rate: float):
        super().__init__()
        self.drop_path = DropPath(drop_path_rate)
        self.ln_1 = RMSNorm(n_embd)
        self.attn = FlexSelfAttention(
            dim=n_embd, 
            num_heads=n_head, 
            max_seq_len=max_seq_len, 
            fp8_out_proj=fp8
        )
        self.ln_2 = RMSNorm(n_embd)
        self.mlp = GLU(
            in_dim=n_embd, 
            out_dim=n_embd, 
            hidden_dim=int(8/3*n_embd), 
            activation="silu", 
            dropout=dropout, 
            fp8=fp8
        )

    def forward(self, x: Tensor, block_mask: BlockMask):
        x = x + self.drop_path(self.attn(self.ln_1(x), block_mask=block_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


##################################################
#################### TESTING ####################
##################################################


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# Helper to create block mask
def get_block_mask(seq_len):
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


def output_validator(
        module: nn.Module,
        inputs: Tuple[Any],
        outputs: Tuple[Any],
) -> None:
    """
    Validates whether the base module output meets expectations.
    """
    input_tensor = inputs[0] 
    output_tensor = outputs[0]
    # Layer preserves shape: (B, T, C) -> (B, T, C)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype


def layer_run_filter(inputs: Tuple[Any]) -> bool:
    # FlexSelfAttention requires CUDA
    if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
        if 'cuda' not in str(inputs[0].device):
            return False
    return True


__competitors__ = {
    'Layer': Competitor(module_class=Layer, run_filter=layer_run_filter),
}


# Test params
dims_to_test = [256]
num_heads_to_test = [4]
max_seq_len_to_test = [128] # Keep it small for quick testing


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='Layer',
    test_cases=[
        {
            'init_args': {
                'n_embd': dim, 
                'n_head': num_heads, 
                'dropout': 0.0,
                'bias': False,
                'max_seq_len': max_seq_len, 
                'fp8': fp8,
                'drop_path_rate': 0.0
            },
            'input_args': (
                torch.randn(1, max_seq_len, dim, requires_grad=True), # x
                get_block_mask(max_seq_len), # block_mask
            ),
            'output_validator': output_validator,
            'tolerances_fn': lambda x: {'atol': 1e-2, 'rtol': 1e-2},
            'case_descriptor': f'dim={dim}_heads={num_heads}_seq={max_seq_len}_fp8={fp8}',
        }
        for dim in dims_to_test
        for num_heads in num_heads_to_test
        for max_seq_len in max_seq_len_to_test
        for fp8 in ([True, False] if is_hopper_available() else [False])
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict) -> tuple:
    max_seq_len = init_args['max_seq_len']
    dim = init_args['n_embd']
    # FlexSelfAttention usually requires batch_size=1
    return (
        torch.randn(1, max_seq_len, dim, requires_grad=True),
        get_block_mask(max_seq_len)
    )


__benchmark_config__ = BenchmarkConfig(
    module_name='Layer',
    competitors=__competitors__,
    parameter_space={
        'n_embd': [256, 512, 1024],
        'n_head': [4, 8],
        'max_seq_len': [512, 1024],
        'fp8': [True, False] if is_hopper_available() else [False],
    },
    init_arg_builder=lambda params: {
        'n_embd': params['n_embd'],
        'n_head': params['n_head'],
        'dropout': 0.0,
        'bias': False,
        'max_seq_len': params['max_seq_len'],
        'fp8': params['fp8'],
        'drop_path_rate': 0.0
    },
    input_provider=benchmark_input_provider,
)
