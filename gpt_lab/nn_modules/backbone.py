from typing import Any, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

from gpt_lab.nn_modules.catalog_utils import (
    ignore_if_no_cuda, 
    ModuleTestConfig, 
    BenchmarkConfig, 
    Competitor
)
from gpt_lab.nn_modules.channel_mixing.fp8_linear import is_hopper_available

# Check for CUDA availability before importing CUDA-specific modules
ignore_if_no_cuda()

from gpt_lab.nn_modules.layer import Layer


class DS2DSBackbone(nn.Module):
    def __init__(
        self,
        num_layers: int, 
        layer_repeat: int,
        model_dim: int, 
        num_heads: int,
        max_seq_len: int, 
        dropout: float,
        drop_path_rate: float,
        fp8: bool = False,
        **kwargs
    ):
        # if kwargs:
        #     print(f"[{self.__class__.__name__}] Unused kwargs passed to constructor: {kwargs}")
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        
        self.layers = nn.ModuleList([Layer(
            n_embd=model_dim, 
            n_head=num_heads, 
            max_seq_len=max_seq_len,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
        ) for _ in range(num_layers)])
        
        # Repeat the sequence of layers layer_repeat times (allowing for recurrence/weight sharing)
        self.layers_repeated = [self.layers[i // layer_repeat] for i in range(num_layers * layer_repeat)]

    def forward(self, x: Tensor, block_mask: Any):
        """
        Args:
            x: Input embeddings of shape (B, T, C) or (Total_Tokens, C)
            block_mask: The attention block mask (passed in, not created here)
        """
        for layer in self.layers_repeated:
            x = layer(
                x=x, 
                block_mask=block_mask
            )
        return x


##################################################
#################### TESTING ####################
##################################################


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


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
    output_tensor = outputs
    # Handle potential tuple return if harness wraps it differently, though usually direct
    if isinstance(output_tensor, tuple):
        output_tensor = output_tensor[0]
        
    # Backbone preserves shape: (B, T, C) -> (B, T, C)
    assert output_tensor.shape == input_tensor.shape, f"Expected output shape {input_tensor.shape}, but got {output_tensor.shape}"
    assert output_tensor.dtype == input_tensor.dtype


def backbone_run_filter(inputs: Tuple[Any]) -> bool:
    # FlexSelfAttention requires CUDA
    if len(inputs) > 0 and isinstance(inputs[0], torch.Tensor):
        if 'cuda' not in str(inputs[0].device):
            return False
    return True


__competitors__ = {
    'DS2DSBackbone': Competitor(module_class=DS2DSBackbone, run_filter=backbone_run_filter),
}


dims_to_test = [256]
num_heads_to_test = [4]
max_seq_len_to_test = [128]


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='DS2DSBackbone',
    test_cases=[
        {
            'init_args': {
                'num_layers': 2,
                'layer_repeat': 1,
                'model_dim': dim,
                'num_heads': num_heads,
                'max_seq_len': max_seq_len,
                'dropout': 0.0,
                'drop_path_rate': 0.0,
                'fp8': fp8,
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
    dim = init_args['model_dim']
    # FlexSelfAttention requires batch_size=1
    return (
        torch.randn(1, max_seq_len, dim, requires_grad=True),
        get_block_mask(max_seq_len)
    )


__benchmark_config__ = BenchmarkConfig(
    module_name='DS2DSBackbone',
    competitors=__competitors__,
    parameter_space={
        'model_dim': [512, 1024],
        'num_heads': [8],
        'max_seq_len': [1024, 4096],
        'num_layers': [4, 8],
        'fp8': [True, False] if is_hopper_available() else [False],
    },
    init_arg_builder=lambda params: {
        'num_layers': params['num_layers'],
        'layer_repeat': 1,
        'model_dim': params['model_dim'],
        'num_heads': params['num_heads'],
        'max_seq_len': params['max_seq_len'],
        'dropout': 0.0,
        'drop_path_rate': 0.0,
        'fp8': params['fp8'],
    },
    input_provider=benchmark_input_provider,
)
