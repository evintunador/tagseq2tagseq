from typing import Dict, Any, Type, Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from gpt_lab.nn_modules.norms.rms_norm import RMSNorm
from gpt_lab.nn_modules.losses.fused_cross_entropy import FusedLinearCELoss
from gpt_lab.nn_modules.catalog_utils import (
    ignore_if_no_cuda,
    ModuleTestConfig,
    BenchmarkConfig,
    Competitor
)
from gpt_lab.nn_modules.channel_mixing.fp8_linear import is_hopper_available
from .backbone import DS2DSBackbone

ignore_if_no_cuda()


# Placeholder for the standalone block mask creation function
def create_dag_block_mask(batch: Dict[str, Any], device: torch.device) -> Any:
    # Implement your complicated cool DAG block mask logic here
    return None


class DS2DSTrainingModule(nn.Module):
    def __init__(
        self,
        block_mask_creator: Callable[[], BlockMask],
        vocab_size: int,
        num_layers: int, 
        model_dim: int, 
        num_heads: int,
        max_seq_len: int, 
        dropout: float,
        drop_path_rate: float,
        fp8: bool = False,
        weight_tying: bool = True,
        ignore_index: int = -100,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.weight_tying = weight_tying

        self.block_mask_creator = block_mask_creator
        
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.backbone = DS2DSBackbone(
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            fp8=fp8,
        )
        self.norm = RMSNorm(model_dim)
        loss_weight = self.embedding.weight if weight_tying else None
        self.loss_fn = FusedLinearCELoss(
            D=model_dim,
            V=vocab_size,
            dtype=dtype,
            ignore_index=ignore_index,
            weight=loss_weight
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Type[Tensor] | Any]:
        tokens = batch['tokens']
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]

        block_mask = self.block_mask_creator(**batch)
        
        x = self.embedding(input_ids)
        x = self.backbone(x, block_mask=block_mask)
        x = self.norm(x)
        
        loss = self.loss_fn(x, target_ids)
        
        return {'loss': loss, 'ce_loss': loss}


##################################################
#################### TESTING ####################
##################################################


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def simple_block_mask_creator(**kwargs) -> BlockMask:
    tokens = kwargs['tokens']
    # input sequence length is tokens.shape[1] - 1
    seq_len = tokens.shape[1] - 1
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


def output_validator(
        module: nn.Module,
        inputs: Tuple[Any],
        outputs: Tuple[Any],
) -> None:
    """
    Validates whether the base module output meets expectations.
    """
    output_dict = outputs[0] if isinstance(outputs, tuple) else outputs
    assert isinstance(output_dict, dict), "Expected output to be a dictionary"
    assert 'loss' in output_dict, "Expected 'loss' in output dictionary"
    loss = output_dict['loss']
    assert loss.ndim == 0, f"Expected scalar loss, but got shape {loss.shape}"


def training_module_run_filter(inputs: Tuple[Any]) -> bool:
    # Check inputs for device. inputs is (batch_dict, )
    if len(inputs) > 0 and isinstance(inputs[0], dict):
        batch = inputs[0]
        if 'tokens' in batch:
            if 'cuda' not in str(batch['tokens'].device):
                return False
    return True


__competitors__ = {
    'DS2DSTrainingModule': Competitor(module_class=DS2DSTrainingModule, run_filter=training_module_run_filter),
}


test_dims = [256]
test_heads = [4]
test_seq_len = [128]


__test_config__ = ModuleTestConfig(
    competitors=__competitors__,
    reference_competitor='DS2DSTrainingModule',
    test_cases=[
        {
            'init_args': {
                'block_mask_creator': simple_block_mask_creator,
                'vocab_size': 1024,
                'num_layers': 2,
                'model_dim': dim,
                'num_heads': num_heads,
                'max_seq_len': max_seq_len,
                'dropout': 0.0,
                'drop_path_rate': 0.0,
                'fp8': fp8,
                'weight_tying': True,
                'dtype': torch.bfloat16,
            },
            'input_args': ({
                'tokens': torch.randint(0, 1024, (1, max_seq_len + 1)), # Batch size 1 for FlexAttention
            },),
            'output_validator': output_validator,
            'tolerances_fn': lambda x: {'atol': 1e-1, 'rtol': 1e-1},
            'case_descriptor': f'dim={dim}_heads={num_heads}_seq={max_seq_len}_fp8={fp8}',
        }
        for dim in test_dims
        for num_heads in test_heads
        for max_seq_len in test_seq_len
        for fp8 in ([True, False] if is_hopper_available() else [False])
    ]
)


##################################################
################# BENCHMARKING ###################
##################################################


def benchmark_input_provider(init_args: dict) -> tuple:
    max_seq_len = init_args['max_seq_len']
    vocab_size = init_args['vocab_size']
    # Batch size 1 for FlexAttention
    return ({
        'tokens': torch.randint(0, vocab_size, (1, max_seq_len + 1))
    },)


__benchmark_config__ = BenchmarkConfig(
    module_name='DS2DSTrainingModule',
    competitors=__competitors__,
    parameter_space={
        'model_dim': [512, 1024],
        'num_heads': [8],
        'max_seq_len': [1024, 2048],
        'num_layers': [4],
        'fp8': [True, False] if is_hopper_available() else [False],
    },
    init_arg_builder=lambda params: {
        'block_mask_creator': simple_block_mask_creator,
        'vocab_size': 50257,
        'num_layers': params['num_layers'],
        'model_dim': params['model_dim'],
        'num_heads': params['num_heads'],
        'max_seq_len': params['max_seq_len'],
        'dropout': 0.0,
        'drop_path_rate': 0.0,
        'fp8': params['fp8'],
        'weight_tying': True,
        'dtype': torch.bfloat16,
    },
    input_provider=benchmark_input_provider,
)
