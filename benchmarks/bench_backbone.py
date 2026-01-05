import torch
from torch.nn.attention.flex_attention import create_block_mask

from tunalab.benchmarking import ModuleBenchmarkRunner
from tunalab.modules.backbone import DS2DSBackbone


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def get_block_mask(seq_len):
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


def main():
    runner = ModuleBenchmarkRunner()
    
    results = runner.run_module_benchmark(
        module_class=DS2DSBackbone,
        module_name='DS2DSBackbone',
        parameter_space={
            'model_dim': [512, 1024],
            'num_heads': [8, 16],
            'max_seq_len': [1024, 2048, 4096],
            'num_layers': [4, 8],
            'fp8': [False, True],
        },
        init_arg_builder=lambda params: {
            'num_layers': params['num_layers'],
            'model_dim': params['model_dim'],
            'num_heads': params['num_heads'],
            'max_seq_len': params['max_seq_len'],
            'dropout': 0.0,
            'drop_path_rate': 0.0,
            'fp8': params['fp8'],
        },
        input_provider=lambda init_args: (
            torch.randn(1, init_args['max_seq_len'], init_args['model_dim'], requires_grad=True),
            get_block_mask(init_args['max_seq_len']),
        ),
        devices=['cuda'],  # FlexAttention requires CUDA
    )
    
    print(f"\nBenchmarked {len(results)} configurations for DS2DSBackbone")
    print(f"Results saved to artifacts/modules/DS2DSBackbone_*.csv")


if __name__ == '__main__':
    main()
