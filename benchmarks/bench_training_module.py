import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from tunalab.benchmarking import ModuleBenchmarkRunner
from tunalab.modules.training_module import DS2DSTrainingModule


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def simple_block_mask_creator(**kwargs) -> BlockMask:
    tokens = kwargs['tokens']
    seq_len = tokens.shape[1] - 1
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


def main():
    runner = ModuleBenchmarkRunner()
    
    results = runner.run_module_benchmark(
        module_class=DS2DSTrainingModule,
        module_name='DS2DSTrainingModule',
        parameter_space={
            'model_dim': [512, 1024],
            'num_heads': [8],
            'max_seq_len': [1024, 2048],
            'num_layers': [4, 8],
            'fp8': [False, True],
        },
        init_arg_builder=lambda params: {
            'block_mask_creator': simple_block_mask_creator,
            'vocab_size': 50257,  # GPT-2 vocab size
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
        input_provider=lambda init_args: (
            {
                'tokens': torch.randint(
                    0, 
                    50257, 
                    (1, init_args['max_seq_len'] + 1)
                )
            },
        ),
        devices=['cuda'],  # FlexAttention requires CUDA
    )
    
    print(f"\nBenchmarked {len(results)} configurations for DS2DSTrainingModule")
    print(f"Results saved to artifacts/modules/DS2DSTrainingModule_*.csv")


if __name__ == '__main__':
    main()
