import torch
from torch.nn.attention.flex_attention import create_block_mask

from tunalab.benchmarking import ModuleBenchmarkRunner
from tunalab.modules.layer import Layer


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def get_block_mask(seq_len):
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


def main():
    runner = ModuleBenchmarkRunner()
    
    results = runner.run_module_benchmark(
        module_class=Layer,
        module_name='DS2DS_Layer',
        parameter_space={
            'n_embd': [256, 512, 1024],
            'n_head': [4, 8],
            'max_seq_len': [512, 1024, 2048],
            'fp8': [False, True],
        },
        init_arg_builder=lambda params: {
            'n_embd': params['n_embd'],
            'n_head': params['n_head'],
            'dropout': 0.0,
            'max_seq_len': params['max_seq_len'],
            'fp8': params['fp8'],
            'drop_path_rate': 0.0,
        },
        input_provider=lambda init_args: (
            torch.randn(1, init_args['max_seq_len'], init_args['n_embd'], requires_grad=True),
            get_block_mask(init_args['max_seq_len']),
        ),
        devices=['cuda'],  # FlexAttention requires CUDA
    )
    
    print(f"\nBenchmarked {len(results)} configurations for DS2DS Layer")
    print(f"Results saved to artifacts/modules/DS2DS_Layer_*.csv")


if __name__ == '__main__':
    main()
