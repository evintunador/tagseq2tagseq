import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from model.modules.training_module import TS2TSTrainingModule


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def simple_block_mask_creator(**kwargs) -> BlockMask:
    """Simple causal block mask creator for testing.

    Note: the training module passes input_ids (already sliced) to the mask creator,
    so tokens.shape[1] is the actual input sequence length.
    """
    tokens = kwargs['tokens']
    seq_len = tokens.shape[1]
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


@pytest.mark.parametrize("model_dim,vocab_size,num_layers", [
    (256, 1024, 2),
    (512, 2048, 4),
])
def test_training_module_forward(model_dim, vocab_size, num_layers, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    module = TS2TSTrainingModule.from_config(
        block_mask_creator=simple_block_mask_creator,
        vocab_size=vocab_size,
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=4,
        max_seq_len=128,
        dropout=0.0,
        drop_path_rate=0.0,
        fp8=False,
        weight_tying=True,
        dtype=torch.bfloat16
    ).to(device, torch.bfloat16)

    # Create batch dict with tokens (seq_len + 1 for input and target)
    batch = {
        'tokens': torch.randint(0, vocab_size, (1, 129), device=device)
    }
    
    out = module(batch)

    assert isinstance(out, torch.Tensor), "Expected output to be a Tensor"
    assert out.ndim == 0, f"Expected scalar loss, got shape {out.shape}"


@pytest.mark.parametrize("weight_tying", [True, False])
def test_training_module_weight_tying(weight_tying, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    model_dim = 256
    vocab_size = 1024
    
    module = TS2TSTrainingModule.from_config(
        block_mask_creator=simple_block_mask_creator,
        vocab_size=vocab_size,
        num_layers=2,
        model_dim=model_dim,
        num_heads=4,
        max_seq_len=128,
        dropout=0.0,
        drop_path_rate=0.0,
        fp8=False,
        weight_tying=weight_tying,
        dtype=torch.bfloat16
    ).to(device, torch.bfloat16)

    if weight_tying:
        assert module.loss_fn.weight is module.embedding.weight
    else:
        assert module.loss_fn.weight is not module.embedding.weight
    
    batch = {
        'tokens': torch.randint(0, vocab_size, (1, 129), device=device)
    }
    out = module(batch)
    assert isinstance(out, torch.Tensor) and out.ndim == 0


def test_training_module_backward_pass(device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    model_dim = 256
    vocab_size = 1024
    
    module = TS2TSTrainingModule.from_config(
        block_mask_creator=simple_block_mask_creator,
        vocab_size=vocab_size,
        num_layers=2,
        model_dim=model_dim,
        num_heads=4,
        max_seq_len=128,
        dropout=0.0,
        drop_path_rate=0.0,
        fp8=False,
        weight_tying=True,
        dtype=torch.bfloat16
    ).to(device, torch.bfloat16)

    batch = {
        'tokens': torch.randint(0, vocab_size, (1, 129), device=device)
    }

    loss = module(batch)

    loss.backward()
    
    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.parametrize("max_seq_len", [64, 128, 256])
def test_training_module_variable_sequence_lengths(max_seq_len, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    module = TS2TSTrainingModule.from_config(
        block_mask_creator=simple_block_mask_creator,
        vocab_size=1024,
        num_layers=2,
        model_dim=256,
        num_heads=4,
        max_seq_len=max_seq_len,
        dropout=0.0,
        drop_path_rate=0.0,
        fp8=False,
        weight_tying=True,
        dtype=torch.bfloat16
    ).to(device, torch.bfloat16)
    
    batch = {
        'tokens': torch.randint(0, 1024, (1, max_seq_len + 1), device=device)
    }
    
    out = module(batch)
    assert out.ndim == 0
