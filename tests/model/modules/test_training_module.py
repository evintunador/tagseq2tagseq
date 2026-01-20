import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask

from model.modules.training_module import DS2DSTrainingModule


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def simple_block_mask_creator(**kwargs) -> BlockMask:
    """Simple causal block mask creator for testing."""
    tokens = kwargs['tokens']
    # Input sequence length is tokens.shape[1] - 1 (we shift for targets)
    seq_len = tokens.shape[1] - 1
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


@pytest.mark.parametrize("model_dim,vocab_size,num_layers", [
    (256, 1024, 2),
    (512, 2048, 4),
])
def test_training_module_forward(model_dim, vocab_size, num_layers, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    module = DS2DSTrainingModule(
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
    ).to(device)
    
    # Create batch dict with tokens (seq_len + 1 for input and target)
    batch = {
        'tokens': torch.randint(0, vocab_size, (1, 129), device=device)
    }
    
    out = module(batch)
    
    assert isinstance(out, dict), "Expected output to be a dictionary"
    assert 'loss' in out, "Expected 'loss' in output dictionary"
    assert 'ce_loss' in out, "Expected 'ce_loss' in output dictionary"
    assert out['loss'].ndim == 0, f"Expected scalar loss, got shape {out['loss'].shape}"
    assert out['ce_loss'].ndim == 0, f"Expected scalar ce_loss, got shape {out['ce_loss'].shape}"


@pytest.mark.parametrize("weight_tying", [True, False])
def test_training_module_weight_tying(weight_tying, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    model_dim = 256
    vocab_size = 1024
    
    module = DS2DSTrainingModule(
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
    ).to(device)
    
    if weight_tying:
        assert module.loss_fn.weight is module.embedding.weight
    else:
        assert module.loss_fn.weight is not module.embedding.weight
    
    batch = {
        'tokens': torch.randint(0, vocab_size, (1, 129), device=device)
    }
    out = module(batch)
    assert 'loss' in out


def test_training_module_backward_pass(device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    model_dim = 256
    vocab_size = 1024
    
    module = DS2DSTrainingModule(
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
    ).to(device)
    
    batch = {
        'tokens': torch.randint(0, vocab_size, (1, 129), device=device)
    }
    
    out = module(batch)
    loss = out['loss']
    
    loss.backward()
    
    for name, param in module.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.parametrize("max_seq_len", [64, 128, 256])
def test_training_module_variable_sequence_lengths(max_seq_len, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    module = DS2DSTrainingModule(
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
    ).to(device)
    
    batch = {
        'tokens': torch.randint(0, 1024, (1, max_seq_len + 1), device=device)
    }
    
    out = module(batch)
    assert out['loss'].ndim == 0
