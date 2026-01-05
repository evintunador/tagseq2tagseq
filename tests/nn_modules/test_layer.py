import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from tunalab.modules.layer import Layer


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def get_block_mask(seq_len):
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


@pytest.mark.parametrize("n_embd,n_head,max_seq_len", [
    (256, 4, 128),
    (512, 8, 256),
])
def test_layer_forward_backward(n_embd, n_head, max_seq_len, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    module = Layer(
        n_embd=n_embd,
        n_head=n_head,
        dropout=0.0,
        max_seq_len=max_seq_len,
        fp8=False,
        drop_path_rate=0.0
    ).to(device, dtype)
    
    x = torch.randn(1, max_seq_len, n_embd, device=device, dtype=dtype, requires_grad=True)
    block_mask = get_block_mask(max_seq_len)
    
    out = module(x, block_mask)
    
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    assert out.dtype == x.dtype
    
    out.sum().backward()
    assert x.grad is not None
    for param in module.parameters():
        assert param.grad is not None


@pytest.mark.parametrize("n_embd,hidden_mult", [
    (256, 2),
    (512, 3),
])
def test_layer_mlp_dimensions(n_embd, hidden_mult, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    max_seq_len = 128
    module = Layer(
        n_embd=n_embd,
        n_head=4,
        dropout=0.0,
        max_seq_len=max_seq_len,
        fp8=False,
        drop_path_rate=0.0
    ).to(device, dtype)
    
    # Check that MLP hidden dim is roughly 8/3 * n_embd
    expected_hidden = int(8/3 * n_embd)
    # GLU uses 2 * hidden_dim internally
    assert module.mlp.hidden_dim == expected_hidden or abs(module.mlp.hidden_dim - expected_hidden) < 128
    
    x = torch.randn(1, max_seq_len, n_embd, device=device, dtype=dtype)
    block_mask = get_block_mask(max_seq_len)
    out = module(x, block_mask)
    assert out.shape == x.shape


@pytest.mark.parametrize("drop_path_rate", [0.0, 0.1, 0.3])
def test_layer_drop_path(drop_path_rate, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    n_embd = 256
    max_seq_len = 128
    
    module = Layer(
        n_embd=n_embd,
        n_head=4,
        dropout=0.0,
        max_seq_len=max_seq_len,
        fp8=False,
        drop_path_rate=drop_path_rate
    ).to(device, dtype)
    
    module.train()
    x = torch.randn(1, max_seq_len, n_embd, device=device, dtype=dtype)
    block_mask = get_block_mask(max_seq_len)
    out = module(x, block_mask)
    assert out.shape == x.shape
