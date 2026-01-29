import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from model.modules.backbone import DS2DSBackbone


def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def get_block_mask(seq_len):
    return create_block_mask(causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len)


@pytest.mark.parametrize("model_dim,num_heads,num_layers,max_seq_len", [
    (256, 4, 4, 128),
    (512, 8, 6, 256),
])
def test_backbone_forward_backward(model_dim, num_heads, num_layers, max_seq_len, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    module = DS2DSBackbone(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=0.0,
        drop_path_rate=0.0,
        fp8=False,
    ).to(device, dtype)
    
    x = torch.randn(1, max_seq_len, model_dim, device=device, dtype=dtype, requires_grad=True)
    block_mask = get_block_mask(max_seq_len)
    
    out = module(x, block_mask)
    
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    assert out.dtype == x.dtype
    
    out.sum().backward()
    assert x.grad is not None
    for param in module.parameters():
        assert param.grad is not None


@pytest.mark.parametrize("num_layers", [2, 4, 8])
def test_backbone_skip_connections(num_layers, device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    model_dim = 256
    max_seq_len = 128
    
    module = DS2DSBackbone(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=4,
        max_seq_len=max_seq_len,
        dropout=0.0,
        drop_path_rate=0.0,
        fp8=False,
    ).to(device, dtype)
    
    assert hasattr(module, 'skip_weights')
    expected_skip_weights = num_layers // 2
    assert module.skip_weights.shape == (expected_skip_weights,)
    
    x = torch.randn(1, max_seq_len, model_dim, device=device, dtype=dtype)
    block_mask = get_block_mask(max_seq_len)
    out = module(x, block_mask)
    assert out.shape == x.shape


def test_backbone_module_list_structure(device, dtype):
    if device != 'cuda':
        pytest.skip("FlexAttention requires CUDA")
    
    num_layers = 4
    module = DS2DSBackbone(
        num_layers=num_layers,
        model_dim=256,
        num_heads=4,
        max_seq_len=128,
        dropout=0.0,
        drop_path_rate=0.0,
        fp8=False,
    ).to(device, dtype)
    
    assert hasattr(module, 'layers')
    assert len(module.layers) == num_layers
    
    from tunalab.modules.layer import Layer
    for layer in module.layers:
        assert isinstance(layer, Layer)
