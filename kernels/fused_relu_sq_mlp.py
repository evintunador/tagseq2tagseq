"""Fused ReLU² MLP Triton kernel for A100 (Ampere / sm80).

Computes the forward and backward of:
    post = relu(x @ W1.T) ** 2        (fwd)
    d_pre = (grad @ W2.T) * 2 * relu(pre)   (bwd activation, fused with W2 linear backward)

Weight conventions (differ from nn.Linear to avoid .T.contiguous() in backward):
    W1 : (H, C)  — standard nn.Linear layout (out_features, in_features)
    W2 : (H, C)  — TRANSPOSED from standard: forward uses  post @ W2  (not F.linear)

The kernel is written for A100 using tl.make_block_ptr (software block pointers,
not TMA), autotuned over a sensible set of configs for large bf16 matmuls.

Public API
----------
fused_relu_sq(a, b, pre_saved=None) -> (pre, post) | d_pre
    Shared entry point for forward and backward.
    - Forward  (pre_saved=None):  a:(M,K), b:(N,K)  →  pre:(M,N), post:(M,N)
    - Backward (pre_saved given): a:(M,K), b:(N,K)  →  d_pre:(M,N)
    Both modes compute  a @ b.T  in fp32, then apply a different epilogue.

FusedReLUSquaredFunction  — torch.autograd.Function wrapping the full MLP pass.

FusedReLUSquaredMLP       — nn.Module (drop-in experiment; integrates into Layer).
"""

import math

import torch
import torch.nn as nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 32,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=5, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8}, num_stages=4, num_warps=8),
    ],
    key=["M", "N", "K", "FORWARD"],
)
@triton.jit
def _relu_sq_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr, aux_ptr,
    # Dimensions
    M, N, K,
    # Strides for a (M, K)
    stride_am, stride_ak,
    # Strides for b (N, K) — accessed as b.T = (K, N) by swapping shape/strides
    stride_bn, stride_bk,
    # Strides for c and aux (M, N)
    stride_cm, stride_cn,
    stride_auxm, stride_auxn,
    # Tiling / mode
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    FORWARD: tl.constexpr,
):
    """Core kernel: computes  acc = a @ b.T  in fp32, then applies epilogue.

    Forward epilogue  (FORWARD=True):
        saves  pre = acc.to(bf16)  in aux
        stores post = relu(pre)^2  in c

    Backward epilogue (FORWARD=False):
        loads  pre_saved  from aux   (pre-relu activation from forward pass)
        stores d_pre = acc * 2 * relu(pre_saved)  in c
        (acc here is the gradient through the second linear layer)
    """
    # -------------------------------------------------------------------
    # Grouped swizzle for improved L2 cache reuse (from Triton matmul tutorial)
    # -------------------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -------------------------------------------------------------------
    # Block pointers
    # -------------------------------------------------------------------
    # a: (M, K) row-major
    a_block = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    # b accessed as b.T = (K, N): swap shape[0/1] and strides[0/1]
    b_T_block = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(0, 1),
    )

    # -------------------------------------------------------------------
    # Matmul accumulation: acc = a @ b.T  in fp32
    # -------------------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        b_t = tl.load(b_T_block, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b_t, acc)
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_T_block = tl.advance(b_T_block, (BLOCK_K, 0))

    # -------------------------------------------------------------------
    # Output block pointers (c and aux share the same (M, N) layout)
    # -------------------------------------------------------------------
    c_block = tl.make_block_ptr(
        base=c_ptr, shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    aux_block = tl.make_block_ptr(
        base=aux_ptr, shape=(M, N),
        strides=(stride_auxm, stride_auxn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )

    lin_out = acc.to(tl.bfloat16)  # linear output (bf16), shared name in both branches

    if FORWARD:
        # Save pre-relu activation for use in backward pass
        tl.store(aux_block, lin_out, boundary_check=(0, 1))
        # post = relu(lin_out)^2  (work in fp32 to avoid bf16 multiply precision issues)
        lin_f32 = lin_out.to(tl.float32)
        act = tl.maximum(lin_f32, 0.)
        tl.store(c_block, (act * act).to(tl.bfloat16), boundary_check=(0, 1))
    else:
        # Load pre-relu activation saved during forward
        pre_f32 = tl.load(aux_block, boundary_check=(0, 1)).to(tl.float32)
        # d(relu(x)^2)/dx = 2 * relu(x) = 2 * x * (x > 0)
        gate = tl.where(pre_f32 > 0., pre_f32, tl.zeros_like(pre_f32))
        d_pre = acc * (2.0 * gate)
        tl.store(c_block, d_pre.to(tl.bfloat16), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def fused_relu_sq(a: torch.Tensor, b: torch.Tensor,
                  pre_saved: torch.Tensor | None = None):
    """Compute  a @ b.T  with a fused relu²-related epilogue.

    Both a and b must be 2-D, contiguous, bf16, on the same CUDA device.

    Forward mode (pre_saved=None):
        Returns (pre, post) where
            pre  = a @ b.T           (bf16, saved for backward)
            post = relu(pre) ** 2    (bf16)

    Backward mode (pre_saved given):
        Returns d_pre where
            d_pre = (a @ b.T) * 2 * relu(pre_saved)   (bf16)
        This fuses the gradient-through-linear-W2 with the activation gradient.
    """
    assert a.ndim == 2 and b.ndim == 2, "a and b must be 2-D"
    assert a.is_contiguous() and b.is_contiguous(), "a and b must be contiguous"
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16

    M, K = a.shape
    N, K2 = b.shape
    assert K == K2

    FORWARD = pre_saved is None
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)

    if FORWARD:
        aux = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
    else:
        assert pre_saved is not None
        assert pre_saved.shape == (M, N)
        assert pre_saved.is_contiguous() and pre_saved.dtype == torch.bfloat16
        aux = pre_saved  # read-only in backward mode

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _relu_sq_kernel[grid](
        a, b, c, aux,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        aux.stride(0), aux.stride(1),
        FORWARD=FORWARD,
    )

    if FORWARD:
        return c, aux   # (post, pre)  — note: aux = pre
    else:
        return c        # d_pre


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------

class FusedReLUSquaredFunction(torch.autograd.Function):
    """Full MLP pass:  out = relu(x @ W1.T)^2 @ W2

    Weight layouts:
        W1 : (H, C)  — standard nn.Linear weight (out, in)
        W2 : (H, C)  — TRANSPOSED from standard; forward: post @ W2 (not F.linear)

    This avoids .T.contiguous() in the backward path.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor):
        x_flat = x.reshape(-1, x.shape[-1]).contiguous()
        post, pre = fused_relu_sq(x_flat, W1)   # returns (post, pre) = (c, aux)
        out_flat = post @ W2                      # (M, H) @ (H, C) = (M, C)
        ctx.save_for_backward(x_flat, W1, W2, pre, post)
        return out_flat.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_flat, W1, W2, pre, post = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        # Gradient w.r.t. W2:  post.T @ grad  (H, M) @ (M, C) = (H, C)
        dW2 = post.t() @ grad_flat

        # Fused: gradient through W2's backward + relu² activation backward
        # kernel computes:  (grad @ W2.T) * 2 * relu(pre)
        # = (M,C) @ (H,C).T * ... = (M,C) @ (C,H) * ... = (M,H)
        d_pre = fused_relu_sq(grad_flat, W2, pre_saved=pre)

        # Gradient w.r.t. W1:  d_pre.T @ x  (H, M) @ (M, C) = (H, C)
        dW1 = d_pre.t() @ x_flat

        # Gradient w.r.t. x:  d_pre @ W1  (M, H) @ (H, C) = (M, C)
        dx = d_pre @ W1

        return dx.reshape(grad_output.shape), dW1, dW2


# ---------------------------------------------------------------------------
# nn.Module
# ---------------------------------------------------------------------------

class FusedReLUSquaredMLP(nn.Module):
    """Drop-in MLP replacement using fused relu² kernel.

    Expansion ratio: 4× (model_dim → 4*model_dim → model_dim).

    Note: W2 is stored in (hidden_dim, model_dim) layout rather than the
    standard (model_dim, hidden_dim) to avoid transposition in the backward
    path.  The forward pass is:  post @ self.W2  (not F.linear(post, self.W2)).
    """

    def __init__(self, model_dim: int):
        super().__init__()
        hidden_dim = 4 * model_dim
        # Standard nn.Linear layout for W1: (H, C)
        self.W1 = nn.Parameter(torch.empty(hidden_dim, model_dim))
        # Transposed-from-standard layout for W2: (H, C)
        self.W2 = nn.Parameter(torch.empty(hidden_dim, model_dim))
        self._reset_parameters(model_dim, hidden_dim)

    def _reset_parameters(self, model_dim: int, hidden_dim: int) -> None:
        # Kaiming uniform for W1 (fan_in = model_dim)
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        # Zero-init for W2 (output projection) — modded-nanogpt style, muP-friendly
        nn.init.zeros_(self.W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return FusedReLUSquaredFunction.apply(x, self.W1, self.W2)
