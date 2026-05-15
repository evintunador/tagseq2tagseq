"""
Polar Express orthogonalization kernels.

Three Triton helper kernels required by the Polar Express algorithm
(https://arxiv.org/pdf/2505.16932):
  - XXT(X, out)           — computes X @ X.T into a pre-allocated buffer
  - XTX(X, out)           — computes X.T @ X (tall-matrix variant)
  - ba_plus_cAA(A, ...)   — fused beta*A + alpha*(A@A)

Both symmetric matmuls exploit upper-triangle symmetry to skip ~half the work.

The polar_express() function fuses Nesterov momentum + 5-iteration Polar Express
orthogonalization (replacing Newton-Schulz).  It is decorated with
@torch.compile(dynamic=False, fullgraph=True): shape specialization produces
the fastest possible kernels; fullgraph=True fails loudly if a graph break
sneaks in.

momentum_t must be a 0-D CPU tensor (not a Python float) — this prevents
torch.compile from recompiling when the momentum value changes (e.g. during
momentum warmup), while still being fully compilable as a graph node.

Hardcoded tile configs are from H100 autotuning; they remain functional
(though not optimal) for our A100 cluster.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Shared helper: map a flat program ID to a 2-D block index with swizzling
# ---------------------------------------------------------------------------

@triton.jit
def _pid_to_block(pid, M,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  GROUP_SIZE_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
    batch_idx  = pid // (num_pid_m * num_pid_n)
    pid        = pid  % (num_pid_m * num_pid_n)
    pid_m = pid // num_pid_n
    pid_n = pid  % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    return batch_idx, pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N


# ---------------------------------------------------------------------------
# XXT — symmetric matmul C = A @ A.T  (A: [M, K], C: [M, M])
# ---------------------------------------------------------------------------

@triton.jit
def XXT_kernel(
    A_ptr, C_ptr, M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    skip_below = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_above = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_below or skip_above:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs  = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_rem = K - k * BLOCK_SIZE_K
        a  = tl.load(a_ptrs,  mask=offs_k[None, :] < k_rem, other=0.0)
        at = tl.trans(tl.load(at_ptrs, mask=offs_k[None, :] < k_rem, other=0.0))
        acc = tl.dot(a, at, acc)
        a_ptrs  += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out = acc.to(C_ptr.dtype.element_ty)
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c
    tl.store(c_ptrs, out, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < M))
    c_ptrs_t = C_ptr + offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c
    tl.store(c_ptrs_t, out.T, mask=(offs_cn[:, None] < M) & (offs_cm[None, :] < M))


def XXT(A: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute out = A @ A.T  (out must be pre-allocated, shape [..., M, M])."""
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    batch = A.size(0) if A.ndim == 3 else 1
    if K == 768:
        BM, BN, BK, stages, warps = 128, 128, 64, 4, 8
    else:
        BM, BN, BK, stages, warps = 64, 128, 128, 4, 8
    grid = (batch * triton.cdiv(M, BM) * triton.cdiv(M, BN),)
    XXT_kernel[grid](
        A, out, M, K,
        A.stride(0) if A.ndim == 3 else 0, A.stride(-2), A.stride(-1),
        out.stride(0) if out.ndim == 3 else 0, out.stride(-2), out.stride(-1),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=8, LOWER_UPPER=1,
        num_stages=stages, num_warps=warps,
    )
    return out


# ---------------------------------------------------------------------------
# XTX — symmetric matmul C = A.T @ A  (A: [M, K], C: [K, K])
# ---------------------------------------------------------------------------

@triton.jit
def XTX_kernel(
    A_ptr, C_ptr, M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, k_idx, n_idx = _pid_to_block(pid, K, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    skip_below = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= k_idx)
    skip_above = (LOWER_UPPER != 0) and (k_idx + BLOCK_SIZE_M <= n_idx)
    if skip_below or skip_above:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_k = (k_idx + tl.arange(0, BLOCK_SIZE_M)) % K
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % K
    offs_m = tl.arange(0, BLOCK_SIZE_K)

    at_ptrs = A_ptr + offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c
    a_ptrs  = A_ptr + offs_m[:, None] * a_stride_r + offs_n[None, :] * a_stride_c

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for m in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_rem = M - m * BLOCK_SIZE_K
        at = tl.load(at_ptrs, mask=offs_m[:, None] < m_rem, other=0.0)
        a  = tl.load(a_ptrs,  mask=offs_m[:, None] < m_rem, other=0.0)
        acc = tl.dot(at.T, a, acc)
        at_ptrs += BLOCK_SIZE_K * a_stride_r
        a_ptrs  += BLOCK_SIZE_K * a_stride_r

    out = acc.to(C_ptr.dtype.element_ty)
    offs_ck = k_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_ck[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c
    tl.store(c_ptrs, out, mask=(offs_ck[:, None] < K) & (offs_cn[None, :] < K))
    c_ptrs_t = C_ptr + offs_cn[:, None] * c_stride_r + offs_ck[None, :] * c_stride_c
    tl.store(c_ptrs_t, out.T, mask=(offs_cn[:, None] < K) & (offs_ck[None, :] < K))


def XTX(A: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute out = A.T @ A  (out must be pre-allocated, shape [..., K, K])."""
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    batch = A.size(0) if A.ndim == 3 else 1
    if K == 768:
        BM, BN, BK, stages, warps = 128, 128, 64, 4, 8
    else:
        BM, BN, BK, stages, warps = 64, 128, 128, 4, 8
    grid = (batch * triton.cdiv(K, BM) * triton.cdiv(K, BN),)
    XTX_kernel[grid](
        A, out, M, K,
        A.stride(0) if A.ndim == 3 else 0, A.stride(-2), A.stride(-1),
        out.stride(0) if out.ndim == 3 else 0, out.stride(-2), out.stride(-1),
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=8, LOWER_UPPER=1,
        num_stages=stages, num_warps=warps,
    )
    return out


# ---------------------------------------------------------------------------
# ba_plus_cAA — fused C = beta*A + alpha*(A@A)  (A must be square)
# ---------------------------------------------------------------------------

@triton.jit
def ba_plus_cAA_kernel(
    A_ptr, C_ptr, M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    skip_below = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_above = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_below or skip_above:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs  = A_ptr + offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c
    at_ptrs = A_ptr + offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        k_rem = M - k * BLOCK_SIZE_K
        a  = tl.load(a_ptrs,  mask=offs_k[None, :] < k_rem, other=0.0)
        at = tl.trans(tl.load(at_ptrs, mask=offs_k[None, :] < k_rem, other=0.0))
        acc = tl.dot(a, at, acc)
        a_ptrs  += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add = tl.load(
        A_ptr + offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c,
        mask=(offs_am[:, None] < M) & (offs_an[None, :] < M), other=0.0,
    ).to(tl.float32)
    acc = acc * alpha + a_add * beta

    out = acc.to(C_ptr.dtype.element_ty)
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c
    tl.store(c_ptrs, out, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < M))
    c_ptrs_t = C_ptr + offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c
    tl.store(c_ptrs_t, out.T, mask=(offs_cn[:, None] < M) & (offs_cm[None, :] < M))


def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor) -> torch.Tensor:
    """Compute out = beta*A + alpha*(A@A)  (A must be square)."""
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    assert M == K
    batch = A.size(0) if A.ndim == 3 else 1
    BM, BN, BK, stages, warps = 128, 128, 64, 4, 8
    grid = (batch * triton.cdiv(M, BM) * triton.cdiv(M, BN),)
    ba_plus_cAA_kernel[grid](
        A, out, M,
        A.stride(0) if A.ndim == 3 else 0, A.stride(-2), A.stride(-1),
        out.stride(0) if out.ndim == 3 else 0, out.stride(-2), out.stride(-1),
        alpha=alpha, beta=beta,
        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
        GROUP_SIZE_M=8, LOWER_UPPER=1,
        num_stages=stages, num_warps=warps,
    )
    return out


# ---------------------------------------------------------------------------
# Polar Express: fused Nesterov momentum + orthogonalization
# ---------------------------------------------------------------------------

# Precomputed for num_iters=5, safety_factor=2e-2
polar_express_coeffs = [
    (8.156554524902461,  -22.48329292557795,   15.878769915207462),
    (4.042929935166739,   -2.808917465908714,   0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,   0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376,  0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081,  0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def polar_express(
    grad_chunk: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum_t: torch.Tensor,
    split_baddbmm: bool = False,
) -> torch.Tensor:
    """
    Fused Nesterov momentum + Polar Express orthogonalization.

    Nesterov momentum runs in FP32; orthogonalization casts to BF16 for speed.
    The result is in BF16 (same as Newton-Schulz output).

    momentum_t must be a 0-D CPU tensor so torch.compile does not recompile
    when momentum changes (e.g. during warmup ramp).

    split_baddbmm=True splits the fused addmm into two calls to avoid
    PyTorch's defensive copy in baddbmm for large matrices (M > 1024).
    """
    momentum = momentum_t.to(grad_chunk.dtype)
    momentum_buffer.lerp_(grad_chunk, 1 - momentum)
    g = grad_chunk.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    is_tall = g.size(-2) > g.size(-1)

    # Ensure spectral norm is at most 1 (with small safety margin)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()

    if is_tall:
        # Tall matrix: intermediate A is (K, K), cheaper than (M, M)
        A = torch.empty((*X.shape[:-2], X.size(-1), X.size(-1)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)
        if split_baddbmm:
            XB_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_XB = torch.baddbmm if X.ndim > 2 else torch.addmm
        for a, b, c in polar_express_coeffs:
            XTX(X, out=A)
            ba_plus_cAA(A, alpha=c, beta=b, out=B)
            if split_baddbmm:
                XB_matmul(X, B, out=C)
                C.add_(X, alpha=a)
            else:
                aX_plus_XB(X, X, B, beta=a, out=C)
            X, C = C, X
    else:
        # Wide (or square) matrix: intermediate A is (M, M)
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)
        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm
        for a, b, c in polar_express_coeffs:
            XXT(X, out=A)
            ba_plus_cAA(A, alpha=c, beta=b, out=B)
            if split_baddbmm:
                BX_matmul(B, X, out=C)
                C.add_(X, alpha=a)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)
            X, C = C, X

    return X
