"""
varlen_bim_v2 — doc-causal attention using BIM block dispatch + v10 optimizations.

v2 adds a nan_to_num guard in backward (same fix as cdb_bim_v18): the flash-attn
forward initialises M=-1e6; if a Q block attends to zero valid KV positions the
sentinel LSE ≈ -1e6 remains.  exp2(score - (-1e6)) overflows bfloat16 to ∞, and
∞ × 0 = NaN when dLdO is zero.  nan_to_num zeroes these out post-kernel.

Motivation:
  The existing triton_varlen kernel predates v10-v12 and is catastrophically slow
  at T=32k (31ms forward vs vslf's 0.54ms).  FlexAttention is 2.3× slower than
  vslf.  vslf backward produces large gradient errors on real packed data and
  cannot be used for training.

  This kernel achieves vslf-class forward speed with reliable gradients by
  combining the BIM-based block dispatch from v12 with simpler doc-causal-only
  inner kernels (no bitmask logic, no cross-doc machinery).

Design:
  Same structure as v12 (v11 kernels + BIM_BLOCK_SIZE=128) but the inner
  sub-kernels for non-full blocks check only `same_doc & causal` — no
  q_bitmasks, kv_bitmasks, or n_chunks arguments.

  For the common training case (doc_len=512, BIM_BS=128):
    - All off-diagonal same-doc blocks are "full" → _attn_fwd_inner_full_v10
    - Only the diagonal block uses the doc-causal inner path
    - Zero bitmask loads anywhere

Interface:
  triton_attn_doc_causal_bim_v1(q, k, v, cu_seqlens, max_seqlen, scale=None)
    q/k/v: (T, H, Dh)  bf16 or fp16
    cu_seqlens: (n_docs+1,) int32 — cumulative sequence lengths
    Returns: (T, H, Dh) output tensor
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from .cross_doc_bitmask_attn import _attn_backward_preprocess_cdb
from .cross_doc_bitmask_bim_v10 import (
    _attn_fwd_inner_full_v10,
    _bwd_kv_full_v10,
    _bwd_q_full_v10,
)


# ===========================================================================
# BIM construction from cu_seqlens (no grants)
# ===========================================================================

def build_doc_causal_bim(
    cu_seqlens: torch.Tensor,
    seq_len: int,
    device: torch.device,
    block_size: int = 128,
):
    """Build a BlockInteractionMask for pure doc-causal attention from cu_seqlens.

    cu_seqlens: [0, len0, len0+len1, ...] — cumulative doc lengths (docs contiguous from 0).
    Returns (BlockInteractionMask, doc_ids).
    """
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator

    n_docs = len(cu_seqlens) - 1
    doc_ids_np = np.full(seq_len, -1, dtype=np.int32)
    for d in range(n_docs):
        s = int(cu_seqlens[d])
        e = int(cu_seqlens[d + 1])
        doc_ids_np[s:e] = d
    doc_ids = torch.from_numpy(doc_ids_np).to(device)
    return build_doc_causal_bim_from_doc_ids(doc_ids, seq_len, device, block_size)


def build_doc_causal_bim_from_doc_ids(
    doc_ids: torch.Tensor,
    seq_len: int,
    device: torch.device,
    block_size: int = 128,
):
    """Build a BlockInteractionMask from a pre-built doc_ids tensor.

    doc_ids: (T,) int32 — doc index per position (-1 for padding/layout gaps).
    Returns BlockInteractionMask.  doc_ids is also returned for convenience.
    """
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator

    q_bms  = [torch.zeros(seq_len, dtype=torch.int64, device=device)]
    kv_bms = [torch.zeros(seq_len, dtype=torch.int64, device=device)]

    creator = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
    creator.triton_block_size = block_size
    creator._n_chunks = 1
    bim = CrossDocLinkMaskCreator._build_block_interaction_mask(
        creator, seq_len, doc_ids, q_bms, kv_bms, device
    )
    return bim, doc_ids


# ===========================================================================
# Forward inner — doc-causal only (same_doc + optional causal, no bitmask)
# ===========================================================================

@triton.jit
def _attn_fwd_inner_doc_causal_v1(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    lo, hi,
    softmax_scale,
    stride_K_N, stride_V_N,
    doc_ids_ptr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N,
    offsets_KV_N,
    N: tl.constexpr,
    Dh: tl.constexpr,
):
    """doc-causal masked block: same_doc | (same_doc & causal). No bitmask."""
    K_T_offsets  += lo * stride_K_N
    V_offsets    += lo * stride_V_N
    offsets_KV_N += lo

    mask_QO_N = offsets_QO_N < N
    doc_q = tl.load(doc_ids_ptr + offsets_QO_N, mask=mask_QO_N, other=-1)

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV  = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        mask_KV_N = offsets_KV_N < N

        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)
        S   = tl.dot(Q, K_T) * softmax_scale

        doc_kv   = tl.load(doc_ids_ptr + offsets_KV_N, mask=mask_KV_N, other=-2)
        attend   = (doc_q[:, None] == doc_kv[None, :])

        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= offsets_KV_N[None, :]
            S += tl.where(causal_mask & attend, 0, -1.0e6)
        else:
            S += tl.where(attend, 0, -1.0e6)

        M_new = tl.maximum(M, tl.max(S, axis=1))
        S    -= M_new[:, None]
        P     = tl.exp2(S)
        L_new = tl.sum(P, axis=1)
        alpha = tl.exp2(M - M_new)
        L     = L * alpha + L_new
        V     = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.)
        O     = O * alpha[:, None]
        O     = tl.dot(P.to(Q.dtype), V, acc=O)
        M     = M_new
        K_T_offsets  += BLOCK_SIZE_KV * stride_K_N
        V_offsets    += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M


# ===========================================================================
# Forward outer kernel
# ===========================================================================

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_KV": BK}, num_stages=ns, num_warps=nw)
        for BK in [32, 64]
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_fwd_doc_causal_v1(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, LSE_ptr,
    softmax_scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    doc_ids_ptr,
    T,
    q_kv_counts_ptr, q_kv_ptrs_ptr, q_kv_indices_ptr,
    q_kv_n_full_ptr,
    B,
    H: tl.constexpr, N: tl.constexpr,
    Dh: tl.constexpr,
    BIM_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
):
    rln2: tl.constexpr = 1.4426950408889634
    softmax_scale *= rln2
    tl.static_assert(BLOCK_SIZE_KV <= Dh)
    tl.static_assert(BIM_BLOCK_SIZE % BLOCK_SIZE_KV == 0)

    BLOCK_SIZE_QO: tl.constexpr = BIM_BLOCK_SIZE

    block_index_QO = tl.program_id(0)
    index_BH = tl.program_id(1)
    index_B  = index_BH // H
    index_H  = index_BH % H

    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh   = tl.arange(0, Dh)

    Q_offsets   = offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh
    K_T_offsets = offsets_Dh[:, None]   * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N
    V_offsets   = offsets_KV_N[:, None] * stride_V_N  + offsets_Dh[None, :] * stride_V_Dh

    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.)

    M = tl.full([BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)
    L = tl.full([BLOCK_SIZE_QO], value=1.0,  dtype=tl.float32)
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)

    q_kv_start = tl.load(q_kv_ptrs_ptr   + block_index_QO)
    num_kv     = tl.load(q_kv_counts_ptr  + block_index_QO)
    n_full     = tl.load(q_kv_n_full_ptr  + block_index_QO)

    # Full same-doc off-diagonal blocks: no masking
    for i in range(n_full):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        lo   = kv_b * BIM_BLOCK_SIZE
        O, L, M = _attn_fwd_inner_full_v10(
            Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
            lo, lo + BIM_BLOCK_SIZE, softmax_scale, stride_K_N, stride_V_N,
            offsets_KV_N, N, BLOCK_SIZE_QO, BLOCK_SIZE_KV, Dh,
        )

    # Off-diagonal non-full blocks: same_doc mask (boundary blocks only)
    for i in range(n_full, num_kv - 1):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        lo   = kv_b * BIM_BLOCK_SIZE
        O, L, M = _attn_fwd_inner_doc_causal_v1(
            Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
            lo, lo + BIM_BLOCK_SIZE,
            softmax_scale, stride_K_N, stride_V_N,
            doc_ids_ptr,
            BLOCK_SIZE_QO, BLOCK_SIZE_KV, False,
            offsets_QO_N, offsets_KV_N, N, Dh,
        )

    # Diagonal block: same_doc & causal
    kv_b_diag = tl.load(q_kv_indices_ptr + q_kv_start + num_kv - 1)
    lo_diag   = kv_b_diag * BIM_BLOCK_SIZE
    O, L, M = _attn_fwd_inner_doc_causal_v1(
        Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
        lo_diag, lo_diag + BIM_BLOCK_SIZE,
        softmax_scale, stride_K_N, stride_V_N,
        doc_ids_ptr,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV, True,
        offsets_QO_N, offsets_KV_N, N, Dh,
    )

    O   = O / L[:, None]
    LSE = M + tl.math.log2(L)

    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=offsets_QO_N < N)
    O_offsets = offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh
    tl.store(O_ptr + O_offsets, O.to(O_ptr.dtype.element_ty), mask=mask_QO_N[:, None])


# ===========================================================================
# Backward inner sub-kernels — doc-causal only (no bitmask)
# ===========================================================================

@triton.jit
def _bwd_kv_doc_causal_v1(
    K, V, dLdK, dLdV,
    Q_ptr, dLdO_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr,
):
    """dLdK/dLdV: same_doc mask (+ causal if MASK=True). No bitmask."""
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh  = tl.arange(0, Dh)

    Q_T_offsets  = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N  + offsets_Dh[None, :] * stride_Dh

    doc_col = tl.load(doc_ids_ptr + offsets_COL, mask=offsets_COL < N, other=-1)

    for _ in range(num_steps):
        mask_N = offsets_ROW < N

        Q_T   = tl.load(Q_ptr    + Q_T_offsets,  mask=mask_N[None, :], other=0.)
        LSE   = tl.load(LSE_ptr  + offsets_ROW,   mask=mask_N, other=0.)
        dLdO  = tl.load(dLdO_ptr + dLdO_offsets,  mask=mask_N[:, None], other=0.)
        Delta = tl.load(Delta_ptr + offsets_ROW,   mask=mask_N, other=0.)

        S_T    = tl.dot(K, Q_T) * (scale * rln2)
        P_T    = tl.exp2(S_T - LSE[None, :])

        doc_row  = tl.load(doc_ids_ptr + offsets_ROW, mask=mask_N, other=-2)
        attend   = (doc_col[:, None] == doc_row[None, :])

        if MASK:
            causal = (offsets_COL[:, None] <= offsets_ROW[None, :])
            P_T = tl.where(causal & attend, P_T, 0.)
        else:
            P_T = tl.where(attend, P_T, 0.)

        P_T_c  = P_T.to(K.dtype)
        dLdV   = tl.dot(P_T_c, dLdO, acc=dLdV)
        dLdP_T = tl.dot(V, tl.trans(dLdO))
        dLdS_T = P_T * (dLdP_T - Delta[None, :]) * ln2
        dLdK   = tl.dot(dLdS_T.to(K.dtype), tl.trans(Q_T), acc=dLdK)

        offsets_ROW  += BLOCK_SIZE_ROW
        Q_ptr        += BLOCK_SIZE_ROW * stride_N
        dLdO_ptr     += BLOCK_SIZE_ROW * stride_N

    return dLdK, dLdV


@triton.jit
def _bwd_q_doc_causal_v1(
    dLdQ, Q, dLdO, LSE,
    K_ptr, V_ptr, Delta_ptr,
    doc_ids_ptr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr,
):
    """dLdQ: same_doc mask (+ causal if MASK=True). No bitmask."""
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh  = tl.arange(0, Dh)

    KV_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N
    Delta  = tl.load(Delta_ptr + offsets_ROW,  mask=offsets_ROW < N, other=0.)
    doc_row = tl.load(doc_ids_ptr + offsets_ROW, mask=offsets_ROW < N, other=-1)

    for _ in range(num_steps):
        col_mask = offsets_COL < N

        K_T = tl.load(K_ptr + KV_T_offsets, mask=col_mask[None, :], other=0.)
        V_T = tl.load(V_ptr + KV_T_offsets, mask=col_mask[None, :], other=0.)

        S = tl.dot(Q, K_T) * (scale * rln2)
        P = tl.exp2(S - LSE)

        doc_col  = tl.load(doc_ids_ptr + offsets_COL, mask=col_mask, other=-2)
        attend   = (doc_row[:, None] == doc_col[None, :])

        if MASK:
            causal = (offsets_ROW[:, None] >= offsets_COL[None, :])
            P = tl.where(causal & attend, P, 0.)
        else:
            P = tl.where(attend, P, 0.)

        dLdP = tl.dot(dLdO, V_T)
        dLdS = P * (dLdP - Delta[:, None]) * ln2
        dLdQ += tl.dot(dLdS.to(Q.dtype), tl.trans(K_T))

        offsets_COL += BLOCK_SIZE_COL
        K_ptr       += BLOCK_SIZE_COL * stride_N
        V_ptr       += BLOCK_SIZE_COL * stride_N

    return dLdQ


# ===========================================================================
# Backward outer kernels
# ===========================================================================

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for m in [16, 32, 64, 128]
        for ns in [3, 4, 5]
        for nw in [4, 8]
        if m <= 128
    ],
    key=["N", "Dh", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_KV_doc_causal_v1(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr,
    T,
    kv_q_counts_ptr, kv_q_ptrs_ptr, kv_q_indices_ptr,
    kv_q_n_full_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BIM_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
    """Compute dLdK and dLdV. Doc-causal only (no bitmask)."""
    ln2:  tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    BLOCK_SIZE_MACRO: tl.constexpr = BIM_BLOCK_SIZE
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // H
    idx_head  = idx_batch_head % H
    bh = idx_batch * stride_B + idx_head * stride_H
    Q_ptr    += bh;  K_ptr    += bh;  V_ptr    += bh
    dLdO_ptr += bh;  dLdK_ptr += bh;  dLdV_ptr += bh
    bh_lse = idx_batch_head * N
    LSE_ptr   += bh_lse
    Delta_ptr += bh_lse

    offsets_Dh = tl.arange(0, Dh)
    pid = tl.program_id(0)

    BLOCK_SIZE_ROW: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL: tl.constexpr = BLOCK_SIZE_MACRO
    num_micro: tl.constexpr = BLOCK_SIZE_COL // BLOCK_SIZE_ROW

    start_COL   = pid * BLOCK_SIZE_COL
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    KV_offsets  = offsets_COL[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask     = offsets_COL[:, None] < N

    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.)

    dLdK = tl.zeros([BLOCK_SIZE_COL, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL, Dh], dtype=tl.float32)

    kv_q_start   = tl.load(kv_q_ptrs_ptr   + pid)
    num_q_macros = tl.load(kv_q_counts_ptr  + pid)
    n_full_kv    = tl.load(kv_q_n_full_ptr  + pid)

    # Diagonal (same_doc & causal)
    q_b_diag = tl.load(kv_q_indices_ptr + kv_q_start)
    dLdK, dLdV = _bwd_kv_doc_causal_v1(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        doc_ids_ptr,
        stride_N, stride_Dh, N, Dh,
        BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
        q_b_diag * BLOCK_SIZE_COL, start_COL, num_micro,
        scale, ln2, rln2, MASK=True,
    )

    # Full same-doc Q-blocks: no masking
    for i in range(1, 1 + n_full_kv):
        q_b = tl.load(kv_q_indices_ptr + kv_q_start + i)
        dLdK, dLdV = _bwd_kv_full_v10(
            K, V, dLdK, dLdV,
            Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
            stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            q_b * BLOCK_SIZE_COL, start_COL, num_micro,
            scale, ln2, rln2,
        )

    # Off-diagonal non-full Q-blocks: same_doc only (boundary blocks)
    for i in range(1 + n_full_kv, num_q_macros):
        q_b = tl.load(kv_q_indices_ptr + kv_q_start + i)
        dLdK, dLdV = _bwd_kv_doc_causal_v1(
            K, V, dLdK, dLdV,
            Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
            doc_ids_ptr,
            stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            q_b * BLOCK_SIZE_COL, start_COL, num_micro,
            scale, ln2, rln2, MASK=False,
        )

    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK.to(dLdK_ptr.dtype.element_ty), mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV.to(dLdV_ptr.dtype.element_ty), mask=KV_mask)


@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for m in [16, 32, 64, 128]
        for ns in [3, 4, 5]
        for nw in [4, 8]
        if m <= 128
    ],
    key=["N", "Dh", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_Q_doc_causal_v1(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdQ_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr,
    T,
    q_kv_counts_ptr, q_kv_ptrs_ptr, q_kv_indices_ptr,
    q_kv_n_full_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BIM_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
    """Compute dLdQ. Doc-causal only (no bitmask)."""
    ln2:  tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    BLOCK_SIZE_MACRO: tl.constexpr = BIM_BLOCK_SIZE
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // H
    idx_head  = idx_batch_head % H
    bh = idx_batch * stride_B + idx_head * stride_H
    Q_ptr    += bh;  K_ptr    += bh;  V_ptr    += bh
    dLdO_ptr += bh;  dLdQ_ptr += bh
    bh_lse = idx_batch_head * N
    LSE_ptr   += bh_lse
    Delta_ptr += bh_lse

    offsets_Dh = tl.arange(0, Dh)
    pid = tl.program_id(0)

    BLOCK_SIZE_ROW: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL: tl.constexpr = BLOCK_SIZE_MICRO
    num_micro: tl.constexpr = BLOCK_SIZE_ROW // BLOCK_SIZE_COL

    start_ROW   = pid * BLOCK_SIZE_ROW
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    QO_offsets  = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW    = offsets_ROW < N

    Q    = tl.load(Q_ptr    + QO_offsets, mask=mask_ROW[:, None], other=0.)
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.)
    LSE  = tl.load(LSE_ptr  + offsets_ROW, mask=mask_ROW, other=0.)[:, None]
    dLdQ = tl.zeros([BLOCK_SIZE_ROW, Dh], dtype=tl.float32)

    q_kv_start    = tl.load(q_kv_ptrs_ptr   + pid)
    num_kv_macros = tl.load(q_kv_counts_ptr  + pid)
    n_full_q      = tl.load(q_kv_n_full_ptr  + pid)

    # Full same-doc KV-blocks: no masking
    for i in range(n_full_q):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        dLdQ = _bwd_q_full_v10(
            dLdQ, Q, dLdO, LSE,
            K_ptr, V_ptr, Delta_ptr,
            stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            start_ROW, kv_b * BLOCK_SIZE_ROW, num_micro,
            scale, ln2, rln2,
        )

    # Off-diagonal non-full KV-blocks: same_doc only (boundary blocks)
    for i in range(n_full_q, num_kv_macros - 1):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        dLdQ = _bwd_q_doc_causal_v1(
            dLdQ, Q, dLdO, LSE,
            K_ptr, V_ptr, Delta_ptr,
            doc_ids_ptr,
            stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            start_ROW, kv_b * BLOCK_SIZE_ROW, num_micro,
            scale, ln2, rln2, MASK=False,
        )

    # Diagonal: same_doc & causal
    kv_b_diag = tl.load(q_kv_indices_ptr + q_kv_start + num_kv_macros - 1)
    dLdQ = _bwd_q_doc_causal_v1(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        doc_ids_ptr,
        stride_N, stride_Dh, N, Dh,
        BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
        start_ROW, kv_b_diag * BLOCK_SIZE_ROW, num_micro,
        scale, ln2, rln2, MASK=True,
    )

    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ.to(dLdQ_ptr.dtype.element_ty), mask=mask_ROW[:, None])


# ===========================================================================
# Autograd function
# ===========================================================================

# Caches keyed on (tensor_id, seq_len) to avoid rebuilding every forward pass.
_bim_cache: dict = {}

def _get_bim_and_doc_ids(cu_seqlens: torch.Tensor, seq_len: int, device: torch.device):
    key = (id(cu_seqlens), seq_len)
    if key not in _bim_cache:
        bim, doc_ids = build_doc_causal_bim(cu_seqlens, seq_len, device, block_size=128)
        _bim_cache[key] = (bim, doc_ids)
    return _bim_cache[key]

_bim_doc_ids_cache: dict = {}

def _get_bim_from_doc_ids(doc_ids: torch.Tensor, seq_len: int, device: torch.device):
    key = (id(doc_ids), seq_len)
    if key not in _bim_doc_ids_cache:
        bim, _ = build_doc_causal_bim_from_doc_ids(doc_ids, seq_len, device, block_size=128)
        _bim_doc_ids_cache[key] = bim
    return _bim_doc_ids_cache[key]


class _VarlenBIMv2(torch.autograd.Function):
    """Doc-causal attention: BIM dispatch + v10 optimizations, no bitmask."""

    @staticmethod
    def forward(ctx, q, k, v, doc_ids, bim, scale):
        T, H, Dh = q.shape
        bim_bs   = bim.block_size
        n_blocks = bim.n_blocks
        assert bim_bs == 128, f"varlen_bim_v1 requires BIM_BLOCK_SIZE=128, got {bim_bs}"
        assert bim.q_kv_n_full is not None

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        sT, sH, sDh = q.stride(0), q.stride(1), q.stride(2)

        O   = torch.empty_like(q)
        LSE = torch.empty(H, T, device=q.device, dtype=torch.float32)

        grid = (n_blocks, H)
        _attn_fwd_doc_causal_v1[grid](
            q, k, v, O, LSE, scale,
            0, sH, sT, sDh,
            0, k.stride(1), k.stride(0), k.stride(2),
            0, v.stride(1), v.stride(0), v.stride(2),
            0, O.stride(1), O.stride(0), O.stride(2),
            H * T, T, 1,
            doc_ids, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            1, H, T, Dh, bim_bs,
        )

        ctx.save_for_backward(q, k, v, O, LSE, doc_ids)
        ctx.bim   = bim
        ctx.T, ctx.H, ctx.Dh = T, H, Dh
        ctx.scale = scale
        ctx.strides = (sT, sH, sDh)
        return O

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE, doc_ids = ctx.saved_tensors
        bim              = ctx.bim
        T, H, Dh         = ctx.T, ctx.H, ctx.Dh
        scale            = ctx.scale
        sT, sH, sDh      = ctx.strides

        dLdO = dLdO.contiguous()

        dLdq = torch.empty_like(q)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)
        Delta = torch.empty_like(LSE)

        pre_grid = lambda meta: (triton.cdiv(T, meta["PRE_BLOCK_SIZE_ROW"]), H)
        _attn_backward_preprocess_cdb[pre_grid](
            O, dLdO, Delta,
            0, sH, sT, sDh,
            0, dLdO.stride(1), dLdO.stride(0), dLdO.stride(2),
            H * T, T, 1,
            T, Dh,
        )

        s = (0, sH, sT, sDh)
        grid = (bim.n_blocks, H)

        _attn_backward_KV_doc_causal_v1[grid](
            q, k, v, dLdO, dLdk, dLdv, LSE, Delta,
            doc_ids, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices,
            bim.kv_q_n_full,
            scale, *s,
            H, T, Dh, bim.block_size,
        )

        _attn_backward_Q_doc_causal_v1[grid](
            q, k, v, dLdO, dLdq, LSE, Delta,
            doc_ids, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            scale, *s,
            H, T, Dh, bim.block_size,
        )

        # Sentinel-LSE NaN guard: same fix as cdb_bim_v18.
        dLdq = torch.nan_to_num(dLdq, nan=0., posinf=0., neginf=0.)
        dLdk = torch.nan_to_num(dLdk, nan=0., posinf=0., neginf=0.)
        dLdv = torch.nan_to_num(dLdv, nan=0., posinf=0., neginf=0.)

        return dLdq, dLdk, dLdv, None, None, None


def triton_attn_doc_causal_bim_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Doc-causal attention: BIM dispatch + v10 opts (bf16 TC, no copies).

    q/k/v: (T, H, Dh)   bf16 or fp16
    cu_seqlens: (n_docs+1,) int32  — cumulative lengths from 0
    Returns: (T, H, Dh)
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    T = q.shape[0]
    bim, doc_ids = _get_bim_and_doc_ids(cu_seqlens, T, q.device)
    return _VarlenBIMv2.apply(q, k, v, doc_ids, bim, scale)


def triton_attn_doc_causal_bim_v2_from_doc_ids(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Doc-causal attention from a pre-built doc_ids tensor.

    q/k/v: (T, H, Dh)   bf16 or fp16
    doc_ids: (T,) int32  — doc index per position (-1 for layout/padding gaps)
    Returns: (T, H, Dh)

    Use this when doc spans are non-contiguous (training with layout tokens).
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    T = q.shape[0]
    bim = _get_bim_from_doc_ids(doc_ids, T, q.device)
    return _VarlenBIMv2.apply(q, k, v, doc_ids, bim, scale)
