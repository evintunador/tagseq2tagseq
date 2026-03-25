"""
cross_doc_bitmask BIM v13 — pure-cross block classification.

Changes vs v12:
  Non-full off-diagonal block pairs are split into two sub-categories:

  pure_cross: both Q-block and KV-block lie entirely within distinct single
    documents (blk_is_pure=True for both, different doc IDs). For these pairs,
    same_doc=False for every (q_pos, kv_pos) element pair by construction, so
    we only need the bitmask check: attend = in_grant. No doc_id loads, no
    same_doc computation.

  boundary: at least one block straddles a doc boundary (blk_is_pure=False),
    or both blocks are single-doc but the same doc (shouldn't happen for
    non-full off-diagonal, but kept for safety). Requires full masking:
    attend = same_doc | in_grant.

  Row ordering in BIM CSR (set by _build_block_interaction_mask v13 extension):
    q_kv: [full...(0..n_full-1), pure_cross...(n_full..n_full+n_pc-1),
           boundary...(n_full+n_pc..count-2), diagonal(count-1)]
    kv_q: [diagonal(0), full...(1..1+n_full-1), pure_cross...(1+n_full..1+n_full+n_pc-1),
           boundary...(1+n_full+n_pc..count-1)]

  At BIM_BS=128, doc_len=512 (aligned): 100% of non-full off-diagonal blocks are
  pure_cross — no doc_id loads anywhere except the diagonal block.

Builds on v12 (v11 kernels + BIM_BLOCK_SIZE=128). All inner sub-kernels from
v10 are reused; two new pure-cross sub-kernels are added.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .cross_doc_bitmask_attn import _attn_backward_preprocess_cdb
from .cross_doc_bitmask_bim_v10 import (
    _attn_fwd_cdb_bim_v10,          # forward outer — reused as-is
    _attn_backward_KV_cdb_bim_v10,  # backward dKV outer — reused
    _attn_backward_Q_cdb_bim_v10,   # backward dQ outer — reused
    _attn_fwd_inner_full_v10,
    _attn_fwd_inner_cdb_v10,
    _bwd_kv_full_v10,
    _bwd_q_full_v10,
    _bwd_kv_cdb_v10,
    _bwd_q_cdb_v10,
)

if TYPE_CHECKING:
    from model.graph_traversal.cross_doc_mask import BlockInteractionMask


# ===========================================================================
# New pure-cross inner sub-kernels (bitmask only, no same_doc check)
# ===========================================================================

@triton.jit
def _attn_fwd_inner_pure_cross_v13(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    lo, hi,
    softmax_scale,
    stride_K_N, stride_V_N,
    q_bitmasks_ptr,
    kv_bitmasks_ptr,
    T,
    n_chunks: tl.constexpr,
    offsets_QO_N,
    offsets_KV_N,
    N: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    Dh: tl.constexpr,
):
    """Pure-cross block: bitmask masking only, no same_doc check. Native dtype."""
    K_T_offsets  += lo * stride_K_N
    V_offsets    += lo * stride_V_N
    offsets_KV_N += lo

    mask_QO_N = offsets_QO_N < N

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV  = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        mask_KV_N = offsets_KV_N < N

        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)
        S   = tl.dot(Q, K_T) * softmax_scale

        in_grant = tl.zeros([BLOCK_SIZE_QO, BLOCK_SIZE_KV], dtype=tl.int1)
        for c in tl.static_range(n_chunks):
            q_bm  = tl.load(q_bitmasks_ptr  + c * T + offsets_QO_N, mask=mask_QO_N, other=0)
            kv_bm = tl.load(kv_bitmasks_ptr + c * T + offsets_KV_N, mask=mask_KV_N, other=0)
            in_grant = in_grant | ((q_bm[:, None] & kv_bm[None, :]) != 0)

        # No causal mask needed (off-diagonal: kv_b < q_b → all positions causal)
        S += tl.where(in_grant, 0, -1.0e6)

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


@triton.jit
def _bwd_kv_pure_cross_v13(
    K, V, dLdK, dLdV,
    Q_ptr, dLdO_ptr,
    LSE_ptr, Delta_ptr,
    q_bitmasks_ptr, kv_bitmasks_ptr, T,
    n_chunks: tl.constexpr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
):
    """dLdK/dLdV: pure-cross block — bitmask masking only, no same_doc check."""
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh  = tl.arange(0, Dh)

    Q_T_offsets  = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N  + offsets_Dh[None, :] * stride_Dh

    for _ in range(num_steps):
        mask_N = offsets_ROW < N

        Q_T   = tl.load(Q_ptr    + Q_T_offsets,  mask=mask_N[None, :], other=0.)
        LSE   = tl.load(LSE_ptr  + offsets_ROW,   mask=mask_N, other=0.)
        dLdO  = tl.load(dLdO_ptr + dLdO_offsets,  mask=mask_N[:, None], other=0.)
        Delta = tl.load(Delta_ptr + offsets_ROW,   mask=mask_N, other=0.)

        S_T = tl.dot(K, Q_T) * (scale * rln2)
        P_T = tl.exp2(S_T - LSE[None, :])

        # Pure-cross: only bitmask check (no same_doc computation)
        in_grant_T = tl.zeros([BLOCK_SIZE_COL, BLOCK_SIZE_ROW], dtype=tl.int1)
        for c in tl.static_range(n_chunks):
            kv_bm = tl.load(kv_bitmasks_ptr + c * T + offsets_COL,
                            mask=offsets_COL < N, other=0)
            q_bm  = tl.load(q_bitmasks_ptr  + c * T + offsets_ROW,
                            mask=mask_N, other=0)
            in_grant_T = in_grant_T | ((kv_bm[:, None] & q_bm[None, :]) != 0)

        P_T = tl.where(in_grant_T, P_T, 0.)

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
def _bwd_q_pure_cross_v13(
    dLdQ, Q, dLdO, LSE,
    K_ptr, V_ptr, Delta_ptr,
    q_bitmasks_ptr, kv_bitmasks_ptr, T,
    n_chunks: tl.constexpr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
):
    """dLdQ: pure-cross block — bitmask masking only, no same_doc check."""
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh  = tl.arange(0, Dh)

    KV_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N
    Delta = tl.load(Delta_ptr + offsets_ROW, mask=offsets_ROW < N, other=0.)

    for _ in range(num_steps):
        col_mask = offsets_COL < N

        K_T = tl.load(K_ptr + KV_T_offsets, mask=col_mask[None, :], other=0.)
        V_T = tl.load(V_ptr + KV_T_offsets, mask=col_mask[None, :], other=0.)

        S = tl.dot(Q, K_T) * (scale * rln2)
        P = tl.exp2(S - LSE)

        in_grant = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.int1)
        for c in tl.static_range(n_chunks):
            q_bm  = tl.load(q_bitmasks_ptr  + c * T + offsets_ROW,
                            mask=offsets_ROW < N, other=0)
            kv_bm = tl.load(kv_bitmasks_ptr + c * T + offsets_COL,
                            mask=col_mask, other=0)
            in_grant = in_grant | ((q_bm[:, None] & kv_bm[None, :]) != 0)

        P = tl.where(in_grant, P, 0.)

        dLdP = tl.dot(dLdO, V_T)
        dLdS = P * (dLdP - Delta[:, None]) * ln2
        dLdQ += tl.dot(dLdS.to(Q.dtype), tl.trans(K_T))

        offsets_COL += BLOCK_SIZE_COL
        K_ptr       += BLOCK_SIZE_COL * stride_N
        V_ptr       += BLOCK_SIZE_COL * stride_N

    return dLdQ


# ===========================================================================
# Forward outer kernel (with pure-cross dispatch)
# ===========================================================================

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_KV": BK}, num_stages=ns, num_warps=nw)
        for BK in [32, 64]
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_fwd_cdb_bim_v13(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, LSE_ptr,
    softmax_scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    doc_ids_ptr,
    q_bitmasks_ptr, kv_bitmasks_ptr,
    T,
    q_kv_counts_ptr, q_kv_ptrs_ptr, q_kv_indices_ptr,
    q_kv_n_full_ptr,
    q_kv_n_pure_cross_ptr,
    B,
    H: tl.constexpr, N: tl.constexpr,
    Dh: tl.constexpr,
    n_chunks: tl.constexpr,
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

    q_kv_start = tl.load(q_kv_ptrs_ptr         + block_index_QO)
    num_kv     = tl.load(q_kv_counts_ptr        + block_index_QO)
    n_full     = tl.load(q_kv_n_full_ptr        + block_index_QO)
    n_pc       = tl.load(q_kv_n_pure_cross_ptr  + block_index_QO)

    # Full same-doc off-diagonal blocks: no masking
    for i in range(n_full):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        lo   = kv_b * BIM_BLOCK_SIZE
        O, L, M = _attn_fwd_inner_full_v10(
            Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
            lo, lo + BIM_BLOCK_SIZE, softmax_scale, stride_K_N, stride_V_N,
            offsets_KV_N, N, BLOCK_SIZE_QO, BLOCK_SIZE_KV, Dh,
        )

    # Pure-cross off-diagonal blocks: bitmask masking only
    for i in range(n_full, n_full + n_pc):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        lo   = kv_b * BIM_BLOCK_SIZE
        O, L, M = _attn_fwd_inner_pure_cross_v13(
            Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
            lo, lo + BIM_BLOCK_SIZE,
            softmax_scale, stride_K_N, stride_V_N,
            q_bitmasks_ptr, kv_bitmasks_ptr, T, n_chunks,
            offsets_QO_N, offsets_KV_N, N, BLOCK_SIZE_QO, BLOCK_SIZE_KV, Dh,
        )

    # Boundary off-diagonal blocks: full masking (same_doc | in_grant)
    for i in range(n_full + n_pc, num_kv - 1):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        lo   = kv_b * BIM_BLOCK_SIZE
        O, L, M = _attn_fwd_inner_cdb_v10(
            Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
            lo, lo + BIM_BLOCK_SIZE,
            softmax_scale, stride_K_N, stride_V_N,
            doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T, n_chunks,
            BLOCK_SIZE_QO, BLOCK_SIZE_KV, False,
            offsets_QO_N, offsets_KV_N, N, Dh,
        )

    # Diagonal block: full masking + causal
    kv_b_diag = tl.load(q_kv_indices_ptr + q_kv_start + num_kv - 1)
    lo_diag   = kv_b_diag * BIM_BLOCK_SIZE
    O, L, M = _attn_fwd_inner_cdb_v10(
        Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
        lo_diag, lo_diag + BIM_BLOCK_SIZE,
        softmax_scale, stride_K_N, stride_V_N,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T, n_chunks,
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
# Backward outer kernels (with pure-cross dispatch)
# ===========================================================================

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for m in [16, 32, 64, 128]
        for ns in [3, 4, 5]
        for nw in [4, 8]
        if m <= 128
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_KV_cdb_bim_v13(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
    kv_q_counts_ptr, kv_q_ptrs_ptr, kv_q_indices_ptr,
    kv_q_n_full_ptr,
    kv_q_n_pure_cross_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    n_chunks: tl.constexpr,
    BIM_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
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

    kv_q_start   = tl.load(kv_q_ptrs_ptr        + pid)
    num_q_macros = tl.load(kv_q_counts_ptr       + pid)
    n_full_kv    = tl.load(kv_q_n_full_ptr       + pid)
    n_pc_kv      = tl.load(kv_q_n_pure_cross_ptr + pid)

    # Diagonal: first entry
    q_b_diag = tl.load(kv_q_indices_ptr + kv_q_start)
    dLdK, dLdV = _bwd_kv_cdb_v10(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
        n_chunks, stride_N, stride_Dh, N, Dh,
        BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
        q_b_diag * BLOCK_SIZE_COL, start_COL, num_micro,
        scale, ln2, rln2, MASK=True,
    )

    # Full same-doc Q blocks
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

    # Pure-cross Q blocks: bitmask masking only
    for i in range(1 + n_full_kv, 1 + n_full_kv + n_pc_kv):
        q_b = tl.load(kv_q_indices_ptr + kv_q_start + i)
        dLdK, dLdV = _bwd_kv_pure_cross_v13(
            K, V, dLdK, dLdV,
            Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
            q_bitmasks_ptr, kv_bitmasks_ptr, T,
            n_chunks, stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            q_b * BLOCK_SIZE_COL, start_COL, num_micro,
            scale, ln2, rln2,
        )

    # Boundary Q blocks: full masking
    for i in range(1 + n_full_kv + n_pc_kv, num_q_macros):
        q_b = tl.load(kv_q_indices_ptr + kv_q_start + i)
        dLdK, dLdV = _bwd_kv_cdb_v10(
            K, V, dLdK, dLdV,
            Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
            doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
            n_chunks, stride_N, stride_Dh, N, Dh,
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
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_Q_cdb_bim_v13(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdQ_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
    q_kv_counts_ptr, q_kv_ptrs_ptr, q_kv_indices_ptr,
    q_kv_n_full_ptr,
    q_kv_n_pure_cross_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    n_chunks: tl.constexpr,
    BIM_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
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

    q_kv_start    = tl.load(q_kv_ptrs_ptr        + pid)
    num_kv_macros = tl.load(q_kv_counts_ptr       + pid)
    n_full_q      = tl.load(q_kv_n_full_ptr       + pid)
    n_pc_q        = tl.load(q_kv_n_pure_cross_ptr + pid)

    # Full same-doc KV blocks
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

    # Pure-cross KV blocks: bitmask masking only
    for i in range(n_full_q, n_full_q + n_pc_q):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        dLdQ = _bwd_q_pure_cross_v13(
            dLdQ, Q, dLdO, LSE,
            K_ptr, V_ptr, Delta_ptr,
            q_bitmasks_ptr, kv_bitmasks_ptr, T,
            n_chunks, stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            start_ROW, kv_b * BLOCK_SIZE_ROW, num_micro,
            scale, ln2, rln2,
        )

    # Boundary KV blocks: full masking
    for i in range(n_full_q + n_pc_q, num_kv_macros - 1):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        dLdQ = _bwd_q_cdb_v10(
            dLdQ, Q, dLdO, LSE,
            K_ptr, V_ptr, Delta_ptr,
            doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
            n_chunks, stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            start_ROW, kv_b * BLOCK_SIZE_ROW, num_micro,
            scale, ln2, rln2, MASK=False,
        )

    # Diagonal KV block: last entry, full masking + causal
    kv_b_diag = tl.load(q_kv_indices_ptr + q_kv_start + num_kv_macros - 1)
    dLdQ = _bwd_q_cdb_v10(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
        n_chunks, stride_N, stride_Dh, N, Dh,
        BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
        start_ROW, kv_b_diag * BLOCK_SIZE_ROW, num_micro,
        scale, ln2, rln2, MASK=True,
    )

    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ.to(dLdQ_ptr.dtype.element_ty), mask=mask_ROW[:, None])


# ===========================================================================
# Autograd function
# ===========================================================================

def _build_bim_128(seq_len, document_ids, q_bitmasks, kv_bitmasks, device, n_chunks):
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
    creator = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
    creator.triton_block_size = 128
    creator._n_chunks = n_chunks
    return CrossDocLinkMaskCreator._build_block_interaction_mask(
        creator, seq_len, document_ids, list(q_bitmasks), list(kv_bitmasks), device,
    )


class _CDBBIMv13(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
        T, H, Dh = q.shape
        n_chunks = q_bitmasks.shape[0]
        bim_bs   = bim.block_size
        n_blocks = bim.n_blocks
        assert bim.q_kv_n_full is not None
        assert bim.q_kv_n_pure_cross is not None, \
            "BIM missing q_kv_n_pure_cross — rebuild with _build_block_interaction_mask"

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        q_bm_c = q_bitmasks.contiguous(); kv_bm_c = kv_bitmasks.contiguous()

        sT, sH, sDh = q.stride(0), q.stride(1), q.stride(2)
        O   = torch.empty_like(q)
        LSE = torch.empty(H, T, device=q.device, dtype=torch.float32)

        grid_fwd = (n_blocks, H)
        _attn_fwd_cdb_bim_v13[grid_fwd](
            q, k, v, O, LSE, scale,
            0, sH, sT, sDh,
            0, k.stride(1), k.stride(0), k.stride(2),
            0, v.stride(1), v.stride(0), v.stride(2),
            0, O.stride(1), O.stride(0), O.stride(2),
            H * T, T, 1,
            document_ids, q_bm_c, kv_bm_c, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full, bim.q_kv_n_pure_cross,
            1, H, T, Dh, n_chunks, bim_bs,
        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.document_ids = document_ids
        ctx.bim = bim; ctx.q_bitmasks = q_bm_c; ctx.kv_bitmasks = kv_bm_c
        ctx.T, ctx.H, ctx.Dh = T, H, Dh
        ctx.n_chunks = n_chunks; ctx.scale = scale
        ctx.strides = (sT, sH, sDh)
        return O

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        document_ids = ctx.document_ids; bim = ctx.bim
        q_bm, kv_bm = ctx.q_bitmasks, ctx.kv_bitmasks
        T, H, Dh = ctx.T, ctx.H, ctx.Dh
        n_chunks = ctx.n_chunks; scale = ctx.scale
        sT, sH, sDh = ctx.strides

        dLdO = dLdO.contiguous()
        dLdq = torch.empty_like(q); dLdk = torch.empty_like(k); dLdv = torch.empty_like(v)
        Delta = torch.empty_like(LSE)

        pre_grid = lambda meta: (triton.cdiv(T, meta["PRE_BLOCK_SIZE_ROW"]), H)
        _attn_backward_preprocess_cdb[pre_grid](
            O, dLdO, Delta,
            0, sH, sT, sDh,
            0, dLdO.stride(1), dLdO.stride(0), dLdO.stride(2),
            H * T, T, 1, T, Dh,
        )

        s = (0, sH, sT, sDh)
        grid = (bim.n_blocks, H)

        _attn_backward_KV_cdb_bim_v13[grid](
            q, k, v, dLdO, dLdk, dLdv, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices,
            bim.kv_q_n_full, bim.kv_q_n_pure_cross,
            scale, *s, H, T, Dh, n_chunks, bim.block_size,
        )

        _attn_backward_Q_cdb_bim_v13[grid](
            q, k, v, dLdO, dLdq, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full, bim.q_kv_n_pure_cross,
            scale, *s, H, T, Dh, n_chunks, bim.block_size,
        )

        return dLdq, dLdk, dLdv, None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v13(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask | None" = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v13: v12 + pure-cross block dispatch (bitmask-only inner kernel)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if bim is None or bim.block_size != 128:
        bim = _build_bim_128(
            q.shape[0], document_ids, q_bitmasks, kv_bitmasks, q.device,
            q_bitmasks.shape[0],
        )
    return _CDBBIMv13.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
