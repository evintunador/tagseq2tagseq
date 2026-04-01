"""
cross_doc_bitmask BIM v10 — native-dtype matmuls (no fp32 cast on load).

Changes vs v4:
  Replace all explicit `.to(tl.float32)` loads for Q/K/V/dLdO with native-dtype
  (bf16/fp16) loads.  `tl.dot` accumulates bf16×bf16 → fp32 via TC natively.

  Intermediates that feed back into matmuls (P_T, dLdS_T, P, dLdS) are cast to
  the native dtype just before each `tl.dot` call; all other arithmetic (softmax,
  exp2, scale multiply, masking arithmetic) stays in fp32.

  Benefits:
    1. Eliminates per-element upcast at every K/V/Q/dLdO load site.
    2. Register pressure halved for K/V/Q/dLdO tiles (bf16 = 2B vs fp32 = 4B).
    3. Higher SM occupancy → better memory-latency hiding.

  Scale handling: v4 pre-scales K with `K *= scale * rln2` and Q with
  `Q *= scale * rln2` before passing them to inner sub-kernels.  v10 keeps K/Q
  unscaled in bf16 and instead applies `* (scale * rln2)` to the fp32 matmul
  output.  The final `dLdK *= scale * rln2` and `dLdQ *= scale * rln2` are
  unchanged.

Forward: new outer kernel (_attn_fwd_cdb_bim_v10) + two new inner sub-kernels
  (_attn_fwd_inner_full_v10 for unmasked same-doc blocks,
   _attn_fwd_inner_cdb_v10 for masked/diagonal blocks).
Backward: two split kernels like v4 (dKV + dQ), each with new v10 inner
  sub-kernels (_bwd_kv_full_v10, _bwd_kv_cdb_v10, _bwd_q_full_v10,
  _bwd_q_cdb_v10).
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .cross_doc_bitmask_attn import _attn_backward_preprocess_cdb

if TYPE_CHECKING:
    from model.graph_traversal.cross_doc_mask import BlockInteractionMask


# ===========================================================================
# Forward — inner sub-kernels
# ===========================================================================

@triton.jit
def _attn_fwd_inner_full_v10(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    lo, hi,
    softmax_scale,
    stride_K_N, stride_V_N,
    offsets_KV_N,
    N: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    Dh: tl.constexpr,
):
    """Full same-doc block: no masking. K_T and V loaded in native dtype."""
    K_T_offsets += lo * stride_K_N
    V_offsets   += lo * stride_V_N
    offsets_KV_N += lo

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV  = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        mask_KV_N = offsets_KV_N < N

        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.)
        S   = tl.dot(Q, K_T) * softmax_scale          # bf16 TC → fp32, then scale
        M_new = tl.maximum(M, tl.max(S, axis=1))
        S    -= M_new[:, None]
        P     = tl.exp2(S)
        L_new = tl.sum(P, axis=1)
        alpha = tl.exp2(M - M_new)
        L     = L * alpha + L_new
        V     = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.)
        O     = O * alpha[:, None]
        O     = tl.dot(P.to(Q.dtype), V, acc=O)       # cast P fp32→bf16 for TC
        M     = M_new
        K_T_offsets  += BLOCK_SIZE_KV * stride_K_N
        V_offsets    += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M


@triton.jit
def _attn_fwd_inner_cdb_v10(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    lo, hi,
    softmax_scale,
    stride_K_N, stride_V_N,
    doc_ids_ptr,
    q_bitmasks_ptr,
    kv_bitmasks_ptr,
    T,
    n_chunks: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N,
    offsets_KV_N,
    N: tl.constexpr,
    Dh: tl.constexpr,
):
    """Masked block (same_doc | in_grant | optional causal). Native dtype."""
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
        same_doc = (doc_q[:, None] == doc_kv[None, :])

        in_grant = tl.zeros([BLOCK_SIZE_QO, BLOCK_SIZE_KV], dtype=tl.int1)
        for c in tl.static_range(n_chunks):
            q_bm  = tl.load(q_bitmasks_ptr  + c * T + offsets_QO_N, mask=mask_QO_N, other=0)
            kv_bm = tl.load(kv_bitmasks_ptr + c * T + offsets_KV_N, mask=mask_KV_N, other=0)
            in_grant = in_grant | ((q_bm[:, None] & kv_bm[None, :]) != 0)

        attend = same_doc | in_grant

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
# Forward — outer kernel
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
def _attn_fwd_cdb_bim_v10(
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
    # KEY CHANGE: no .to(tl.float32) — Q stays in native dtype (bf16/fp16)
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

    # Off-diagonal non-full blocks: full masking (same_doc | in_grant)
    for i in range(n_full, num_kv - 1):
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
# Backward — inner sub-kernels
# ===========================================================================

@triton.jit
def _bwd_kv_full_v10(
    K, V, dLdK, dLdV,
    Q_ptr, dLdO_ptr,
    LSE_ptr, Delta_ptr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
):
    """dLdK/dLdV for a full same-doc block: no masking. Native dtype throughout."""
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_Dh  = tl.arange(0, Dh)

    Q_T_offsets  = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N  + offsets_Dh[None, :] * stride_Dh

    for _ in range(num_steps):
        mask_N = offsets_ROW < N

        Q_T   = tl.load(Q_ptr    + Q_T_offsets,  mask=mask_N[None, :], other=0.)
        LSE   = tl.load(LSE_ptr  + offsets_ROW,   mask=mask_N, other=0.)
        dLdO  = tl.load(dLdO_ptr + dLdO_offsets,  mask=mask_N[:, None], other=0.)
        Delta = tl.load(Delta_ptr + offsets_ROW,   mask=mask_N, other=0.)

        S_T    = tl.dot(K, Q_T) * (scale * rln2)      # bf16 TC → fp32, then scale
        P_T    = tl.exp2(S_T - LSE[None, :])           # fp32; no masking
        P_T_c  = P_T.to(K.dtype)                       # fp32 → bf16 for TC
        dLdV   = tl.dot(P_T_c, dLdO, acc=dLdV)
        dLdP_T = tl.dot(V, tl.trans(dLdO))             # both bf16 → fp32
        dLdS_T = P_T * (dLdP_T - Delta[None, :]) * ln2
        dLdK   = tl.dot(dLdS_T.to(K.dtype), tl.trans(Q_T), acc=dLdK)

        offsets_ROW  += BLOCK_SIZE_ROW
        Q_ptr        += BLOCK_SIZE_ROW * stride_N
        dLdO_ptr     += BLOCK_SIZE_ROW * stride_N

    return dLdK, dLdV


@triton.jit
def _bwd_kv_cdb_v10(
    K, V, dLdK, dLdV,
    Q_ptr, dLdO_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
    n_chunks: tl.constexpr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr,
):
    """dLdK/dLdV: full masking (same_doc | in_grant | optional causal). Native dtype."""
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

        S_T = tl.dot(K, Q_T) * (scale * rln2)
        P_T = tl.exp2(S_T - LSE[None, :])

        doc_row  = tl.load(doc_ids_ptr + offsets_ROW, mask=mask_N, other=-2)
        same_doc = (doc_col[:, None] == doc_row[None, :])

        in_grant_T = tl.zeros([BLOCK_SIZE_COL, BLOCK_SIZE_ROW], dtype=tl.int1)
        for c in tl.static_range(n_chunks):
            kv_bm = tl.load(kv_bitmasks_ptr + c * T + offsets_COL,
                            mask=offsets_COL < N, other=0)
            q_bm  = tl.load(q_bitmasks_ptr  + c * T + offsets_ROW,
                            mask=mask_N, other=0)
            in_grant_T = in_grant_T | ((kv_bm[:, None] & q_bm[None, :]) != 0)

        attend = same_doc | in_grant_T

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
def _bwd_q_full_v10(
    dLdQ, Q, dLdO, LSE,
    K_ptr, V_ptr, Delta_ptr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
):
    """dLdQ for a full same-doc block: no masking. Native dtype throughout."""
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh  = tl.arange(0, Dh)

    KV_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N
    Delta = tl.load(Delta_ptr + offsets_ROW, mask=offsets_ROW < N, other=0.)

    for _ in range(num_steps):
        col_mask = offsets_COL < N

        K_T = tl.load(K_ptr + KV_T_offsets, mask=col_mask[None, :], other=0.)
        V_T = tl.load(V_ptr + KV_T_offsets, mask=col_mask[None, :], other=0.)

        S    = tl.dot(Q, K_T) * (scale * rln2)
        P    = tl.exp2(S - LSE)                        # fp32; no masking
        dLdP = tl.dot(dLdO, V_T)                       # both bf16 → fp32
        dLdS = P * (dLdP - Delta[:, None]) * ln2
        dLdQ += tl.dot(dLdS.to(Q.dtype), tl.trans(K_T))

        offsets_COL += BLOCK_SIZE_COL
        K_ptr       += BLOCK_SIZE_COL * stride_N
        V_ptr       += BLOCK_SIZE_COL * stride_N

    return dLdQ


@triton.jit
def _bwd_q_cdb_v10(
    dLdQ, Q, dLdO, LSE,
    K_ptr, V_ptr, Delta_ptr,
    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
    n_chunks: tl.constexpr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr,
):
    """dLdQ: full masking (same_doc | in_grant | optional causal). Native dtype."""
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
        same_doc = (doc_row[:, None] == doc_col[None, :])

        in_grant = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.int1)
        for c in tl.static_range(n_chunks):
            q_bm  = tl.load(q_bitmasks_ptr  + c * T + offsets_ROW,
                            mask=offsets_ROW < N, other=0)
            kv_bm = tl.load(kv_bitmasks_ptr + c * T + offsets_COL,
                            mask=col_mask, other=0)
            in_grant = in_grant | ((q_bm[:, None] & kv_bm[None, :]) != 0)

        attend = same_doc | in_grant

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
# Backward — dK/dV outer kernel
# ===========================================================================

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for m in [16, 32, 64]  # 128 excluded: violates BIM_BLOCK_SIZE%BLOCK_SIZE_MICRO==0 when BIM_BS=64 (v18 bwd)
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_KV_cdb_bim_v10(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
    kv_q_counts_ptr, kv_q_ptrs_ptr, kv_q_indices_ptr,
    kv_q_n_full_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    n_chunks: tl.constexpr,
    BIM_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
    """Compute dLdK and dLdV. K and V stay in native dtype (no pre-scaling)."""
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

    # KEY CHANGE: load K and V in native dtype, no pre-scaling of K
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.)

    dLdK = tl.zeros([BLOCK_SIZE_COL, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL, Dh], dtype=tl.float32)

    kv_q_start   = tl.load(kv_q_ptrs_ptr   + pid)
    num_q_macros = tl.load(kv_q_counts_ptr  + pid)
    n_full_kv    = tl.load(kv_q_n_full_ptr  + pid)

    # Diagonal: first entry in kv_q
    q_b_diag = tl.load(kv_q_indices_ptr + kv_q_start)
    dLdK, dLdV = _bwd_kv_cdb_v10(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
        n_chunks,
        stride_N, stride_Dh, N, Dh,
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

    # Off-diagonal non-full Q blocks: full masking
    for i in range(1 + n_full_kv, num_q_macros):
        q_b = tl.load(kv_q_indices_ptr + kv_q_start + i)
        dLdK, dLdV = _bwd_kv_cdb_v10(
            K, V, dLdK, dLdV,
            Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
            doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
            n_chunks,
            stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            q_b * BLOCK_SIZE_COL, start_COL, num_micro,
            scale, ln2, rln2, MASK=False,
        )

    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK.to(dLdK_ptr.dtype.element_ty), mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV.to(dLdV_ptr.dtype.element_ty), mask=KV_mask)


# ===========================================================================
# Backward — dQ outer kernel
# ===========================================================================

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for m in [16, 32, 64]  # 128 excluded: violates BIM_BLOCK_SIZE%BLOCK_SIZE_MICRO==0 when BIM_BS=64 (v18 bwd)
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_Q_cdb_bim_v10(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdQ_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
    q_kv_counts_ptr, q_kv_ptrs_ptr, q_kv_indices_ptr,
    q_kv_n_full_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    n_chunks: tl.constexpr,
    BIM_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
    """Compute dLdQ. Q and dLdO stay in native dtype (no pre-scaling)."""
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

    # KEY CHANGE: load Q and dLdO in native dtype, no pre-scaling
    Q    = tl.load(Q_ptr    + QO_offsets, mask=mask_ROW[:, None], other=0.)
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.)
    LSE  = tl.load(LSE_ptr  + offsets_ROW, mask=mask_ROW, other=0.)[:, None]
    dLdQ = tl.zeros([BLOCK_SIZE_ROW, Dh], dtype=tl.float32)

    q_kv_start    = tl.load(q_kv_ptrs_ptr   + pid)
    num_kv_macros = tl.load(q_kv_counts_ptr  + pid)
    n_full_q      = tl.load(q_kv_n_full_ptr  + pid)

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

    # Off-diagonal non-full KV blocks: full masking
    for i in range(n_full_q, num_kv_macros - 1):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        dLdQ = _bwd_q_cdb_v10(
            dLdQ, Q, dLdO, LSE,
            K_ptr, V_ptr, Delta_ptr,
            doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
            n_chunks,
            stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            start_ROW, kv_b * BLOCK_SIZE_ROW, num_micro,
            scale, ln2, rln2, MASK=False,
        )

    # Diagonal KV block: last entry
    kv_b_diag = tl.load(q_kv_indices_ptr + q_kv_start + num_kv_macros - 1)
    dLdQ = _bwd_q_cdb_v10(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
        n_chunks,
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

class _CDBBIMv10(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
        T, H, Dh = q.shape
        n_chunks  = q_bitmasks.shape[0]
        bim_bs    = bim.block_size
        n_blocks  = bim.n_blocks
        assert bim.q_kv_n_full is not None, \
            "BIM missing q_kv_n_full — rebuild with updated CrossDocLinkMaskCreator"

        q_f  = q.permute(1, 0, 2).unsqueeze(0).contiguous()
        k_f  = k.permute(1, 0, 2).unsqueeze(0).contiguous()
        v_f  = v.permute(1, 0, 2).unsqueeze(0).contiguous()
        q_bm_c  = q_bitmasks.contiguous()
        kv_bm_c = kv_bitmasks.contiguous()

        B_k = 1
        O   = torch.empty_like(q_f)
        LSE = torch.empty(B_k, H, T, device=q.device, dtype=torch.float32)

        grid_fwd = (n_blocks, B_k * H)
        _attn_fwd_cdb_bim_v10[grid_fwd](
            q_f, k_f, v_f, O, LSE, scale,
            q_f.stride(0), q_f.stride(1), q_f.stride(2), q_f.stride(3),
            k_f.stride(0), k_f.stride(1), k_f.stride(2), k_f.stride(3),
            v_f.stride(0), v_f.stride(1), v_f.stride(2), v_f.stride(3),
            O.stride(0),   O.stride(1),   O.stride(2),   O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            document_ids, q_bm_c, kv_bm_c, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            B_k, H, T, Dh, n_chunks, bim_bs,
        )

        ctx.save_for_backward(q_f, k_f, v_f, O, LSE)
        ctx.document_ids = document_ids
        ctx.bim          = bim
        ctx.q_bitmasks   = q_bm_c
        ctx.kv_bitmasks  = kv_bm_c
        ctx.T, ctx.H, ctx.Dh = T, H, Dh
        ctx.n_chunks = n_chunks
        ctx.scale    = scale
        return O.squeeze(0).permute(1, 0, 2)

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        document_ids     = ctx.document_ids
        bim              = ctx.bim
        q_bm, kv_bm      = ctx.q_bitmasks, ctx.kv_bitmasks
        T, H, Dh         = ctx.T, ctx.H, ctx.Dh
        n_chunks         = ctx.n_chunks
        scale            = ctx.scale
        B_k              = 1

        dLdO_f = dLdO.permute(1, 0, 2).unsqueeze(0).contiguous()
        assert q.stride() == k.stride() == v.stride() == O.stride() == dLdO_f.stride()

        dLdq = torch.empty_like(q)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)
        Delta = torch.empty_like(LSE)

        pre_grid = lambda meta: (triton.cdiv(T, meta["PRE_BLOCK_SIZE_ROW"]), B_k * H)
        _attn_backward_preprocess_cdb[pre_grid](
            O, dLdO_f, Delta,
            O.stride(0),      O.stride(1),      O.stride(2),      O.stride(3),
            dLdO_f.stride(0), dLdO_f.stride(1), dLdO_f.stride(2), dLdO_f.stride(3),
            Delta.stride(0),  Delta.stride(1),  Delta.stride(2),
            T, Dh,
        )

        grid = (bim.n_blocks, B_k * H)

        _attn_backward_KV_cdb_bim_v10[grid](
            q, k, v, dLdO_f, dLdk, dLdv, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices,
            bim.kv_q_n_full,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, T, Dh, n_chunks, bim.block_size,
        )

        _attn_backward_Q_cdb_bim_v10[grid](
            q, k, v, dLdO_f, dLdq, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, T, Dh, n_chunks, bim.block_size,
        )

        to_thd = lambda t: t.squeeze(0).permute(1, 0, 2)
        return to_thd(dLdq), to_thd(dLdk), to_thd(dLdv), None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v10(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask",
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v10: native-dtype matmuls (bf16 TC) — no fp32 cast on load."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _CDBBIMv10.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
