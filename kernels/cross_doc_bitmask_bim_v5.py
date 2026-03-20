"""
cross_doc_bitmask BIM v5 — free-tile backward.

Root cause of backward gap vs varlen: BLOCK_SIZE_MACRO was locked == BIM_BLOCK_SIZE=64.
varlen backward autotuned BLOCK_SIZE_MACRO up to 128, giving 128×128×Dh matmuls.
Arithmetic intensity scales with BLOCK_SIZE_MACRO; going 64→128 doubles it,
moving from 20% to 41% of A100 roofline.

Fix:
  Forward:  unchanged — reuses v3 BIM-guided kernel (already 30% faster than flex).
  Backward: two split kernels with BLOCK_SIZE_MACRO free-tuned over {64, 128}.
    • Same-doc sparsity: cu_seqlens range-limit (exactly like varlen backward).
    • Cross-doc sparsity: bitmask block-union scan at BLOCK_SIZE_MACRO granularity
      (T/BLOCK_SIZE_MACRO = 256 iterations at T=32768, not 1024).
    • No BIM needed in backward → fewer kernel args → lower register pressure.

Cross-doc scan: for each BLOCK_SIZE_MACRO-sized Q chunk outside the same-doc range,
  OR-reduce q_bitmasks over the chunk and AND with kv_union (precomputed once per CTA).
  If non-zero, process the full BLOCK_SIZE_MACRO chunk with bitmask per-element masking.
  At T=32768 with 64 grants, ~254 checks per CTA; <1% are non-zero.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .cross_doc_bitmask_attn import _attn_backward_preprocess_cdb, _bit_or_combine
from .cross_doc_bitmask_bim_v3 import _attn_fwd_cdb_bim_v3

if TYPE_CHECKING:
    from model.graph_traversal.cross_doc_mask import BlockInteractionMask


# ---------------------------------------------------------------------------
# dK/dV inner sub-kernel
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_kv_inner_v5(
    K, V, dLdK, dLdV,
    Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr,
    T, n_chunks: tl.constexpr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,   # = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL: tl.constexpr,   # = BLOCK_SIZE_MACRO
    start_ROW, start_COL, num_steps,
    ln2: tl.constexpr,
    MASK: tl.constexpr,       # True = causal (diagonal tile)
    CROSS_DOC: tl.constexpr,  # True = bitmask grant masking, False = same_doc masking
):
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh  = tl.arange(0, Dh)

    Q_T_offsets  = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N  + offsets_Dh[None, :] * stride_Dh

    if not CROSS_DOC:
        doc_col = tl.load(doc_ids_ptr + offsets_COL, mask=offsets_COL < N, other=-1)

    for _ in range(num_steps):
        mask_N = offsets_ROW < N
        Q_T   = tl.load(Q_ptr    + Q_T_offsets,  mask=mask_N[None, :], other=0.).to(tl.float32)
        LSE   = tl.load(LSE_ptr  + offsets_ROW,   mask=mask_N, other=0.)
        dLdO  = tl.load(dLdO_ptr + dLdO_offsets,  mask=mask_N[:, None], other=0.).to(tl.float32)
        Delta = tl.load(Delta_ptr + offsets_ROW,   mask=mask_N, other=0.)

        S_T = tl.dot(K, Q_T)
        P_T = tl.exp2(S_T - LSE[None, :])

        if CROSS_DOC:
            in_grant_T = tl.zeros([BLOCK_SIZE_COL, BLOCK_SIZE_ROW], dtype=tl.int1)
            for c in tl.static_range(n_chunks):
                kv_bm = tl.load(kv_bitmasks_ptr + c * T + offsets_COL,
                                mask=offsets_COL < N, other=0)
                q_bm  = tl.load(q_bitmasks_ptr  + c * T + offsets_ROW,
                                mask=mask_N, other=0)
                in_grant_T = in_grant_T | ((kv_bm[:, None] & q_bm[None, :]) != 0)
            P_T = tl.where(in_grant_T, P_T, 0.)
        else:
            doc_row  = tl.load(doc_ids_ptr + offsets_ROW, mask=mask_N, other=-2)
            same_doc = (doc_col[:, None] == doc_row[None, :])
            if MASK:
                causal = (offsets_COL[:, None] <= offsets_ROW[None, :])
                P_T = tl.where(causal & same_doc, P_T, 0.)
            else:
                P_T = tl.where(same_doc, P_T, 0.)

        dLdV   = tl.dot(P_T, dLdO, acc=dLdV)
        dLdP_T = tl.dot(V, tl.trans(dLdO))
        dLdS_T = P_T * (dLdP_T - Delta[None, :]) * ln2
        dLdK   = tl.dot(dLdS_T, tl.trans(Q_T), acc=dLdK)

        offsets_ROW  += BLOCK_SIZE_ROW
        Q_ptr        += BLOCK_SIZE_ROW * stride_N
        dLdO_ptr     += BLOCK_SIZE_ROW * stride_N

    return dLdK, dLdV


# ---------------------------------------------------------------------------
# dQ inner sub-kernel
# ---------------------------------------------------------------------------

@triton.jit
def _bwd_q_inner_v5(
    dLdQ, Q, dLdO, LSE,
    K_ptr, V_ptr, Delta_ptr,
    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr,
    T, n_chunks: tl.constexpr,
    stride_N, stride_Dh,
    N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,   # = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL: tl.constexpr,   # = BLOCK_SIZE_MICRO
    start_ROW, start_COL, num_steps,
    ln2: tl.constexpr,
    MASK: tl.constexpr,
    CROSS_DOC: tl.constexpr,
):
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh  = tl.arange(0, Dh)

    K_and_V_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N
    Delta   = tl.load(Delta_ptr   + offsets_ROW, mask=offsets_ROW < N, other=0.)
    doc_row = tl.load(doc_ids_ptr + offsets_ROW, mask=offsets_ROW < N, other=-1)

    for _ in range(num_steps):
        col_mask = offsets_COL < N
        K_T = tl.load(K_ptr + K_and_V_T_offsets, mask=col_mask[None, :], other=0.).to(tl.float32)
        V_T = tl.load(V_ptr + K_and_V_T_offsets, mask=col_mask[None, :], other=0.).to(tl.float32)

        S = tl.dot(Q, K_T)
        P = tl.exp2(S - LSE)

        if CROSS_DOC:
            in_grant = tl.zeros([BLOCK_SIZE_ROW, BLOCK_SIZE_COL], dtype=tl.int1)
            for c in tl.static_range(n_chunks):
                q_bm  = tl.load(q_bitmasks_ptr  + c * T + offsets_ROW,
                                mask=offsets_ROW < N, other=0)
                kv_bm = tl.load(kv_bitmasks_ptr + c * T + offsets_COL,
                                mask=col_mask, other=0)
                in_grant = in_grant | ((q_bm[:, None] & kv_bm[None, :]) != 0)
            P = tl.where(in_grant, P, 0.)
        else:
            doc_col  = tl.load(doc_ids_ptr + offsets_COL, mask=col_mask, other=-2)
            same_doc = (doc_row[:, None] == doc_col[None, :])
            if MASK:
                causal = (offsets_ROW[:, None] >= offsets_COL[None, :])
                P = tl.where(causal & same_doc, P, 0.)
            else:
                P = tl.where(same_doc, P, 0.)

        dLdP  = tl.dot(dLdO, V_T)
        dLdS  = P * (dLdP - Delta[:, None]) * ln2
        dLdQ += tl.dot(dLdS, tl.trans(K_T))

        offsets_COL += BLOCK_SIZE_COL
        K_ptr       += BLOCK_SIZE_COL * stride_N
        V_ptr       += BLOCK_SIZE_COL * stride_N

    return dLdQ


# ---------------------------------------------------------------------------
# dK/dV backward kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MACRO": M, "BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for M, m in [(64, 16), (64, 32), (64, 64), (128, 32), (128, 64), (128, 128)]
        for ns in [3, 4, 5]
        for nw in [4, 8]
        if M % m == 0
    ],
    key=["N", "Dh", "n_chunks"],
)
@triton.jit
def _attn_backward_KV_v5(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr, cu_seqlens_ptr,
    q_bitmasks_ptr, kv_bitmasks_ptr,
    T,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    n_chunks: tl.constexpr,
    BLOCK_SIZE_MACRO: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
    ln2:  tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    idx_bh = tl.program_id(1)
    idx_b  = idx_bh // H
    idx_h  = idx_bh % H
    bh = idx_b * stride_B + idx_h * stride_H
    Q_ptr += bh; K_ptr += bh; V_ptr += bh
    dLdO_ptr += bh; dLdK_ptr += bh; dLdV_ptr += bh
    LSE_ptr   += idx_bh * N
    Delta_ptr += idx_bh * N

    offsets_Dh = tl.arange(0, Dh)
    pid = tl.program_id(0)

    BLOCK_SIZE_ROW: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL: tl.constexpr = BLOCK_SIZE_MACRO
    num_micro: tl.constexpr = BLOCK_SIZE_COL // BLOCK_SIZE_ROW

    start_COL    = pid * BLOCK_SIZE_COL
    offsets_COL  = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    KV_offsets   = offsets_COL[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask      = offsets_COL[:, None] < N
    K  = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.).to(tl.float32)
    V  = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.).to(tl.float32)
    K *= scale * rln2

    dLdK = tl.zeros([BLOCK_SIZE_COL, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL, Dh], dtype=tl.float32)

    kv_doc_id    = tl.load(doc_ids_ptr + start_COL).to(tl.int32)
    doc_kv_start = tl.load(cu_seqlens_ptr + kv_doc_id).to(tl.int32)
    doc_kv_end   = tl.load(cu_seqlens_ptr + kv_doc_id + 1).to(tl.int32)

    # Precompute KV bitmask union over the whole BLOCK_SIZE_COL chunk (once per CTA).
    kv_union_0 = tl.reduce(
        tl.load(kv_bitmasks_ptr + 0 * T + offsets_COL, mask=offsets_COL < N, other=0),
        0, _bit_or_combine,
    )
    if n_chunks >= 2:
        kv_union_1 = tl.reduce(
            tl.load(kv_bitmasks_ptr + 1 * T + offsets_COL, mask=offsets_COL < N, other=0),
            0, _bit_or_combine,
        )
    if n_chunks >= 3:
        kv_union_2 = tl.reduce(
            tl.load(kv_bitmasks_ptr + 2 * T + offsets_COL, mask=offsets_COL < N, other=0),
            0, _bit_or_combine,
        )
    if n_chunks >= 4:
        kv_union_3 = tl.reduce(
            tl.load(kv_bitmasks_ptr + 3 * T + offsets_COL, mask=offsets_COL < N, other=0),
            0, _bit_or_combine,
        )

    # ── 1. Diagonal BLOCK_SIZE_COL macro-tile (MASK=True, same-doc) ─────
    dLdK, dLdV = _bwd_kv_inner_v5(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr,
        T, n_chunks, stride_N, stride_Dh, N, Dh,
        BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
        start_COL, start_COL, num_micro,
        ln2, MASK=True, CROSS_DOC=False,
    )

    # ── 2. Same-doc off-diagonal Q blocks after diagonal ─────────────────
    start_after_diag = start_COL + BLOCK_SIZE_COL
    doc_kv_end_aligned = tl.cdiv(doc_kv_end, BLOCK_SIZE_ROW) * BLOCK_SIZE_ROW
    num_same = tl.maximum((doc_kv_end_aligned - start_after_diag) // BLOCK_SIZE_ROW, 0)
    dLdK, dLdV = _bwd_kv_inner_v5(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr,
        T, n_chunks, stride_N, stride_Dh, N, Dh,
        BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
        start_after_diag, start_COL, num_same,
        ln2, MASK=False, CROSS_DOC=False,
    )

    # ── 3. Cross-doc Q tiles (scan at BLOCK_SIZE_COL granularity) ───────
    # Cross-doc grants point FROM a later document TO this KV block.
    # Only Q macro-tiles entirely AFTER this KV doc can carry such grants.
    # Tiles before or within the KV doc are handled in steps 1 & 2.
    N_macro_aligned = ((N + BLOCK_SIZE_COL - 1) // BLOCK_SIZE_COL) * BLOCK_SIZE_COL
    doc_kv_end_macro = tl.cdiv(doc_kv_end, BLOCK_SIZE_COL) * BLOCK_SIZE_COL
    cur_q_macro = 0
    for _step in range(N_macro_aligned // BLOCK_SIZE_COL):
        is_after_kv_doc = cur_q_macro >= doc_kv_end_macro
        if is_after_kv_doc:
            q_macro_offsets = cur_q_macro + tl.arange(0, BLOCK_SIZE_COL)
            q_union_0 = tl.reduce(
                tl.load(q_bitmasks_ptr + 0 * T + q_macro_offsets,
                        mask=q_macro_offsets < N, other=0),
                0, _bit_or_combine,
            )
            can_interact = (kv_union_0 & q_union_0) != 0
            if n_chunks >= 2:
                q_union_1 = tl.reduce(
                    tl.load(q_bitmasks_ptr + 1 * T + q_macro_offsets,
                            mask=q_macro_offsets < N, other=0),
                    0, _bit_or_combine,
                )
                can_interact = can_interact | ((kv_union_1 & q_union_1) != 0)
            if n_chunks >= 3:
                q_union_2 = tl.reduce(
                    tl.load(q_bitmasks_ptr + 2 * T + q_macro_offsets,
                            mask=q_macro_offsets < N, other=0),
                    0, _bit_or_combine,
                )
                can_interact = can_interact | ((kv_union_2 & q_union_2) != 0)
            if n_chunks >= 4:
                q_union_3 = tl.reduce(
                    tl.load(q_bitmasks_ptr + 3 * T + q_macro_offsets,
                            mask=q_macro_offsets < N, other=0),
                    0, _bit_or_combine,
                )
                can_interact = can_interact | ((kv_union_3 & q_union_3) != 0)
            if can_interact:
                dLdK, dLdV = _bwd_kv_inner_v5(
                    K, V, dLdK, dLdV,
                    Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
                    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr,
                    T, n_chunks, stride_N, stride_Dh, N, Dh,
                    BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
                    cur_q_macro, start_COL, num_micro,
                    ln2, MASK=False, CROSS_DOC=True,
                )
        cur_q_macro += BLOCK_SIZE_COL

    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK.to(dLdK_ptr.dtype.element_ty), mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV.to(dLdV_ptr.dtype.element_ty), mask=KV_mask)


# ---------------------------------------------------------------------------
# dQ backward kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MACRO": M, "BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for M, m in [(64, 16), (64, 32), (64, 64), (128, 32), (128, 64), (128, 128)]
        for ns in [3, 4, 5]
        for nw in [4, 8]
        if M % m == 0
    ],
    key=["N", "Dh", "n_chunks"],
)
@triton.jit
def _attn_backward_Q_v5(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdQ_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr, cu_seqlens_ptr,
    q_bitmasks_ptr, kv_bitmasks_ptr,
    T,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    n_chunks: tl.constexpr,
    BLOCK_SIZE_MACRO: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
    ln2:  tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    idx_bh = tl.program_id(1)
    idx_b  = idx_bh // H
    idx_h  = idx_bh % H
    bh = idx_b * stride_B + idx_h * stride_H
    Q_ptr    += bh; K_ptr += bh; V_ptr += bh
    dLdO_ptr += bh; dLdQ_ptr += bh
    LSE_ptr   += idx_bh * N
    Delta_ptr += idx_bh * N

    offsets_Dh = tl.arange(0, Dh)
    pid = tl.program_id(0)

    BLOCK_SIZE_ROW: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL: tl.constexpr = BLOCK_SIZE_MICRO
    num_micro: tl.constexpr = BLOCK_SIZE_ROW // BLOCK_SIZE_COL

    start_ROW   = pid * BLOCK_SIZE_ROW
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    QO_offsets  = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW    = offsets_ROW < N
    Q    = tl.load(Q_ptr    + QO_offsets, mask=mask_ROW[:, None], other=0.).to(tl.float32)
    Q   *= scale * rln2
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.).to(tl.float32)
    LSE  = tl.load(LSE_ptr  + offsets_ROW, mask=mask_ROW, other=0.)[:, None]
    dLdQ = tl.zeros([BLOCK_SIZE_ROW, Dh], dtype=tl.float32)

    q_doc_id    = tl.load(doc_ids_ptr + start_ROW).to(tl.int32)
    doc_q_start = tl.load(cu_seqlens_ptr + q_doc_id).to(tl.int32)

    # Precompute Q bitmask union
    q_union_0 = tl.reduce(
        tl.load(q_bitmasks_ptr + 0 * T + offsets_ROW, mask=mask_ROW, other=0),
        0, _bit_or_combine,
    )
    if n_chunks >= 2:
        q_union_1 = tl.reduce(
            tl.load(q_bitmasks_ptr + 1 * T + offsets_ROW, mask=mask_ROW, other=0),
            0, _bit_or_combine,
        )
    if n_chunks >= 3:
        q_union_2 = tl.reduce(
            tl.load(q_bitmasks_ptr + 2 * T + offsets_ROW, mask=mask_ROW, other=0),
            0, _bit_or_combine,
        )
    if n_chunks >= 4:
        q_union_3 = tl.reduce(
            tl.load(q_bitmasks_ptr + 3 * T + offsets_ROW, mask=mask_ROW, other=0),
            0, _bit_or_combine,
        )

    # ── 1. Same-doc KV blocks before diagonal ────────────────────────────
    doc_q_start_aligned = (doc_q_start // BLOCK_SIZE_COL) * BLOCK_SIZE_COL
    num_pre = (start_ROW - doc_q_start_aligned) // BLOCK_SIZE_COL
    dLdQ = _bwd_q_inner_v5(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr,
        T, n_chunks, stride_N, stride_Dh, N, Dh,
        BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
        start_ROW, doc_q_start_aligned, num_pre,
        ln2, MASK=False, CROSS_DOC=False,
    )

    # ── 2. Diagonal (MASK=True, same-doc) ────────────────────────────────
    dLdQ = _bwd_q_inner_v5(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr,
        doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr,
        T, n_chunks, stride_N, stride_Dh, N, Dh,
        BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
        start_ROW, start_ROW, num_micro,
        ln2, MASK=True, CROSS_DOC=False,
    )

    # ── 3. Cross-doc KV tiles (scan at BLOCK_SIZE_ROW granularity) ───────
    # Cross-doc grants point FROM this Q block TO an earlier document.
    # Only KV macro-tiles entirely BEFORE this Q doc carry such grant targets.
    # Tiles from doc_q_start onward are same-doc (step 1) or diagonal (step 2).
    N_macro_aligned = ((N + BLOCK_SIZE_ROW - 1) // BLOCK_SIZE_ROW) * BLOCK_SIZE_ROW
    doc_q_start_macro = (doc_q_start // BLOCK_SIZE_ROW) * BLOCK_SIZE_ROW
    cur_kv_macro = 0
    for _step in range(N_macro_aligned // BLOCK_SIZE_ROW):
        is_before_q_doc = cur_kv_macro < doc_q_start_macro
        if is_before_q_doc:
            kv_macro_offsets = cur_kv_macro + tl.arange(0, BLOCK_SIZE_ROW)
            kv_union_0 = tl.reduce(
                tl.load(kv_bitmasks_ptr + 0 * T + kv_macro_offsets,
                        mask=kv_macro_offsets < N, other=0),
                0, _bit_or_combine,
            )
            can_interact = (q_union_0 & kv_union_0) != 0
            if n_chunks >= 2:
                kv_union_1 = tl.reduce(
                    tl.load(kv_bitmasks_ptr + 1 * T + kv_macro_offsets,
                            mask=kv_macro_offsets < N, other=0),
                    0, _bit_or_combine,
                )
                can_interact = can_interact | ((q_union_1 & kv_union_1) != 0)
            if n_chunks >= 3:
                kv_union_2 = tl.reduce(
                    tl.load(kv_bitmasks_ptr + 2 * T + kv_macro_offsets,
                            mask=kv_macro_offsets < N, other=0),
                    0, _bit_or_combine,
                )
                can_interact = can_interact | ((q_union_2 & kv_union_2) != 0)
            if n_chunks >= 4:
                kv_union_3 = tl.reduce(
                    tl.load(kv_bitmasks_ptr + 3 * T + kv_macro_offsets,
                            mask=kv_macro_offsets < N, other=0),
                    0, _bit_or_combine,
                )
                can_interact = can_interact | ((q_union_3 & kv_union_3) != 0)
            if can_interact:
                dLdQ = _bwd_q_inner_v5(
                    dLdQ, Q, dLdO, LSE,
                    K_ptr, V_ptr, Delta_ptr,
                    doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr,
                    T, n_chunks, stride_N, stride_Dh, N, Dh,
                    BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
                    start_ROW, cur_kv_macro, num_micro,
                    ln2, MASK=False, CROSS_DOC=True,
                )
        cur_kv_macro += BLOCK_SIZE_ROW

    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ.to(dLdQ_ptr.dtype.element_ty), mask=mask_ROW[:, None])


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

def _build_cu_seqlens_v5(document_ids: torch.Tensor) -> torch.Tensor:
    n_docs = int(document_ids.max().item()) + 1
    cu = torch.zeros(n_docs + 1, dtype=torch.int32, device=document_ids.device)
    cu[1:] = torch.bincount(document_ids.long(), minlength=n_docs).to(torch.int32).cumsum(0)
    return cu


class _CDBBIMv5(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
        T, H, Dh = q.shape
        n_chunks  = q_bitmasks.shape[0]
        bim_bs    = bim.block_size
        n_blocks  = bim.n_blocks
        assert bim.q_kv_n_full is not None

        q_f  = q.permute(1, 0, 2).unsqueeze(0).contiguous()
        k_f  = k.permute(1, 0, 2).unsqueeze(0).contiguous()
        v_f  = v.permute(1, 0, 2).unsqueeze(0).contiguous()
        q_bm_c  = q_bitmasks.contiguous()
        kv_bm_c = kv_bitmasks.contiguous()

        B_k = 1
        O   = torch.empty_like(q_f)
        LSE = torch.empty(B_k, H, T, device=q.device, dtype=torch.float32)

        _attn_fwd_cdb_bim_v3[(n_blocks, B_k * H)](
            q_f, k_f, v_f, O, LSE, scale,
            q_f.stride(0), q_f.stride(1), q_f.stride(2), q_f.stride(3),
            k_f.stride(0), k_f.stride(1), k_f.stride(2), k_f.stride(3),
            v_f.stride(0), v_f.stride(1), v_f.stride(2), v_f.stride(3),
            O.stride(0),   O.stride(1),   O.stride(2),   O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            document_ids, q_bm_c, kv_bm_c, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices, bim.q_kv_n_full,
            B_k, H, T, Dh, n_chunks, bim_bs,
        )

        cu_seqlens = _build_cu_seqlens_v5(document_ids)
        ctx.save_for_backward(q_f, k_f, v_f, O, LSE)
        ctx.document_ids = document_ids
        ctx.cu_seqlens   = cu_seqlens
        ctx.q_bitmasks   = q_bm_c
        ctx.kv_bitmasks  = kv_bm_c
        ctx.T, ctx.H, ctx.Dh = T, H, Dh
        ctx.n_chunks = n_chunks
        ctx.scale    = scale
        return O.squeeze(0).permute(1, 0, 2)

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        document_ids = ctx.document_ids
        cu_seqlens   = ctx.cu_seqlens
        q_bm, kv_bm  = ctx.q_bitmasks, ctx.kv_bitmasks
        T, H, Dh     = ctx.T, ctx.H, ctx.Dh
        n_chunks     = ctx.n_chunks
        scale        = ctx.scale
        B_k          = 1

        dLdO_f = dLdO.permute(1, 0, 2).unsqueeze(0).contiguous()
        assert q.stride() == k.stride() == v.stride() == O.stride() == dLdO_f.stride()

        dLdq = torch.empty_like(q)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)
        Delta = torch.empty_like(LSE)

        pre_grid = lambda meta: (triton.cdiv(T, meta["PRE_BLOCK_SIZE_ROW"]), B_k * H)
        _attn_backward_preprocess_cdb[pre_grid](
            O, dLdO_f, Delta,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dLdO_f.stride(0), dLdO_f.stride(1), dLdO_f.stride(2), dLdO_f.stride(3),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            T, Dh,
        )

        kv_grid = lambda meta: (triton.cdiv(T, meta["BLOCK_SIZE_MACRO"]), B_k * H)
        _attn_backward_KV_v5[kv_grid](
            q, k, v, dLdO_f, dLdk, dLdv, LSE, Delta,
            document_ids, cu_seqlens, q_bm, kv_bm, T, scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, T, Dh, n_chunks,
        )

        q_grid = lambda meta: (triton.cdiv(T, meta["BLOCK_SIZE_MACRO"]), B_k * H)
        _attn_backward_Q_v5[q_grid](
            q, k, v, dLdO_f, dLdq, LSE, Delta,
            document_ids, cu_seqlens, q_bm, kv_bm, T, scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, T, Dh, n_chunks,
        )

        to_thd = lambda t: t.squeeze(0).permute(1, 0, 2)
        return to_thd(dLdq), to_thd(dLdk), to_thd(dLdv), None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v5(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask",
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v5: free-tile backward (BLOCK_SIZE_MACRO up to 128, no BIM in bwd)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _CDBBIMv5.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
