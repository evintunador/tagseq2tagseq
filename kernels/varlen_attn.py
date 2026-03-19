"""
Triton flash-attention kernel — doc_causal mask (packed documents).

Extends causal_attn.py: only the mask logic changes.
A precomputed document_ids tensor (shape [T], int32) is used so that two
positions may only attend if they belong to the same document AND the query
position comes no earlier than the key position.

Public interface:
    triton_attn_varlen(q, k, v, cu_seqlens, max_seqlen, scale) → Tensor
    q/k/v shape: (T, H, Dh)  — matches PyTorch VSLF layout for direct comparison.
    Internally reshaped to (1, H, T, Dh) to reuse the causal kernel structure.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Helper: build document_ids from cu_seqlens
# ---------------------------------------------------------------------------

def _build_document_ids(cu_seqlens: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
    """Map each token position to its document index."""
    document_ids = torch.zeros(T, dtype=torch.int32, device=device)
    n_docs = len(cu_seqlens) - 1
    for d in range(n_docs):
        s = int(cu_seqlens[d].item())
        e = int(cu_seqlens[d + 1].item())
        if s < e:
            document_ids[s:e] = d
    return document_ids


# ---------------------------------------------------------------------------
# Forward inner kernel (doc_causal mask)
# ---------------------------------------------------------------------------

@triton.jit
def _attn_fwd_inner_varlen(
    Q, O, L, M,
    K_ptr, V_ptr,
    K_T_offsets, V_offsets,
    block_index_QO,
    softmax_scale,
    stride_K_N, stride_V_N,
    doc_ids_ptr,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
    DIAGONAL: tl.constexpr,
    offsets_QO_N: tl.constexpr, offsets_KV_N: tl.constexpr,
    N: tl.constexpr, Dh: tl.constexpr,
):
    if DIAGONAL:
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        lo = tl.multiple_of(lo, BLOCK_SIZE_QO)
    else:
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO

    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo

    # Doc-ids for the fixed Q block (loaded once outside the KV loop)
    mask_QO_N = offsets_QO_N < N
    doc_q = tl.load(doc_ids_ptr + offsets_QO_N, mask=mask_QO_N, other=-1)  # (BLOCK_SIZE_QO,)

    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        start_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        mask_KV_N = offsets_KV_N < N

        K_T = tl.load(K_ptr + K_T_offsets, mask=mask_KV_N[None, :], other=0.).to(tl.float32)
        S = tl.dot(Q, K_T) * softmax_scale

        # Doc-ids for the current KV block
        doc_kv = tl.load(doc_ids_ptr + offsets_KV_N, mask=mask_KV_N, other=-2)  # (BLOCK_SIZE_KV,)
        same_doc = (doc_q[:, None] == doc_kv[None, :])  # (BLOCK_SIZE_QO, BLOCK_SIZE_KV)

        if DIAGONAL:
            causal_mask = offsets_QO_N[:, None] >= offsets_KV_N[None, :]
            S += tl.where(causal_mask & same_doc, 0, -1.0e6)
        else:
            # Causal is trivially true below diagonal; apply doc mask only
            S += tl.where(same_doc, 0, -1.0e6)

        M_new = tl.maximum(M, tl.max(S, axis=1))
        S -= M_new[:, None]
        P = tl.exp2(S)
        L_new = tl.sum(P, axis=1)
        alpha = tl.exp2(M - M_new)
        L = L * alpha + L_new
        V = tl.load(V_ptr + V_offsets, mask=mask_KV_N[:, None], other=0.).to(tl.float32)
        O = O * alpha[:, None]
        O = tl.dot(P, V, acc=O)
        M = M_new
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV

    return O, L, M


# ---------------------------------------------------------------------------
# Forward outer kernel (doc_causal)
# ---------------------------------------------------------------------------

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_QO": BQ, "BLOCK_SIZE_KV": BK}, num_stages=ns, num_warps=nw)
        for BQ, BK in [(32, 32), (64, 32), (64, 64), (128, 32), (128, 64)]
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh"],
)
@triton.jit
def _attn_fwd_varlen(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, LSE_ptr,
    softmax_scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    doc_ids_ptr,
    B,
    H: tl.constexpr, N: tl.constexpr,
    Dh: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr, BLOCK_SIZE_KV: tl.constexpr,
):
    rln2: tl.constexpr = 1.4426950408889634
    softmax_scale *= rln2
    tl.static_assert(BLOCK_SIZE_KV <= Dh)

    block_index_QO = tl.program_id(0)
    index_BH = tl.program_id(1)
    index_B = index_BH // H
    index_H = index_BH % H

    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H

    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)

    Q_offsets = offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh
    K_T_offsets = offsets_Dh[:, None] * stride_K_Dh + offsets_KV_N[None, :] * stride_K_N
    V_offsets = offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh

    mask_QO_N = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask=mask_QO_N[:, None], other=0.).to(tl.float32)

    M = tl.full([BLOCK_SIZE_QO], value=-1e6, dtype=tl.float32)
    L = tl.full([BLOCK_SIZE_QO], value=1.0, dtype=tl.float32)
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype=tl.float32)

    O, L, M = _attn_fwd_inner_varlen(
        Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
        block_index_QO, softmax_scale, stride_K_N, stride_V_N,
        doc_ids_ptr, BLOCK_SIZE_QO, BLOCK_SIZE_KV, False,
        offsets_QO_N, offsets_KV_N, N, Dh,
    )
    O, L, M = _attn_fwd_inner_varlen(
        Q, O, L, M, K_ptr, V_ptr, K_T_offsets, V_offsets,
        block_index_QO, softmax_scale, stride_K_N, stride_V_N,
        doc_ids_ptr, BLOCK_SIZE_QO, BLOCK_SIZE_KV, True,
        offsets_QO_N, offsets_KV_N, N, Dh,
    )

    O = O / L[:, None]
    LSE = M + tl.math.log2(L)

    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    tl.store(LSE_ptr + LSE_offsets, LSE, mask=offsets_QO_N < N)
    O_offsets = offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh
    tl.store(O_ptr + O_offsets, O.to(O_ptr.dtype.element_ty), mask=mask_QO_N[:, None])


# ---------------------------------------------------------------------------
# Backward pre-process (identical to causal, reuse)
# ---------------------------------------------------------------------------

@triton.autotune(
    [
        triton.Config({"PRE_BLOCK_SIZE_ROW": r}, num_stages=ns, num_warps=nw)
        for r in [32, 64, 128]
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh"],
)
@triton.jit
def _attn_backward_preprocess_varlen(
    O_ptr, dLdO_ptr, Delta_ptr,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_dLdO_B, stride_dLdO_H, stride_dLdO_N, stride_dLdO_Dh,
    stride_Delta_B, stride_Delta_H, stride_Delta_N,
    N, Dh: tl.constexpr,
    PRE_BLOCK_SIZE_ROW: tl.constexpr,
):
    index_BH = tl.program_id(1)
    row = tl.program_id(0)
    row_offsets = row * PRE_BLOCK_SIZE_ROW + tl.arange(0, PRE_BLOCK_SIZE_ROW)
    col_offsets = tl.arange(0, Dh)
    mask = row_offsets < N

    O_ptr += index_BH * stride_O_H
    O_offsets = row_offsets[:, None] * stride_O_N + col_offsets[None, :] * stride_O_Dh
    O = tl.load(O_ptr + O_offsets, mask=mask[:, None], other=0.)

    dLdO_ptr += index_BH * stride_dLdO_H
    dLdO_offsets = row_offsets[:, None] * stride_dLdO_N + col_offsets[None, :] * stride_dLdO_Dh
    dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask[:, None], other=0.)

    Delta = tl.sum(dLdO.to(tl.float32) * O.to(tl.float32), axis=1)
    Delta_ptr += index_BH * stride_Delta_H
    tl.store(Delta_ptr + row_offsets, Delta, mask=mask)


# ---------------------------------------------------------------------------
# Backward KV sub-kernel with doc mask
# ---------------------------------------------------------------------------

@triton.jit
def _attn_backward_KV_varlen(
    K, V, dLdK, dLdV,
    Q_ptr, dLdO_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr,
    stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr,
):
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    Q_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_ROW[None, :] * stride_N
    dLdO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh

    # Doc-ids for the fixed KV (COL) block
    doc_col = tl.load(doc_ids_ptr + offsets_COL, mask=offsets_COL < N, other=-1)

    for block_idx in range(num_steps):
        mask_N = offsets_ROW < N
        Q_T = tl.load(Q_ptr + Q_T_offsets, mask=mask_N[None, :], other=0.).to(tl.float32)
        LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_N, other=0.)
        dLdO = tl.load(dLdO_ptr + dLdO_offsets, mask=mask_N[:, None], other=0.).to(tl.float32)
        Delta = tl.load(Delta_ptr + offsets_ROW, mask=mask_N, other=0.)

        S_T = tl.dot(K, Q_T)
        P_T = tl.exp2(S_T - LSE[None, :])

        # Doc-ids for current ROW (Q) block
        doc_row = tl.load(doc_ids_ptr + offsets_ROW, mask=mask_N, other=-2)
        same_doc = (doc_col[:, None] == doc_row[None, :])  # (BLOCK_SIZE_COL, BLOCK_SIZE_ROW)

        if MASK:
            causal = (offsets_COL[:, None] <= offsets_ROW[None, :])
            P_T = tl.where(causal & same_doc, P_T, 0.)
        else:
            P_T = tl.where(same_doc, P_T, 0.)

        dLdV = tl.dot(P_T, dLdO, acc=dLdV)
        dLdP_T = tl.dot(V, tl.trans(dLdO))
        dLdS_T = (P_T * (dLdP_T - Delta[None, :]) * ln2)
        dLdK = tl.dot(dLdS_T, tl.trans(Q_T), acc=dLdK)

        offsets_ROW += BLOCK_SIZE_ROW
        Q_ptr += BLOCK_SIZE_ROW * stride_N
        dLdO_ptr += BLOCK_SIZE_ROW * stride_N

    return dLdK, dLdV


# ---------------------------------------------------------------------------
# Backward Q sub-kernel with doc mask
# ---------------------------------------------------------------------------

@triton.jit
def _attn_backward_Q_varlen(
    dLdQ, Q, dLdO, LSE,
    K_ptr, V_ptr, Delta_ptr,
    doc_ids_ptr,
    stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
    start_ROW, start_COL, num_steps,
    scale, ln2: tl.constexpr, rln2: tl.constexpr,
    MASK: tl.constexpr,
):
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    offsets_COL = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    offsets_Dh = tl.arange(0, Dh)

    K_and_V_T_offsets = offsets_Dh[:, None] * stride_Dh + offsets_COL[None, :] * stride_N
    Delta = tl.load(Delta_ptr + offsets_ROW, mask=offsets_ROW < N, other=0.)

    # Doc-ids for the fixed ROW (Q) block
    doc_row = tl.load(doc_ids_ptr + offsets_ROW, mask=offsets_ROW < N, other=-1)

    for block_idx in range(num_steps):
        K_T = tl.load(K_ptr + K_and_V_T_offsets, mask=(offsets_COL < N)[None, :], other=0.).to(tl.float32)
        V_T = tl.load(V_ptr + K_and_V_T_offsets, mask=(offsets_COL < N)[None, :], other=0.).to(tl.float32)

        S = tl.dot(Q, K_T)
        P = tl.exp2(S - LSE)

        # Doc-ids for current COL (KV) block
        doc_col = tl.load(doc_ids_ptr + offsets_COL, mask=offsets_COL < N, other=-2)
        same_doc = (doc_row[:, None] == doc_col[None, :])  # (BLOCK_SIZE_ROW, BLOCK_SIZE_COL)

        if MASK:
            causal = (offsets_ROW[:, None] >= offsets_COL[None, :])
            P = tl.where(causal & same_doc, P, 0.)
        else:
            P = tl.where(same_doc, P, 0.)

        dLdP = tl.dot(dLdO, V_T)
        dLdS = (P * (dLdP - Delta[:, None]) * ln2)
        dLdQ += tl.dot(dLdS, tl.trans(K_T))

        offsets_COL += BLOCK_SIZE_COL
        K_ptr += BLOCK_SIZE_COL * stride_N
        V_ptr += BLOCK_SIZE_COL * stride_N

    return dLdQ


# ---------------------------------------------------------------------------
# Backward outer kernel (doc_causal)
# ---------------------------------------------------------------------------

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MACRO": M, "BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for M, m in [(32, 16), (64, 16), (64, 32), (128, 32), (128, 64)]
        for ns in [3, 4, 5]
        for nw in [4, 8]
        if M > m and M % m == 0
    ],
    key=["N", "Dh"],
)
@triton.jit
def _attn_backward_varlen(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdQ_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr,
    cu_seqlens_ptr,   # [n_docs+1] int32 — used to skip zero blocks in backward
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
    BLOCK_SIZE_MACRO: tl.constexpr,
):
    ln2: tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634

    idx_batch_head = tl.program_id(1)
    idx_batch = idx_batch_head // H
    idx_head = idx_batch_head % H
    batch_head_jump = idx_batch * stride_B + idx_head * stride_H
    Q_ptr += batch_head_jump
    K_ptr += batch_head_jump
    V_ptr += batch_head_jump
    dLdO_ptr += batch_head_jump
    dLdQ_ptr += batch_head_jump
    dLdK_ptr += batch_head_jump
    dLdV_ptr += batch_head_jump

    batch_head_jump = idx_batch_head * N
    LSE_ptr += batch_head_jump
    Delta_ptr += batch_head_jump

    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)

    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO

    pid = tl.program_id(0)
    start_COL = pid * BLOCK_SIZE_COL_1
    start_ROW = start_COL
    num_steps = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1

    offsets_COL_1 = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    offsets_Dh = tl.arange(0, Dh)
    KV_offsets = offsets_COL_1[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask = offsets_COL_1[:, None] < N
    K = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.).to(tl.float32)
    V = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.).to(tl.float32)
    K *= scale * rln2

    dLdK = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)

    dLdK, dLdV = _attn_backward_KV_varlen(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr, doc_ids_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2, MASK=True,
    )

    start_ROW += BLOCK_SIZE_COL_1
    # Sparsity: only iterate Q blocks within the same document as this K/V block.
    # K/V positions outside this doc have same_doc=False → P_T=0 → zero gradient.
    kv_doc_id = tl.load(doc_ids_ptr + start_COL).to(tl.int32)
    doc_kv_end = tl.load(cu_seqlens_ptr + kv_doc_id + 1).to(tl.int32)
    doc_kv_end_aligned = tl.cdiv(doc_kv_end, BLOCK_SIZE_ROW_1) * BLOCK_SIZE_ROW_1
    num_steps = tl.maximum((doc_kv_end_aligned - start_ROW) // BLOCK_SIZE_ROW_1, 0)

    dLdK, dLdV = _attn_backward_KV_varlen(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr, doc_ids_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2, MASK=False,
    )

    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK.to(dLdK_ptr.dtype.element_ty), mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV.to(dLdV_ptr.dtype.element_ty), mask=KV_mask)

    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO

    start_ROW = pid * BLOCK_SIZE_ROW_2
    start_COL = start_ROW
    num_steps = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2

    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    QO_offsets = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW = offsets_ROW < N
    Q = tl.load(Q_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.).to(tl.float32)
    Q *= scale * rln2
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.).to(tl.float32)
    LSE = tl.load(LSE_ptr + offsets_ROW, mask=mask_ROW, other=0.)[:, None]

    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, Dh], dtype=tl.float32)

    dLdQ = _attn_backward_Q_varlen(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr, doc_ids_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2, MASK=True,
    )

    end_COL = start_COL
    # Sparsity: only iterate K/V blocks within the same document as this Q block.
    q_doc_id = tl.load(doc_ids_ptr + start_ROW).to(tl.int32)
    doc_q_start = tl.load(cu_seqlens_ptr + q_doc_id).to(tl.int32)
    doc_q_start_aligned = (doc_q_start // BLOCK_SIZE_COL_2) * BLOCK_SIZE_COL_2
    start_COL = doc_q_start_aligned
    num_steps = (end_COL - start_COL) // BLOCK_SIZE_COL_2

    dLdQ = _attn_backward_Q_varlen(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr, doc_ids_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, start_COL, num_steps,
        scale, ln2, rln2, MASK=False,
    )

    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ.to(dLdQ_ptr.dtype.element_ty), mask=mask_ROW[:, None])


# ---------------------------------------------------------------------------
# Autograd function  (T, H, Dh) → (T, H, Dh), handles dtype dispatch
# ---------------------------------------------------------------------------

class _VarlenAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, document_ids, cu_seqlens, scale):
        # q/k/v: (T, H, Dh) — kept in native dtype, kernel casts internally
        T, H, Dh = q.shape
        q_f = q.permute(1, 0, 2).unsqueeze(0).contiguous()
        k_f = k.permute(1, 0, 2).unsqueeze(0).contiguous()
        v_f = v.permute(1, 0, 2).unsqueeze(0).contiguous()

        B_k = 1
        O = torch.empty_like(q_f)
        LSE = torch.empty(B_k, H, T, device=q.device, dtype=torch.float32)

        grid = lambda args: (triton.cdiv(T, args["BLOCK_SIZE_QO"]), B_k * H)
        _attn_fwd_varlen[grid](
            q_f, k_f, v_f, O, LSE, scale,
            q_f.stride(0), q_f.stride(1), q_f.stride(2), q_f.stride(3),
            k_f.stride(0), k_f.stride(1), k_f.stride(2), k_f.stride(3),
            v_f.stride(0), v_f.stride(1), v_f.stride(2), v_f.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            LSE.stride(0), LSE.stride(1), LSE.stride(2),
            document_ids,
            B_k, H, T, Dh,
        )

        ctx.save_for_backward(q_f, k_f, v_f, O, LSE)
        ctx.document_ids = document_ids
        ctx.cu_seqlens = cu_seqlens
        ctx.T, ctx.H, ctx.Dh = T, H, Dh
        ctx.scale = scale
        return O.squeeze(0).permute(1, 0, 2)

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        document_ids = ctx.document_ids
        cu_seqlens = ctx.cu_seqlens
        T, H, Dh = ctx.T, ctx.H, ctx.Dh
        scale = ctx.scale
        B_k = 1

        dLdO_f = dLdO.permute(1, 0, 2).unsqueeze(0).contiguous()
        assert q.stride() == k.stride() == v.stride() == O.stride() == dLdO_f.stride()

        dLdq = torch.empty_like(q)
        dLdk = torch.empty_like(k)
        dLdv = torch.empty_like(v)
        Delta = torch.empty_like(LSE)

        pre_grid = lambda meta: (triton.cdiv(T, meta["PRE_BLOCK_SIZE_ROW"]), B_k * H)
        _attn_backward_preprocess_varlen[pre_grid](
            O, dLdO_f, Delta,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            dLdO_f.stride(0), dLdO_f.stride(1), dLdO_f.stride(2), dLdO_f.stride(3),
            Delta.stride(0), Delta.stride(1), Delta.stride(2),
            T, Dh,
        )

        grid = lambda meta: (triton.cdiv(T, meta["BLOCK_SIZE_MACRO"]), B_k * H)
        _attn_backward_varlen[grid](
            q, k, v,
            dLdO_f, dLdq, dLdk, dLdv,
            LSE, Delta, document_ids, cu_seqlens,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, T, Dh,
        )

        to_thd = lambda t: t.squeeze(0).permute(1, 0, 2)
        return to_thd(dLdq), to_thd(dLdk), to_thd(dLdv), None, None, None


def triton_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Doc-causal flash attention for packed sequences.

    Args:
        q, k, v:     shape (T, H, Dh), any dtype (bf16/fp16/fp32).
        cu_seqlens:  int32 tensor [n_docs+1], cumulative document lengths.
        max_seqlen:  maximum document length (unused by kernel, kept for API parity with VSLF).
        scale:       softmax scale; defaults to 1/sqrt(Dh).

    Returns:
        Output tensor of same shape and dtype as q.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    T = q.shape[0]
    document_ids = _build_document_ids(cu_seqlens, T, q.device)
    return _VarlenAttention.apply(q, k, v, document_ids, cu_seqlens, scale)
