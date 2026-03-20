"""
cross_doc_bitmask BIM v6 — double-tile backward via BIM pair merge.

Each backward CTA covers TWO consecutive BIM blocks (b0=2*pid, b1=2*pid+1),
loading a 128-token KV tile and iterating the deduplicated union of
kv_q[b0] and kv_q[b1].  Each Q BIM block (64 tokens) is processed against
the full 128-token KV tile in one sub-kernel call:
    S_T = K(128, Dh) @ Q_T(Dh, 64) → (128, 64)  [vs v4: (64, 16)]
~8× larger matmul → much better MMA utilisation.

Masking: always MASK=True (full same_doc|in_grant|causal).  For off-diagonal
Q blocks (q_b > b1), causal is trivially True for all 128 KV positions, so
MASK=True ≡ MASK=False — no correctness or performance penalty.  For boundary
cases (q_b = b0 or b1), MASK=True correctly zero-masks non-causal pairs.

Grid: ceil(n_blocks/2) × B×H  (half as many CTAs as v4).
Forward: unchanged — reuses v3 BIM-guided kernel.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .cross_doc_bitmask_attn import (
    _attn_backward_preprocess_cdb,
    _attn_backward_KV_cdb,
    _attn_backward_Q_cdb,
)
from .cross_doc_bitmask_bim_v3 import _attn_fwd_cdb_bim_v3

if TYPE_CHECKING:
    from model.graph_traversal.cross_doc_mask import BlockInteractionMask


# ---------------------------------------------------------------------------
# dK/dV backward kernel — 2 BIM blocks per CTA (128-token KV tile)
# ---------------------------------------------------------------------------

@triton.autotune(
    [
        triton.Config({"num_stages": ns, "num_warps": nw}, num_stages=ns, num_warps=nw)
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_KV_v6(
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
):
    ln2:  tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    BLOCK_SIZE_COL: tl.constexpr = BIM_BLOCK_SIZE * 2
    BLOCK_SIZE_ROW: tl.constexpr = BIM_BLOCK_SIZE
    num_micro: tl.constexpr = 1

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
    b0 = pid * 2
    b1 = pid * 2 + 1
    n_blocks = (N + BIM_BLOCK_SIZE - 1) // BIM_BLOCK_SIZE
    b1_valid = b1 < n_blocks

    start_COL    = b0 * BIM_BLOCK_SIZE
    offsets_COL  = start_COL + tl.arange(0, BLOCK_SIZE_COL)
    KV_offsets   = offsets_COL[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask      = offsets_COL[:, None] < N
    K  = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.).to(tl.float32)
    V  = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.).to(tl.float32)
    K *= scale * rln2

    dLdK = tl.zeros([BLOCK_SIZE_COL, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL, Dh], dtype=tl.float32)

    kv_q_start_b0 = tl.load(kv_q_ptrs_ptr  + b0)
    num_q_b0      = tl.load(kv_q_counts_ptr + b0)
    kv_q_start_b1 = tl.load(kv_q_ptrs_ptr  + b1, mask=b1_valid, other=0)
    num_q_b1      = tl.load(kv_q_counts_ptr + b1, mask=b1_valid, other=0)

    # ── b0's kv_q list ────────────────────────────────────────────────────
    # Always MASK=True: for q_b > b1, causal is trivially True; for q_b<=b1
    # the causal check correctly masks non-causal (KV > Q) pairs.
    for i in range(num_q_b0):
        q_b = tl.load(kv_q_indices_ptr + kv_q_start_b0 + i)
        dLdK, dLdV = _attn_backward_KV_cdb(
            K, V, dLdK, dLdV,
            Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
            doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
            0, n_chunks,
            stride_N, stride_Dh, H, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            q_b * BIM_BLOCK_SIZE, start_COL, num_micro,
            scale, ln2, rln2, MASK=True, USE_BIM=True,
        )

    # ── b1's kv_q list (skip entries already in b0's list) ───────────────
    for j in range(num_q_b1):
        q_b_1 = tl.load(kv_q_indices_ptr + kv_q_start_b1 + j, mask=b1_valid, other=-1)
        is_dup = False
        for i in range(num_q_b0):
            if tl.load(kv_q_indices_ptr + kv_q_start_b0 + i) == q_b_1:
                is_dup = True
        if not is_dup:
            dLdK, dLdV = _attn_backward_KV_cdb(
                K, V, dLdK, dLdV,
                Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
                doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
                0, n_chunks,
                stride_N, stride_Dh, H, N, Dh,
                BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
                q_b_1 * BIM_BLOCK_SIZE, start_COL, num_micro,
                scale, ln2, rln2, MASK=True, USE_BIM=True,
            )

    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK.to(dLdK_ptr.dtype.element_ty), mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV.to(dLdV_ptr.dtype.element_ty), mask=KV_mask)


# ---------------------------------------------------------------------------
# dQ backward kernel — 2 BIM blocks per CTA (128-token Q tile)
# ---------------------------------------------------------------------------

@triton.autotune(
    [
        triton.Config({"num_stages": ns, "num_warps": nw}, num_stages=ns, num_warps=nw)
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_Q_v6(
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
):
    ln2:  tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    BLOCK_SIZE_ROW: tl.constexpr = BIM_BLOCK_SIZE * 2
    BLOCK_SIZE_COL: tl.constexpr = BIM_BLOCK_SIZE
    num_micro: tl.constexpr = 1

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
    b0 = pid * 2
    b1 = pid * 2 + 1
    n_blocks = (N + BIM_BLOCK_SIZE - 1) // BIM_BLOCK_SIZE
    b1_valid = b1 < n_blocks

    start_ROW   = b0 * BIM_BLOCK_SIZE
    offsets_ROW = start_ROW + tl.arange(0, BLOCK_SIZE_ROW)
    QO_offsets  = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW    = offsets_ROW < N
    Q    = tl.load(Q_ptr    + QO_offsets, mask=mask_ROW[:, None], other=0.).to(tl.float32)
    Q   *= scale * rln2
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.).to(tl.float32)
    LSE  = tl.load(LSE_ptr  + offsets_ROW, mask=mask_ROW, other=0.)[:, None]
    dLdQ = tl.zeros([BLOCK_SIZE_ROW, Dh], dtype=tl.float32)

    q_kv_start_b0 = tl.load(q_kv_ptrs_ptr  + b0)
    num_kv_b0     = tl.load(q_kv_counts_ptr + b0)
    q_kv_start_b1 = tl.load(q_kv_ptrs_ptr  + b1, mask=b1_valid, other=0)
    num_kv_b1     = tl.load(q_kv_counts_ptr + b1, mask=b1_valid, other=0)

    # ── b0's q_kv list ────────────────────────────────────────────────────
    for i in range(num_kv_b0):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start_b0 + i)
        dLdQ = _attn_backward_Q_cdb(
            dLdQ, Q, dLdO, LSE,
            K_ptr, V_ptr, Delta_ptr,
            doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
            0, n_chunks,
            stride_N, stride_Dh, H, N, Dh,
            BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
            start_ROW, kv_b * BIM_BLOCK_SIZE, num_micro,
            scale, ln2, rln2, MASK=True, USE_BIM=True,
        )

    # ── b1's q_kv list (skip entries already in b0's list) ───────────────
    for j in range(num_kv_b1):
        kv_b_1 = tl.load(q_kv_indices_ptr + q_kv_start_b1 + j, mask=b1_valid, other=-1)
        is_dup = False
        for i in range(num_kv_b0):
            if tl.load(q_kv_indices_ptr + q_kv_start_b0 + i) == kv_b_1:
                is_dup = True
        if not is_dup:
            dLdQ = _attn_backward_Q_cdb(
                dLdQ, Q, dLdO, LSE,
                K_ptr, V_ptr, Delta_ptr,
                doc_ids_ptr, q_bitmasks_ptr, kv_bitmasks_ptr, T,
                0, n_chunks,
                stride_N, stride_Dh, H, N, Dh,
                BLOCK_SIZE_ROW, BLOCK_SIZE_COL,
                start_ROW, kv_b_1 * BIM_BLOCK_SIZE, num_micro,
                scale, ln2, rln2, MASK=True, USE_BIM=True,
            )

    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ.to(dLdQ_ptr.dtype.element_ty), mask=mask_ROW[:, None])


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class _CDBBIMv6(torch.autograd.Function):

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
        document_ids = ctx.document_ids
        bim          = ctx.bim
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

        bim_bs  = bim.block_size
        n_pairs = triton.cdiv(bim.n_blocks, 2)

        _attn_backward_KV_v6[(n_pairs, B_k * H)](
            q, k, v, dLdO_f, dLdk, dLdv, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices, bim.kv_q_n_full,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, T, Dh, n_chunks, bim_bs,
        )

        _attn_backward_Q_v6[(n_pairs, B_k * H)](
            q, k, v, dLdO_f, dLdq, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices, bim.q_kv_n_full,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, T, Dh, n_chunks, bim_bs,
        )

        to_thd = lambda t: t.squeeze(0).permute(1, 0, 2)
        return to_thd(dLdq), to_thd(dLdk), to_thd(dLdv), None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v6(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask",
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v6: double-tile backward (2 BIM blocks per CTA)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _CDBBIMv6.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
