"""
cross_doc_bitmask BIM v8 — varlen-style same-doc + BIM cross-doc.

Root cause of v4's backward gap vs varlen: v4's "full-block loop" iterates the
BIM CSR one entry at a time.  Each entry launches a sub-kernel call with
num_steps=1 — these sequential calls CANNOT be software-pipelined by Triton.
Varlen's backward uses a single sub-kernel call for all same-doc Q blocks
with dynamic num_steps, enabling num_stages=3-5 prefetch pipelining.

Fix:
  Same-doc off-diagonal:   ONE contiguous call like varlen — uses cu_seqlens
                            to compute [start, doc_end) range, fully pipelined.
  Cross-doc:               BIM CSR loop (few entries, not pipelined, negligible).
  Diagonal:                One call with MASK=True (same as v4).

Both dKV and dQ stages benefit.  Combined single kernel (like varlen) to allow
compiler register reuse between stages.  Autotune BLOCK_SIZE_MACRO and
BLOCK_SIZE_MICRO freely.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .cross_doc_bitmask_attn import _attn_backward_preprocess_cdb
from .cross_doc_bitmask_bim_v3 import (
    _attn_fwd_cdb_bim_v3,
    _attn_backward_KV_cross_v3,
    _attn_backward_Q_cross_v3,
)
from .varlen_attn import _attn_backward_KV_varlen, _attn_backward_Q_varlen

if TYPE_CHECKING:
    from model.graph_traversal.cross_doc_mask import BlockInteractionMask


# ---------------------------------------------------------------------------
# Backward kernel — combined dKV + dQ, varlen-style same-doc pipelining
# ---------------------------------------------------------------------------

@triton.autotune(
    [
        # BLOCK_SIZE_MACRO is fixed = BIM_BLOCK_SIZE (grid is one CTA per BIM block).
        # Only BLOCK_SIZE_MICRO is autotuned — it controls the pipelineable inner step.
        triton.Config({"BLOCK_SIZE_MACRO": 64, "BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for m in [16, 32, 64]
        for ns in [3, 4, 5]
        for nw in [4, 8]
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_cdb_bim_v8(
    Q_ptr, K_ptr, V_ptr,
    dLdO_ptr, dLdQ_ptr, dLdK_ptr, dLdV_ptr,
    LSE_ptr, Delta_ptr,
    doc_ids_ptr, cu_seqlens_ptr,
    q_bitmasks_ptr, kv_bitmasks_ptr, T,
    kv_q_counts_ptr, kv_q_ptrs_ptr, kv_q_indices_ptr,
    kv_q_n_full_ptr,
    q_kv_counts_ptr, q_kv_ptrs_ptr, q_kv_indices_ptr,
    q_kv_n_full_ptr,
    scale,
    stride_B, stride_H, stride_N, stride_Dh,
    H, N, Dh: tl.constexpr,
    n_chunks: tl.constexpr,
    BIM_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_MACRO: tl.constexpr,
    BLOCK_SIZE_MICRO: tl.constexpr,
):
    ln2:  tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634
    tl.static_assert(BLOCK_SIZE_MACRO % BLOCK_SIZE_MICRO == 0)
    tl.static_assert(BIM_BLOCK_SIZE % BLOCK_SIZE_MICRO == 0)
    # Grid is (n_BIM_blocks, B*H); each CTA owns one BIM block.
    tl.static_assert(BLOCK_SIZE_MACRO == BIM_BLOCK_SIZE)

    idx_bh = tl.program_id(1)
    idx_b  = idx_bh // H
    idx_h  = idx_bh % H
    bh = idx_b * stride_B + idx_h * stride_H
    Q_ptr += bh; K_ptr += bh; V_ptr += bh
    dLdO_ptr += bh; dLdQ_ptr += bh; dLdK_ptr += bh; dLdV_ptr += bh
    LSE_ptr   += idx_bh * N
    Delta_ptr += idx_bh * N

    offsets_Dh = tl.arange(0, Dh)
    pid = tl.program_id(0)

    # ── STAGE 1: dLdK, dLdV for KV BIM-block pid ──────────────────────────
    BLOCK_SIZE_ROW_1: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL_1: tl.constexpr = BLOCK_SIZE_MACRO
    num_micro_1: tl.constexpr = BLOCK_SIZE_COL_1 // BLOCK_SIZE_ROW_1

    start_COL    = pid * BLOCK_SIZE_COL_1
    offsets_COL  = start_COL + tl.arange(0, BLOCK_SIZE_COL_1)
    KV_offsets   = offsets_COL[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    KV_mask      = offsets_COL[:, None] < N
    K  = tl.load(K_ptr + KV_offsets, mask=KV_mask, other=0.).to(tl.float32)
    V  = tl.load(V_ptr + KV_offsets, mask=KV_mask, other=0.).to(tl.float32)
    K *= scale * rln2

    dLdK = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)
    dLdV = tl.zeros([BLOCK_SIZE_COL_1, Dh], dtype=tl.float32)

    # Doc boundaries for this KV block
    kv_doc_id    = tl.load(doc_ids_ptr + start_COL).to(tl.int32)
    doc_kv_end   = tl.load(cu_seqlens_ptr + kv_doc_id + 1).to(tl.int32)

    kv_q_start   = tl.load(kv_q_ptrs_ptr   + pid)
    num_q_macros = tl.load(kv_q_counts_ptr  + pid)
    n_full_kv    = tl.load(kv_q_n_full_ptr  + pid)

    # Diagonal block — first entry in kv_q (MASK=True)
    q_b_diag = tl.load(kv_q_indices_ptr + kv_q_start)
    dLdK, dLdV = _attn_backward_KV_varlen(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr, doc_ids_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        q_b_diag * BIM_BLOCK_SIZE, start_COL, num_micro_1,
        scale, ln2, rln2, MASK=True,
    )

    # Same-doc off-diagonal Q blocks: ONE pipelined contiguous call.
    # Range: [start_COL + BLOCK_SIZE_COL_1, doc_kv_end) in steps of BLOCK_SIZE_ROW_1.
    start_same_ROW    = start_COL + BLOCK_SIZE_COL_1
    doc_kv_end_aligned = tl.cdiv(doc_kv_end, BLOCK_SIZE_ROW_1) * BLOCK_SIZE_ROW_1
    num_same_1 = tl.maximum((doc_kv_end_aligned - start_same_ROW) // BLOCK_SIZE_ROW_1, 0)
    dLdK, dLdV = _attn_backward_KV_varlen(
        K, V, dLdK, dLdV,
        Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr, doc_ids_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
        start_same_ROW, start_COL, num_same_1,
        scale, ln2, rln2, MASK=False,
    )

    # Cross-doc Q blocks: BIM CSR (few entries, correctly identified)
    for i in range(1 + n_full_kv, num_q_macros):
        q_b = tl.load(kv_q_indices_ptr + kv_q_start + i)
        dLdK, dLdV = _attn_backward_KV_cross_v3(
            K, V, dLdK, dLdV,
            Q_ptr, dLdO_ptr, LSE_ptr, Delta_ptr,
            q_bitmasks_ptr, kv_bitmasks_ptr, T, n_chunks,
            stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW_1, BLOCK_SIZE_COL_1,
            q_b * BIM_BLOCK_SIZE, start_COL, num_micro_1,
            ln2,
        )

    dLdK *= scale * rln2
    tl.store(dLdK_ptr + KV_offsets, dLdK.to(dLdK_ptr.dtype.element_ty), mask=KV_mask)
    tl.store(dLdV_ptr + KV_offsets, dLdV.to(dLdV_ptr.dtype.element_ty), mask=KV_mask)

    # ── STAGE 2: dLdQ for Q BIM-block pid ──────────────────────────────────
    BLOCK_SIZE_ROW_2: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL_2: tl.constexpr = BLOCK_SIZE_MICRO
    num_micro_2: tl.constexpr = BLOCK_SIZE_ROW_2 // BLOCK_SIZE_COL_2

    start_ROW    = pid * BLOCK_SIZE_ROW_2
    offsets_ROW  = start_ROW + tl.arange(0, BLOCK_SIZE_ROW_2)
    QO_offsets   = offsets_ROW[:, None] * stride_N + offsets_Dh[None, :] * stride_Dh
    mask_ROW     = offsets_ROW < N
    Q    = tl.load(Q_ptr    + QO_offsets, mask=mask_ROW[:, None], other=0.).to(tl.float32)
    Q   *= scale * rln2
    dLdO = tl.load(dLdO_ptr + QO_offsets, mask=mask_ROW[:, None], other=0.).to(tl.float32)
    LSE  = tl.load(LSE_ptr  + offsets_ROW, mask=mask_ROW, other=0.)[:, None]
    dLdQ = tl.zeros([BLOCK_SIZE_ROW_2, Dh], dtype=tl.float32)

    q_doc_id    = tl.load(doc_ids_ptr + start_ROW).to(tl.int32)
    doc_q_start = tl.load(cu_seqlens_ptr + q_doc_id).to(tl.int32)

    q_kv_start    = tl.load(q_kv_ptrs_ptr   + pid)
    num_kv_macros = tl.load(q_kv_counts_ptr  + pid)
    n_full_q      = tl.load(q_kv_n_full_ptr  + pid)

    # Same-doc KV blocks before diagonal: ONE pipelined contiguous call.
    doc_q_start_aligned = (doc_q_start // BLOCK_SIZE_COL_2) * BLOCK_SIZE_COL_2
    num_same_2 = tl.maximum((start_ROW - doc_q_start_aligned) // BLOCK_SIZE_COL_2, 0)
    dLdQ = _attn_backward_Q_varlen(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr, doc_ids_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, doc_q_start_aligned, num_same_2,
        scale, ln2, rln2, MASK=False,
    )

    # Diagonal block — last entry in q_kv (MASK=True)
    kv_b_diag = tl.load(q_kv_indices_ptr + q_kv_start + num_kv_macros - 1)
    dLdQ = _attn_backward_Q_varlen(
        dLdQ, Q, dLdO, LSE,
        K_ptr, V_ptr, Delta_ptr, doc_ids_ptr,
        stride_N, stride_Dh, H, N, Dh,
        BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
        start_ROW, kv_b_diag * BIM_BLOCK_SIZE, num_micro_2,
        scale, ln2, rln2, MASK=True,
    )

    # Cross-doc KV blocks: BIM CSR
    for i in range(n_full_q, num_kv_macros - 1):
        kv_b = tl.load(q_kv_indices_ptr + q_kv_start + i)
        dLdQ = _attn_backward_Q_cross_v3(
            dLdQ, Q, dLdO, LSE,
            K_ptr, V_ptr, Delta_ptr,
            q_bitmasks_ptr, kv_bitmasks_ptr, T, n_chunks,
            stride_N, stride_Dh, N, Dh,
            BLOCK_SIZE_ROW_2, BLOCK_SIZE_COL_2,
            start_ROW, kv_b * BIM_BLOCK_SIZE, num_micro_2,
            ln2,
        )

    dLdQ *= scale * rln2
    tl.store(dLdQ_ptr + QO_offsets, dLdQ.to(dLdQ_ptr.dtype.element_ty), mask=mask_ROW[:, None])


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

def _build_cu_seqlens_v8(document_ids: torch.Tensor) -> torch.Tensor:
    n_docs = int(document_ids.max().item()) + 1
    cu = torch.zeros(n_docs + 1, dtype=torch.int32, device=document_ids.device)
    cu[1:] = torch.bincount(document_ids.long(), minlength=n_docs).to(torch.int32).cumsum(0)
    return cu


class _CDBBIMv8(torch.autograd.Function):

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

        cu_seqlens = _build_cu_seqlens_v8(document_ids)
        ctx.save_for_backward(q_f, k_f, v_f, O, LSE)
        ctx.document_ids = document_ids
        ctx.cu_seqlens   = cu_seqlens
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
        cu_seqlens   = ctx.cu_seqlens
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

        bim_bs   = bim.block_size
        n_blocks = bim.n_blocks

        grid = (n_blocks, B_k * H)
        _attn_backward_cdb_bim_v8[grid](
            q, k, v, dLdO_f, dLdq, dLdk, dLdv, LSE, Delta,
            document_ids, cu_seqlens, q_bm, kv_bm, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices, bim.kv_q_n_full,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices, bim.q_kv_n_full,
            scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            H, T, Dh, n_chunks, bim_bs,
        )

        to_thd = lambda t: t.squeeze(0).permute(1, 0, 2)
        return to_thd(dLdq), to_thd(dLdk), to_thd(dLdv), None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask",
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v8: varlen-style pipelined same-doc + BIM cross-doc."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _CDBBIMv8.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
