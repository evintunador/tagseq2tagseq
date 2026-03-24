"""
cross_doc_bitmask BIM v14 — expanded backward autotune search space.

Changes vs v12:
  Same kernels and BIM_BLOCK_SIZE=128 as v12.  The only change is the autotune
  search space for the two backward outer kernels (_attn_backward_KV,
  _attn_backward_Q):

  v12 / v10 autotune:
    num_warps  ∈ {4, 8}
    num_stages ∈ {3, 4, 5}
    → 4 × 3 × 2 = 24 configs per kernel

  v14 autotune:
    num_warps  ∈ {4, 8, 16}
    num_stages ∈ {2, 3, 4, 5, 6}
    → 4 × 5 × 3 = 60 configs per kernel

  With BIM_BS=128 and bf16 register reduction (v10), the register file is less
  stressed — 16 warps may achieve better memory-latency hiding without hurting
  occupancy.  num_stages=2 reduces software-pipeline buffering (saves shared
  memory, potentially higher occupancy); num_stages=6 hides more latency.

  The forward kernel is unchanged (inherits v10's autotune as-is).

All inner sub-kernels (_bwd_kv_full_v10, _bwd_kv_cdb_v10, _bwd_q_full_v10,
_bwd_q_cdb_v10) are imported and inlined from v10 — no code duplication.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from .cross_doc_bitmask_attn import _attn_backward_preprocess_cdb
from .cross_doc_bitmask_bim_v10 import (
    _attn_fwd_cdb_bim_v10,
    _bwd_kv_full_v10,
    _bwd_kv_cdb_v10,
    _bwd_q_full_v10,
    _bwd_q_cdb_v10,
)
from .cross_doc_bitmask_bim_v12 import _build_bim_128

if TYPE_CHECKING:
    from model.graph_traversal.cross_doc_mask import BlockInteractionMask


# ===========================================================================
# Backward — dK/dV outer kernel  (expanded autotune)
# ===========================================================================

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for m in [16, 32, 64, 128]
        for ns in [2, 3, 4, 5, 6]
        for nw in [4, 8, 16]
        if m <= 128
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_KV_cdb_bim_v14(
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
    """Compute dLdK and dLdV. Expanded autotune vs v10/v12."""
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
# Backward — dQ outer kernel  (expanded autotune)
# ===========================================================================

@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE_MICRO": m}, num_stages=ns, num_warps=nw)
        for m in [16, 32, 64, 128]
        for ns in [2, 3, 4, 5, 6]
        for nw in [4, 8, 16]
        if m <= 128
    ],
    key=["N", "Dh", "n_chunks", "BIM_BLOCK_SIZE"],
)
@triton.jit
def _attn_backward_Q_cdb_bim_v14(
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
    """Compute dLdQ. Expanded autotune vs v10/v12."""
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

class _CDBBIMv14(torch.autograd.Function):
    """v12 (BIM_BS=128) + expanded backward autotune."""

    @staticmethod
    def forward(ctx, q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
        T, H, Dh = q.shape
        n_chunks  = q_bitmasks.shape[0]
        bim_bs    = bim.block_size
        n_blocks  = bim.n_blocks
        assert bim.q_kv_n_full is not None, \
            "BIM missing q_kv_n_full — rebuild with updated CrossDocLinkMaskCreator"
        assert bim_bs == 128, f"v14 requires BIM_BLOCK_SIZE=128, got {bim_bs}"

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_bm_c  = q_bitmasks.contiguous()
        kv_bm_c = kv_bitmasks.contiguous()

        sT, sH, sDh = q.stride(0), q.stride(1), q.stride(2)

        O   = torch.empty_like(q)
        LSE = torch.empty(H, T, device=q.device, dtype=torch.float32)

        grid_fwd = (n_blocks, H)
        _attn_fwd_cdb_bim_v10[grid_fwd](
            q, k, v, O, LSE, scale,
            0, sH, sT, sDh,
            0, k.stride(1), k.stride(0), k.stride(2),
            0, v.stride(1), v.stride(0), v.stride(2),
            0, O.stride(1), O.stride(0), O.stride(2),
            H * T, T, 1,
            document_ids, q_bm_c, kv_bm_c, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            1, H, T, Dh, n_chunks, bim_bs,
        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.document_ids = document_ids
        ctx.bim          = bim
        ctx.q_bitmasks   = q_bm_c
        ctx.kv_bitmasks  = kv_bm_c
        ctx.T, ctx.H, ctx.Dh = T, H, Dh
        ctx.n_chunks = n_chunks
        ctx.scale    = scale
        ctx.strides  = (sT, sH, sDh)
        return O

    @staticmethod
    def backward(ctx, dLdO):
        q, k, v, O, LSE = ctx.saved_tensors
        document_ids     = ctx.document_ids
        bim              = ctx.bim
        q_bm, kv_bm      = ctx.q_bitmasks, ctx.kv_bitmasks
        T, H, Dh         = ctx.T, ctx.H, ctx.Dh
        n_chunks         = ctx.n_chunks
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

        _attn_backward_KV_cdb_bim_v14[grid](
            q, k, v, dLdO, dLdk, dLdv, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices,
            bim.kv_q_n_full,
            scale, *s,
            H, T, Dh, n_chunks, bim.block_size,
        )

        _attn_backward_Q_cdb_bim_v14[grid](
            q, k, v, dLdO, dLdq, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            scale, *s,
            H, T, Dh, n_chunks, bim.block_size,
        )

        return dLdq, dLdk, dLdv, None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v14(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask | None" = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v14: v12 kernels + expanded backward autotune (warps=16, stages=2/6)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if bim is None or bim.block_size != 128:
        T = q.shape[0]
        bim = _build_bim_128(
            T, document_ids, q_bitmasks, kv_bitmasks, q.device,
            q_bitmasks.shape[0],
        )
    return _CDBBIMv14.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
