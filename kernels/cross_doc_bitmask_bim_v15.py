"""
cross_doc_bitmask BIM v15 — persistent-CTA backward kernels.

Changes vs v12:
  The backward dKV and dQ outer kernels use atomic work-stealing to address
  load imbalance in the BIM CSR traversal.

Problem in v12 backward:
  At T=32k, the BIM grid is n_blocks=256 × H=16 = 4096 CTAs per backward kernel.
  On A100 with 108 SMs at ~2.7 CTAs/SM occupancy, that's ~14 CTA waves.
  The BIM CSR is heavily triangular: KV-block 0 has only 1 Q entry (diagonal),
  KV-block 255 has up to 256 Q entries.  Early-wave CTAs (handling late KV blocks)
  finish far later than early KV-block CTAs, leaving SMs idle in the last wave.

Solution:
  Launch ctas_per_head = ceil(n_sms * occupancy / H) persistent CTAs per head
  (fills ~1 SM wave).  Each CTA atomically increments a per-head int32 counter
  to claim the next KV block.  Fast CTAs immediately grab new blocks; slow CTAs
  finish one block per slot.  All SMs stay busy until the last block.

Persistent loop design — no `break`:
  Triton may compile `break` inside a runtime-bound `for` loop as an early
  function exit, causing stores to never fire (all gradients remain uninitialized
  / NaN).  Instead we loop exactly max_steps = ceil(n_blocks / ctas_per_head)
  times and guard the entire body with `if pid < n_blocks:`.  The extra atomic
  increments for over-claiming CTAs are cheap no-ops.

All inner sub-kernels are imported from v10.  Forward is unchanged (v10).
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
# Backward — dK/dV persistent outer kernel
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
def _attn_backward_KV_cdb_bim_v15(
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
    n_blocks,            # total KV blocks (runtime)   — before BLOCK_SIZE_MICRO
    max_steps,           # ceil(n_blocks / ctas_per_head) (runtime)
    work_counter_ptr,    # [B*H] int32, zeroed before launch
    BLOCK_SIZE_MICRO: tl.constexpr,  # autotuned — must be last constexpr
):
    """Compute dLdK and dLdV.  Persistent: each CTA claims multiple KV blocks."""
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

    BLOCK_SIZE_ROW: tl.constexpr = BLOCK_SIZE_MICRO
    BLOCK_SIZE_COL: tl.constexpr = BLOCK_SIZE_MACRO
    num_micro: tl.constexpr = BLOCK_SIZE_COL // BLOCK_SIZE_ROW

    # Persistent loop — guard body with `if pid < n_blocks` (no break)
    for _step in range(max_steps):
        pid = tl.atomic_add(work_counter_ptr + idx_batch_head, 1)
        if pid < n_blocks:
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
# Backward — dQ persistent outer kernel
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
def _attn_backward_Q_cdb_bim_v15(
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
    n_blocks,            # total Q blocks (runtime)    — before BLOCK_SIZE_MICRO
    max_steps,           # ceil(n_blocks / ctas_per_head) (runtime)
    work_counter_ptr,    # [B*H] int32, zeroed before launch
    BLOCK_SIZE_MICRO: tl.constexpr,  # autotuned — must be last constexpr
):
    """Compute dLdQ.  Persistent: each CTA claims multiple Q blocks."""
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

    BLOCK_SIZE_ROW: tl.constexpr = BLOCK_SIZE_MACRO
    BLOCK_SIZE_COL: tl.constexpr = BLOCK_SIZE_MICRO
    num_micro: tl.constexpr = BLOCK_SIZE_ROW // BLOCK_SIZE_COL

    # Persistent loop — guard body with `if pid < n_blocks` (no break)
    for _step in range(max_steps):
        pid = tl.atomic_add(work_counter_ptr + idx_batch_head, 1)
        if pid < n_blocks:
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

def _ctas_per_head(n_blocks: int, H: int, device: torch.device) -> int:
    """Number of persistent CTAs to launch per head.

    Targets filling ~1 SM wave: n_sms × 2 / H  (conservative occupancy=2).
    Clamped to [1, n_blocks].
    """
    n_sms = torch.cuda.get_device_properties(device).multi_processor_count
    target = max(1, n_sms * 2 // H)
    return max(1, min(n_blocks, target))


class _CDBBIMv15(torch.autograd.Function):
    """v12 (BIM_BS=128) + persistent-CTA backward (work-stealing dKV + dQ)."""

    @staticmethod
    def forward(ctx, q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
        T, H, Dh = q.shape
        n_chunks  = q_bitmasks.shape[0]
        bim_bs    = bim.block_size
        n_blocks  = bim.n_blocks
        assert bim.q_kv_n_full is not None, \
            "BIM missing q_kv_n_full — rebuild with updated CrossDocLinkMaskCreator"
        assert bim_bs == 128, f"v15 requires BIM_BLOCK_SIZE=128, got {bim_bs}"

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
        n_blocks         = bim.n_blocks
        device           = q.device

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
        ctas = _ctas_per_head(n_blocks, H, device)
        max_steps = (n_blocks + ctas - 1) // ctas   # ceiling division

        # Persistent dKV
        work_kv = torch.zeros(H, dtype=torch.int32, device=device)
        _attn_backward_KV_cdb_bim_v15[(ctas, H)](
            q, k, v, dLdO, dLdk, dLdv, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices,
            bim.kv_q_n_full,
            scale, *s,
            H, T, Dh, n_chunks, bim.block_size,
            n_blocks, max_steps, work_kv,
            # BLOCK_SIZE_MICRO supplied by autotuner as kwarg
        )

        # Persistent dQ
        work_q = torch.zeros(H, dtype=torch.int32, device=device)
        _attn_backward_Q_cdb_bim_v15[(ctas, H)](
            q, k, v, dLdO, dLdq, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            scale, *s,
            H, T, Dh, n_chunks, bim.block_size,
            n_blocks, max_steps, work_q,
            # BLOCK_SIZE_MICRO supplied by autotuner as kwarg
        )

        return dLdq, dLdk, dLdv, None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v15(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask | None" = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v15: v12 + persistent-CTA backward (work-stealing per head)."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if bim is None or bim.block_size != 128:
        T = q.shape[0]
        bim = _build_bim_128(
            T, document_ids, q_bitmasks, kv_bitmasks, q.device,
            q_bitmasks.shape[0],
        )
    return _CDBBIMv15.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
