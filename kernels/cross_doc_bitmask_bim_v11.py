"""
cross_doc_bitmask BIM v11 — eliminate permute/contiguous copies.

Changes vs v10:
  v10 (and all prior versions) convert (T, H, Dh) inputs to (1, H, T, Dh) with
  `.permute(1,0,2).unsqueeze(0).contiguous()` before calling the kernel, and
  permute back at the end.  At T=32k, H=16, Dh=64 this is three 64MB copies on
  the way in and one on the way out — roughly 200+ MB of unnecessary HBM traffic.

  Fix: pass the THD strides (stride_T=H*Dh, stride_H=Dh, stride_Dh=1) directly
  to the kernel as stride_Q_N / stride_Q_H / stride_Q_Dh, letting the kernel
  walk memory in THD layout without any format conversion.

  The forward and backward Triton kernels already accept full stride arguments;
  only the autograd function (Python) changes.  The kernel code is identical to
  v10 — we reuse all its @triton.jit functions unchanged.

Memory savings: ~3-4 × (T × H × Dh × dtype_bytes) per fwd+bwd call.
  T=32768, H=16, Dh=64, bf16 → 3 × 64MB = 192MB saved in fwd alone.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import triton

from .cross_doc_bitmask_attn import _attn_backward_preprocess_cdb
from .cross_doc_bitmask_bim_v10 import (
    _attn_fwd_cdb_bim_v10,
    _attn_backward_KV_cdb_bim_v10,
    _attn_backward_Q_cdb_bim_v10,
)

if TYPE_CHECKING:
    from model.graph_traversal.cross_doc_mask import BlockInteractionMask


class _CDBBIMv11(torch.autograd.Function):
    """v11: same kernels as v10, but no permute/contiguous copies."""

    @staticmethod
    def forward(ctx, q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
        T, H, Dh = q.shape
        n_chunks  = q_bitmasks.shape[0]
        bim_bs    = bim.block_size
        n_blocks  = bim.n_blocks
        assert bim.q_kv_n_full is not None, \
            "BIM missing q_kv_n_full — rebuild with updated CrossDocLinkMaskCreator"

        # Ensure contiguous THD layout (the common case — no copy if already contiguous)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_bm_c  = q_bitmasks.contiguous()
        kv_bm_c = kv_bitmasks.contiguous()

        # THD strides: (H*Dh, Dh, 1) — no permute needed
        sT, sH, sDh = q.stride(0), q.stride(1), q.stride(2)  # H*Dh, Dh, 1

        O   = torch.empty_like(q)           # (T, H, Dh) — same layout as input
        LSE = torch.empty(H, T, device=q.device, dtype=torch.float32)  # (H, T)

        # Pass strides as (stride_B=0, stride_H=sH, stride_N=sT, stride_Dh=sDh).
        # We use B=1 dummy batch: stride_B doesn't matter (pid always picks index_B=0).
        grid_fwd = (n_blocks, H)
        _attn_fwd_cdb_bim_v10[grid_fwd](
            q, k, v, O, LSE, scale,
            # Q strides
            0, sH, sT, sDh,
            # K strides (same layout)
            0, k.stride(1), k.stride(0), k.stride(2),
            # V strides
            0, v.stride(1), v.stride(0), v.stride(2),
            # O strides
            0, O.stride(1), O.stride(0), O.stride(2),
            # LSE strides: (B=1, H, T) → stride_B=H*T, stride_H=T, stride_N=1
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
        Delta = torch.empty_like(LSE)    # (H, T)

        # Preprocess: compute Delta = rowsum(O * dLdO)
        # _attn_backward_preprocess_cdb expects (B, H, T, Dh) layout tensors.
        # We pass our (T, H, Dh) tensors with a fake B=1 dimension as a 4-stride set.
        pre_grid = lambda meta: (triton.cdiv(T, meta["PRE_BLOCK_SIZE_ROW"]), H)
        _attn_backward_preprocess_cdb[pre_grid](
            O, dLdO, Delta,
            # O strides: (B, H, N, Dh) — fake B stride = 0
            0, sH, sT, sDh,
            # dLdO strides
            0, dLdO.stride(1), dLdO.stride(0), dLdO.stride(2),
            # Delta strides: (B=1, H, T) → 0, T, 1
            H * T, T, 1,
            T, Dh,
        )

        # Shared kernel strides for q/k/v/grads: all in THD layout
        s = (0, sH, sT, sDh)   # (stride_B, stride_H, stride_N, stride_Dh)

        grid = (bim.n_blocks, H)

        _attn_backward_KV_cdb_bim_v10[grid](
            q, k, v, dLdO, dLdk, dLdv, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices,
            bim.kv_q_n_full,
            scale,
            *s,
            H, T, Dh, n_chunks, bim.block_size,
        )

        _attn_backward_Q_cdb_bim_v10[grid](
            q, k, v, dLdO, dLdq, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            scale,
            *s,
            H, T, Dh, n_chunks, bim.block_size,
        )

        return dLdq, dLdk, dLdv, None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v11(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask",
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v11: v10 kernels + no permute/contiguous copies."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    return _CDBBIMv11.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
