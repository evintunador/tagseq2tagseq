"""
cross_doc_bitmask BIM v12 — BIM_BLOCK_SIZE=128 with v11 kernels.

Changes vs v11:
  Rebuild the BlockInteractionMask at block_size=128 instead of 64.  The Triton
  kernels are identical to v11 (which reuse v10's @triton.jit functions); Triton
  specialises them on BIM_BLOCK_SIZE as a tl.constexpr so they compile to a
  separate PTX binary with 128-token macro-tiles.

Why this matters:
  Backward dKV: each CTA now holds K(128,Dh) + V(128,Dh) and iterates Q micro-
    blocks of BLOCK_SIZE_MICRO tokens.  With MICRO=32 that's 128/32=4 pipelined
    steps — identical pipeline depth to v11's MICRO=16 on BS=64, but each matmul
    is K(128,Dh)@Q_T(Dh,32) = 4× the arithmetic intensity.
  Forward: BLOCK_SIZE_QO=128 tiles give Q(128,Dh)@K_T(Dh,BK) matmuls, 2× larger
    than v11's 64-token Q tile.

  With v10's bf16 inputs the K/V register footprint is halved vs fp32:
    K(128,64) bf16 = 16KB  (was 32KB fp32)
    V(128,64) bf16 = 16KB
    dLdK(128,64) fp32 = 32KB  (accumulator, unchanged)
    dLdV(128,64) fp32 = 32KB
    Total  96KB vs v4's 128KB — still fits ~2-3 CTAs/SM.

Memory: same as v11 (no copies, 69/138 MB for fwd/bwd).
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


def _build_bim_128(
    seq_len: int,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    device: torch.device,
    n_chunks: int,
):
    """Build a BlockInteractionMask at block_size=128."""
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
    creator = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
    creator.triton_block_size = 128
    creator._n_chunks = n_chunks
    return CrossDocLinkMaskCreator._build_block_interaction_mask(
        creator, seq_len, document_ids, list(q_bitmasks), list(kv_bitmasks), device,
    )


class _CDBBIMv12(torch.autograd.Function):
    """v11 kernels + BIM_BLOCK_SIZE=128."""

    @staticmethod
    def forward(ctx, q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
        T, H, Dh = q.shape
        n_chunks  = q_bitmasks.shape[0]
        bim_bs    = bim.block_size          # 128
        n_blocks  = bim.n_blocks
        assert bim.q_kv_n_full is not None, \
            "BIM missing q_kv_n_full — rebuild with updated CrossDocLinkMaskCreator"
        assert bim_bs == 128, f"v12 requires BIM_BLOCK_SIZE=128, got {bim_bs}"

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

        _attn_backward_KV_cdb_bim_v10[grid](
            q, k, v, dLdO, dLdk, dLdv, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.kv_q_counts, bim.kv_q_ptrs, bim.kv_q_indices,
            bim.kv_q_n_full,
            scale, *s,
            H, T, Dh, n_chunks, bim.block_size,
        )

        _attn_backward_Q_cdb_bim_v10[grid](
            q, k, v, dLdO, dLdq, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim.q_kv_counts, bim.q_kv_ptrs, bim.q_kv_indices,
            bim.q_kv_n_full,
            scale, *s,
            H, T, Dh, n_chunks, bim.block_size,
        )

        return dLdq, dLdk, dLdv, None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v12(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim: "BlockInteractionMask | None" = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v12: v11 kernels (native dtype, no copies) + BIM_BLOCK_SIZE=128."""
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # Build bim128 lazily if not provided
    if bim is None or bim.block_size != 128:
        T = q.shape[0]
        bim = _build_bim_128(
            T, document_ids, q_bitmasks, kv_bitmasks, q.device,
            q_bitmasks.shape[0],
        )
    return _CDBBIMv12.apply(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale)
