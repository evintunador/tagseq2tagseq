"""
cross_doc_bitmask BIM v17 — BIM_BLOCK_SIZE=128 forward, BIM_BLOCK_SIZE=64 backward.

Motivation:
  v12 uses BIM_BLOCK_SIZE=128 everywhere.  The backward outer kernels
  (_attn_backward_KV_cdb_bim_v10, _attn_backward_Q_cdb_bim_v10) hold the
  following tensors live simultaneously per CTA:

    KV backward:  K[BIM_BS, Dh] bf16 + V[BIM_BS, Dh] bf16
                  + dLdK[BIM_BS, Dh] fp32 + dLdV[BIM_BS, Dh] fp32
    Q  backward:  Q[BIM_BS, Dh] bf16 + dLdO[BIM_BS, Dh] bf16
                  + dLdQ[BIM_BS, Dh] fp32

  At BIM_BS=128, Dh=64 the register/SMEM footprint is ~96KB — fine for A100
  (163KB limit).  At BIM_BS=128, Dh=128 it doubles to ~192KB, exceeding the
  limit and triggering:
    triton.runtime.errors.OutOfResources: shared memory Required: 173440,
    Hardware limit: 166912

  Fix: rebuild the BIM at block_size=64 for the backward passes only.  The
  forward still uses BIM_BS=128 (larger Q-tiles, better tensor-core utilisation
  at Dh=128).  At BIM_BS=64, Dh=128:
    K(64,128) bf16 = 16KB  V(64,128) bf16 = 16KB
    dLdK(64,128) fp32 = 32KB  dLdV(64,128) fp32 = 32KB  → 96KB ✓

  Backward tile size reverts to v11 equivalent; forward keeps v12 performance.
  At Dh=128 the reduced register pressure also improves SM occupancy, making
  the backward ~2× faster than v12 despite the smaller tile size.

  On H100 (228KB max SMEM) this fix is unnecessary, but also harmless.
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
    """Build a BlockInteractionMask at block_size=128 (forward tiles)."""
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
    creator = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
    creator.triton_block_size = 128
    creator._n_chunks = n_chunks
    return CrossDocLinkMaskCreator._build_block_interaction_mask(
        creator, seq_len, document_ids, list(q_bitmasks), list(kv_bitmasks), device,
    )


def _build_bim_64(
    seq_len: int,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    device: torch.device,
    n_chunks: int,
):
    """Build a BlockInteractionMask at block_size=64 (backward tiles).

    Halves the per-CTA tile footprint vs BIM_BS=128, keeping backward SMEM
    usage within A100's 163KB limit even at Dh=128.
    """
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
    creator = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
    creator.triton_block_size = 64
    creator._n_chunks = n_chunks
    return CrossDocLinkMaskCreator._build_block_interaction_mask(
        creator, seq_len, document_ids, list(q_bitmasks), list(kv_bitmasks), device,
    )


class _CDBBIMv17(torch.autograd.Function):
    """Forward: BIM_BLOCK_SIZE=128.  Backward: BIM_BLOCK_SIZE=64."""

    @staticmethod
    def forward(ctx, q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim128, bim64, scale):
        T, H, Dh = q.shape
        n_chunks  = q_bitmasks.shape[0]
        bim_bs    = bim128.block_size
        assert bim_bs == 128, f"v17 forward requires BIM_BLOCK_SIZE=128, got {bim_bs}"
        assert bim64.block_size == 64, \
            f"v17 backward requires BIM_BLOCK_SIZE=64, got {bim64.block_size}"
        assert bim128.q_kv_n_full is not None, \
            "BIM128 missing q_kv_n_full — rebuild with updated CrossDocLinkMaskCreator"
        assert bim64.q_kv_n_full is not None, \
            "BIM64 missing q_kv_n_full — rebuild with updated CrossDocLinkMaskCreator"

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_bm_c  = q_bitmasks.contiguous()
        kv_bm_c = kv_bitmasks.contiguous()

        sT, sH, sDh = q.stride(0), q.stride(1), q.stride(2)

        O   = torch.empty_like(q)
        LSE = torch.empty(H, T, device=q.device, dtype=torch.float32)

        grid_fwd = (bim128.n_blocks, H)
        _attn_fwd_cdb_bim_v10[grid_fwd](
            q, k, v, O, LSE, scale,
            0, sH, sT, sDh,
            0, k.stride(1), k.stride(0), k.stride(2),
            0, v.stride(1), v.stride(0), v.stride(2),
            0, O.stride(1), O.stride(0), O.stride(2),
            H * T, T, 1,
            document_ids, q_bm_c, kv_bm_c, T,
            bim128.q_kv_counts, bim128.q_kv_ptrs, bim128.q_kv_indices,
            bim128.q_kv_n_full,
            1, H, T, Dh, n_chunks, bim_bs,
        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.document_ids = document_ids
        ctx.bim64        = bim64
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
        bim64            = ctx.bim64
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
        grid = (bim64.n_blocks, H)

        _attn_backward_KV_cdb_bim_v10[grid](
            q, k, v, dLdO, dLdk, dLdv, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim64.kv_q_counts, bim64.kv_q_ptrs, bim64.kv_q_indices,
            bim64.kv_q_n_full,
            scale, *s,
            H, T, Dh, n_chunks, bim64.block_size,
        )

        _attn_backward_Q_cdb_bim_v10[grid](
            q, k, v, dLdO, dLdq, LSE, Delta,
            document_ids, q_bm, kv_bm, T,
            bim64.q_kv_counts, bim64.q_kv_ptrs, bim64.q_kv_indices,
            bim64.q_kv_n_full,
            scale, *s,
            H, T, Dh, n_chunks, bim64.block_size,
        )

        return dLdq, dLdk, dLdv, None, None, None, None, None, None


def triton_attn_cross_doc_bitmask_bim_v17(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    bim128: "BlockInteractionMask | None" = None,
    bim64: "BlockInteractionMask | None" = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Cross-doc BIM v17: BIM_BS=128 forward, BIM_BS=64 backward.

    Fixes the A100 shared-memory OOM in v12 backward at Dh=128 while keeping
    the larger-tile forward.  At Dh=128 the backward is also ~2× faster than
    v12 due to improved SM occupancy from lower register pressure.
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    T = q.shape[0]
    n_chunks = q_bitmasks.shape[0]
    if bim128 is None or bim128.block_size != 128:
        bim128 = _build_bim_128(
            T, document_ids, q_bitmasks, kv_bitmasks, q.device, n_chunks,
        )
    if bim64 is None or bim64.block_size != 64:
        bim64 = _build_bim_64(
            T, document_ids, q_bitmasks, kv_bitmasks, q.device, n_chunks,
        )
    return _CDBBIMv17.apply(
        q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim128, bim64, scale,
    )
