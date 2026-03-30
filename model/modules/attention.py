"""
BIMv12Attention / BIMv17Attention — drop-in replacements for
FlexSelfAttention using custom Triton cross-doc kernels.

BIMv12Attention: v12 kernel throughout (BIM_BLOCK_SIZE=128 fwd+bwd).
                 Correct at Dh=64; OOMs in backward at Dh=128 on A100.
BIMv17Attention: v17 kernel — BIM_BLOCK_SIZE=128 forward,
                 BIM_BLOCK_SIZE=64 backward.  Fixes the A100 SMEM OOM at
                 Dh=128 and is ~2× faster in backward than BIMv12Attention
                 at Dh=128 (reduced register pressure → better occupancy).

Both accept a TritonMaskInputs bundle instead of a FlexAttention BlockMask.
Inherits QKV projection, RoPE, QK norm, and output projection from
FlexSelfAttention unchanged.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from tunalab.modules.sequence_mixing.flex_self_attention import FlexSelfAttention
from model.graph_traversal.cross_doc_mask import TritonMaskInputs, DocCausalTritonMaskInputs


# Disable dynamo tracing for the Triton kernel call — the autograd Function
# has non-tensor arguments (bim, scale) that cause graph-break issues.
# The surrounding code (QKV proj, RoPE, norm, output proj) is still compiled.
def _triton_attn_v12(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
    from kernels.cross_doc_bitmask_bim_v12 import triton_attn_cross_doc_bitmask_bim_v12
    return triton_attn_cross_doc_bitmask_bim_v12(
        q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale
    )

_triton_attn_v12_disabled = torch._dynamo.disable(_triton_attn_v12)


class BIMv12Attention(FlexSelfAttention):
    """Cross-doc attention using the v12 Triton kernel.

    Identical parameters and weight structure to FlexSelfAttention.
    Expects block_mask to be a TritonMaskInputs (from CrossDocLinkMaskCreator
    with backend="triton_v12").
    """

    def forward(self, x: Tensor, block_mask: TritonMaskInputs) -> Tensor:
        B, T = x.size(0), x.size(1)
        assert B == 1, "BIMv12Attention requires batch size = 1 (packed sequences)"

        q, k, v = (
            F.linear(x, self.Wqkv.flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = self.norm(q), self.norm(k)
        q, k = self.rotary(q), self.rotary(k)

        # q/k/v: (1, T, H, Dh) → (T, H, Dh) for the Triton kernel
        q_thd = q.squeeze(0)
        k_thd = k.squeeze(0)
        v_thd = v.squeeze(0)

        y = _triton_attn_v12_disabled(
            q_thd, k_thd, v_thd,
            block_mask.document_ids,
            block_mask.q_bitmasks,
            block_mask.kv_bitmasks,
            block_mask.bim,
            self.scale,
        )  # (T, H, Dh)

        y = y.unsqueeze(0).reshape(B, T, self.num_heads * self.head_dim)
        return self.Wout(y)


def _triton_attn_v17(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim128, bim64, scale):
    from kernels.cross_doc_bitmask_bim_v17 import triton_attn_cross_doc_bitmask_bim_v17
    return triton_attn_cross_doc_bitmask_bim_v17(
        q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim128, bim64, scale
    )

_triton_attn_v17_disabled = torch._dynamo.disable(_triton_attn_v17)


class BIMv17Attention(FlexSelfAttention):
    """Cross-doc attention using the v17 Triton kernel.

    Forward: BIM_BLOCK_SIZE=128 (identical to BIMv12Attention).
    Backward: BIM_BLOCK_SIZE=64 — fixes the A100 SMEM OOM at Dh=128 and is
    ~2× faster in backward at Dh=128 due to reduced register pressure.

    Expects block_mask to be a TritonMaskInputs with bim64 set (from
    CrossDocLinkMaskCreator with backend="triton_v17").
    """

    def forward(self, x: Tensor, block_mask: TritonMaskInputs) -> Tensor:
        B, T = x.size(0), x.size(1)
        assert B == 1, "BIMv17Attention requires batch size = 1 (packed sequences)"
        assert block_mask.bim64 is not None, \
            "BIMv17Attention requires TritonMaskInputs.bim64 (backend='triton_v17')"

        q, k, v = (
            F.linear(x, self.Wqkv.flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = self.norm(q), self.norm(k)
        q, k = self.rotary(q), self.rotary(k)

        q_thd = q.squeeze(0)
        k_thd = k.squeeze(0)
        v_thd = v.squeeze(0)

        y = _triton_attn_v17_disabled(
            q_thd, k_thd, v_thd,
            block_mask.document_ids,
            block_mask.q_bitmasks,
            block_mask.kv_bitmasks,
            block_mask.bim,
            block_mask.bim64,
            self.scale,
        )  # (T, H, Dh)

        y = y.unsqueeze(0).reshape(B, T, self.num_heads * self.head_dim)
        return self.Wout(y)


def _varlen_bim_v1_attn(q, k, v, doc_ids, scale):
    from kernels.varlen_bim_v1 import triton_attn_doc_causal_bim_v1_from_doc_ids
    return triton_attn_doc_causal_bim_v1_from_doc_ids(q, k, v, doc_ids, scale)

_varlen_bim_v1_attn_disabled = torch._dynamo.disable(_varlen_bim_v1_attn)


class VarlenBIMv1Attention(FlexSelfAttention):
    """Doc-causal attention using varlen_bim_v1 Triton kernel.

    Identical parameters and weight structure to FlexSelfAttention.
    Expects block_mask to be a DocCausalTritonMaskInputs (from the
    doc_causal triton mask creator, attention_backend="varlen_bim_v1").
    """

    def forward(self, x: Tensor, block_mask: DocCausalTritonMaskInputs) -> Tensor:
        B, T = x.size(0), x.size(1)
        assert B == 1, "VarlenBIMv1Attention requires batch size = 1 (packed sequences)"

        q, k, v = (
            F.linear(x, self.Wqkv.flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = self.norm(q), self.norm(k)
        q, k = self.rotary(q), self.rotary(k)

        # q/k/v: (1, T, H, Dh) → (T, H, Dh)
        y = _varlen_bim_v1_attn_disabled(
            q.squeeze(0), k.squeeze(0), v.squeeze(0),
            block_mask.document_ids,
            self.scale,
        )  # (T, H, Dh)

        y = y.unsqueeze(0).reshape(B, T, self.num_heads * self.head_dim)
        return self.Wout(y)
