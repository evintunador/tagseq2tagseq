"""
TS2TSAttention — single attention module that dispatches between flex_attention
and custom Triton kernels based on self.backend.

  backend='flex':   calls flex_attention directly with a FlexAttention BlockMask.
  backend='triton': dispatches to the right custom kernel based on block_mask type:
                      TritonMaskInputs          → cdb_bim_v12 (cross_doc_link)
                      DocCausalTritonMaskInputs → varlen_bim_v1 (doc_causal)

Changing self.backend after construction (e.g. layer.attn.backend = 'flex') switches
inference mode without any weight copying or __class__ reassignment.

We inherit __init__ from FlexSelfAttention only to reuse its weight initialisation
(Wqkv, Wout, QK norm, RoPE, scale). We never call FlexSelfAttention.forward — the
flex path calls flex_attention explicitly, giving us full control over the calling
convention and decoupling us from tunalab's internal API changes.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention as _flex_attention_raw

from tunalab.modules.sequence_mixing.flex_self_attention import FlexSelfAttention
from model.graph_traversal.cross_doc_mask import TritonMaskInputs, DocCausalTritonMaskInputs

# Compile once at module load. dynamic=True handles all sequence lengths without
# recompilation — avoids per-length Triton JIT cost during eval (~20s/unique length).
_flex_attention = torch.compile(_flex_attention_raw, dynamic=True, mode="default")


# ---------------------------------------------------------------------------
# Triton kernel wrappers (dynamo-disabled so non-tensor args don't graph-break)
# ---------------------------------------------------------------------------

def _triton_attn_v12(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale):
    from kernels.cross_doc_bitmask_bim_v12 import triton_attn_cross_doc_bitmask_bim_v12
    return triton_attn_cross_doc_bitmask_bim_v12(
        q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim, scale
    )

_triton_attn_v12_disabled = torch._dynamo.disable(_triton_attn_v12)


def _varlen_bim_v1_attn(q, k, v, doc_ids, scale):
    from kernels.varlen_bim_v1 import triton_attn_doc_causal_bim_v1_from_doc_ids
    return triton_attn_doc_causal_bim_v1_from_doc_ids(q, k, v, doc_ids, scale)

_varlen_bim_v1_attn_disabled = torch._dynamo.disable(_varlen_bim_v1_attn)


# ---------------------------------------------------------------------------
# TS2TSAttention — the one class to rule them all
# ---------------------------------------------------------------------------

class TS2TSAttention(FlexSelfAttention):
    """TS2TS attention with runtime backend switching.

    Inherits weights/init from FlexSelfAttention (Wqkv, Wout, QK norm, RoPE,
    scale). Never calls FlexSelfAttention.forward.

    Args:
        backend: 'triton' (default, uses custom Triton kernels) or 'flex'
                 (calls flex_attention directly). Can be changed after
                 construction: ``layer.attn.backend = 'flex'``.
    """

    def __init__(self, dim: int, num_heads: int, max_seq_len: int,
                 fp8_out_proj: bool = False, backend: str = 'triton'):
        super().__init__(dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
                         fp8_out_proj=fp8_out_proj)
        self.backend = backend

    def forward(self, x: Tensor, block_mask) -> Tensor:
        B, T = x.size(0), x.size(1)
        assert B == 1, "TS2TSAttention requires batch size = 1 (packed sequences)"

        q, k, v = (
            F.linear(x, self.Wqkv.flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = self.norm(q), self.norm(k)
        q, k = self.rotary(q), self.rotary(k)

        if self.backend == 'flex':
            y = _flex_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                block_mask=block_mask,
                scale=self.scale,
            ).transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)

        elif self.backend == 'triton':
            if isinstance(block_mask, TritonMaskInputs):
                # cross_doc_link: v12 BIM kernel
                y = _triton_attn_v12_disabled(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    block_mask.document_ids,
                    block_mask.q_bitmasks,
                    block_mask.kv_bitmasks,
                    block_mask.bim,
                    self.scale,
                )
                y = y.unsqueeze(0).reshape(B, T, self.num_heads * self.head_dim)

            elif isinstance(block_mask, DocCausalTritonMaskInputs):
                # doc_causal: varlen BIM v1 kernel
                y = _varlen_bim_v1_attn_disabled(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    block_mask.document_ids,
                    self.scale,
                )
                y = y.unsqueeze(0).reshape(B, T, self.num_heads * self.head_dim)

            else:
                raise TypeError(
                    f"TS2TSAttention(backend='triton') expects TritonMaskInputs or "
                    f"DocCausalTritonMaskInputs, got {type(block_mask).__name__}. "
                    f"Check that your mask creator uses backend='triton_v12' / "
                    f"'varlen_bim_v1', not 'flex'."
                )
        else:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Expected 'flex' or 'triton'."
            )

        return self.Wout(y)
