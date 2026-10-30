"""
TS2TSAttention — single attention module that dispatches between flex_attention
and custom Triton kernels based on self.backend.

  backend='flex':    calls flex_attention directly with a FlexAttention BlockMask.
  backend='triton':  dispatches to the custom kernel matching the block_mask type:
                       TritonMaskInputs          → cross_doc_bitmask_bim_v18
                                                   (cross_doc_link, doc_concat_link)
                       DocCausalTritonMaskInputs → varlen_bim_v2
                                                   (doc_causal, doc_concatenated)

Changing self.backend after construction (e.g. layer.attn.backend = 'flex') switches
inference mode without any weight copying or __class__ reassignment.

We inherit __init__ from FlexSelfAttention only to reuse its weight initialisation
(Wqkv, Wout, QK norm, RoPE, scale). We never call FlexSelfAttention.forward — the
flex path calls flex_attention explicitly, giving us full control over the calling
convention and decoupling us from tunalab's internal API changes.

Value embeddings (ve):
    forward() accepts optional ``ve`` (T, D) and ``ve_gate_w`` (H, 12) arguments.
    When provided, a per-head gate is computed from the first 6 dims of the
    (already-normed) input and the first 6 dims of ve, and the gated ve is added
    to v before the attention kernel.  This is the "value embeddings" feature from
    modded-nanogpt — the ve contribution is mixed through the attention weights just
    like ordinary value vectors, with no kernel changes required.
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
# Value embedding injection helper
# ---------------------------------------------------------------------------

def _inject_ve(x_norm: Tensor, v: Tensor, ve: Tensor, ve_gate_w: Tensor,
               num_heads: int, head_dim: int) -> Tensor:
    """Add gated value embeddings to v before the attention kernel.

    Args:
        x_norm: Pre-norm attention input (B, T, D) — already normalised.
        v:       Value tensor (B, T, H, Dh).
        ve:      Value embedding lookup for this layer (T, D).
        ve_gate_w: Gate weight (H, 12).
        num_heads, head_dim: Attention head structure.

    Returns:
        v with ve contribution added (same shape).
    """
    B, T = x_norm.size(0), x_norm.size(1)
    gate_input = torch.cat([x_norm[..., :6], ve.unsqueeze(0)[..., :6]], dim=-1)  # (B, T, 12)
    gate = (2 * torch.sigmoid(F.linear(gate_input, ve_gate_w))).view(B, T, num_heads, 1)
    return v + gate * ve.view(1, T, num_heads, head_dim)


# ---------------------------------------------------------------------------
# Triton kernel wrappers (dynamo-disabled so non-tensor args don't graph-break)
# ---------------------------------------------------------------------------

def _triton_attn_v18(q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim128, bim64, scale):
    from kernels.cross_doc_bitmask_bim_v18 import triton_attn_cross_doc_bitmask_bim_v18
    return triton_attn_cross_doc_bitmask_bim_v18(
        q, k, v, document_ids, q_bitmasks, kv_bitmasks, bim128, bim64, scale
    )

_triton_attn_v18_disabled = torch._dynamo.disable(_triton_attn_v18)


def _varlen_bim_v2_attn(q, k, v, doc_ids, scale):
    from kernels.varlen_bim_v2 import triton_attn_doc_causal_bim_v2_from_doc_ids
    return triton_attn_doc_causal_bim_v2_from_doc_ids(q, k, v, doc_ids, scale)

_varlen_bim_v2_attn_disabled = torch._dynamo.disable(_varlen_bim_v2_attn)


# ---------------------------------------------------------------------------
# TS2TSAttention — the one class to rule them all
# ---------------------------------------------------------------------------

class TS2TSAttention(FlexSelfAttention):
    """TS2TS attention with runtime backend switching.

    Inherits weights/init from FlexSelfAttention (Wqkv, Wout, QK norm, RoPE,
    scale). Never calls FlexSelfAttention.forward.

    Args:
        backend: Attention backend. One of:
                   'triton' — custom Triton kernels (v18 for link masks,
                              varlen_bim_v2 for doc-causal masks). Default.
                   'flex'   — flex_attention with a FlexAttention BlockMask.
                 Can be changed after construction: ``layer.attn.backend = 'flex'``.

    Value embeddings:
        forward() accepts optional ``ve`` (T, D) and ``ve_gate_w`` (H, 12).
        When provided, a gated value embedding is added to v before the attention
        kernel (modded-nanogpt style).
    """

    def __init__(self, dim: int, num_heads: int, max_seq_len: int,
                 fp8_out_proj: bool = False, backend: str = 'triton'):
        super().__init__(dim=dim, num_heads=num_heads, max_seq_len=max_seq_len,
                         fp8_out_proj=fp8_out_proj)
        self.backend = backend

    def forward(self, x: Tensor, block_mask,
                ve: Tensor = None, ve_gate_w: Tensor = None) -> Tensor:
        B, T = x.size(0), x.size(1)
        assert B == 1, "TS2TSAttention requires batch size = 1 (packed sequences)"

        q, k, v = (
            F.linear(x, self.Wqkv.flatten(end_dim=1).type_as(x))
            .view(B, T, 3 * self.num_heads, self.head_dim)
            .chunk(3, dim=-2)
        )
        q, k = self.norm(q), self.norm(k)
        q, k = self.rotary(q), self.rotary(k)

        if ve is not None:
            v = _inject_ve(x, v, ve, ve_gate_w, self.num_heads, self.head_dim)

        if self.backend == 'flex':
            y = _flex_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                block_mask=block_mask,
                scale=self.scale,
            ).transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)

        elif self.backend == 'triton':
            if isinstance(block_mask, TritonMaskInputs):
                assert block_mask.bim64 is not None, "v18 requires TritonMaskInputs.bim64"
                y = _triton_attn_v18_disabled(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    block_mask.document_ids,
                    block_mask.q_bitmasks,
                    block_mask.kv_bitmasks,
                    block_mask.bim,
                    block_mask.bim64,
                    self.scale,
                )
            elif isinstance(block_mask, DocCausalTritonMaskInputs):
                y = _varlen_bim_v2_attn_disabled(
                    q.squeeze(0), k.squeeze(0), v.squeeze(0),
                    block_mask.document_ids,
                    self.scale,
                )
            else:
                raise TypeError(
                    f"TS2TSAttention(backend='triton') expects TritonMaskInputs or "
                    f"DocCausalTritonMaskInputs, got {type(block_mask).__name__}. "
                    f"Check that the mask creator uses backend='triton', not 'flex'."
                )
            y = y.unsqueeze(0).reshape(B, T, self.num_heads * self.head_dim)

        else:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Expected 'flex' or 'triton'."
            )

        return self.Wout(y)
