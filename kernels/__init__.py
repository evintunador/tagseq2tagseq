"""Custom Triton attention kernels for tagseq2tagseq.

Kernel hierarchy (each step inherits everything above and changes only the mask):
  causal_attn          — full causal (port of triton_docs_tutorials, + bf16)
  varlen_attn          — doc_causal (packed docs via document_ids tensor)
  cross_doc_naive_attn — cross_doc with dense [T,T] bool mask
  cross_doc_bitmask_attn — cross_doc with [n_chunks,T] int64 bitmasks
"""

from .causal_attn import triton_attn_causal
from .varlen_attn import triton_attn_varlen
from .cross_doc_naive_attn import triton_attn_cross_doc_naive
from .cross_doc_bitmask_attn import triton_attn_cross_doc_bitmask
from .cross_doc_bitmask_bim_v1 import triton_attn_cross_doc_bitmask_bim_v1

__all__ = [
    "triton_attn_causal",
    "triton_attn_varlen",
    "triton_attn_cross_doc_naive",
    "triton_attn_cross_doc_bitmask",         # cdb_v1 — baseline OR-reduction
    "triton_attn_cross_doc_bitmask_bim_v1",  # cdb_bim_v1 — BlockInteractionMask
]
