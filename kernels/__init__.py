"""Custom Triton kernels for tagseq2tagseq.

Attention (block-interaction-mask, "BIM", flash attention):
  cross_doc_bitmask_bim_v18  cross_doc_link / doc_concat_link: doc-causal attention plus
                             cross-document grants encoded as [n_chunks, T] int64 bitmasks.
  varlen_bim_v2              doc_causal / doc_concatenated: doc-causal attention only.
  bim_attn_core              Triton sub-kernels shared by the two above.

Other:
  fused_relu_sq_mlp          fused ReLU^2 MLP.
  polar_express              Gram / quintic kernels for the Muon optimizer.

Modules are imported directly (e.g. ``from kernels.varlen_bim_v2 import ...``); this
package deliberately imports nothing eagerly. Design notes and the version history are
in KERNELS.md.
"""
