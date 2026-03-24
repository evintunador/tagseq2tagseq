"""Attention kernel harness — correctness tests + memory + runtime benchmarks.

Validates and benchmarks all four kernel variants against SDPA ground truth,
building complexity step by step:
  causal → varlen (doc_causal) → cross_doc_naive → cross_doc_bitmask

Usage:
    # Step 1 – causal kernel correctness
    python benchmarks/attention_harness.py correctness --mask-types full_causal

    # Step 2 – varlen kernel correctness
    python benchmarks/attention_harness.py correctness --mask-types doc_causal

    # Step 3 – full correctness suite
    python benchmarks/attention_harness.py correctness \\
        --dataset-dir data/pretokenized_datasets/simplewiki

    # Step 4 – production benchmark
    python benchmarks/attention_harness.py bench \\
        --seq-lens 32768 --num-heads 16 --head-dim 64 \\
        --dataset-dir data/pretokenized_datasets/simplewiki

    # Full run
    python benchmarks/attention_harness.py all \\
        --dataset-dir data/pretokenized_datasets/simplewiki
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Add project root to sys.path so we can import from kernels/
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compiled flex_attention (amortises triton JIT across calls)
_compiled_flex = torch.compile(flex_attention, dynamic=True)


# ---------------------------------------------------------------------------
# Mask types
# ---------------------------------------------------------------------------

class MaskType(str, Enum):
    FULL_CAUSAL = "full_causal"
    DOC_CAUSAL = "doc_causal"
    CROSS_DOC_NAIVE = "cross_doc_naive"
    CROSS_DOC_BITMASK = "cross_doc_bitmask"


# ---------------------------------------------------------------------------
# MaskInputs dataclass
# ---------------------------------------------------------------------------

@dataclass
class MaskInputs:
    seq_len: int
    doc_spans: List[SimpleNamespace]           # each has .start, .end, .doc_id
    cu_seqlens: torch.Tensor                   # [n_docs+1] int32
    max_seqlen: int
    document_ids: torch.Tensor                 # [T] int32 — doc index per position
    dense_mask: Optional[torch.Tensor]         # [T,T] bool — cross-doc links only (no same_doc/causal)
    q_bitmasks: Optional[torch.Tensor]         # [n_chunks,T] int64 — for CROSS_DOC_BITMASK
    kv_bitmasks: Optional[torch.Tensor]        # [n_chunks,T] int64
    flex_doc_causal_block_mask: Optional[Any]  # BlockMask for doc_causal flex
    flex_cross_doc_block_mask: Optional[Any]   # BlockMask for cross_doc flex (None if no links)
    bim: Optional[Any]                         # BlockInteractionMask (None if no links)


# ---------------------------------------------------------------------------
# Fixtures: synthetic
# ---------------------------------------------------------------------------

def _make_doc_spans(seq_len: int, doc_len: int) -> List[SimpleNamespace]:
    spans = []
    doc_id = 0
    for start in range(0, seq_len, doc_len):
        end = min(start + doc_len, seq_len)
        if end > start:
            spans.append(SimpleNamespace(doc_id=doc_id, start=start, end=end))
            doc_id += 1
    return spans


def _spans_to_cu_seqlens(spans: List[SimpleNamespace], device: torch.device) -> Tuple[torch.Tensor, int]:
    lengths = [s.end - s.start for s in spans]
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0).to(device)
    return cu, max(lengths)


def _spans_to_document_ids(spans: List[SimpleNamespace], seq_len: int, device: torch.device) -> torch.Tensor:
    doc_ids = torch.zeros(seq_len, dtype=torch.int32, device=device)
    for span in spans:
        doc_ids[span.start:span.end] = span.doc_id
    return doc_ids


def _build_cross_doc_masks(
    seq_len: int,
    doc_spans: List[SimpleNamespace],
    link_to_target: Dict[int, List[int]],
    device: torch.device,
    max_grants: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (dense_mask [T,T], q_bitmasks [n_chunks,T], kv_bitmasks [n_chunks,T])."""
    n_chunks = max(1, (max_grants + 63) // 64)

    # Dense mask
    dense = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    q_bm_list = [torch.zeros(seq_len, dtype=torch.int64, device=device) for _ in range(n_chunks)]
    kv_bm_list = [torch.zeros(seq_len, dtype=torch.int64, device=device) for _ in range(n_chunks)]

    grant_idx = 0
    for link_pos, target_doc_ids in sorted(link_to_target.items()):
        for target_doc_id in target_doc_ids:
            link_span = next((s for s in doc_spans if s.start < link_pos <= s.end), None)
            target_span = next((s for s in doc_spans if s.doc_id == target_doc_id), None)
            if link_span is None or target_span is None:
                continue

            gs = link_pos
            ge = min(seq_len, link_span.end)
            ts = max(0, target_span.start)
            te = min(seq_len, target_span.end)
            if gs >= ge or ts >= te:
                continue

            if grant_idx < max_grants:
                # dense_mask is kept consistent with bitmasks — both encode
                # exactly the same grants (truncated at max_grants).
                dense[gs:ge, ts:te] = True
                chunk = grant_idx // 64
                bit_pos = grant_idx % 64
                bit = (1 << bit_pos) if bit_pos < 63 else -(1 << 63)
                q_bm_list[chunk][gs:ge] |= bit
                kv_bm_list[chunk][ts:te] |= bit
                grant_idx += 1

    q_bitmasks = torch.stack(q_bm_list)
    kv_bitmasks = torch.stack(kv_bm_list)
    return dense, q_bitmasks, kv_bitmasks


def _build_flex_cross_doc_block_mask(
    seq_len: int,
    document_ids: torch.Tensor,
    q_bitmasks: torch.Tensor,
    kv_bitmasks: torch.Tensor,
    device: torch.device,
) -> Any:
    """Build FlexAttention BlockMask from bitmasks."""
    q_bms = list(q_bitmasks)   # list of (T,) int64
    kv_bms = list(kv_bitmasks)

    def cross_doc_link_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_doc = document_ids[q_idx] == document_ids[kv_idx]
        in_grant = (q_bms[0][q_idx] & kv_bms[0][kv_idx]) != 0
        for i in range(1, len(q_bms)):
            in_grant = in_grant | ((q_bms[i][q_idx] & kv_bms[i][kv_idx]) != 0)
        return causal & (same_doc | in_grant)

    return create_block_mask(
        cross_doc_link_mod, B=None, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )


def make_synthetic_batch(
    seq_len: int,
    doc_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    n_links: int = 0,
    max_grants: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, MaskInputs]:
    """Create synthetic uniform-length packed batch.

    Returns (q, k, v, MaskInputs).
    q/k/v shape: (T, H, Dh) — VSLF layout.
    """
    doc_spans = _make_doc_spans(seq_len, doc_len)
    cu_seqlens, max_seqlen = _spans_to_cu_seqlens(doc_spans, device)
    document_ids = _spans_to_document_ids(doc_spans, seq_len, device)

    q = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(seq_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)

    # Doc-causal block mask (always built)
    def _doc_causal_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (document_ids[q_idx] == document_ids[kv_idx])

    flex_doc_causal_bm = create_block_mask(
        _doc_causal_mod, B=None, H=None,
        Q_LEN=seq_len, KV_LEN=seq_len, device=device,
    )

    # Cross-doc masks (optional)
    dense_mask = q_bitmasks = kv_bitmasks = flex_cross_doc_bm = bim = None

    if n_links > 0 and len(doc_spans) >= 2:
        import random as _random
        _rng = _random.Random(42)
        link_to_target: Dict[int, List[int]] = {}
        n_docs = len(doc_spans)
        for _ in range(n_links):
            src_idx = _rng.randint(1, n_docs - 1)   # source doc
            tgt_idx = _rng.randint(0, src_idx - 1)  # target must be earlier
            src_span = doc_spans[src_idx]
            link_pos = _rng.randint(src_span.start + 1, src_span.end)
            link_to_target.setdefault(link_pos, []).append(doc_spans[tgt_idx].doc_id)

        dense_mask, q_bitmasks, kv_bitmasks = _build_cross_doc_masks(
            seq_len, doc_spans, link_to_target, device, max_grants=max_grants,
        )
        flex_cross_doc_bm = _build_flex_cross_doc_block_mask(
            seq_len, document_ids, q_bitmasks, kv_bitmasks, device,
        )
        # BlockInteractionMask for triton_cross_doc_bitmask_bim
        try:
            from model.graph_traversal.cross_doc_mask import BlockInteractionMask
            # Use doc_spans SimpleNamespaces (harness format) with the creator helper
            class _FakeCreator:
                triton_block_size = 64
                _n_chunks = q_bitmasks.shape[0]
                def _build_block_interaction_mask(self, sl, doc_ids, q_bms, kv_bms, dev):
                    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
                    c = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
                    c.triton_block_size = self.triton_block_size
                    c._n_chunks = self._n_chunks
                    return CrossDocLinkMaskCreator._build_block_interaction_mask(
                        c, sl, doc_ids, q_bms, kv_bms, dev)
            bim = _FakeCreator()._build_block_interaction_mask(
                seq_len, document_ids, list(q_bitmasks), list(kv_bitmasks), device)
        except Exception:
            bim = None

    mask_inputs = MaskInputs(
        seq_len=seq_len,
        doc_spans=doc_spans,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        document_ids=document_ids,
        dense_mask=dense_mask,
        q_bitmasks=q_bitmasks,
        kv_bitmasks=kv_bitmasks,
        flex_doc_causal_block_mask=flex_doc_causal_bm,
        flex_cross_doc_block_mask=flex_cross_doc_bm,
        bim=bim,
    )
    return q, k, v, mask_inputs


# ---------------------------------------------------------------------------
# Real pack fixtures — permanent artifacts in tests/fixtures/real_packs/
# ---------------------------------------------------------------------------
# Generated by scripts/generate_pack_fixtures.py
# (simplewiki, BFS, prefer_targets_first, token_budget=1024, seed=7).
#
# dense uses max_grants=4 < n_grants=14 → exercises bitmask truncation path.

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "real_packs"


@dataclass
class FixtureMeta:
    """Metadata for a pre-sampled pack fixture."""
    label: str           # "sparse" | "medium" | "dense"
    n_grants: int        # pre-sampled grant count (resolved cross-doc grants)
    max_grants: int      # max_grants used when building masks; < n_grants → truncation
    kv_block_count: int  # analytical block count
    seq_len: int
    n_docs: int

    @property
    def path(self) -> Path:
        return FIXTURES_DIR / f"{self.label}.pt"

    def data_label(self) -> str:
        trunc = f",trunc@{self.max_grants}" if self.max_grants < self.n_grants else ""
        return (f"real:simplewiki/{self.label}"
                f"(T={self.seq_len},docs={self.n_docs},grants={self.n_grants}{trunc})")


REAL_PACK_FIXTURES: List[FixtureMeta] = [
    FixtureMeta(label="sparse", n_grants=0,   max_grants=64,  kv_block_count=31653, seq_len=32768, n_docs=2),
    FixtureMeta(label="medium", n_grants=25,  max_grants=64,  kv_block_count=8157,  seq_len=32768, n_docs=17),
    FixtureMeta(label="dense",  n_grants=609, max_grants=128, kv_block_count=10680, seq_len=32768, n_docs=191),
]


def load_fixture_batch(
    meta: FixtureMeta,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, MaskInputs]:
    """Load a pre-sampled pack fixture and build MaskInputs from it.

    q/k/v are deterministically random (seeded by fixture label) — only the
    mask uses real token data.  meta.max_grants controls how many cross-doc
    grants are encoded; for the dense fixture max_grants=4 < n_grants=14,
    exercising the bitmask truncation codepath.
    """
    data = torch.load(str(meta.path), weights_only=False)

    doc_spans_dicts: List[Dict] = data["doc_spans"]
    actual_T = int(data["seq_len"])

    # Remap raw GraphIndex doc_ids to pack-order 0-indexed values.
    # Kernels that index cu_seqlens by doc_id (varlen bwd, cdn/cdb bwd) require
    # 0-indexed ids that exactly match cu_seqlens offsets.  The BIM same-doc
    # range-overlap check also requires monotonically non-decreasing ids.
    # Raw GraphIndex doc_ids are neither sequential nor guaranteed monotonic
    # (e.g. dense fixture: doc_ids 23658, 119129, 272582, 239182 — the last is
    # smaller than the third, so blk_min/max ranges would be wrong).
    raw_to_idx: Dict[int, int] = {
        d["doc_id"]: i for i, d in enumerate(doc_spans_dicts)
    }
    doc_spans = [
        SimpleNamespace(doc_id=raw_to_idx[d["doc_id"]], start=d["start"], end=d["end"])
        for d in doc_spans_dicts
    ]
    # Remap link_to_target values to 0-indexed doc positions.
    link_to_target: Dict[int, List[int]] = {
        int(k): [raw_to_idx[int(v)] for v in vs]
        for k, vs in data["link_to_target"].items()
    }

    cu_seqlens, max_seqlen = _spans_to_cu_seqlens(doc_spans, device)
    document_ids = _spans_to_document_ids(doc_spans, actual_T, device)

    dense_mask, q_bitmasks, kv_bitmasks = _build_cross_doc_masks(
        actual_T, doc_spans, link_to_target, device, max_grants=meta.max_grants,
    )

    flex_cross_doc_bm = None
    if link_to_target:
        flex_cross_doc_bm = _build_flex_cross_doc_block_mask(
            actual_T, document_ids, q_bitmasks, kv_bitmasks, device,
        )

    def _dc_mod(b, h, qi, ki):
        return (qi >= ki) & (document_ids[qi] == document_ids[ki])
    flex_doc_causal_bm = create_block_mask(
        _dc_mod, B=None, H=None, Q_LEN=actual_T, KV_LEN=actual_T, device=device,
    )

    bim = None
    try:
        from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
        c = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
        c.triton_block_size = 64
        c._n_chunks = max(1, (meta.max_grants + 63) // 64)
        bim = CrossDocLinkMaskCreator._build_block_interaction_mask(
            c, actual_T, document_ids, list(q_bitmasks), list(kv_bitmasks), device)
    except Exception:
        pass

    # Deterministic random q/k/v — seeded by label for cross-run reproducibility
    seed_val = hash(meta.label) & 0xFFFFFFFF
    gen = torch.Generator(device=device)
    gen.manual_seed(seed_val)
    q = torch.randn(actual_T, num_heads, head_dim, dtype=dtype, device=device,
                    generator=gen).requires_grad_(True)
    k = torch.randn(actual_T, num_heads, head_dim, dtype=dtype, device=device,
                    generator=gen).requires_grad_(True)
    v = torch.randn(actual_T, num_heads, head_dim, dtype=dtype, device=device,
                    generator=gen).requires_grad_(True)

    mask_inputs = MaskInputs(
        seq_len=actual_T,
        doc_spans=doc_spans,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        document_ids=document_ids,
        dense_mask=dense_mask,
        q_bitmasks=q_bitmasks,
        kv_bitmasks=kv_bitmasks,
        flex_doc_causal_block_mask=flex_doc_causal_bm,
        flex_cross_doc_block_mask=flex_cross_doc_bm,
        bim=bim,
    )
    return q, k, v, mask_inputs


# ---------------------------------------------------------------------------
# Ground-truth reference implementations
# ---------------------------------------------------------------------------

def _to_bhnd(x: torch.Tensor) -> torch.Tensor:
    """(T, H, Dh) → (1, H, T, Dh)"""
    return x.permute(1, 0, 2).unsqueeze(0)


def _to_thd(x: torch.Tensor) -> torch.Tensor:
    """(1, H, T, Dh) → (T, H, Dh)"""
    return x.squeeze(0).permute(1, 0, 2)


_CHUNKED_THRESHOLD = 4096  # use chunked naive above this T to avoid OOM


def compute_reference(
    mask_type: MaskType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_inputs: MaskInputs,
    scale: float,
) -> torch.Tensor:
    """Ground-truth reference: naive O(T²) attention in the input dtype.

    Uses plain matmul → masked_fill → softmax → matmul with no flash tiling or
    reordering.  This is the mathematical definition of attention, directly
    applicable to every mask type without per-case special-casing.  Any deviation
    a flash-based implementation shows from this IS its numerical error budget.

    At T > _CHUNKED_THRESHOLD uses a chunked variant with identical per-element
    arithmetic but avoids materialising the full T×T score matrix.

    Returns (T, H, Dh) output.
    """
    T = q.shape[0]
    bool_mask = _build_bool_mask(mask_type, T, mask_inputs, q.device)
    fwd = _chunked_naive_forward if T > _CHUNKED_THRESHOLD else _naive_forward
    return _to_thd(fwd(_to_bhnd(q), _to_bhnd(k), _to_bhnd(v), bool_mask, scale))


# ---------------------------------------------------------------------------
# Implementation registry
# ---------------------------------------------------------------------------

# Each entry: callable(q, k, v, mask_inputs, scale) → (T, H, Dh) Tensor
# Returns None if mask_type unsupported.

def _impl_sdpa(q, k, v, mask_inputs, scale):
    """SDPA with is_causal=True (full causal only)."""
    q4, k4, v4 = _to_bhnd(q), _to_bhnd(k), _to_bhnd(v)
    return _to_thd(F.scaled_dot_product_attention(q4, k4, v4, is_causal=True, scale=scale))


def _impl_vslf(q, k, v, mask_inputs, scale):
    """PyTorch built-in varlen flash (doc_causal via cu_seqlens)."""
    _vslf = torch.ops.aten._flash_attention_forward
    # Build a fresh tensor via torch.tensor() — same pattern as
    # _impl_vslf_full_causal which is known to have stable autograd.
    # Using mask_inputs.cu_seqlens directly (or a .clone()) causes backward
    # instability with this op for reasons that appear to be internal to
    # _flash_attention_forward's autograd implementation.
    cu = torch.tensor(
        mask_inputs.cu_seqlens.tolist(),
        dtype=torch.int32, device=q.device,
    )
    out, *_ = _vslf(
        q, k, v,
        cu, cu,
        mask_inputs.max_seqlen, mask_inputs.max_seqlen,
        0.0, True, False, scale=scale,
    )
    return out


def _impl_vslf_full_causal(q, k, v, mask_inputs, scale):
    """VSLF with a single doc spanning the whole sequence (= full causal)."""
    T = q.shape[0]
    cu = torch.tensor([0, T], dtype=torch.int32, device=q.device)
    _vslf = torch.ops.aten._flash_attention_forward
    out, *_ = _vslf(q, k, v, cu, cu, T, T, 0.0, True, False, scale=scale)
    return out


def _impl_flex_full_causal(q, k, v, mask_inputs, scale):
    """FlexAttention with a plain full-causal BlockMask (no doc boundaries)."""
    T = q.shape[0]
    device = q.device
    bm = create_block_mask(
        lambda b, h, qi, ki: qi >= ki,
        B=None, H=None, Q_LEN=T, KV_LEN=T, device=device,
    )
    q4, k4, v4 = _to_bhnd(q), _to_bhnd(k), _to_bhnd(v)
    out4 = _compiled_flex(q4, k4, v4, block_mask=bm, scale=scale)
    return _to_thd(out4)


def _impl_flex_doc_causal(q, k, v, mask_inputs, scale):
    """FlexAttention with doc_causal BlockMask."""
    assert mask_inputs.flex_doc_causal_block_mask is not None
    q4, k4, v4 = _to_bhnd(q), _to_bhnd(k), _to_bhnd(v)
    out4 = _compiled_flex(q4, k4, v4, block_mask=mask_inputs.flex_doc_causal_block_mask, scale=scale)
    return _to_thd(out4)


def _impl_flex_cross_doc(q, k, v, mask_inputs, scale):
    """FlexAttention with cross_doc BlockMask."""
    assert mask_inputs.flex_cross_doc_block_mask is not None, \
        "cross_doc BlockMask not built — pass n_links>0 to make_synthetic_batch"
    q4, k4, v4 = _to_bhnd(q), _to_bhnd(k), _to_bhnd(v)
    out4 = _compiled_flex(q4, k4, v4, block_mask=mask_inputs.flex_cross_doc_block_mask, scale=scale)
    return _to_thd(out4)


def _impl_triton_causal(q, k, v, mask_inputs, scale):
    from kernels.causal_attn import triton_attn_causal
    q4, k4, v4 = _to_bhnd(q), _to_bhnd(k), _to_bhnd(v)
    return _to_thd(triton_attn_causal(q4, k4, v4, scale))


def _impl_triton_varlen(q, k, v, mask_inputs, scale):
    from kernels.varlen_attn import triton_attn_varlen
    return triton_attn_varlen(q, k, v, mask_inputs.cu_seqlens, mask_inputs.max_seqlen, scale)


def _impl_triton_cross_doc_naive(q, k, v, mask_inputs, scale):
    from kernels.cross_doc_naive_attn import triton_attn_cross_doc_naive
    assert mask_inputs.dense_mask is not None
    return triton_attn_cross_doc_naive(
        q, k, v, mask_inputs.document_ids, mask_inputs.dense_mask, scale,
    )


def _impl_cdb_v1(q, k, v, mask_inputs, scale):
    """cdb_v1: OR-reduction block-skip, no BIM required."""
    from kernels.cross_doc_bitmask_attn import triton_attn_cross_doc_bitmask
    assert mask_inputs.q_bitmasks is not None
    return triton_attn_cross_doc_bitmask(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, scale,
    )


def _impl_cdb_bim_v1(q, k, v, mask_inputs, scale):
    """cdb_bim_v1: CSR index lists, BIM fwd + bwd (no OR-reductions)."""
    from kernels.cross_doc_bitmask_bim_v1 import triton_attn_cross_doc_bitmask_bim_v1
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None, \
        "BlockInteractionMask not built — use make_synthetic_batch with n_links>0"
    return triton_attn_cross_doc_bitmask_bim_v1(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _rebuild_bim(mask_inputs, block_size: int):
    """Rebuild BlockInteractionMask at a different block_size (for block-size experiments)."""
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
    creator = CrossDocLinkMaskCreator.__new__(CrossDocLinkMaskCreator)
    creator.triton_block_size = block_size
    creator._n_chunks = mask_inputs.q_bitmasks.shape[0]
    return creator._build_block_interaction_mask(
        mask_inputs.seq_len,
        mask_inputs.document_ids,
        list(mask_inputs.q_bitmasks),
        list(mask_inputs.kv_bitmasks),
        mask_inputs.document_ids.device,
    )


def _impl_cdb_bim_v2(q, k, v, mask_inputs, scale):
    """cdb_bim_v2: diagonal-first backward (H2 fix). Forward identical to v1."""
    from kernels.cross_doc_bitmask_bim_v2 import triton_attn_cross_doc_bitmask_bim_v2
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None, \
        "BlockInteractionMask not built — use make_synthetic_batch with n_links>0"
    return triton_attn_cross_doc_bitmask_bim_v2(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _impl_cdb_bim_v3(q, k, v, mask_inputs, scale):
    """cdb_bim_v3: full/partial block split (H1) + diagonal-first bwd (H2)."""
    from kernels.cross_doc_bitmask_bim_v3 import triton_attn_cross_doc_bitmask_bim_v3
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None, \
        "BlockInteractionMask not built — use make_synthetic_batch with n_links>0"
    return triton_attn_cross_doc_bitmask_bim_v3(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _impl_cdb_bim_v4(q, k, v, mask_inputs, scale):
    """cdb_bim_v4: split dK/dV and dQ backward kernels (H5) + H1 + H2."""
    from kernels.cross_doc_bitmask_bim_v4 import triton_attn_cross_doc_bitmask_bim_v4
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None, \
        "BlockInteractionMask not built — use make_synthetic_batch with n_links>0"
    return triton_attn_cross_doc_bitmask_bim_v4(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _impl_cdb_bim_v4_bs128(q, k, v, mask_inputs, scale):
    """cdb_bim_v4 with BIM_BLOCK_SIZE=128 (flex's native tile size)."""
    from kernels.cross_doc_bitmask_bim_v4 import triton_attn_cross_doc_bitmask_bim_v4
    assert mask_inputs.q_bitmasks is not None
    bim128 = _rebuild_bim(mask_inputs, 128)
    return triton_attn_cross_doc_bitmask_bim_v4(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, scale,
    )


def _impl_cdb_bim_v5(q, k, v, mask_inputs, scale):
    """cdb_bim_v5: free-tile backward (BLOCK_SIZE_MACRO up to 128, no BIM in bwd)."""
    from kernels.cross_doc_bitmask_bim_v5 import triton_attn_cross_doc_bitmask_bim_v5
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None
    return triton_attn_cross_doc_bitmask_bim_v5(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _impl_cdb_bim_v6(q, k, v, mask_inputs, scale):
    """cdb_bim_v6: double-tile backward (2 BIM blocks/CTA → 128-token tiles)."""
    from kernels.cross_doc_bitmask_bim_v6 import triton_attn_cross_doc_bitmask_bim_v6
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None
    return triton_attn_cross_doc_bitmask_bim_v6(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _impl_cdb_bim_v7(q, k, v, mask_inputs, scale):
    """cdb_bim_v7: precomputed coarse 128-token backward tiles."""
    from kernels.cross_doc_bitmask_bim_v7 import triton_attn_cross_doc_bitmask_bim_v7
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None
    return triton_attn_cross_doc_bitmask_bim_v7(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _impl_cdb_bim_v8(q, k, v, mask_inputs, scale):
    """cdb_bim_v8: varlen-style pipelined same-doc + BIM cross-doc."""
    from kernels.cross_doc_bitmask_bim_v8 import triton_attn_cross_doc_bitmask_bim_v8
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None
    return triton_attn_cross_doc_bitmask_bim_v8(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _impl_cdb_bim_v9(q, k, v, mask_inputs, scale):
    """cdb_bim_v9: split pipelined backward (varlen-style same-doc + BIM cross-doc)."""
    from kernels.cross_doc_bitmask_bim_v9 import triton_attn_cross_doc_bitmask_bim_v9
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None
    return triton_attn_cross_doc_bitmask_bim_v9(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim,
        mask_inputs.cu_seqlens, scale,
    )


def _impl_cdb_bim_v10(q, k, v, mask_inputs, scale):
    """cdb_bim_v10: native-dtype matmuls (bf16 TC) — no fp32 cast on load."""
    from kernels.cross_doc_bitmask_bim_v10 import triton_attn_cross_doc_bitmask_bim_v10
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None
    return triton_attn_cross_doc_bitmask_bim_v10(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


def _impl_cdb_bim_v11(q, k, v, mask_inputs, scale):
    """cdb_bim_v11: v10 + no permute/contiguous copies (pass THD strides directly)."""
    from kernels.cross_doc_bitmask_bim_v11 import triton_attn_cross_doc_bitmask_bim_v11
    assert mask_inputs.q_bitmasks is not None
    assert mask_inputs.bim is not None
    return triton_attn_cross_doc_bitmask_bim_v11(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, mask_inputs.bim, scale,
    )


_bim128_cache: Dict[int, Any] = {}

def _get_bim128(mask_inputs):
    """Build (and cache) a BIM at block_size=128 for the given mask_inputs."""
    from kernels.cross_doc_bitmask_bim_v12 import _build_bim_128
    key = id(mask_inputs)
    if key not in _bim128_cache:
        _bim128_cache[key] = _build_bim_128(
            mask_inputs.seq_len, mask_inputs.document_ids,
            mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks,
            mask_inputs.document_ids.device, mask_inputs.q_bitmasks.shape[0],
        )
    return _bim128_cache[key]


def _impl_cdb_bim_v12(q, k, v, mask_inputs, scale):
    """cdb_bim_v12: v11 + BIM_BLOCK_SIZE=128 (larger matmul tiles)."""
    from kernels.cross_doc_bitmask_bim_v12 import (
        triton_attn_cross_doc_bitmask_bim_v12, _build_bim_128,
    )
    assert mask_inputs.q_bitmasks is not None
    bim128 = _get_bim128(mask_inputs)
    return triton_attn_cross_doc_bitmask_bim_v12(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, scale,
    )


def _impl_cdb_bim_v13(q, k, v, mask_inputs, scale):
    """cdb_bim_v13: v12 + pure-cross dispatch (bitmask-only inner kernel)."""
    from kernels.cross_doc_bitmask_bim_v13 import triton_attn_cross_doc_bitmask_bim_v13
    assert mask_inputs.q_bitmasks is not None
    bim128 = _get_bim128(mask_inputs)
    return triton_attn_cross_doc_bitmask_bim_v13(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, scale,
    )


def _impl_cdb_bim_v14(q, k, v, mask_inputs, scale):
    """cdb_bim_v14: v12 + expanded backward autotune (num_warps=16, num_stages={2,6})."""
    from kernels.cross_doc_bitmask_bim_v14 import triton_attn_cross_doc_bitmask_bim_v14
    assert mask_inputs.q_bitmasks is not None
    bim128 = _get_bim128(mask_inputs)
    return triton_attn_cross_doc_bitmask_bim_v14(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, scale,
    )


def _impl_cdb_bim_v15(q, k, v, mask_inputs, scale):
    """cdb_bim_v15: v12 + persistent-CTA backward (work-stealing dKV + dQ)."""
    from kernels.cross_doc_bitmask_bim_v15 import triton_attn_cross_doc_bitmask_bim_v15
    assert mask_inputs.q_bitmasks is not None
    bim128 = _get_bim128(mask_inputs)
    return triton_attn_cross_doc_bitmask_bim_v15(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, scale,
    )


def _impl_cdb_bim_v16(q, k, v, mask_inputs, scale):
    """cdb_bim_v16: v12 + .cg cache hints on bitmask loads."""
    from kernels.cross_doc_bitmask_bim_v16 import triton_attn_cross_doc_bitmask_bim_v16
    assert mask_inputs.q_bitmasks is not None
    bim128 = _get_bim128(mask_inputs)
    return triton_attn_cross_doc_bitmask_bim_v16(
        q, k, v, mask_inputs.document_ids,
        mask_inputs.q_bitmasks, mask_inputs.kv_bitmasks, bim128, scale,
    )


# ---------------------------------------------------------------------------
# Naive PyTorch baselines (O(T²), exact numerics, no flash tricks)
# ---------------------------------------------------------------------------

def _build_bool_mask(
    mask_type: MaskType,
    T: int,
    mask_inputs: MaskInputs,
    device: torch.device,
) -> torch.Tensor:
    """Build a (T, T) bool attention mask for the given mask type.
    True  = this (query, key) pair is allowed to attend.
    False = masked out (will become -inf before softmax).
    """
    qi = torch.arange(T, device=device)
    causal = qi.unsqueeze(1) >= qi.unsqueeze(0)   # lower-triangular including diagonal

    if mask_type == MaskType.FULL_CAUSAL:
        return causal

    doc_ids = mask_inputs.document_ids
    same_doc = doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0)

    if mask_type == MaskType.DOC_CAUSAL:
        return causal & same_doc

    # CROSS_DOC_NAIVE or CROSS_DOC_BITMASK — same semantic mask, different encoding
    assert mask_inputs.dense_mask is not None, \
        "dense_mask required for cross_doc naive baselines"
    return causal & (same_doc | mask_inputs.dense_mask)


def _naive_forward(
    q4: torch.Tensor,
    k4: torch.Tensor,
    v4: torch.Tensor,
    bool_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Dense O(T²) attention — the numerically cleanest baseline.

    All arithmetic is in the dtype of q4/k4/v4 (no flash algorithm, no tiling).
    q4/k4/v4: (1, H, T, Dh);  bool_mask: (T, T) True=attend.
    Returns (1, H, T, Dh).
    """
    scores = (q4 @ k4.transpose(-2, -1)) * scale           # (1, H, T, T)
    scores = scores.masked_fill(~bool_mask[None, None], float("-inf"))
    return scores.softmax(dim=-1) @ v4


def _chunked_naive_forward(
    q4: torch.Tensor,
    k4: torch.Tensor,
    v4: torch.Tensor,
    bool_mask: torch.Tensor,
    scale: float,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Chunked naive reference — same arithmetic as _naive_forward, avoids OOM at large T.

    Processes Q in non-overlapping chunks so peak scores tensor is
    [1, H, chunk_size, T] instead of [1, H, T, T].
    q4/k4/v4: (1, H, T, Dh);  bool_mask: (T, T) True=attend.
    Returns (1, H, T, Dh).
    """
    B, H, T, Dh = q4.shape
    out = torch.zeros_like(q4)
    for qs in range(0, T, chunk_size):
        qe = min(qs + chunk_size, T)
        scores = (q4[:, :, qs:qe, :] @ k4.transpose(-2, -1)) * scale  # (1, H, chunk, T)
        scores = scores.masked_fill(~bool_mask[qs:qe, :][None, None], float("-inf"))
        out[:, :, qs:qe, :] = scores.softmax(dim=-1) @ v4
    return out


# Compiled variant — same arithmetic, but torch.compile may fuse/reorder ops.
# Dynamic=True so different (T, H, Dh) shapes don't cause recompiles.
_compiled_naive_forward = torch.compile(_naive_forward, dynamic=True)


def _make_naive_impl(mask_type: MaskType, compiled: bool = False) -> Callable:
    """Factory: returns an impl_fn for the given mask_type, naive or compiled."""
    _fn = _compiled_naive_forward if compiled else _naive_forward

    def impl(q, k, v, mask_inputs, scale):
        T, device = q.shape[0], q.device
        bool_mask = _build_bool_mask(mask_type, T, mask_inputs, device)
        return _to_thd(_fn(_to_bhnd(q), _to_bhnd(k), _to_bhnd(v), bool_mask, scale))

    label = f"naive_{'compiled' if compiled else 'pytorch'}_{mask_type.value}"
    impl.__name__ = label
    return impl


# ---------------------------------------------------------------------------
# Benchmark reference — the impl whose time is the denominator for ratios.
# Chosen to be the fastest "trusted" flash baseline for each mask type so
# ratios are meaningful (>1 = slower than the trusted baseline).
# Naive impls are excluded from BENCH_SKIP so they don't OOM at large T.
# ---------------------------------------------------------------------------

# Impl names to omit from runtime/memory benchmarks (O(T²), will OOM at 32k).
BENCH_SKIP: frozenset = frozenset({"naive_pytorch", "naive_compiled"})

# Same impls are also skipped from correctness checks when T > _CHUNKED_THRESHOLD.
CORRECTNESS_SKIP_LARGE_T: frozenset = frozenset({"naive_pytorch", "naive_compiled"})

# ---------------------------------------------------------------------------
# Tolerance philosophy
# ---------------------------------------------------------------------------
# Current approach: fixed fwd_atol / bwd_atol, calibrated to what PyTorch's
# own sdpa/flex show vs the chunked naive reference at T=32k.  At that scale
# both hit ~9e-2 bwd error, so bwd_atol=2e-1 gives a safe margin.
#
# Future improvement — witness-based dynamic tolerance:
#   Rather than a hard constant, gate on  impl_err <= C * max(witness_errs),
#   where "witnesses" are trusted PyTorch implementations (sdpa, flex) whose
#   errors are computed on the same fixture in the same check_correctness call.
#   C=1.5 would mean "our kernel is allowed to be at most 50% noisier than the
#   noisiest PyTorch impl on this fixture."  This self-calibrates across T,
#   dtype, and doc-length distribution without ever needing to update a constant.
#   The only scenario it misses is a systematic bias where ALL witnesses and the
#   impl under test are wrong in the same direction relative to true math — but
#   the chunked naive reference (not flash-based) guards against that already.
# ---------------------------------------------------------------------------

# Impl names whose backward is not checked for correctness.
# vslf = torch.ops.aten._flash_attention_forward — PyTorch internal op used as a
# timing baseline.  Its autograd backward is unreliable on real-data packs
# (non-uniform doc lengths, T≥1024): after exhaustive investigation and multiple
# fix attempts (clone, fresh torch.tensor, per-doc loop), it produces large errors
# on real fixtures while passing on synthetic.  Root cause is PyTorch-internal.
# Our custom triton_varlen kernel covers the doc_causal backward correctly.
SKIP_BWD_CHECK: frozenset = frozenset({"vslf"})

# Which impl to use as the ratio denominator in bench mode.
BENCH_REFERENCE: Dict[MaskType, str] = {
    MaskType.FULL_CAUSAL:      "sdpa",
    MaskType.DOC_CAUSAL:       "vslf",
    MaskType.CROSS_DOC_NAIVE:  "flex",
    MaskType.CROSS_DOC_BITMASK: "flex",
}

# Registry: mask_type → list of (name, impl_fn)
#
# Ordering convention within each mask type:
#   1. naive_pytorch          — O(T²) exact reference, no flash tricks
#   2. naive_compiled         — same + torch.compile
#   3. sdpa / vslf            — PyTorch's own fast kernels (certified error budget)
#   4. flex                   — FlexAttention
#   5. triton_*               — our custom kernels
REGISTRY: Dict[MaskType, List[Tuple[str, Callable]]] = {
    MaskType.FULL_CAUSAL: [
        ("naive_pytorch",      _make_naive_impl(MaskType.FULL_CAUSAL, compiled=False)),
        ("naive_compiled",     _make_naive_impl(MaskType.FULL_CAUSAL, compiled=True)),
        ("sdpa",               _impl_sdpa),
        ("vslf",               _impl_vslf_full_causal),
        ("flex",               _impl_flex_full_causal),
        ("triton_causal",      _impl_triton_causal),
    ],
    MaskType.DOC_CAUSAL: [
        ("naive_pytorch",      _make_naive_impl(MaskType.DOC_CAUSAL, compiled=False)),
        ("naive_compiled",     _make_naive_impl(MaskType.DOC_CAUSAL, compiled=True)),
        ("vslf",               _impl_vslf),
        ("flex",               _impl_flex_doc_causal),
        ("triton_varlen",      _impl_triton_varlen),
    ],
    MaskType.CROSS_DOC_NAIVE: [
        ("naive_pytorch",          _make_naive_impl(MaskType.CROSS_DOC_NAIVE, compiled=False)),
        ("naive_compiled",         _make_naive_impl(MaskType.CROSS_DOC_NAIVE, compiled=True)),
        ("flex",                   _impl_flex_cross_doc),
        ("triton_cross_doc_naive", _impl_triton_cross_doc_naive),
    ],
    MaskType.CROSS_DOC_BITMASK: [
        ("naive_pytorch",    _make_naive_impl(MaskType.CROSS_DOC_BITMASK, compiled=False)),
        ("naive_compiled",   _make_naive_impl(MaskType.CROSS_DOC_BITMASK, compiled=True)),
        ("flex",             _impl_flex_cross_doc),
        ("cdb_v1",           _impl_cdb_v1),           # baseline: OR-reduction skip
        ("cdb_bim_v1",       _impl_cdb_bim_v1),       # BIM fwd + bwd
        ("cdb_bim_v2",       _impl_cdb_bim_v2),       # BIM fwd + diagonal-first bwd (H2)
        ("cdb_bim_v3",       _impl_cdb_bim_v3),       # full/partial split (H1) + H2
        ("cdb_bim_v4",       _impl_cdb_bim_v4),       # split dKV/dQ bwd (H5) + H1 + H2
        ("cdb_bim_v4_bs128", _impl_cdb_bim_v4_bs128), # v4 with BIM_BLOCK_SIZE=128
        ("cdb_bim_v5",       _impl_cdb_bim_v5),       # free-tile bwd
        ("cdb_bim_v6",       _impl_cdb_bim_v6),       # double-tile bwd
        ("cdb_bim_v7",       _impl_cdb_bim_v7),       # coarse 128-token bwd tiles
        ("cdb_bim_v8",       _impl_cdb_bim_v8),       # varlen-style pipelined same-doc
        ("cdb_bim_v9",       _impl_cdb_bim_v9),       # split pipelined same-doc + BIM cross-doc
        ("cdb_bim_v10",      _impl_cdb_bim_v10),      # native-dtype matmuls (bf16 TC)
        ("cdb_bim_v11",      _impl_cdb_bim_v11),      # v10 + no permute copies
        ("cdb_bim_v12",      _impl_cdb_bim_v12),      # v11 + BIM_BLOCK_SIZE=128
        ("cdb_bim_v13",      _impl_cdb_bim_v13),      # v12 + pure-cross dispatch
        ("cdb_bim_v14",      _impl_cdb_bim_v14),      # v12 + expanded bwd autotune
        ("cdb_bim_v15",      _impl_cdb_bim_v15),      # v12 + persistent-CTA backward
        ("cdb_bim_v16",      _impl_cdb_bim_v16),      # v12 + .cg cache hints on bitmask loads
    ],
}


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def time_fn(fn: Callable, warmup: int, iters: int) -> Tuple[float, float]:
    """Returns (median_ms, stdev_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times), (statistics.stdev(times) if len(times) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Correctness checking
# ---------------------------------------------------------------------------

@dataclass
class ImplResult:
    name: str
    fwd_max_err: float
    fwd_p99_err: float
    fwd_mean_err: float
    fwd_pass: bool
    bwd_q_max_err: float
    bwd_k_max_err: float
    bwd_v_max_err: float
    bwd_mean_err: float   # max of dQ/dK/dV mean abs errors
    bwd_p99_err: float    # max of dQ/dK/dV 99th-percentile abs errors
    bwd_pass: bool
    error: Optional[str] = None


@dataclass
class CorrectnessReport:
    mask_type: MaskType
    seq_len: int
    data_label: str = "synthetic"
    results: List[ImplResult] = field(default_factory=list)

    def print(self):
        _sep = "=" * 110
        print(f"\n{_sep}")
        print(f"  Correctness: {self.mask_type.value}  seq_len={self.seq_len}  data={self.data_label}")
        print(_sep)
        hdr = (f"  {'impl':30s}"
               f"  {'fwd_max':>10}  {'fwd_p99':>10}  {'fwd_mean':>10}  {'fwd':>6}"
               f"  {'bwd_max':>10}  {'bwd_p99':>10}  {'bwd_mean':>10}  {'bwd':>6}")
        print(hdr)
        print(f"  {'-'*30}"
              f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}"
              f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
        for r in self.results:
            if r.error:
                print(f"  {r.name:30s}  {'ERROR':>10}  {'':>10}  {'':>10}  {'':>6}"
                      f"  {'':>10}  {'':>10}  {'':>10}  {'':>6}  [{r.error}]")
                continue
            bwd_max = max(r.bwd_q_max_err, r.bwd_k_max_err, r.bwd_v_max_err)
            fwd_sym = "PASS" if r.fwd_pass else "FAIL"
            bwd_skipped = r.name in SKIP_BWD_CHECK
            bwd_sym = "skip" if bwd_skipped else ("PASS" if r.bwd_pass else "FAIL")
            suffix = "  [bwd skipped]" if bwd_skipped else ""
            print(f"  {r.name:30s}"
                  f"  {r.fwd_max_err:>10.2e}  {r.fwd_p99_err:>10.2e}  {r.fwd_mean_err:>10.2e}  {fwd_sym:>6}"
                  f"  {bwd_max:>10.2e}  {r.bwd_p99_err:>10.2e}  {r.bwd_mean_err:>10.2e}  {bwd_sym:>6}"
                  f"{suffix}")
        print(_sep)


def _clone_requires_grad(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().requires_grad_(True)


def _p99(t: torch.Tensor, max_elems: int = 4_000_000) -> float:
    """99th-percentile of abs-error tensor. Subsamples if needed (torch.quantile limit)."""
    flat = t.reshape(-1).float()
    if flat.numel() > max_elems:
        idx = torch.randperm(flat.numel(), device=flat.device)[:max_elems]
        flat = flat[idx]
    return torch.quantile(flat, 0.99).item()


def check_correctness(
    mask_type: MaskType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_inputs: MaskInputs,
    fwd_atol: float = 5e-2,
    bwd_atol: float = 2e-1,
    check_backward: bool = True,
    data_label: str = "synthetic",
    impls_filter: Optional[List[str]] = None,
) -> CorrectnessReport:
    """Compare each applicable implementation against ground-truth chunked naive.

    Pass/fail is gated on max error (fwd_atol / bwd_atol).
    p99 and mean are reported as diagnostics to distinguish isolated spikes
    (mask boundary bugs) from distributed numerical noise (bf16 rounding).
    """
    scale = q.shape[-1] ** -0.5
    report = CorrectnessReport(mask_type=mask_type, seq_len=mask_inputs.seq_len,
                               data_label=data_label)

    # Ground truth
    q_ref = _clone_requires_grad(q)
    k_ref = _clone_requires_grad(k)
    v_ref = _clone_requires_grad(v)
    try:
        ref_out = compute_reference(mask_type, q_ref, k_ref, v_ref, mask_inputs, scale)
        if check_backward:
            grad = torch.ones_like(ref_out)
            ref_out.backward(grad)
            ref_dq = q_ref.grad.clone()
            ref_dk = k_ref.grad.clone()
            ref_dv = v_ref.grad.clone()
    except Exception as e:
        print(f"  ERROR: ground truth failed: {e}")
        return report

    skip_large = CORRECTNESS_SKIP_LARGE_T if mask_inputs.seq_len > _CHUNKED_THRESHOLD else frozenset()
    _impls_set = set(impls_filter) if impls_filter else None
    for name, impl_fn in REGISTRY.get(mask_type, []):
        if _impls_set is not None and name not in _impls_set:
            continue
        if name in skip_large:
            continue
        q_i = _clone_requires_grad(q)
        k_i = _clone_requires_grad(k)
        v_i = _clone_requires_grad(v)
        try:
            out_i = impl_fn(q_i, k_i, v_i, mask_inputs, scale)
        except Exception as e:
            report.results.append(ImplResult(
                name=name,
                fwd_max_err=0, fwd_p99_err=0, fwd_mean_err=0, fwd_pass=False,
                bwd_q_max_err=0, bwd_k_max_err=0, bwd_v_max_err=0,
                bwd_mean_err=0, bwd_p99_err=0, bwd_pass=False,
                error=str(e)[:80],
            ))
            continue

        fwd_diff = (out_i.float() - ref_out.detach().float()).abs()
        fwd_max  = fwd_diff.max().item()
        fwd_p99  = _p99(fwd_diff)
        fwd_mean = fwd_diff.mean().item()
        fwd_pass = bool(fwd_max <= fwd_atol)

        bwd_q_max = bwd_k_max = bwd_v_max = 0.0
        bwd_mean = bwd_p99 = 0.0
        bwd_pass = True
        if check_backward and name not in SKIP_BWD_CHECK:
            try:
                out_i.backward(torch.ones_like(out_i))
                dq_diff = (q_i.grad.float() - ref_dq.float()).abs()
                dk_diff = (k_i.grad.float() - ref_dk.float()).abs()
                dv_diff = (v_i.grad.float() - ref_dv.float()).abs()
                bwd_q_max = dq_diff.max().item()
                bwd_k_max = dk_diff.max().item()
                bwd_v_max = dv_diff.max().item()
                bwd_mean  = max(dq_diff.mean().item(), dk_diff.mean().item(), dv_diff.mean().item())
                bwd_p99   = max(_p99(dq_diff), _p99(dk_diff), _p99(dv_diff))
                bwd_pass  = all(x <= bwd_atol for x in [bwd_q_max, bwd_k_max, bwd_v_max])
            except Exception as e:
                bwd_pass = False
                bwd_q_max = bwd_k_max = bwd_v_max = float("nan")
                bwd_mean = bwd_p99 = float("nan")

        report.results.append(ImplResult(
            name=name,
            fwd_max_err=fwd_max, fwd_p99_err=fwd_p99, fwd_mean_err=fwd_mean, fwd_pass=fwd_pass,
            bwd_q_max_err=bwd_q_max, bwd_k_max_err=bwd_k_max, bwd_v_max_err=bwd_v_max,
            bwd_mean_err=bwd_mean, bwd_p99_err=bwd_p99, bwd_pass=bwd_pass,
        ))

    return report


# ---------------------------------------------------------------------------
# Memory benchmarking
# ---------------------------------------------------------------------------

@dataclass
class MemResult:
    name: str
    fwd_mb: float
    bwd_mb: float


def bench_memory(
    name: str,
    fn_fwd: Callable,
    fn_bwd: Optional[Callable],
) -> MemResult:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn_fwd()
    torch.cuda.synchronize()
    fwd_mb = (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()) / 1e6

    bwd_mb = 0.0
    if fn_bwd is not None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fn_bwd()
        torch.cuda.synchronize()
        bwd_mb = (torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()) / 1e6

    return MemResult(name=name, fwd_mb=fwd_mb, bwd_mb=bwd_mb)


# ---------------------------------------------------------------------------
# Runtime benchmarking
# ---------------------------------------------------------------------------

@dataclass
class BenchRow:
    mask_type: str
    name: str
    seq_len: int
    doc_len: int         # -1 for real-data fixtures (variable doc lengths)
    fwd_ms: float
    bwd_ms: float
    fwd_mb: float        # peak HBM delta during forward (MB)
    bwd_mb: float        # peak HBM delta during backward (MB)
    fwd_ratio: float     # fwd_ms / reference fwd_ms  (>1 = slower than reference)
    fwdbwd_ratio: float  # (fwd+bwd) / reference (fwd+bwd)
    data_label: str = "synthetic"


def run_benchmarks(
    mask_types: List[MaskType],
    seq_lens: List[int],
    doc_lens: List[int],
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    dataset_dir: Optional[str] = None,
    data_source: str = "synthetic",
    impls_filter: Optional[List[str]] = None,
) -> List[BenchRow]:
    rows: List[BenchRow] = []
    scale = head_dim ** -0.5

    sep = "=" * 110

    for seq_len in seq_lens:
        for doc_len in doc_lens:
            if doc_len > seq_len:
                continue

            print(f"\n{sep}")
            print(f"  seq_len={seq_len}  doc_len={doc_len}  "
                  f"dtype={dtype}  heads={num_heads}  Dh={head_dim}")
            print(sep)
            print(f"  {'mask_type / impl':42s}  {'fwd_ms':>8}  {'bwd_ms':>8}  "
                  f"{'fwd_MB':>7}  {'bwd_MB':>7}  {'ratio_fwd':>10}  {'ratio_fwdbwd':>13}")
            print(f"  {'-'*42}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*13}")

            # Build fixture once per (seq_len, doc_len)
            n_links = max(8, seq_len // doc_len)
            q, k, v, mask_inputs = make_synthetic_batch(
                seq_len, doc_len, num_heads, head_dim, dtype, DEVICE,
                n_links=n_links,
            )

            _impls_set = set(impls_filter) if impls_filter else None
            for mask_type in mask_types:
                all_impls = REGISTRY.get(mask_type, [])
                # Exclude O(T²) naive impls from bench — they OOM at large T
                bench_impls = [(n, f) for n, f in all_impls
                               if n not in BENCH_SKIP
                               and (_impls_set is None or n in _impls_set)]
                if not bench_impls:
                    continue

                # Locate reference impl for ratio denominator
                ref_name = BENCH_REFERENCE.get(mask_type)
                ref_fn = next((f for n, f in bench_impls if n == ref_name), None)
                if ref_fn is None:
                    ref_name, ref_fn = bench_impls[0]  # fallback: first available

                def _make_fwd(fn, qi, ki, vi):
                    def fwd():
                        return fn(qi, ki, vi, mask_inputs, scale)
                    return fwd

                def _make_fwdbwd(fn, qi, ki, vi):
                    def fwdbwd():
                        qi.grad = ki.grad = vi.grad = None
                        out = fn(qi, ki, vi, mask_inputs, scale)
                        out.sum().backward()
                    return fwdbwd

                # Time the reference
                q_r = _clone_requires_grad(q)
                k_r = _clone_requires_grad(k)
                v_r = _clone_requires_grad(v)
                try:
                    ref_fwd_ms, _ = time_fn(_make_fwd(ref_fn, q_r, k_r, v_r), warmup, iters)
                    ref_fb_ms, _ = time_fn(_make_fwdbwd(ref_fn, q_r, k_r, v_r), warmup, iters)
                except Exception as e:
                    print(f"  [{mask_type.value}] {ref_name}: ERROR (reference) — {e}")
                    continue

                for name, impl_fn in bench_impls:
                    q_i = _clone_requires_grad(q)
                    k_i = _clone_requires_grad(k)
                    v_i = _clone_requires_grad(v)
                    try:
                        fwd_ms, _ = time_fn(_make_fwd(impl_fn, q_i, k_i, v_i), warmup, iters)
                        fb_ms, _ = time_fn(_make_fwdbwd(impl_fn, q_i, k_i, v_i), warmup, iters)
                        bwd_ms = fb_ms - fwd_ms

                        # Memory: peak HBM delta (fwd pass only, then bwd pass only)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        baseline_mem = torch.cuda.memory_allocated()
                        _make_fwd(impl_fn, q_i, k_i, v_i)()
                        torch.cuda.synchronize()
                        fwd_mb = (torch.cuda.max_memory_allocated() - baseline_mem) / 1e6

                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        baseline_mem = torch.cuda.memory_allocated()
                        _make_fwdbwd(impl_fn, q_i, k_i, v_i)()
                        torch.cuda.synchronize()
                        bwd_mb = (torch.cuda.max_memory_allocated() - baseline_mem) / 1e6

                        fwd_ratio = fwd_ms / ref_fwd_ms
                        fb_ratio = fb_ms / ref_fb_ms
                        label = f"[{mask_type.value}] {name}"
                        print(
                            f"  {label:42s}  {fwd_ms:8.2f}  {bwd_ms:8.2f}  "
                            f"{fwd_mb:7.0f}  {bwd_mb:7.0f}  "
                            f"{fwd_ratio:9.2f}x  {fb_ratio:12.2f}x"
                            + ("  ← ref" if name == ref_name else ""),
                            flush=True,
                        )
                        rows.append(BenchRow(
                            mask_type=mask_type.value, name=name,
                            seq_len=seq_len, doc_len=doc_len,
                            fwd_ms=fwd_ms, bwd_ms=bwd_ms,
                            fwd_mb=fwd_mb, bwd_mb=bwd_mb,
                            fwd_ratio=fwd_ratio, fwdbwd_ratio=fb_ratio,
                            data_label="synthetic",
                        ))
                    except Exception as e:
                        print(f"  [{mask_type.value}] {name}: ERROR — {e}")

    print(f"\n{sep}")
    ref_str = "  ".join(f"{k.value}→{v}" for k, v in BENCH_REFERENCE.items())
    print(f"  ratio = impl / reference  (>1 = slower)   references: {ref_str}")
    print(sep)
    return rows


def bench_real_fixtures(
    mask_types: List[MaskType],
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    impls_filter: Optional[List[str]] = None,
) -> List[BenchRow]:
    """Benchmark all applicable impls on each pre-sampled real pack fixture.

    Uses the same timing / memory measurement logic as run_benchmarks but with
    fixed MaskInputs loaded from the .pt artifacts instead of synthetic data.
    Skips fixtures whose .pt file is not present.
    """
    available = [m for m in REAL_PACK_FIXTURES if m.path.exists()]
    if not available:
        print("\n  [real fixtures] skipped — run scripts/generate_pack_fixtures.py first")
        return []

    rows: List[BenchRow] = []
    scale = head_dim ** -0.5
    sep = "=" * 110

    for meta in available:
        print(f"\n{sep}")
        print(f"  REAL FIXTURE: {meta.data_label()}")
        print(sep)
        print(f"  {'mask_type / impl':42s}  {'fwd_ms':>8}  {'bwd_ms':>8}  "
              f"{'fwd_MB':>7}  {'bwd_MB':>7}  {'ratio_fwd':>10}  {'ratio_fwdbwd':>13}")
        print(f"  {'-'*42}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*13}")

        try:
            q, k, v, mask_inputs = load_fixture_batch(meta, num_heads, head_dim, dtype, DEVICE)
        except Exception as e:
            print(f"  [fixture {meta.label}] load error: {e}")
            continue

        _cross_doc_types = {MaskType.CROSS_DOC_NAIVE, MaskType.CROSS_DOC_BITMASK}
        for mask_type in mask_types:
            # Skip cross-doc mask types when the fixture has no links — the mask
            # degenerates to doc_causal and flex_cross_doc_block_mask is None,
            # which would error when used as the benchmark reference.
            if mask_type in _cross_doc_types and not mask_inputs.flex_cross_doc_block_mask:
                print(f"  [{mask_type.value}] skipped — fixture has no cross-doc links (degenerates to doc_causal)")
                continue
            _impls_set_r = set(impls_filter) if impls_filter else None
            all_impls = REGISTRY.get(mask_type, [])
            bench_impls = [(n, f) for n, f in all_impls
                           if n not in BENCH_SKIP
                           and (_impls_set_r is None or n in _impls_set_r)]
            if not bench_impls:
                continue

            ref_name = BENCH_REFERENCE.get(mask_type)
            ref_fn = next((f for n, f in bench_impls if n == ref_name), None)
            if ref_fn is None:
                ref_name, ref_fn = bench_impls[0]

            def _make_fwd(fn, qi, ki, vi):
                def fwd():
                    return fn(qi, ki, vi, mask_inputs, scale)
                return fwd

            def _make_fwdbwd(fn, qi, ki, vi):
                def fwdbwd():
                    qi.grad = ki.grad = vi.grad = None
                    out = fn(qi, ki, vi, mask_inputs, scale)
                    out.sum().backward()
                return fwdbwd

            q_r = _clone_requires_grad(q)
            k_r = _clone_requires_grad(k)
            v_r = _clone_requires_grad(v)
            try:
                ref_fwd_ms, _ = time_fn(_make_fwd(ref_fn, q_r, k_r, v_r), warmup, iters)
                ref_fb_ms, _ = time_fn(_make_fwdbwd(ref_fn, q_r, k_r, v_r), warmup, iters)
            except Exception as e:
                print(f"  [{mask_type.value}] {ref_name}: ERROR (reference) — {e}")
                continue

            for name, impl_fn in bench_impls:
                q_i = _clone_requires_grad(q)
                k_i = _clone_requires_grad(k)
                v_i = _clone_requires_grad(v)
                try:
                    fwd_ms, _ = time_fn(_make_fwd(impl_fn, q_i, k_i, v_i), warmup, iters)
                    fb_ms, _ = time_fn(_make_fwdbwd(impl_fn, q_i, k_i, v_i), warmup, iters)
                    bwd_ms = fb_ms - fwd_ms

                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    baseline_mem = torch.cuda.memory_allocated()
                    _make_fwd(impl_fn, q_i, k_i, v_i)()
                    torch.cuda.synchronize()
                    fwd_mb = (torch.cuda.max_memory_allocated() - baseline_mem) / 1e6

                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    baseline_mem = torch.cuda.memory_allocated()
                    _make_fwdbwd(impl_fn, q_i, k_i, v_i)()
                    torch.cuda.synchronize()
                    bwd_mb = (torch.cuda.max_memory_allocated() - baseline_mem) / 1e6

                    fwd_ratio = fwd_ms / ref_fwd_ms
                    fb_ratio = fb_ms / ref_fb_ms
                    label = f"[{mask_type.value}] {name}"
                    print(
                        f"  {label:42s}  {fwd_ms:8.2f}  {bwd_ms:8.2f}  "
                        f"{fwd_mb:7.0f}  {bwd_mb:7.0f}  "
                        f"{fwd_ratio:9.2f}x  {fb_ratio:12.2f}x"
                        + ("  ← ref" if name == ref_name else ""),
                        flush=True,
                    )
                    rows.append(BenchRow(
                        mask_type=mask_type.value, name=name,
                        seq_len=meta.seq_len, doc_len=-1,
                        fwd_ms=fwd_ms, bwd_ms=bwd_ms,
                        fwd_mb=fwd_mb, bwd_mb=bwd_mb,
                        fwd_ratio=fwd_ratio, fwdbwd_ratio=fb_ratio,
                        data_label=meta.data_label(),
                    ))
                except Exception as e:
                    print(f"  [{mask_type.value}] {name}: ERROR — {e}")

    print(f"\n{sep}")
    ref_str = "  ".join(f"{k.value}→{v}" for k, v in BENCH_REFERENCE.items())
    print(f"  ratio = impl / reference  (>1 = slower)   references: {ref_str}")
    print(sep)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--mask-types", nargs="+",
                   default=[m.value for m in MaskType],
                   choices=[m.value for m in MaskType],
                   help="Mask types to test/bench (default: all)")
    p.add_argument("--impls", nargs="+", default=None,
                   help="Impl names to include (default: all). E.g. --impls flex cdb_bim_v12 cdb_bim_v14")
    p.add_argument("--dataset-dir", type=str, default=None,
                   help="Path to pretokenized dataset (enables real-data fixtures for cross_doc)")
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", type=str, default="bfloat16")


def _add_correctness_args(p: argparse.ArgumentParser):
    p.add_argument("--fwd-atol", type=float, default=5e-2,
                   help="Max abs error threshold for forward pass "
                        "(default 5e-2; fwd errors are stable across T, "
                        "typically ~1.6e-2 in bf16)")
    p.add_argument("--bwd-atol", type=float, default=2e-1,
                   help="Max abs error threshold for backward pass "
                        "(default 2e-1, calibrated to PyTorch's own sdpa/flex "
                        "error floor at T=32k on real data; at T<=1024 bwd errors "
                        "are typically <6e-2)")
    p.add_argument("--no-backward", action="store_true",
                   help="Skip backward correctness check")
    p.add_argument("--seq-len", type=int, default=32768,
                   help="Sequence length for correctness test")
    p.add_argument("--doc-len", type=int, default=128,
                   help="Document length for correctness test")


def _add_bench_args(p: argparse.ArgumentParser):
    p.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    p.add_argument("--doc-lens", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--data-source", type=str, default="all",
                   choices=["synthetic", "real", "all"],
                   help="Data to benchmark: synthetic grid, real pack fixtures, or both (default: all)")
    p.add_argument(
        "--autotune-cache",
        type=str,
        default="benchmarks/.autotune_cache.json",
        metavar="PATH",
        help="JSON file to load/save Triton autotune winning configs. "
             "If the file exists the warmup search is skipped and saved configs "
             "are used directly (much faster second run). "
             "Set to '' to disable caching. (default: benchmarks/.autotune_cache.json)",
    )


def cmd_correctness(args):
    dtype = getattr(torch, args.dtype)
    mask_types = [MaskType(m) for m in args.mask_types]
    seq_len = args.seq_len
    doc_len = args.doc_len
    impls_filter = args.impls  # None = all

    any_fail = False

    # ── Synthetic data ────────────────────────────────────────────────────
    _cross_doc_types = {MaskType.CROSS_DOC_NAIVE, MaskType.CROSS_DOC_BITMASK}
    needs_cross = bool(set(mask_types) & _cross_doc_types)
    n_links = max(4, seq_len // doc_len) if needs_cross else 0
    syn_label = f"synthetic(T={seq_len},doc={doc_len},links={n_links})"

    print(f"\n{'='*80}")
    print(f"  DATA SOURCE: {syn_label}")
    print('='*80)

    q, k, v, mask_inputs = make_synthetic_batch(
        seq_len, doc_len, args.num_heads, args.head_dim, dtype, DEVICE,
        n_links=n_links,
    )
    for mask_type in mask_types:
        report = check_correctness(
            mask_type, q, k, v, mask_inputs,
            fwd_atol=args.fwd_atol, bwd_atol=args.bwd_atol,
            check_backward=not args.no_backward,
            data_label=syn_label,
            impls_filter=impls_filter,
        )
        report.print()
        for r in report.results:
            if not r.error and (not r.fwd_pass or not r.bwd_pass):
                any_fail = True

    # ── Real pack fixtures ────────────────────────────────────────────────
    available = [m for m in REAL_PACK_FIXTURES if m.path.exists()]
    if not available:
        print("\n  [real fixtures] skipped — run scripts/generate_pack_fixtures.py first")
    else:
        for meta in available:
            print(f"\n{'='*80}")
            print(f"  DATA SOURCE: {meta.data_label()}")
            print('='*80)
            try:
                q_r, k_r, v_r, mask_r = load_fixture_batch(
                    meta, args.num_heads, args.head_dim, dtype, DEVICE,
                )
            except Exception as e:
                print(f"  [fixture {meta.label}] load error: {e}")
                continue
            for mask_type in mask_types:
                report = check_correctness(
                    mask_type, q_r, k_r, v_r, mask_r,
                    fwd_atol=args.fwd_atol, bwd_atol=args.bwd_atol,
                    check_backward=not args.no_backward,
                    data_label=meta.data_label(),
                    impls_filter=impls_filter,
                )
                report.print()
                for r in report.results:
                    if not r.error and (not r.fwd_pass or not r.bwd_pass):
                        any_fail = True

    if any_fail:
        print("\nSome checks FAILED.")
        sys.exit(1)
    else:
        print("\nAll checks PASSED.")


# ---------------------------------------------------------------------------
# Autotune cache — persist winning Triton configs between runs
# ---------------------------------------------------------------------------
# Triton compiles PTX to disk (~/.triton/cache/) so recompilation is fast.
# But the autotune *search* (benchmarking every config to find the winner)
# runs fresh every process launch.  We persist the winning configs to a JSON
# file so subsequent runs skip the search entirely and just warm up the
# winning compiled kernel.
#
# Format:  { "kernel_name": { "(N, Dh)": {"kwargs": {...}, "num_warps": N, "num_stages": N} } }

def _get_autotune_kernels() -> Dict[str, Any]:
    """Return all @triton.autotune-decorated kernel functions keyed by name.

    Each key is used as the cache key in .autotune_cache.json.  Versioned
    BIM kernels get separate entries so their tuning is tracked independently.
    """
    from kernels.causal_attn import (
        _attn_fwd, _attn_backward_preprocess, _attn_backward,
    )
    from kernels.varlen_attn import (
        _attn_fwd_varlen, _attn_backward_preprocess_varlen, _attn_backward_varlen,
    )
    from kernels.cross_doc_naive_attn import (
        _attn_fwd_cross_doc_naive, _attn_backward_preprocess_cdn, _attn_backward_cdn,
    )
    from kernels.cross_doc_bitmask_attn import (
        _attn_fwd_cdb, _attn_backward_preprocess_cdb, _attn_backward_cdb,
    )
    from kernels.cross_doc_bitmask_bim_v1 import (
        _attn_fwd_cdb_bim_v1, _attn_backward_cdb_bim_v1,
    )
    from kernels.cross_doc_bitmask_bim_v2 import (
        _attn_backward_cdb_bim_v2,
    )
    from kernels.cross_doc_bitmask_bim_v3 import (
        _attn_fwd_cdb_bim_v3, _attn_backward_cdb_bim_v3,
    )
    from kernels.cross_doc_bitmask_bim_v4 import (
        _attn_backward_KV_cdb_bim_v4, _attn_backward_Q_cdb_bim_v4,
    )
    from kernels.cross_doc_bitmask_bim_v5 import (
        _attn_backward_KV_v5, _attn_backward_Q_v5,
    )
    from kernels.cross_doc_bitmask_bim_v6 import (
        _attn_backward_KV_v6, _attn_backward_Q_v6,
    )
    from kernels.cross_doc_bitmask_bim_v7 import (
        _attn_backward_KV_v7, _attn_backward_Q_v7,
    )
    from kernels.cross_doc_bitmask_bim_v8 import _attn_backward_cdb_bim_v8
    from kernels.cross_doc_bitmask_bim_v9 import _attn_backward_KV_v9, _attn_backward_Q_v9
    from kernels.cross_doc_bitmask_bim_v10 import (
        _attn_fwd_cdb_bim_v10,
        _attn_backward_KV_cdb_bim_v10, _attn_backward_Q_cdb_bim_v10,
    )
    from kernels.cross_doc_bitmask_bim_v14 import (
        _attn_backward_KV_cdb_bim_v14, _attn_backward_Q_cdb_bim_v14,
    )
    from kernels.cross_doc_bitmask_bim_v15 import (
        _attn_backward_KV_cdb_bim_v15, _attn_backward_Q_cdb_bim_v15,
    )
    from kernels.cross_doc_bitmask_bim_v16 import (
        _attn_fwd_cdb_bim_v16,
        _attn_backward_KV_cdb_bim_v16, _attn_backward_Q_cdb_bim_v16,
    )
    kernels = {
        "causal_fwd":        _attn_fwd,
        "causal_pre":        _attn_backward_preprocess,
        "causal_bwd":        _attn_backward,
        "varlen_fwd":        _attn_fwd_varlen,
        "varlen_pre":        _attn_backward_preprocess_varlen,
        "varlen_bwd":        _attn_backward_varlen,
        "cdn_fwd":           _attn_fwd_cross_doc_naive,
        "cdn_pre":           _attn_backward_preprocess_cdn,
        "cdn_bwd":           _attn_backward_cdn,
        "cdb_v1_fwd":        _attn_fwd_cdb,
        "cdb_v1_pre":        _attn_backward_preprocess_cdb,
        "cdb_v1_bwd":        _attn_backward_cdb,
        "cdb_bim_v1_fwd":    _attn_fwd_cdb_bim_v1,
        "cdb_bim_v1_bwd":    _attn_backward_cdb_bim_v1,
        # v2 shares v1 forward; only backward is new
        "cdb_bim_v2_bwd":    _attn_backward_cdb_bim_v2,
        "cdb_bim_v3_fwd":    _attn_fwd_cdb_bim_v3,
        "cdb_bim_v3_bwd":    _attn_backward_cdb_bim_v3,
        # v4 shares v3 forward; backward is two separate kernels
        "cdb_bim_v4_KV_bwd": _attn_backward_KV_cdb_bim_v4,
        "cdb_bim_v4_Q_bwd":  _attn_backward_Q_cdb_bim_v4,
        "cdb_bim_v9_KV_bwd": _attn_backward_KV_v9,
        "cdb_bim_v9_Q_bwd":  _attn_backward_Q_v9,
        "cdb_bim_v10_fwd":   _attn_fwd_cdb_bim_v10,
        "cdb_bim_v10_KV_bwd": _attn_backward_KV_cdb_bim_v10,
        "cdb_bim_v10_Q_bwd":  _attn_backward_Q_cdb_bim_v10,
        "cdb_bim_v8_bwd":    _attn_backward_cdb_bim_v8,
        "cdb_bim_v7_KV_bwd": _attn_backward_KV_v7,
        "cdb_bim_v7_Q_bwd":  _attn_backward_Q_v7,
        "cdb_bim_v6_KV_bwd": _attn_backward_KV_v6,
        "cdb_bim_v6_Q_bwd":  _attn_backward_Q_v6,
        "cdb_bim_v5_KV_bwd": _attn_backward_KV_v5,
        "cdb_bim_v5_Q_bwd":  _attn_backward_Q_v5,
        # bs128 shares same kernels; autotune handles BIM_BLOCK_SIZE=128 via key
        "cdb_bim_v14_KV_bwd": _attn_backward_KV_cdb_bim_v14,
        "cdb_bim_v14_Q_bwd":  _attn_backward_Q_cdb_bim_v14,
        "cdb_bim_v15_KV_bwd": _attn_backward_KV_cdb_bim_v15,
        "cdb_bim_v15_Q_bwd":  _attn_backward_Q_cdb_bim_v15,
        "cdb_bim_v16_fwd":    _attn_fwd_cdb_bim_v16,
        "cdb_bim_v16_KV_bwd": _attn_backward_KV_cdb_bim_v16,
        "cdb_bim_v16_Q_bwd":  _attn_backward_Q_cdb_bim_v16,
    }
    return kernels


def save_autotune_cache(path: str) -> int:
    """Serialize each autotuned kernel's winning configs to JSON.

    Returns the number of (kernel, key) entries saved.
    """
    try:
        kernels = _get_autotune_kernels()
    except Exception as e:
        print(f"  [autotune cache] could not import kernels: {e}")
        return 0

    cache: Dict[str, Dict[str, Any]] = {}
    total = 0
    for name, fn in kernels.items():
        fn_cache = getattr(fn, "cache", None)
        if fn_cache is None:
            continue
        entries: Dict[str, Any] = {}
        for key, config in fn_cache.items():
            entries[repr(key)] = {
                "kwargs":      config.kwargs,
                "num_warps":   config.num_warps,
                "num_stages":  config.num_stages,
            }
            total += 1
        if entries:
            cache[name] = entries

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)
    return total


def load_autotune_cache(path: str) -> int:
    """Pre-fill each autotuned kernel's cache from a saved JSON file.

    Returns the number of (kernel, key) entries loaded.  The loaded configs are
    set directly into the in-memory Autotuner.cache dict, bypassing the search
    entirely on the first call for those (N, Dh) combos.
    """
    import triton

    if not Path(path).exists():
        return 0

    try:
        kernels = _get_autotune_kernels()
    except Exception as e:
        print(f"  [autotune cache] could not import kernels: {e}")
        return 0

    with open(path) as f:
        cache = json.load(f)

    total = 0
    for name, entries in cache.items():
        fn = kernels.get(name)
        if fn is None or not hasattr(fn, "cache"):
            continue
        for key_repr, cfg in entries.items():
            try:
                key = ast.literal_eval(key_repr)
                config = triton.Config(cfg["kwargs"],
                                       num_warps=cfg["num_warps"],
                                       num_stages=cfg["num_stages"])
                fn.cache[key] = config
                total += 1
            except Exception:
                pass  # skip malformed entries without crashing

    return total


def warm_all_kernels(
    mask_types: List[MaskType],
    seq_lens: List[int],
    doc_lens: List[int],
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    impls_filter: Optional[List[str]] = None,
) -> None:
    """Pre-trigger Triton autotuning for every (seq_len, doc_len, mask_type) combination.

    Triton's @triton.autotune runs a search over all registered configs on the first
    call for each unique key (N, Dh, ...).  Running this before the timed benchmark
    ensures that the benchmark numbers reflect steady-state kernel performance, not
    the one-time search cost.  Results are cached in-process by Triton so the
    subsequent benchmark calls use the winning config immediately.

    Also warms up torch.compile'd flex_attention so JIT latency doesn't bleed into
    the first benchmark entry.
    """
    scale = head_dim ** -0.5

    # 1. flex_attention compile warmup
    print("  [warmup] flex_attention compile...", flush=True)
    _sq = torch.randn(1, num_heads, min(seq_lens), head_dim, dtype=dtype, device=DEVICE, requires_grad=True)
    _sk, _sv = torch.randn_like(_sq, requires_grad=True), torch.randn_like(_sq, requires_grad=True)
    _bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, B=None, H=None,
                            Q_LEN=min(seq_lens), KV_LEN=min(seq_lens), device=DEVICE)
    _out = _compiled_flex(_sq, _sk, _sv, block_mask=_bm, scale=scale)
    _out.sum().backward()
    torch.cuda.synchronize()

    # 2. Triton kernel autotune at each (seq_len, doc_len)
    for seq_len in seq_lens:
        for doc_len in doc_lens:
            if doc_len > seq_len:
                continue
            print(f"  [warmup] triton autotune  seq_len={seq_len}  doc_len={doc_len}  ...",
                  end=" ", flush=True)
            n_links = max(8, seq_len // doc_len)
            q, k, v, mask_inputs = make_synthetic_batch(
                seq_len, doc_len, num_heads, head_dim, dtype, DEVICE, n_links=n_links,
            )
            _impls_set_w = set(impls_filter) if impls_filter else None
            for mask_type in mask_types:
                bench_impls = [(n, f) for n, f in REGISTRY.get(mask_type, [])
                               if n not in BENCH_SKIP
                               and (_impls_set_w is None or n in _impls_set_w)]
                for _name, impl_fn in bench_impls:
                    qi = _clone_requires_grad(q)
                    ki = _clone_requires_grad(k)
                    vi = _clone_requires_grad(v)
                    try:
                        out = impl_fn(qi, ki, vi, mask_inputs, scale)
                        out.sum().backward()
                        torch.cuda.synchronize()
                    except Exception:
                        pass  # errors will surface again during timed bench
            print("done", flush=True)


def cmd_bench(args):
    dtype = getattr(torch, args.dtype)
    mask_types = [MaskType(m) for m in args.mask_types]
    cache_path = getattr(args, "autotune_cache", "") or ""
    impls_filter = args.impls  # None = all

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    print("=" * 60)
    print("  WARMUP  (Triton autotune + torch.compile)")
    print("=" * 60)

    # Try to load previously saved winning configs — if successful, the
    # autotune search is skipped for those (N, Dh) combos this run.
    cache_loaded = 0
    if cache_path:
        cache_loaded = load_autotune_cache(cache_path)
        if cache_loaded:
            print(f"  [autotune cache] loaded {cache_loaded} winning configs from {cache_path}")
        else:
            print(f"  [autotune cache] no cache found at {cache_path} — will run full search")

    warm_all_kernels(
        mask_types=mask_types,
        seq_lens=args.seq_lens,
        doc_lens=args.doc_lens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dtype=dtype,
        impls_filter=impls_filter,
    )

    # Save winning configs so the next run can skip the search.
    if cache_path:
        n_saved = save_autotune_cache(cache_path)
        print(f"  [autotune cache] saved {n_saved} winning configs → {cache_path}")
    print()

    data_source = args.data_source
    if data_source in ("synthetic", "all"):
        run_benchmarks(
            mask_types=mask_types,
            seq_lens=args.seq_lens,
            doc_lens=args.doc_lens,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            dataset_dir=args.dataset_dir,
            data_source=data_source,
            impls_filter=impls_filter,
        )
    if data_source in ("real", "all"):
        bench_real_fixtures(
            mask_types=mask_types,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            warmup=args.warmup,
            iters=args.iters,
            impls_filter=impls_filter,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # correctness
    p_corr = sub.add_parser("correctness", help="Run correctness checks")
    _add_common_args(p_corr)
    _add_correctness_args(p_corr)

    # bench
    p_bench = sub.add_parser("bench", help="Run memory + runtime benchmarks")
    _add_common_args(p_bench)
    _add_bench_args(p_bench)

    # all
    p_all = sub.add_parser("all", help="Run correctness then benchmarks")
    _add_common_args(p_all)
    _add_correctness_args(p_all)
    _add_bench_args(p_all)

    args = parser.parse_args()

    if args.command == "correctness":
        cmd_correctness(args)
    elif args.command == "bench":
        cmd_bench(args)
    elif args.command == "all":
        cmd_correctness(args)
        cmd_bench(args)


if __name__ == "__main__":
    main()
