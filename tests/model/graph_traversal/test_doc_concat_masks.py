"""
Tests for the two concat-baseline mask conditions:

  * ``doc_concatenated``  — merge each connected document component into one
    causally concatenated super-doc. Built by ``create_doc_concat_*`` from
    ``component_id``; reuses the doc-causal varlen kernel.
  * ``doc_concat_link``   — a detected link concatenates the *entire* source doc
    onto its target. Built by ``CrossDocLinkMaskCreator(whole_doc_grant=True)``;
    reuses the cross-doc-link grant kernel.

These run on CPU (dense / flex BlockMask reconstruction); they do not require CUDA.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List

import pytest
import torch

from model.graph_traversal.cross_doc_mask import (
    CrossDocLinkMaskCreator,
    _kv_block_count_analytical,
)
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
from model.graph_traversal.block_mask_creator import (
    _build_component_document_ids,
    create_doc_concat_triton_mask,
)

CPU = torch.device("cpu")


@dataclass
class MockDocSpan:
    doc_id: int
    raw_identifier: str
    start: int
    end: int
    truncated: bool = False
    outgoing_identifiers: List[str] = field(default_factory=list)
    component_id: int = -1


def _creator(whole_doc_grant: bool, max_grants: int = 64) -> CrossDocLinkMaskCreator:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return CrossDocLinkMaskCreator(
        link_detector=MarkdownLinkDetector(decode_fn=enc.decode),
        max_grants=max_grants,
        whole_doc_grant=whole_doc_grant,
    )


# ---------------------------------------------------------------------------
# 1. whole_doc_grant grant semantics (doc_concat_link)
# ---------------------------------------------------------------------------

class TestWholeDocGrant:
    def test_grant_spans_whole_source_doc(self):
        """whole_doc_grant grants from source.start; gated grants from link_pos."""
        seq_len = 16
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=5),
            MockDocSpan(doc_id=1, raw_identifier="B", start=5, end=16),
        ]
        link_to_target = {8: [0]}  # link at pos 8 inside B, target doc A

        gated = _creator(whole_doc_grant=False)._build_cross_doc_mask(
            seq_len, doc_spans, link_to_target, CPU
        )
        whole = _creator(whole_doc_grant=True)._build_cross_doc_mask(
            seq_len, doc_spans, link_to_target, CPU
        )

        # Gated: only positions [8, 16) of B can see A.
        assert gated[8, 2].item() is True
        assert gated[5, 2].item() is False   # before the link position
        assert gated[7, 2].item() is False

        # Whole-doc: the ENTIRE source doc B [5, 16) can see A.
        assert whole[5, 2].item() is True
        assert whole[7, 2].item() is True
        assert whole[8, 2].item() is True
        # Positions before B (i.e. in A itself) still cannot see A via a grant.
        assert whole[4, 2].item() is False

        # whole_doc strictly contains gated.
        assert (whole | gated).equal(whole)
        assert whole.sum() > gated.sum()

    def test_whole_doc_grant_bitmasks_match_dense(self):
        """The bitmask path agrees with the dense path under whole_doc_grant."""
        seq_len = 30
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=8),
            MockDocSpan(doc_id=1, raw_identifier="B", start=8, end=18),
            MockDocSpan(doc_id=2, raw_identifier="C", start=18, end=30),
        ]
        link_to_target = {12: [0], 22: [0], 25: [1]}
        c = _creator(whole_doc_grant=True)
        dense = c._build_cross_doc_mask(seq_len, doc_spans, link_to_target, CPU)
        q_bms, kv_bms = c._build_grant_bitmasks(seq_len, doc_spans, link_to_target, CPU)
        bitmask_dense = (q_bms[0][:, None] & kv_bms[0][None, :]) != 0
        for q_bm, kv_bm in zip(q_bms[1:], kv_bms[1:]):
            bitmask_dense = bitmask_dense | ((q_bm[:, None] & kv_bm[None, :]) != 0)
        assert torch.equal(dense, bitmask_dense)

    def test_analytical_count_matches_geometry_whole_doc(self):
        """_kv_block_count_analytical(whole_doc_grant=True) counts the wider grant."""
        seq_len = 384
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=128),
            MockDocSpan(doc_id=1, raw_identifier="B", start=128, end=384),
        ]
        link_to_target = {300: [0]}
        gated = _kv_block_count_analytical(doc_spans, link_to_target, seq_len, whole_doc_grant=False)
        whole = _kv_block_count_analytical(doc_spans, link_to_target, seq_len, whole_doc_grant=True)
        # The whole-doc grant covers more query blocks, never fewer.
        assert whole >= gated


# ---------------------------------------------------------------------------
# 2. component relabeling + contiguity assertion (doc_concatenated)
# ---------------------------------------------------------------------------

class TestComponentDocumentIds:
    def test_contiguous_components_merge(self):
        tokens = torch.zeros(1, 6, dtype=torch.long)
        spans = [
            MockDocSpan(10, "a", 0, 2, component_id=0),
            MockDocSpan(11, "b", 2, 4, component_id=0),
            MockDocSpan(20, "c", 4, 6, component_id=1),
        ]
        ids = _build_component_document_ids(tokens, spans)
        assert ids.tolist() == [0, 0, 0, 0, 1, 1]

    def test_interleaved_components_raise(self):
        tokens = torch.zeros(1, 6, dtype=torch.long)
        spans = [
            MockDocSpan(10, "a", 0, 2, component_id=0),
            MockDocSpan(20, "b", 2, 4, component_id=1),
            MockDocSpan(11, "c", 4, 6, component_id=0),  # component 0 split!
        ]
        with pytest.raises(AssertionError, match="contiguous"):
            _build_component_document_ids(tokens, spans)

    def test_component_id_minus_one_falls_back_to_doc_id(self):
        tokens = torch.zeros(1, 6, dtype=torch.long)
        spans = [
            MockDocSpan(10, "a", 0, 3, component_id=-1),
            MockDocSpan(20, "b", 3, 6, component_id=-1),
        ]
        ids = _build_component_document_ids(tokens, spans)
        assert ids.tolist() == [10, 10, 10, 20, 20, 20]

    def test_triton_mask_inputs_carry_component_ids(self):
        tokens = torch.zeros(1, 6, dtype=torch.long)
        spans = [
            MockDocSpan(10, "a", 0, 3, component_id=7),
            MockDocSpan(11, "b", 3, 6, component_id=7),
        ]
        out = create_doc_concat_triton_mask(tokens, spans)
        assert out.document_ids.tolist() == [7, 7, 7, 7, 7, 7]

    def test_concat_merges_more_than_doc_causal(self):
        """Two linked docs in one component attend across the boundary; doc_causal does not."""
        tokens = torch.zeros(1, 6, dtype=torch.long)
        spans = [
            MockDocSpan(10, "a", 0, 3, component_id=0),
            MockDocSpan(11, "b", 3, 6, component_id=0),
        ]
        comp_ids = _build_component_document_ids(tokens, spans)
        # doc-causal labels positions by doc_id.
        doc_ids = torch.tensor([10, 10, 10, 11, 11, 11])
        q = torch.arange(6).unsqueeze(1)
        kv = torch.arange(6).unsqueeze(0)
        causal = q >= kv
        concat_mask = causal & (comp_ids.unsqueeze(1) == comp_ids.unsqueeze(0))
        doc_causal_mask = causal & (doc_ids.unsqueeze(1) == doc_ids.unsqueeze(0))
        # Position 4 (doc B) can attend to position 1 (doc A) under concat, not doc_causal.
        assert concat_mask[4, 1].item() is True
        assert doc_causal_mask[4, 1].item() is False
        # concat is a strict superset of doc_causal.
        assert (concat_mask | doc_causal_mask).equal(concat_mask)


# ---------------------------------------------------------------------------
# 3. flex BlockMask agrees with the dense reconstruction (CUDA only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_doc_concat_flex_blockmask_matches_dense():
    """Run flex_attention with the concat BlockMask and compare to a manually
    masked reference. BlockMask.to_dense() is block-granular (128-block), so we
    validate at element granularity via the actual attention output instead.
    """
    from torch.nn.attention.flex_attention import flex_attention
    from model.graph_traversal.block_mask_creator import create_doc_concat_block_mask

    device = torch.device("cuda")
    # Use a seq_len > one flex block so the mask actually partitions the sequence.
    seq_len = 256
    comp_ids = torch.zeros(seq_len, dtype=torch.long)
    comp_ids[:160] = 0   # component 0 spans docs A+B (a merged super-doc)
    comp_ids[160:] = 1   # component 1 is a separate doc
    spans = [
        MockDocSpan(0, "a", 0, 96, component_id=0),
        MockDocSpan(1, "b", 96, 160, component_id=0),
        MockDocSpan(2, "c", 160, seq_len, component_id=1),
    ]
    tokens = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    bm = create_doc_concat_block_mask(tokens, spans)

    torch.manual_seed(0)
    H, D = 2, 32
    q = torch.randn(1, H, seq_len, D, device=device, dtype=torch.float32)
    k = torch.randn(1, H, seq_len, D, device=device, dtype=torch.float32)
    v = torch.randn(1, H, seq_len, D, device=device, dtype=torch.float32)

    out_flex = flex_attention(q, k, v, block_mask=bm)

    # Reference: dense masked attention with the same causal + same-component mask.
    comp = comp_ids.to(device)
    qi = torch.arange(seq_len, device=device).unsqueeze(1)
    ki = torch.arange(seq_len, device=device).unsqueeze(0)
    allow = (qi >= ki) & (comp.unsqueeze(1) == comp.unsqueeze(0))   # [T, T]
    scores = (q @ k.transpose(-1, -2)) / (D ** 0.5)                 # [1, H, T, T]
    scores = scores.masked_fill(~allow, float("-inf"))
    out_ref = torch.softmax(scores, dim=-1) @ v

    assert torch.allclose(out_flex, out_ref, atol=1e-3, rtol=1e-3)
