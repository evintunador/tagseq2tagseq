"""
Tests proving numerical equivalence of _build_cross_doc_mask vs _build_grant_bitmasks.

The dense method builds an O(seq_len²) bool tensor.
The bitmask method builds two O(seq_len) int64 tensors.
Both encode identical attention grants up to the 63-grant limit.

bitmask_to_dense materialises the bitmask pair as a [T, T] bool tensor using:
    (q_bitmask[:, None] & kv_bitmask[None, :]) != 0

which is non-zero iff the same grant bit is set in both masks — i.e. q is in the
query range AND kv is in the target range of that grant, exactly what the dense
mask encodes.
"""

import random
import pytest
import torch
from dataclasses import dataclass, field
from typing import Dict, List

from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector


# ---------------------------------------------------------------------------
# Helpers shared by all tests
# ---------------------------------------------------------------------------

@dataclass
class MockDocSpan:
    doc_id: int
    raw_identifier: str
    start: int
    end: int
    truncated: bool = False
    outgoing_identifiers: List[str] = field(default_factory=list)


def _dummy_creator(max_grants: int = 64) -> CrossDocLinkMaskCreator:
    """Return a CrossDocLinkMaskCreator; the link_detector is not called in these tests."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return CrossDocLinkMaskCreator(
        link_detector=MarkdownLinkDetector(decode_fn=enc.decode),
        max_grants=max_grants,
    )


def bitmask_to_dense(
    q_bitmasks: List[torch.Tensor],
    kv_bitmasks: List[torch.Tensor],
) -> torch.Tensor:
    """Materialise the bitmask lists as a [T, T] bool tensor.

    ORs across all chunks: non-zero iff the same grant bit is set in both
    bitmasks for some chunk, i.e. there exists a grant k such that q is in
    grant_k_q_range and kv is in grant_k_kv_range.
    """
    result = (q_bitmasks[0][:, None] & kv_bitmasks[0][None, :]) != 0
    for q_bm, kv_bm in zip(q_bitmasks[1:], kv_bitmasks[1:]):
        result = result | ((q_bm[:, None] & kv_bm[None, :]) != 0)
    return result


def _both_masks(
    creator: CrossDocLinkMaskCreator,
    seq_len: int,
    doc_spans: List[MockDocSpan],
    link_to_target: Dict[int, List[int]],
    device: torch.device,
):
    """Call both internal methods with the same arguments and return (dense, bitmask_dense)."""
    dense = creator._build_cross_doc_mask(seq_len, doc_spans, link_to_target, device)
    q_bms, kv_bms = creator._build_grant_bitmasks(seq_len, doc_spans, link_to_target, device)
    bitmask_dense = bitmask_to_dense(q_bms, kv_bms)
    return dense, bitmask_dense


def _make_n_grant_scenario(n_grants: int):
    """Build a scenario with exactly n_grants resolved cross-doc links.

    One shared target doc (doc 0, 4 tokens) and n_grants source docs
    (each 4 tokens wide), each contributing exactly one link to doc 0.
    Source docs have non-overlapping spans so no aliasing is possible.
    """
    target_width = 4
    src_width = 4
    seq_len = target_width + n_grants * src_width
    doc_spans = [MockDocSpan(doc_id=0, raw_identifier="target", start=0, end=target_width)]
    pos = target_width
    for i in range(1, n_grants + 1):
        doc_spans.append(MockDocSpan(
            doc_id=i, raw_identifier=f"src_{i}", start=pos, end=pos + src_width,
        ))
        pos += src_width
    # link_pos = span.start + 1 is inside (start, end], giving a non-empty grant range
    link_to_target = {span.start + 1: [0] for span in doc_spans[1:]}
    return seq_len, doc_spans, link_to_target


CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Synthetic unit tests
# ---------------------------------------------------------------------------

class TestSyntheticEquivalence:
    """Synthetic cases comparing _build_cross_doc_mask to _build_grant_bitmasks."""

    def setup_method(self):
        self.creator = _dummy_creator()

    def test_empty_no_links(self):
        seq_len = 20
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=10),
            MockDocSpan(doc_id=1, raw_identifier="B", start=10, end=20),
        ]
        link_to_target: Dict[int, List[int]] = {}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert not dense.any(), "Dense mask should be all-False with no links"
        assert torch.equal(dense, bitmask_dense)

    def test_single_grant(self):
        """One link at position 8 in doc B (tokens 5–15), targeting doc A (tokens 0–5)."""
        seq_len = 16
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=5),
            MockDocSpan(doc_id=1, raw_identifier="B", start=5, end=16),
        ]
        # link_pos=8 is inside span B (5 < 8 <= 16); target is doc 0
        link_to_target = {8: [0]}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert torch.equal(dense, bitmask_dense)
        # Spot check: q=8, kv=2 should be True (8 in [8,16), 2 in [0,5))
        assert dense[8, 2].item() is True
        assert bitmask_dense[8, 2].item() is True
        # q=5 (before link_pos), kv=2 should be False
        assert dense[5, 2].item() is False
        assert bitmask_dense[5, 2].item() is False

    def test_two_nonoverlapping_grants(self):
        """Two links with completely separate q- and kv-ranges."""
        seq_len = 30
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=5),
            MockDocSpan(doc_id=1, raw_identifier="B", start=5, end=15),
            MockDocSpan(doc_id=2, raw_identifier="C", start=15, end=30),
        ]
        # Doc B links to doc A at pos 10; Doc C links to doc B at pos 20
        link_to_target = {10: [0], 20: [1]}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert torch.equal(dense, bitmask_dense)

    def test_overlapping_q_ranges(self):
        """Same source doc has two links → same query range, two different target ranges."""
        seq_len = 30
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=5),
            MockDocSpan(doc_id=1, raw_identifier="B", start=5, end=10),
            MockDocSpan(doc_id=2, raw_identifier="C", start=10, end=30),
        ]
        # Doc C links to both A (at pos 15) and B (at pos 18) — both in [10, 30)
        link_to_target = {15: [0], 18: [1]}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert torch.equal(dense, bitmask_dense)
        # q=20 can attend to kv=3 (in A) and kv=7 (in B)
        assert dense[20, 3].item() is True
        assert dense[20, 7].item() is True

    def test_overlapping_kv_ranges(self):
        """Two different source docs both link to the same target doc."""
        seq_len = 30
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=5),
            MockDocSpan(doc_id=1, raw_identifier="B", start=5, end=15),
            MockDocSpan(doc_id=2, raw_identifier="C", start=15, end=30),
        ]
        # Both B and C link to A
        link_to_target = {10: [0], 20: [0]}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert torch.equal(dense, bitmask_dense)

    def test_link_at_span_end_boundary(self):
        """link_pos == span.end — containment check is span.start < link_pos <= span.end."""
        seq_len = 20
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=5),
            MockDocSpan(doc_id=1, raw_identifier="B", start=5, end=12),
        ]
        # link_pos exactly at span.end of B: 5 < 12 <= 12, so still inside B
        link_to_target = {12: [0]}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert torch.equal(dense, bitmask_dense)
        # grant_start = 12, grant_end = min(20, 12) = 12 → empty range → no grant
        assert not dense.any()

    def test_link_just_before_span_end(self):
        """link_pos == span.end - 1 — strictly within span, non-empty grant range."""
        seq_len = 20
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=5),
            MockDocSpan(doc_id=1, raw_identifier="B", start=5, end=12),
        ]
        # link_pos=11, span.end=12 → grant_start=11, grant_end=12 (one token)
        link_to_target = {11: [0]}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert torch.equal(dense, bitmask_dense)
        # Only position 11 should have cross-doc access, not 10
        assert dense[11, 2].item() is True
        assert dense[10, 2].item() is False

    def test_three_doc_chain(self):
        """A→∅, B→A, C→{A,B}: exercises multi-grant with different q/kv combinations."""
        seq_len = 30
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=8),
            MockDocSpan(doc_id=1, raw_identifier="B", start=8, end=18),
            MockDocSpan(doc_id=2, raw_identifier="C", start=18, end=30),
        ]
        # B links to A at pos 12; C links to A at pos 22 and B at pos 25
        link_to_target = {12: [0], 22: [0], 25: [1]}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert torch.equal(dense, bitmask_dense)

    def test_empty_grant_range(self):
        """link_pos past span.end → grant_start >= grant_end → treated as no-op."""
        seq_len = 20
        doc_spans = [
            MockDocSpan(doc_id=0, raw_identifier="A", start=0, end=5),
            MockDocSpan(doc_id=1, raw_identifier="B", start=5, end=10),
        ]
        # link_pos=15 is not inside any span (no span has start < 15 <= end for the spans above)
        # Both implementations should emit a warning and produce no grant
        link_to_target = {15: [0]}
        dense, bitmask_dense = _both_masks(self.creator, seq_len, doc_spans, link_to_target, CPU)
        assert torch.equal(dense, bitmask_dense)
        assert not dense.any()


# ---------------------------------------------------------------------------
# 2. Fuzz tests (random, CPU)
# ---------------------------------------------------------------------------

def _random_scenario(rng: random.Random):
    """Generate a random (seq_len, doc_spans, link_to_target) tuple."""
    n_docs = rng.randint(2, 8)
    # Assign random widths summing to seq_len, minimum 2 tokens per doc
    widths = [rng.randint(2, 20) for _ in range(n_docs)]
    seq_len = sum(widths)

    doc_spans = []
    pos = 0
    for doc_id, w in enumerate(widths):
        doc_spans.append(MockDocSpan(
            doc_id=doc_id,
            raw_identifier=f"doc_{doc_id}",
            start=pos,
            end=pos + w,
        ))
        pos += w

    # Draw random links; each link is (link_pos, target_doc_id) with DAG property
    n_links = rng.randint(1, 10)
    link_to_target: Dict[int, List[int]] = {}
    for _ in range(n_links):
        src_doc_id = rng.randint(1, n_docs - 1)   # must have at least one earlier doc
        src_span = doc_spans[src_doc_id]
        # link_pos inside (start, end]
        link_pos = rng.randint(src_span.start + 1, src_span.end)
        # target must be strictly earlier
        tgt_doc_id = rng.randint(0, src_doc_id - 1)
        link_to_target.setdefault(link_pos, []).append(tgt_doc_id)

    return seq_len, doc_spans, link_to_target


@pytest.mark.parametrize("seed", range(20))
def test_fuzz_equivalence(seed):
    """Random batches: bitmask_to_dense must match _build_cross_doc_mask element-wise."""
    creator = _dummy_creator()
    rng = random.Random(seed)
    seq_len, doc_spans, link_to_target = _random_scenario(rng)

    dense, bitmask_dense = _both_masks(creator, seq_len, doc_spans, link_to_target, CPU)
    assert torch.equal(dense, bitmask_dense), (
        f"seed={seed}: dense and bitmask differ at "
        f"{(dense != bitmask_dense).nonzero()[:5].tolist()}"
    )


# ---------------------------------------------------------------------------
# 3. Dataset integration tests (conditional on simplewiki being present)
# ---------------------------------------------------------------------------

SIMPLEWIKI_DIR = "data/pretokenized_datasets/simplewiki"


def _simplewiki_available() -> bool:
    import os
    return os.path.isdir(SIMPLEWIKI_DIR)


@pytest.mark.skipif(not _simplewiki_available(), reason="simplewiki dataset not present")
def test_real_batches_equivalence():
    """5 real simplewiki batches: bitmask method must match dense method element-wise."""
    import itertools
    import tiktoken
    from data.dataset import GraphIndex, PretokShardedBackend
    from data.packed_dataset import PackedSequenceDataset
    from data.pack_sampler import PackBatchSampler
    from data.traversal import DFSStrategy
    from data.layout import make_layout_policy

    enc = tiktoken.get_encoding("gpt2")
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    graph_index = GraphIndex(SIMPLEWIKI_DIR)
    backend = PretokShardedBackend(graph_index)
    layout_policy = make_layout_policy("null", encode_fn=enc.encode_ordinary)
    pack_sampler = PackBatchSampler(
        graph=graph_index,
        strategy_factory=lambda: DFSStrategy(edge_mode="outgoing"),
        token_budget=512,
        seed=42,
        layout_policy=layout_policy,
    )
    dataset = PackedSequenceDataset(
        graph=graph_index,
        backend=backend,
        pack_sampler=pack_sampler,
        layout_policy=layout_policy,
    )

    for batch_idx, batch in enumerate(itertools.islice(dataset, 5)):
        tokens = batch["tokens"]  # [1, T]
        doc_spans = batch["doc_spans"]

        # Detect links and match to targets exactly as __call__ would
        input_ids = tokens[0]
        links = detector.detect_links(input_ids)
        link_to_target = creator._match_links_to_docs(links, doc_spans)

        seq_len = tokens.shape[-1]
        dense, bitmask_dense = _both_masks(creator, seq_len, doc_spans, link_to_target, CPU)

        assert torch.equal(dense, bitmask_dense), (
            f"Batch {batch_idx}: dense and bitmask differ at "
            f"{(dense != bitmask_dense).nonzero()[:5].tolist()}"
        )


# ---------------------------------------------------------------------------
# 4. Grant-limit equivalence: parametrized over 64 / 128 / 256
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_grants", [64, 128, 256])
def test_at_capacity(max_grants):
    """Exactly max_grants grants → bitmask == dense (all grants encoded)."""
    creator = _dummy_creator(max_grants=max_grants)
    seq_len, doc_spans, link_to_target = _make_n_grant_scenario(max_grants)
    dense, bitmask_dense = _both_masks(creator, seq_len, doc_spans, link_to_target, CPU)
    assert torch.equal(dense, bitmask_dense), (
        f"max_grants={max_grants}: at-capacity scenario must be exactly equal"
    )


@pytest.mark.parametrize("max_grants", [64, 128, 256])
def test_one_over_limit_truncates(max_grants):
    """max_grants+1 grants → bitmask silently drops the last grant, diverging from dense."""
    creator = _dummy_creator(max_grants=max_grants)
    seq_len, doc_spans, link_to_target = _make_n_grant_scenario(max_grants + 1)
    dense, bitmask_dense = _both_masks(creator, seq_len, doc_spans, link_to_target, CPU)
    assert not torch.equal(dense, bitmask_dense), (
        f"max_grants={max_grants}: one-over-limit scenario must diverge "
        "(bitmask truncates the last grant, dense does not)"
    )


@pytest.mark.parametrize("n_grants,max_grants,expect_equal", [
    # within each limit
    (64,  64,  True),
    (128, 128, True),
    (256, 256, True),
    # one over each limit — truncation fires
    (65,  64,  False),
    (129, 128, False),
    (257, 256, False),
    # a count that busts a lower limit but fits in the next higher one
    (65,  128, True),
    (129, 256, True),
])
def test_truncation_boundary(n_grants, max_grants, expect_equal):
    """Cross-limit matrix: verify equal/unequal at each (n_grants, max_grants) pair."""
    creator = _dummy_creator(max_grants=max_grants)
    seq_len, doc_spans, link_to_target = _make_n_grant_scenario(n_grants)
    dense, bitmask_dense = _both_masks(creator, seq_len, doc_spans, link_to_target, CPU)
    if expect_equal:
        assert torch.equal(dense, bitmask_dense), (
            f"n_grants={n_grants}, max_grants={max_grants}: expected equal"
        )
    else:
        assert not torch.equal(dense, bitmask_dense), (
            f"n_grants={n_grants}, max_grants={max_grants}: expected unequal (truncation)"
        )
