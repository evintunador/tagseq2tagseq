"""
Tests for the three kv_block_count methods (Step 1 of the density-aware
batching plan).

Method A — _kv_block_count_from_dense : build full dense bool mask, count blocks
Method B — block_mask.kv_num_blocks.sum() : GPU FlexAttention BlockMask (ground truth)
Method C — _kv_block_count_analytical   : set-based analytical count (exact)

A and C both run on CPU; B requires CUDA.  Tests that compare against B are
skipped when CUDA is unavailable.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List

import pytest
import torch

from model.graph_traversal.cross_doc_mask import (
    CrossDocLinkMaskCreator,
    _kv_block_count_analytical,
    _kv_block_count_from_dense,
)
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector

cuda_available = torch.cuda.is_available()
CUDA_MARK = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

SIMPLEWIKI_DIR = "data/pretokenized_datasets/simplewiki"
STACK10M_DIR = "data/pretokenized_datasets/stack_10m"


# ---------------------------------------------------------------------------
# Shared helpers (mirrored from test_grant_bitmask_equivalence.py)
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
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return CrossDocLinkMaskCreator(
        link_detector=MarkdownLinkDetector(decode_fn=enc.decode),
        max_grants=max_grants,
    )


def _build_full_dense_mask(
    creator: CrossDocLinkMaskCreator,
    seq_len: int,
    doc_spans: List[MockDocSpan],
    link_to_target: Dict[int, List[int]],
    device: torch.device,
) -> torch.Tensor:
    """Build full causal+same_doc+cross_doc dense mask [seq_len, seq_len]."""
    cross_doc = creator._build_cross_doc_mask(seq_len, doc_spans, link_to_target, device)
    document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
    for span in doc_spans:
        s, e = max(0, span.start), min(seq_len, span.end)
        if s < e:
            document_ids[s:e] = span.doc_id
    q = torch.arange(seq_len, device=device).unsqueeze(1)
    k = torch.arange(seq_len, device=device).unsqueeze(0)
    causal = q >= k
    same_doc = document_ids.unsqueeze(1) == document_ids.unsqueeze(0)
    return causal & (same_doc | cross_doc)


def _method_b(
    creator: CrossDocLinkMaskCreator,
    seq_len: int,
    doc_spans: List[MockDocSpan],
    link_to_target: Dict[int, List[int]],
) -> int:
    """Method B: total non-empty blocks from BlockMask on CUDA.

    FlexAttention splits non-empty blocks into two categories:
      kv_num_blocks       — partial blocks (mixed masked/unmasked entries)
      full_kv_num_blocks  — full blocks   (entire tile is unmasked)
    The backward FLOP cost depends on the SUM of both.
    """
    device = torch.device("cuda")
    tokens = torch.zeros(1, seq_len, dtype=torch.long, device=device)
    block_mask = creator(tokens, doc_spans, link_to_target=link_to_target)
    total = block_mask.kv_num_blocks.sum() + block_mask.full_kv_num_blocks.sum()
    return int(total.item())


def _random_scenario(rng: random.Random):
    """Generate a random (seq_len, doc_spans, link_to_target) tuple."""
    n_docs = rng.randint(2, 8)
    widths = [rng.randint(2, 20) for _ in range(n_docs)]
    seq_len = sum(widths)
    doc_spans = []
    pos = 0
    for doc_id, w in enumerate(widths):
        doc_spans.append(MockDocSpan(
            doc_id=doc_id, raw_identifier=f"doc_{doc_id}",
            start=pos, end=pos + w,
        ))
        pos += w
    n_links = rng.randint(1, 10)
    link_to_target: Dict[int, List[int]] = {}
    for _ in range(n_links):
        src = rng.randint(1, n_docs - 1)
        src_span = doc_spans[src]
        link_pos = rng.randint(src_span.start + 1, src_span.end)
        tgt = rng.randint(0, src - 1)
        link_to_target.setdefault(link_pos, []).append(tgt)
    return seq_len, doc_spans, link_to_target


def _block_aligned_scenario():
    """Three docs each occupying exactly one 128-token block; one cross-doc link.

    Block layout: [0,128) doc0 | [128,256) doc1 | [256,384) doc2
    Link at pos 260 (inside doc2), target is doc0.
    Grant q-range: [260, 384), kv-range: [0, 128).

    Expected non-empty blocks:
      same_doc:    (0,0), (1,1), (2,2)
      cross_doc:   (2,0)
    Total = 4.
    """
    doc_spans = [
        MockDocSpan(doc_id=0, raw_identifier="A", start=0,   end=128),
        MockDocSpan(doc_id=1, raw_identifier="B", start=128, end=256),
        MockDocSpan(doc_id=2, raw_identifier="C", start=256, end=384),
    ]
    link_to_target = {260: [0]}
    seq_len = 384
    return seq_len, doc_spans, link_to_target, 4


# ---------------------------------------------------------------------------
# TestKVBlockCountMethods
# ---------------------------------------------------------------------------

class TestKVBlockCountMethods:

    def setup_method(self):
        self.creator = _dummy_creator()
        self.cpu = torch.device("cpu")

    # ---- A == B (CUDA required) ----------------------------------------

    @CUDA_MARK
    def test_dense_matches_blockmask_synthetic(self):
        """Method A == Method B on block-aligned scenario."""
        seq_len, doc_spans, link_to_target, expected = _block_aligned_scenario()
        full_mask = _build_full_dense_mask(
            self.creator, seq_len, doc_spans, link_to_target, self.cpu
        )
        a = _kv_block_count_from_dense(full_mask)
        b = _method_b(self.creator, seq_len, doc_spans, link_to_target)
        assert a == expected, f"Method A={a}, expected={expected}"
        assert a == b, f"Method A={a} != Method B={b}"

    @CUDA_MARK
    def test_analytical_matches_blockmask_synthetic(self):
        """Method C == Method B on block-aligned scenario."""
        seq_len, doc_spans, link_to_target, expected = _block_aligned_scenario()
        c = _kv_block_count_analytical(doc_spans, link_to_target, seq_len)
        b = _method_b(self.creator, seq_len, doc_spans, link_to_target)
        assert c == expected, f"Method C={c}, expected={expected}"
        assert c == b, f"Method C={c} != Method B={b}"

    # ---- A == C (CPU only, always runs) -----------------------------------

    @pytest.mark.parametrize("seed", range(20))
    def test_dense_equals_analytical_fuzz(self, seed):
        """Method A == Method C on random scenarios (CPU, no CUDA needed)."""
        rng = random.Random(seed)
        seq_len, doc_spans, link_to_target = _random_scenario(rng)
        full_mask = _build_full_dense_mask(
            self.creator, seq_len, doc_spans, link_to_target, self.cpu
        )
        a = _kv_block_count_from_dense(full_mask)
        c = _kv_block_count_analytical(doc_spans, link_to_target, seq_len)
        assert a == c, (
            f"seed={seed}: Method A={a} != Method C={c} for seq_len={seq_len}"
        )

    # ---- all three (CUDA required) ----------------------------------------

    @CUDA_MARK
    @pytest.mark.parametrize("seed", range(20))
    def test_fuzz_all_methods_agree(self, seed):
        """All three methods agree on random scenarios (requires CUDA for B)."""
        rng = random.Random(seed)
        seq_len, doc_spans, link_to_target = _random_scenario(rng)
        full_mask = _build_full_dense_mask(
            self.creator, seq_len, doc_spans, link_to_target, self.cpu
        )
        a = _kv_block_count_from_dense(full_mask)
        b = _method_b(self.creator, seq_len, doc_spans, link_to_target)
        c = _kv_block_count_analytical(doc_spans, link_to_target, seq_len)
        assert a == b, f"seed={seed}: A={a} != B={b}"
        assert a == c, f"seed={seed}: A={a} != C={c}"

    # ---- real dataset batches (CUDA + dataset required) -------------------

    @CUDA_MARK
    @pytest.mark.skipif(
        not __import__("os").path.isdir(SIMPLEWIKI_DIR),
        reason="simplewiki dataset not present",
    )
    def test_real_batches_all_methods_agree(self):
        """5 real simplewiki batches: all three methods must agree."""
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

        graph = GraphIndex(SIMPLEWIKI_DIR)
        backend = PretokShardedBackend(graph)
        layout = make_layout_policy("null", encode_fn=enc.encode_ordinary)
        sampler = PackBatchSampler(
            graph=graph,
            strategy_factory=lambda: DFSStrategy(edge_mode="outgoing"),
            token_budget=512,
            seed=42,
            layout_policy=layout,
        )
        dataset = PackedSequenceDataset(
            graph=graph, backend=backend, pack_sampler=sampler,
            layout_policy=layout,
        )

        cpu = torch.device("cpu")
        for batch in itertools.islice(dataset, 5):
            tokens = batch["tokens"]
            doc_spans = batch["doc_spans"]
            seq_len = tokens.shape[-1]
            links = detector.detect_links(tokens[0])
            link_to_target = creator._match_links_to_docs(links, doc_spans)

            full_mask = _build_full_dense_mask(creator, seq_len, doc_spans, link_to_target, cpu)
            a = _kv_block_count_from_dense(full_mask)
            b = _method_b(creator, seq_len, doc_spans, link_to_target)
            c = _kv_block_count_analytical(doc_spans, link_to_target, seq_len)

            assert a == b, f"real batch: A={a} != B={b}"
            assert a == c, f"real batch: A={a} != C={c}"

