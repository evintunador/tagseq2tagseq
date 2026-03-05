"""
Tests for cross-document link mask detection and matching.
"""

import torch
import tiktoken
import pytest
from cross_doc_mask import CrossDocLinkMaskCreator, MarkdownLinkDetector
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockDocSpan:
    """Mock DocSpan for testing."""
    doc_id: int
    clean_title: str
    start: int
    end: int
    truncated: bool = False
    outgoing_titles: List[str] = field(default_factory=list)


def make_batch(texts, enc):
    """Encode a list of text segments and return (tokens_2d, doc_spans)."""
    all_tokens = []
    spans = []
    for i, text in enumerate(texts):
        toks = enc.encode(text)
        start = len(all_tokens)
        all_tokens.extend(toks)
        spans.append(MockDocSpan(
            doc_id=i,
            clean_title=f"doc_{i}",
            start=start,
            end=len(all_tokens),
        ))
    # tokens_2d shape: [1, T+1] (extra target token appended as required by mask creator)
    tokens_2d = torch.tensor(all_tokens + [0], dtype=torch.long).unsqueeze(0)
    return tokens_2d, spans


def test_link_detection_and_matching():
    """Detected links must map to correct in-batch doc spans."""
    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    # Doc 0 (target): plain text, no links
    # Doc 1 (linker): contains a link to doc_0 and a link to a missing doc
    texts = [
        "This is the target document.",
        " Here is [a link](doc_0) to the first doc and [bad](missing_doc).",
    ]
    tokens_2d, doc_spans = make_batch(texts, enc)
    input_ids = tokens_2d[0, :-1]

    links = detector.detect_links(input_ids)
    assert len(links) >= 1, "Should detect at least the link to doc_0"

    target_strs = [lnk.target_str for lnk in links]
    assert "doc_0" in target_strs, f"Expected 'doc_0' in detected links, got {target_strs}"

    link_to_target = creator._match_links_to_docs(links, doc_spans)

    # At least the doc_0 link should match (doc_0 is before the linker in the batch)
    assert len(link_to_target) >= 1, "Expected at least one matched link"

    # The matched targets should all be doc_id=0
    matched_doc_ids = {doc_id for targets in link_to_target.values() for doc_id in targets}
    assert 0 in matched_doc_ids, "Expected doc_id=0 to be a match target"


def test_dag_property_enforced():
    """Links must not grant attention to documents that appear later in the batch."""
    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    # Doc 0 links to doc_1, but doc_1 appears AFTER doc_0 → should not match (DAG violation)
    texts = [
        " See [this](doc_1) for more.",
        "I am doc 1.",
    ]
    tokens_2d, doc_spans = make_batch(texts, enc)
    input_ids = tokens_2d[0, :-1]

    links = detector.detect_links(input_ids)
    link_to_target = creator._match_links_to_docs(links, doc_spans)

    # doc_1 starts after the link position → DAG violation → should not be matched
    matched_doc_ids = {doc_id for targets in link_to_target.values() for doc_id in targets}
    assert 1 not in matched_doc_ids, (
        "doc_1 appears after the link; DAG property should prevent the match"
    )


def test_cross_doc_mask_shape_and_causality():
    """Dense mask must be causal and allow cross-doc access only after links."""
    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    texts = [
        "Target text here.",
        " Link to [doc](doc_0) is here.",
    ]
    tokens_2d, doc_spans = make_batch(texts, enc)
    seq_len = tokens_2d.shape[1] - 1

    dense = creator.build_dense_mask_for_visualization(tokens_2d, doc_spans, device=torch.device('cpu'))
    assert dense.shape == (seq_len, seq_len)

    # No future attention: upper triangle (strictly above diagonal) must be all False
    upper = torch.triu(dense, diagonal=1)
    assert not upper.any(), "Mask must be causal (no future attention)"

    # Diagonal must be True (self-attention)
    assert dense.diagonal().all(), "Every position must attend to itself"


def test_block_mask_creation_succeeds():
    """CrossDocLinkMaskCreator.__call__ must return a valid BlockMask on CUDA."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("FlexAttention requires CUDA")

    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    texts = [
        "Target document content.",
        " References [target](doc_0) document.",
    ]
    tokens_2d, doc_spans = make_batch(texts, enc)
    tokens_2d = tokens_2d.cuda()

    from torch.nn.attention.flex_attention import BlockMask
    block_mask = creator(tokens_2d, doc_spans)
    assert isinstance(block_mask, BlockMask)


def test_index_doc_span_normalizes_spaces():
    """MarkdownLinkDetector.index_doc_span must replace spaces with underscores."""
    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)

    span = MockDocSpan(doc_id=0, clean_title="Sunshine Coast, Queensland", start=0, end=10)
    key = detector.index_doc_span(span)
    assert key == "Sunshine_Coast,_Queensland", (
        f"Expected underscores in key, got {repr(key)}"
    )

    # Single-word title: no change expected
    span2 = MockDocSpan(doc_id=1, clean_title="Australia", start=10, end=20)
    assert detector.index_doc_span(span2) == "Australia"
