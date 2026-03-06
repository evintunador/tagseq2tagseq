"""
Tests for cross-document link mask detection and matching.

Link targets in Wikipedia .md files use the original article title with spaces
(e.g. ``(Sunshine Coast, Queensland)``), matching DocSpan.raw_identifier directly.
"""

import torch
import tiktoken
import pytest
from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
from dataclasses import dataclass, field
from typing import List


@dataclass
class MockDocSpan:
    """Mock DocSpan for testing."""
    doc_id: int
    raw_identifier: str
    start: int
    end: int
    truncated: bool = False
    outgoing_identifiers: List[str] = field(default_factory=list)


def make_batch(texts, titles, enc):
    """Encode a list of text segments and return (tokens_2d, doc_spans)."""
    all_tokens = []
    spans = []
    for i, (text, title) in enumerate(zip(texts, titles)):
        toks = enc.encode(text)
        start = len(all_tokens)
        all_tokens.extend(toks)
        spans.append(MockDocSpan(
            doc_id=i,
            raw_identifier=title,
            start=start,
            end=len(all_tokens),
        ))
    # tokens_2d shape: [1, T+1] (extra target token appended as required by mask creator)
    tokens_2d = torch.tensor(all_tokens + [0], dtype=torch.long).unsqueeze(0)
    return tokens_2d, spans


def test_link_detection_and_matching():
    """Detected links map to correct in-batch doc spans.

    Link targets use the original title with spaces, matching raw_identifier directly.
    """
    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    texts = [
        "This is the target document.",
        " Here is [a link](Target Doc) to the first doc and [bad](Missing Doc).",
    ]
    titles = ["Target Doc", "Linker Doc"]
    tokens_2d, doc_spans = make_batch(texts, titles, enc)
    input_ids = tokens_2d[0, :-1]

    links = detector.detect_links(input_ids)
    assert len(links) >= 1, "Should detect at least the link to Target Doc"

    target_strs = [lnk.target_str for lnk in links]
    assert "Target Doc" in target_strs, (
        f"Expected 'Target Doc' in detected links, got {target_strs}"
    )

    link_to_target = creator._match_links_to_docs(links, doc_spans)

    assert len(link_to_target) >= 1, "Expected at least one matched link"
    matched_doc_ids = {doc_id for targets in link_to_target.values() for doc_id in targets}
    assert 0 in matched_doc_ids, "Expected doc_id=0 (Target Doc) to be a match target"


def test_dag_property_enforced():
    """Links must not grant attention to documents that appear later in the batch."""
    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    # Doc 0 links to "Doc One", but Doc One appears AFTER Doc Zero → DAG violation
    texts = [
        " See [this](Doc One) for more.",
        "I am Doc One.",
    ]
    titles = ["Doc Zero", "Doc One"]
    tokens_2d, doc_spans = make_batch(texts, titles, enc)
    input_ids = tokens_2d[0, :-1]

    links = detector.detect_links(input_ids)
    link_to_target = creator._match_links_to_docs(links, doc_spans)

    matched_doc_ids = {doc_id for targets in link_to_target.values() for doc_id in targets}
    assert 1 not in matched_doc_ids, (
        "Doc One appears after the link; DAG property should prevent the match"
    )


def test_cross_doc_mask_shape_and_causality():
    """Dense mask must be causal and allow cross-doc access only after links."""
    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    texts = [
        "Target text here.",
        " Link to [doc](Target Doc) is here.",
    ]
    titles = ["Target Doc", "Linker Doc"]
    tokens_2d, doc_spans = make_batch(texts, titles, enc)
    seq_len = tokens_2d.shape[1]

    dense = creator.build_dense_mask_for_visualization(tokens_2d, doc_spans, device=torch.device('cpu'))
    assert dense.shape == (seq_len, seq_len)

    # No future attention: upper triangle (strictly above diagonal) must be all False
    upper = torch.triu(dense, diagonal=1)
    assert not upper.any(), "Mask must be causal (no future attention)"

    # Diagonal must be True (self-attention)
    assert dense.diagonal().all(), "Every position must attend to itself"


def test_block_mask_creation_succeeds():
    """CrossDocLinkMaskCreator.__call__ must return a valid BlockMask on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("FlexAttention requires CUDA")

    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)
    creator = CrossDocLinkMaskCreator(link_detector=detector)

    texts = [
        "Target document content.",
        " References [target](Target Doc) document.",
    ]
    titles = ["Target Doc", "Linker Doc"]
    tokens_2d, doc_spans = make_batch(texts, titles, enc)
    tokens_2d = tokens_2d.cuda()

    from torch.nn.attention.flex_attention import BlockMask
    block_mask = creator(tokens_2d, doc_spans)
    assert isinstance(block_mask, BlockMask)


def test_detect_links_space_form_target():
    """detect_links returns the link target exactly as it appears in the token stream.

    With the fixed raw_link_target, Wikipedia .md files store link targets with
    spaces matching raw_identifier directly — no normalization needed.
    """
    enc = tiktoken.get_encoding('gpt2')
    detector = MarkdownLinkDetector(decode_fn=enc.decode)

    text = "See [Queensland](Sunshine Coast, Queensland) for details."
    input_ids = torch.tensor(enc.encode(text), dtype=torch.long)

    links = detector.detect_links(input_ids)
    target_strs = [lnk.target_str for lnk in links]

    assert any("Sunshine Coast, Queensland" == t for t in target_strs), (
        f"Expected space-form target matching raw_identifier; got {target_strs}"
    )

    # index_doc_span returns raw_identifier unchanged
    span = MockDocSpan(doc_id=0, raw_identifier="Sunshine Coast, Queensland", start=0, end=10)
    assert detector.index_doc_span(span) == "Sunshine Coast, Queensland"
