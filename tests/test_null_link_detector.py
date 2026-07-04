r"""
Tests for null_link_detector.py

Coverage:
  - NullLinkDetector.detect_links always returns an empty list (edgeless corpora)
  - index_doc_span returns raw_identifier verbatim (must be defined explicitly:
    the LinkDetector Protocol default is not inherited, and the mask creator
    indexes every doc span even when no links are detected)
  - make_link_detector('null') constructs a NullLinkDetector
"""
from types import SimpleNamespace

import tiktoken
import torch

from model.graph_traversal.null_link_detector import NullLinkDetector
from model.graph_traversal.link_detector import make_link_detector


def test_detect_links_always_empty():
    enc = tiktoken.get_encoding("gpt2")
    det = NullLinkDetector(decode_fn=enc.decode)
    # Text that WOULD trip the markdown / arxiv / python detectors.
    text = r"See [foo](Bar) and \cite{Baz} plus `import os`"
    ids = torch.tensor(enc.encode(text), dtype=torch.long)
    assert det.detect_links(ids) == []
    assert det.detect_links(torch.empty(0, dtype=torch.long)) == []


def test_index_doc_span_returns_raw_identifier():
    det = NullLinkDetector()
    span = SimpleNamespace(raw_identifier="Some Title", normed_identifier="some_title")
    assert det.index_doc_span(span) == "Some Title"


def test_make_link_detector_null():
    enc = tiktoken.get_encoding("gpt2")
    det = make_link_detector("null", enc.decode)
    assert isinstance(det, NullLinkDetector)
    assert det.detect_links(torch.tensor(enc.encode("hi"), dtype=torch.long)) == []
