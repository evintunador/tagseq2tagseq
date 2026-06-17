r"""
Tests for arxiv_cite_detector.py

Coverage:
  - ArxivCiteDetector.detect_links: \cite{Title} detection end-to-end with tiktoken GPT-2
  - variant commands (\citep, \citet) and optional [note] groups
  - empty \cite{} placeholders skipped (out-of-corpus markers)
  - multiple citations, titles with spaces/punctuation
  - link_end_pos lands just after the closing brace
  - index_doc_span returns raw_identifier verbatim
"""
import re
from types import SimpleNamespace

import pytest
import tiktoken
import torch

from model.graph_traversal.arxiv_cite_detector import ArxivCiteDetector, _CITE_RE


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return ArxivCiteDetector(decode_fn=enc.decode)


def _encode(enc, text: str) -> torch.Tensor:
    return torch.tensor(enc.encode(text), dtype=torch.long)


class TestCiteRegex:
    def test_basic(self):
        assert _CITE_RE.findall(r"\cite{Attention Is All You Need}") == [
            "Attention Is All You Need"
        ]

    def test_variant_commands(self):
        assert _CITE_RE.findall(r"\citep{Foo}") == ["Foo"]
        assert _CITE_RE.findall(r"\citet{Bar}") == ["Bar"]

    def test_optional_note_group(self):
        assert _CITE_RE.findall(r"\cite[p.~5]{Some Title}") == ["Some Title"]

    def test_empty(self):
        assert _CITE_RE.findall(r"\cite{}") == [""]

    def test_multiple(self):
        text = r"see \cite{A Paper} and \cite{B Paper}"
        assert _CITE_RE.findall(text) == ["A Paper", "B Paper"]


class TestDetectLinks:
    def test_single_citation(self, detector, enc):
        title = "Deep Residual Learning for Image Recognition"
        ids = _encode(enc, f"As shown in \\cite{{{title}}}, we observe...")
        links = detector.detect_links(ids)
        assert len(links) == 1
        assert links[0].target_str == title

    def test_empty_placeholder_skipped(self, detector, enc):
        # Out-of-corpus citations are emitted as \cite{} and must NOT produce a link.
        ids = _encode(enc, r"Prior work \cite{} did not address this.")
        assert detector.detect_links(ids) == []

    def test_multiple_citations(self, detector, enc):
        ids = _encode(enc, r"Both \cite{Title One} and \cite{Title Two} apply.")
        links = detector.detect_links(ids)
        assert [l.target_str for l in links] == ["Title One", "Title Two"]

    def test_mixed_empty_and_real(self, detector, enc):
        ids = _encode(enc, r"\cite{Real Paper} but not \cite{} here.")
        links = detector.detect_links(ids)
        assert [l.target_str for l in links] == ["Real Paper"]

    def test_title_with_punctuation(self, detector, enc):
        title = "BERT: Pre-training of Deep Bidirectional Transformers"
        ids = _encode(enc, f"\\cite{{{title}}}")
        links = detector.detect_links(ids)
        assert len(links) == 1
        assert links[0].target_str == title

    def test_link_end_pos_after_brace(self, detector, enc):
        # The token at link_end_pos should be at/after the closing brace — i.e. the
        # decoded prefix up to link_end_pos contains the full \cite{...}.
        title = "Some Title"
        text = f"x \\cite{{{title}}} y"
        ids = _encode(enc, text)
        links = detector.detect_links(ids)
        assert len(links) == 1
        prefix = enc.decode(ids[: links[0].link_end_pos].tolist())
        assert "}" in prefix
        assert title in prefix

    def test_no_citations(self, detector, enc):
        ids = _encode(enc, "This text has no citations at all.")
        assert detector.detect_links(ids) == []

    def test_variant_citep(self, detector, enc):
        ids = _encode(enc, r"\citep{Vaswani Transformer}")
        links = detector.detect_links(ids)
        assert len(links) == 1
        assert links[0].target_str == "Vaswani Transformer"


class TestIndexDocSpan:
    def test_returns_raw_identifier(self, detector):
        span = SimpleNamespace(raw_identifier="Attention Is All You Need", normed_identifier="x")
        assert detector.index_doc_span(span) == "Attention Is All You Need"
