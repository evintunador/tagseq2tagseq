"""Tests for DocLayoutPolicy implementations."""
import pytest
from data.layout import (
    BOSEOSLayoutPolicy,
    DocLayoutInfo,
    IdentifierPrefixLayoutPolicy,
    NullLayoutPolicy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_info(raw="", normed="", outgoing=None, incoming=None, body_tokens=None):
    return DocLayoutInfo(
        raw_identifier=raw,
        normed_identifier=normed,
        outgoing_identifiers=outgoing or [],
        incoming_identifiers=incoming or [],
        body_tokens=body_tokens,
    )


# ---------------------------------------------------------------------------
# DocLayoutInfo
# ---------------------------------------------------------------------------

def test_doc_layout_info_defaults():
    info = DocLayoutInfo(raw_identifier="Python", normed_identifier="python_abc123")
    assert info.outgoing_identifiers == []
    assert info.incoming_identifiers == []
    assert info.body_tokens is None


def test_doc_layout_info_full_construction():
    info = DocLayoutInfo(
        raw_identifier="Python",
        normed_identifier="python_abc123",
        outgoing_identifiers=["ruby_def456"],
        incoming_identifiers=["java_ghi789"],
        body_tokens=[1, 2, 3],
    )
    assert info.raw_identifier == "Python"
    assert info.normed_identifier == "python_abc123"
    assert info.outgoing_identifiers == ["ruby_def456"]
    assert info.incoming_identifiers == ["java_ghi789"]
    assert info.body_tokens == [1, 2, 3]


# ---------------------------------------------------------------------------
# NullLayoutPolicy
# ---------------------------------------------------------------------------

def test_null_policy_lengths_are_zero():
    p = NullLayoutPolicy()
    assert p.prefix_length(make_info("anything")) == 0
    assert p.suffix_length(make_info("anything", "normed")) == 0


def test_null_policy_tokens_are_empty():
    p = NullLayoutPolicy()
    assert p.prefix_tokens(make_info("anything")) == []
    assert p.suffix_tokens(make_info("anything", "normed")) == []


# ---------------------------------------------------------------------------
# BOSEOSLayoutPolicy
# ---------------------------------------------------------------------------

def test_bos_eos_lengths():
    p = BOSEOSLayoutPolicy(bos_token_id=1, eos_token_id=2)
    assert p.prefix_length(make_info("doc")) == 1
    assert p.suffix_length(make_info("doc")) == 1


def test_bos_eos_tokens():
    p = BOSEOSLayoutPolicy(bos_token_id=50256, eos_token_id=50257)
    assert p.prefix_tokens(make_info("doc")) == [50256]
    assert p.suffix_tokens(make_info("doc")) == [50257]


def test_bos_eos_ignores_all_info_fields():
    p = BOSEOSLayoutPolicy(bos_token_id=1, eos_token_id=2)
    info_a = make_info("Python", outgoing=["ruby"], body_tokens=[1, 2, 3])
    info_b = make_info("Ruby", incoming=["java"])
    assert p.prefix_tokens(info_a) == p.prefix_tokens(info_b)
    assert p.suffix_tokens(info_a) == p.suffix_tokens(info_b)


# ---------------------------------------------------------------------------
# IdentifierPrefixLayoutPolicy
# ---------------------------------------------------------------------------

def _simple_encode(text: str):
    """Trivial encode: one token per character (ord value)."""
    return [ord(c) for c in text]


def test_identifier_prefix_format():
    p = IdentifierPrefixLayoutPolicy(_simple_encode)
    tokens = p.prefix_tokens(make_info("Python"))
    expected = _simple_encode("# Python\n\n")
    assert tokens == expected


def test_identifier_prefix_length_matches_tokens():
    p = IdentifierPrefixLayoutPolicy(_simple_encode)
    info = make_info("Some Title")
    assert p.prefix_length(info) == len(p.prefix_tokens(info))


def test_identifier_prefix_empty_identifier():
    p = IdentifierPrefixLayoutPolicy(_simple_encode)
    tokens = p.prefix_tokens(make_info(""))
    expected = _simple_encode("# \n\n")
    assert tokens == expected
    assert p.prefix_length(make_info("")) == len(expected)


def test_identifier_prefix_suffix_is_empty():
    p = IdentifierPrefixLayoutPolicy(_simple_encode)
    assert p.suffix_tokens(make_info("anything")) == []
    assert p.suffix_length(make_info("anything")) == 0


def test_identifier_prefix_caches_results():
    calls = [0]

    def counting_encode(text):
        calls[0] += 1
        return _simple_encode(text)

    p = IdentifierPrefixLayoutPolicy(counting_encode)
    p.prefix_tokens(make_info("Python"))
    p.prefix_tokens(make_info("Python"))
    p.prefix_length(make_info("Python"))
    assert calls[0] == 1  # encode called once despite three lookups


def test_identifier_prefix_different_identifiers_encoded_separately():
    p = IdentifierPrefixLayoutPolicy(_simple_encode)
    assert p.prefix_tokens(make_info("Python")) != p.prefix_tokens(make_info("Ruby"))


def test_identifier_prefix_uses_raw_identifier_not_normed():
    p = IdentifierPrefixLayoutPolicy(_simple_encode)
    # normed_identifier is irrelevant — prefix derived from raw only
    info_a = make_info("Python", "python_abc123")
    info_b = make_info("Python", "something_completely_different")
    assert p.prefix_tokens(info_a) == p.prefix_tokens(info_b)


def test_identifier_prefix_ignores_body_and_links():
    p = IdentifierPrefixLayoutPolicy(_simple_encode)
    info_a = make_info("Python", outgoing=["ruby"], body_tokens=[1, 2, 3])
    info_b = make_info("Python", incoming=["java"])
    assert p.prefix_tokens(info_a) == p.prefix_tokens(info_b)
