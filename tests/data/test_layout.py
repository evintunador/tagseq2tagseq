"""Tests for DocLayoutPolicy implementations."""
import pytest
from data.layout import (
    EOSLayoutPolicy,
    IdentifierPrefixEOSLayoutPolicy,
    IdentifierPrefixLayoutPolicy,
    LatexCommentPrefixLayoutPolicy,
    NullLayoutPolicy,
    StochasticLatexCommentPrefixLayoutPolicy,
    make_layout_policy,
    inference_layout_for_detector,
    DocLayoutInfo,
    _latex_comment_card,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_info(raw="", normed="", outgoing=None, incoming=None, body_tokens=None, categories=""):
    return DocLayoutInfo(
        raw_identifier=raw,
        normed_identifier=normed,
        outgoing_identifiers=outgoing or [],
        incoming_identifiers=incoming or [],
        body_tokens=body_tokens,
        categories=categories,
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
# EOSLayoutPolicy
# ---------------------------------------------------------------------------

def test_eos_lengths():
    p = EOSLayoutPolicy(eos_token_id=2)
    assert p.prefix_length(make_info("doc")) == 0
    assert p.suffix_length(make_info("doc")) == 1


def test_eos_tokens():
    p = EOSLayoutPolicy(eos_token_id=50256)
    assert p.prefix_tokens(make_info("doc")) == []
    assert p.suffix_tokens(make_info("doc")) == [50256]


def test_eos_ignores_all_info_fields():
    p = EOSLayoutPolicy(eos_token_id=2)
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
    info_a = make_info("Python", "python_abc123")
    info_b = make_info("Python", "something_completely_different")
    assert p.prefix_tokens(info_a) == p.prefix_tokens(info_b)


def test_identifier_prefix_ignores_body_and_links():
    p = IdentifierPrefixLayoutPolicy(_simple_encode)
    info_a = make_info("Python", outgoing=["ruby"], body_tokens=[1, 2, 3])
    info_b = make_info("Python", incoming=["java"])
    assert p.prefix_tokens(info_a) == p.prefix_tokens(info_b)


# ---------------------------------------------------------------------------
# IdentifierPrefixEOSLayoutPolicy
# ---------------------------------------------------------------------------

def test_identifier_prefix_eos_prefix_is_title_only():
    p = IdentifierPrefixEOSLayoutPolicy(_simple_encode, eos_token_id=2)
    tokens = p.prefix_tokens(make_info("Python"))
    assert tokens == _simple_encode("# Python\n\n")


def test_identifier_prefix_eos_prefix_contains_title():
    p = IdentifierPrefixEOSLayoutPolicy(_simple_encode, eos_token_id=2)
    tokens = p.prefix_tokens(make_info("Python"))
    assert tokens == _simple_encode("# Python\n\n")


def test_identifier_prefix_eos_suffix_is_eos():
    p = IdentifierPrefixEOSLayoutPolicy(_simple_encode, eos_token_id=2)
    assert p.suffix_tokens(make_info("Python")) == [2]


def test_identifier_prefix_eos_lengths_match_tokens():
    p = IdentifierPrefixEOSLayoutPolicy(_simple_encode, eos_token_id=2)
    info = make_info("Some Title")
    assert p.prefix_length(info) == len(p.prefix_tokens(info))
    assert p.suffix_length(info) == len(p.suffix_tokens(info))


def test_identifier_prefix_eos_caches_title():
    calls = [0]
    def counting_encode(text):
        calls[0] += 1
        return _simple_encode(text)
    p = IdentifierPrefixEOSLayoutPolicy(counting_encode, eos_token_id=2)
    p.prefix_tokens(make_info("Python"))
    p.prefix_tokens(make_info("Python"))
    p.prefix_length(make_info("Python"))
    assert calls[0] == 1


# ---------------------------------------------------------------------------
# make_layout_policy factory
# ---------------------------------------------------------------------------

def test_factory_null():
    p = make_layout_policy("null")
    assert isinstance(p, NullLayoutPolicy)


def test_factory_eos():
    p = make_layout_policy("eos", eos_token_id=8)
    assert isinstance(p, EOSLayoutPolicy)
    assert p.prefix_tokens(make_info("x")) == []
    assert p.suffix_tokens(make_info("x")) == [8]


def test_factory_identifier_prefix():
    p = make_layout_policy("identifier_prefix", encode_fn=_simple_encode)
    assert isinstance(p, IdentifierPrefixLayoutPolicy)
    assert p.prefix_tokens(make_info("Foo")) == _simple_encode("# Foo\n\n")


def test_factory_identifier_prefix_eos():
    p = make_layout_policy("identifier_prefix_eos", encode_fn=_simple_encode, eos_token_id=2)
    assert isinstance(p, IdentifierPrefixEOSLayoutPolicy)
    assert p.prefix_tokens(make_info("Foo")) == _simple_encode("# Foo\n\n")
    assert p.suffix_tokens(make_info("Foo")) == [2]


def test_factory_requires_encode_fn_for_prefix_policies():
    with pytest.raises(ValueError, match="encode_fn"):
        make_layout_policy("identifier_prefix")
    with pytest.raises(ValueError, match="encode_fn"):
        make_layout_policy("identifier_prefix_eos")


def test_factory_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown layout_policy"):
        make_layout_policy("banana")


# ---------------------------------------------------------------------------
# _latex_comment_card helper
# ---------------------------------------------------------------------------

def test_latex_card_is_title_only():
    # Title-only by design: the card never emits a "% Categories:" line, even when
    # categories are known. The \cite{Title} link carries only the title, so seeding
    # an aux doc at inference can only reconstruct the title — emitting categories at
    # training time would create a train/inference mismatch.
    card = _latex_comment_card(make_info("Attention Is All You Need", categories="cs.CL cs.LG"))
    assert card == "% Title: Attention Is All You Need\n\n"


def test_latex_card_ignores_categories():
    # Same output whether or not categories are present.
    card = _latex_comment_card(make_info("Some Paper", categories=""))
    assert card == "% Title: Some Paper\n\n"


# ---------------------------------------------------------------------------
# LatexCommentPrefixLayoutPolicy (deterministic, for inference)
# ---------------------------------------------------------------------------

def test_latex_comment_prefix_format():
    p = LatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=2)
    info = make_info("My Paper", "arxiv_123", categories="cs.LG")
    # Title-only: categories are ignored even when present.
    assert p.prefix_tokens(info) == _simple_encode("% Title: My Paper\n\n")


def test_latex_comment_prefix_no_categories():
    p = LatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=2)
    info = make_info("My Paper", "arxiv_123", categories="")
    assert p.prefix_tokens(info) == _simple_encode("% Title: My Paper\n\n")


def test_latex_comment_prefix_suffix_is_eos():
    p = LatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=42)
    assert p.suffix_tokens(make_info("x")) == [42]
    assert p.suffix_length(make_info("x")) == 1


def test_latex_comment_prefix_length_matches_tokens():
    p = LatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=2)
    info = make_info("Title", "n", categories="cs.AI")
    assert p.prefix_length(info) == len(p.prefix_tokens(info))


def test_latex_comment_prefix_caches():
    calls = [0]
    def counting_encode(text):
        calls[0] += 1
        return _simple_encode(text)
    p = LatexCommentPrefixLayoutPolicy(counting_encode, eos_token_id=2)
    info = make_info("Title", "n", categories="cs.AI")
    p.prefix_tokens(info)
    p.prefix_tokens(info)
    p.prefix_length(info)
    assert calls[0] == 1


# ---------------------------------------------------------------------------
# StochasticLatexCommentPrefixLayoutPolicy
# ---------------------------------------------------------------------------

def test_stochastic_latex_prefix_length_matches_tokens_both_branches():
    # For every doc, in either include/exclude state, prefix_length must equal
    # len(prefix_tokens) — the shared _include_prefix guarantees agreement.
    p = StochasticLatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=2)
    for i in range(50):
        info = make_info(f"Title {i}", f"arxiv_{i}", categories="cs.LG")
        assert p.prefix_length(info) == len(p.prefix_tokens(info))


def test_stochastic_latex_prefix_deterministic_across_instances():
    a = StochasticLatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=2)
    b = StochasticLatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=2)
    for i in range(30):
        info = make_info(f"T{i}", f"arxiv_{i}", categories="cs.LG")
        assert a.prefix_tokens(info) == b.prefix_tokens(info)


def test_stochastic_latex_prefix_both_outcomes_occur():
    # Across many docs, the coin flip should produce both included and omitted cards.
    p = StochasticLatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=2)
    lengths = {
        p.prefix_length(make_info(f"T{i}", f"arxiv_{i}", categories="cs.LG")) > 0
        for i in range(100)
    }
    assert lengths == {True, False}


def test_stochastic_latex_prefix_epoch_changes_decision():
    # The same doc can flip inclusion across epochs (at least one differs over a span).
    p = StochasticLatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=2)
    info = make_info("Title", "arxiv_42", categories="cs.LG")
    p.set_epoch(0)
    e0 = [p.prefix_length(make_info("T", f"arxiv_{i}", categories="c")) > 0 for i in range(20)]
    p.set_epoch(1)
    e1 = [p.prefix_length(make_info("T", f"arxiv_{i}", categories="c")) > 0 for i in range(20)]
    assert e0 != e1  # epoch participates in the hash, so the pattern shifts


def test_stochastic_latex_prefix_always_eos_suffix():
    p = StochasticLatexCommentPrefixLayoutPolicy(_simple_encode, eos_token_id=99)
    # Suffix is EOS regardless of prefix inclusion.
    for i in range(20):
        info = make_info(f"T{i}", f"arxiv_{i}")
        assert p.suffix_tokens(info) == [99]


def test_factory_latex_comment_prefix():
    p = make_layout_policy("latex_comment_prefix", encode_fn=_simple_encode, eos_token_id=2)
    assert isinstance(p, LatexCommentPrefixLayoutPolicy)


def test_factory_stochastic_latex_comment_prefix():
    p = make_layout_policy(
        "stochastic_latex_comment_prefix", encode_fn=_simple_encode, eos_token_id=2
    )
    assert isinstance(p, StochasticLatexCommentPrefixLayoutPolicy)


def test_factory_latex_policies_require_encode_fn():
    with pytest.raises(ValueError, match="encode_fn"):
        make_layout_policy("latex_comment_prefix")
    with pytest.raises(ValueError, match="encode_fn"):
        make_layout_policy("stochastic_latex_comment_prefix")


# ---------------------------------------------------------------------------
# inference_layout_for_detector
# ---------------------------------------------------------------------------

def test_inference_layout_for_detector_known():
    assert inference_layout_for_detector("python") == "identifier_prefix_eos"
    assert inference_layout_for_detector("markdown") == "identifier_prefix_eos"
    assert inference_layout_for_detector("arxiv") == "latex_comment_prefix"
    assert inference_layout_for_detector("null") == "eos"


def test_inference_layout_for_detector_none_is_eos():
    assert inference_layout_for_detector(None) == "eos"


def test_inference_layout_for_detector_unknown_raises():
    with pytest.raises(ValueError, match="No inference layout mapping"):
        inference_layout_for_detector("bogus")
