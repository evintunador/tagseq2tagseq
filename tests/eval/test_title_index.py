"""
tests/eval/test_title_index.py — unit tests for eval.title_index.
"""
import pytest
from eval.title_index import HashNormTitleIndex, TitleIndex


# ── Protocol compliance ───────────────────────────────────────────────────────

def test_satisfies_protocol():
    idx = HashNormTitleIndex([])
    assert isinstance(idx, TitleIndex)


# ── Basic lookup ─────────────────────────────────────────────────────────────

def test_exact_match():
    idx = HashNormTitleIndex(["Python (programming language)"])
    assert idx.lookup("Python (programming language)") == "Python (programming language)"


def test_case_insensitive():
    idx = HashNormTitleIndex(["Python (programming language)"])
    assert idx.lookup("python (programming language)") == "Python (programming language)"


def test_mixed_case_generation():
    idx = HashNormTitleIndex(["Python (programming language)"])
    assert idx.lookup("PYTHON (PROGRAMMING LANGUAGE)") == "Python (programming language)"


def test_miss_returns_none():
    idx = HashNormTitleIndex(["Python (programming language)"])
    assert idx.lookup("Rust (programming language)") is None


def test_empty_index():
    idx = HashNormTitleIndex([])
    assert idx.lookup("anything") is None


def test_empty_query():
    idx = HashNormTitleIndex(["Python"])
    assert idx.lookup("") is None


# ── Normalization behaviour ───────────────────────────────────────────────────

def test_hyphen_stripped():
    # normalize_identifier strips hyphens (non-alphanumeric) but keeps underscores.
    # "Spider-Man" → "spiderman"; "Spider Man" → "spider_man" — different keys, no match.
    idx = HashNormTitleIndex(["Spider-Man"])
    assert idx.lookup("Spider Man") is None
    assert idx.lookup("Spider-Man") == "Spider-Man"
    assert idx.lookup("SpiderMan") == "Spider-Man"  # same stripped-norm "spiderman"


def test_punctuation_stripped():
    idx = HashNormTitleIndex(["C++"])
    assert idx.lookup("C") == "C++"


def test_disambiguation_survives():
    # Parens are stripped by normalize_identifier, but the inner text is kept,
    # so disambiguation text still differentiates entries.
    idx = HashNormTitleIndex(["Mercury (planet)", "Mercury (element)"])
    assert idx.lookup("Mercury (planet)") == "Mercury (planet)"
    assert idx.lookup("Mercury (element)") == "Mercury (element)"


# ── Collision (first-wins) ────────────────────────────────────────────────────

def test_collision_first_wins():
    # "C++" and "C" both normalize to "c"; first entry should win.
    idx = HashNormTitleIndex(["C++", "C"])
    result = idx.lookup("C")
    assert result == "C++"  # "C++" was inserted first


def test_collision_second_not_reachable():
    idx = HashNormTitleIndex(["C++", "C"])
    # "C" resolves to "C++" (first-wins), so "C" is unreachable by its own name
    assert idx.lookup("C") == "C++"


# ── __len__ ───────────────────────────────────────────────────────────────────

def test_len_deduplicates_collisions():
    # "C++" and "C" collide → only 1 entry stored
    idx = HashNormTitleIndex(["C++", "C", "Python"])
    assert len(idx) == 2


def test_len_no_collisions():
    idx = HashNormTitleIndex(["Mercury (planet)", "Mercury (element)", "Python"])
    assert len(idx) == 3


# ── MarkdownPromptAnnotator integration ──────────────────────────────────────

def test_annotator_uses_title_index_on_corpus_miss(monkeypatch):
    """When the model generates a case-variant title, title_index resolves it."""
    from unittest.mock import MagicMock
    from eval.link_annotator import MarkdownPromptAnnotator

    corpus = MagicMock()
    # has_document is case-sensitive verbatim — only matches exact raw_identifier
    corpus.has_document.side_effect = lambda s: s == "Python (programming language)"
    corpus.get_document.return_value = iter([1, 2, 3])

    idx = HashNormTitleIndex(["Python (programming language)"])
    ann = MarkdownPromptAnnotator(
        corpus=corpus,
        title_index=idx,
        link_retrieval_mode="corpus_only",
    )

    # Model generates lowercase variant — verbatim lookup would miss, index should hit
    aux_lists, aux_ids, fired = ann._fetch_aux(
        model=MagicMock(), target_str="python (programming language)", device="cpu"
    )
    assert fired is True
    assert aux_ids == ["Python (programming language)"]


def test_annotator_without_title_index_uses_verbatim(monkeypatch):
    """Without a title_index, annotator falls back to verbatim has_document."""
    from unittest.mock import MagicMock
    from eval.link_annotator import MarkdownPromptAnnotator

    corpus = MagicMock()
    corpus.has_document.side_effect = lambda s: s == "Python (programming language)"
    corpus.get_document.return_value = iter([1, 2, 3])

    ann = MarkdownPromptAnnotator(
        corpus=corpus,
        title_index=None,
        link_retrieval_mode="corpus_only",
    )

    # Verbatim miss
    _, _, fired_miss = ann._fetch_aux(
        model=MagicMock(), target_str="python (programming language)", device="cpu"
    )
    assert fired_miss is False

    # Verbatim hit
    _, _, fired_hit = ann._fetch_aux(
        model=MagicMock(), target_str="Python (programming language)", device="cpu"
    )
    assert fired_hit is True
