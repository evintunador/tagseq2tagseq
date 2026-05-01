"""
tests/eval/test_title_index.py — unit tests for eval.title_index.
"""
import pytest
from eval.title_index import HashNormTitleIndex, TitleIndex, _DEFAULT_STRATEGIES, _is_contiguous_subsequence


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


def test_punct_only_query_no_spurious_match():
    # "." and "](" normalize to empty string — must not collide with
    # corpus entries whose titles also normalize to empty (e.g. non-ASCII-only titles).
    idx = HashNormTitleIndex(["ظ", "Python"])
    assert idx.lookup(".") is None
    assert idx.lookup("](") is None


# ── Normalization behaviour ───────────────────────────────────────────────────

def test_hyphen_stripped():
    idx = HashNormTitleIndex(["Spider-Man"])
    # "Spider Man" hits via word_overlap_ordered: ["spider", "man"] is a
    # contiguous subsequence of Spider-Man's word list ["spider", "man"].
    assert idx.lookup("Spider Man") == "Spider-Man"
    assert idx.lookup("Spider-Man") == "Spider-Man"
    assert idx.lookup("SpiderMan") == "Spider-Man"


def test_punctuation_stripped():
    idx = HashNormTitleIndex(["C++"])
    assert idx.lookup("C") == "C++"


def test_disambiguation_survives():
    idx = HashNormTitleIndex(["Mercury (planet)", "Mercury (element)"])
    assert idx.lookup("Mercury (planet)") == "Mercury (planet)"
    assert idx.lookup("Mercury (element)") == "Mercury (element)"


# ── Collision (first-wins) ────────────────────────────────────────────────────

def test_collision_exact_wins():
    idx = HashNormTitleIndex(["C++", "C"])
    assert idx.lookup("C") == "C"
    assert idx.lookup("C++") == "C++"


def test_exact_wins_over_norm_collision():
    idx = HashNormTitleIndex(["C++", "C"])
    assert idx.lookup("C") == "C"


# ── __len__ ───────────────────────────────────────────────────────────────────

def test_len_counts_distinct_raws():
    idx = HashNormTitleIndex(["C++", "C", "Python"])
    assert len(idx) == 3


def test_len_no_collisions():
    idx = HashNormTitleIndex(["Mercury (planet)", "Mercury (element)", "Python"])
    assert len(idx) == 3


# ── Exact-match priority ─────────────────────────────────────────────────────

def test_exact_fires_before_norm():
    idx = HashNormTitleIndex(["C++", "C"])
    assert idx.lookup("C") == "C"
    assert idx.lookup("c") == "C"


def test_exact_case_insensitive_no_norm_needed():
    idx = HashNormTitleIndex(["Python"])
    assert idx.lookup("PYTHON") == "Python"


# ── Strategy configuration ────────────────────────────────────────────────────

def test_default_strategies():
    idx = HashNormTitleIndex(["Python"])
    assert idx.strategies == ("exact", "norm", "word_overlap_ordered")


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown strategies"):
        HashNormTitleIndex(["Python"], strategies=("exact", "bogus"))


def test_old_word_overlap_name_raises():
    # "word_overlap" was the old name — now split into _ordered / _unordered.
    with pytest.raises(ValueError, match="Unknown strategies"):
        HashNormTitleIndex(["Python"], strategies=("exact", "word_overlap"))


def test_exact_only_skips_norm():
    idx = HashNormTitleIndex(["C++", "C"], strategies=("exact",))
    assert idx.lookup("C") == "C"
    assert idx.lookup("C++") == "C++"
    assert idx.lookup("SPIDERMAN") is None


def test_norm_only_skips_exact():
    idx = HashNormTitleIndex(["Spider-Man"], strategies=("norm",))
    assert idx.lookup("SpiderMan") == "Spider-Man"


# ── _is_contiguous_subsequence helper ────────────────────────────────────────

def test_contiguous_subseq_exact():
    assert _is_contiguous_subsequence(["a", "b"], ["a", "b", "c"]) is True


def test_contiguous_subseq_at_end():
    assert _is_contiguous_subsequence(["b", "c"], ["a", "b", "c"]) is True


def test_contiguous_subseq_not_contiguous():
    assert _is_contiguous_subsequence(["a", "c"], ["a", "b", "c"]) is False


def test_contiguous_subseq_wrong_order():
    assert _is_contiguous_subsequence(["b", "a"], ["a", "b", "c"]) is False


def test_contiguous_subseq_longer_than_haystack():
    assert _is_contiguous_subsequence(["a", "b", "c", "d"], ["a", "b", "c"]) is False


# ── word_overlap_ordered strategy ─────────────────────────────────────────────

def test_ordered_partial_title():
    idx = HashNormTitleIndex(
        ["Russian Civil War"],
        strategies=("word_overlap_ordered",),
    )
    assert idx.lookup("Russian Civil") == "Russian Civil War"


def test_ordered_rejects_wrong_order():
    idx = HashNormTitleIndex(
        ["Russian Civil War"],
        strategies=("word_overlap_ordered",),
    )
    assert idx.lookup("Civil Russian") is None


def test_ordered_rejects_non_contiguous():
    # "Russian War" skips "Civil" — not contiguous, must miss.
    idx = HashNormTitleIndex(
        ["Russian Civil War"],
        strategies=("word_overlap_ordered",),
    )
    assert idx.lookup("Russian War") is None


def test_ordered_prefers_shorter_on_tie():
    # "Indianapolis Motor" is a prefix of both — pick shorter.
    idx = HashNormTitleIndex(
        ["Indianapolis Motor Speedway", "Indianapolis Motor Sports"],
        strategies=("word_overlap_ordered",),
    )
    result = idx.lookup("Indianapolis Motor")
    assert result in ("Indianapolis Motor Speedway", "Indianapolis Motor Sports")
    assert len(result) == min(
        len("Indianapolis Motor Speedway"), len("Indianapolis Motor Sports")
    )


def test_ordered_in_default():
    idx = HashNormTitleIndex(["Russian Civil War"])
    assert idx.lookup("Russian Civil") == "Russian Civil War"


# ── word_overlap_unordered strategy ──────────────────────────────────────────

def test_unordered_partial_title():
    idx = HashNormTitleIndex(
        ["United States of America"],
        strategies=("word_overlap_unordered",),
    )
    assert idx.lookup("United States") == "United States of America"


def test_unordered_accepts_reordered_words():
    idx = HashNormTitleIndex(
        ["Russian Civil War"],
        strategies=("word_overlap_unordered",),
    )
    # Unordered: "Civil Russian" still hits because both words are present.
    assert idx.lookup("Civil Russian") == "Russian Civil War"


def test_unordered_all_words_required():
    idx = HashNormTitleIndex(
        ["United Nations", "United Kingdom"],
        strategies=("word_overlap_unordered",),
    )
    assert idx.lookup("United Kingdom") == "United Kingdom"
    assert idx.lookup("United Nations") == "United Nations"


def test_unordered_no_match_on_disjoint():
    idx = HashNormTitleIndex(
        ["United States of America"],
        strategies=("word_overlap_unordered",),
    )
    assert idx.lookup("Python programming") is None


def test_unordered_not_in_default():
    # "United States" hits via word_overlap_ordered: ["united", "states"] is a
    # contiguous prefix of ["united", "states", "of", "america"]. Unordered is
    # still not needed for this case; ordered covers it.
    idx = HashNormTitleIndex(["United States of America"])
    assert idx.lookup("United States") == "United States of America"


def test_word_overlap_ordered_in_default_unordered_not():
    idx = HashNormTitleIndex(["Russian Civil War", "United States of America"])
    assert "word_overlap_ordered" in idx.strategies
    assert "word_overlap_unordered" not in idx.strategies


def test_unordered_empty_query_no_crash():
    idx = HashNormTitleIndex(["Python"], strategies=("word_overlap_unordered",))
    assert idx.lookup("") is None


def test_ordered_empty_query_no_crash():
    idx = HashNormTitleIndex(["Python"], strategies=("word_overlap_ordered",))
    assert idx.lookup("") is None


def test_ordered_punct_only_query():
    idx = HashNormTitleIndex(["Python"], strategies=("word_overlap_ordered",))
    assert idx.lookup("...") is None


# ── edit_distance strategy ───────────────────────────────────────────────────

def test_ed_accent_variant():
    idx = HashNormTitleIndex(["Réunion"], strategies=("edit_distance",))
    assert idx.lookup("Reunion") == "Réunion"


def test_ed_typo_one_char():
    idx = HashNormTitleIndex(["Python"], strategies=("edit_distance",))
    assert idx.lookup("Pythn") == "Python"


def test_ed_typo_two_chars():
    idx = HashNormTitleIndex(["Machine Learning"], strategies=("edit_distance",))
    assert idx.lookup("Machin Lerning") == "Machine Learning"


def test_ed_miss_when_too_different():
    idx = HashNormTitleIndex(["Python"], strategies=("edit_distance",))
    assert idx.lookup("Ruby") is None


def test_ed_short_query_skipped():
    # "Arc" is 3 chars, below the default min_chars=5 — must not match "Art".
    idx = HashNormTitleIndex(["Art"], strategies=("edit_distance",))
    assert idx.lookup("Arc") is None


def test_ed_threshold_boundary():
    # "Pythox" vs normed "Python" → 1 char differs out of 6 → similarity ~83%.
    # Default threshold=0.2 means cutoff=80%; 83% > 80% so it should match.
    idx = HashNormTitleIndex(["Python"], strategies=("edit_distance",))
    assert idx.lookup("Pythox") == "Python"
    # "Pxthox" — 2 chars differ out of 6 → similarity ~67%, below 80% cutoff.
    assert idx.lookup("Pxthox") is None


def test_ed_not_in_default_strategies():
    # Default strategies do not include edit_distance — typo must miss.
    idx = HashNormTitleIndex(["Python"])
    assert idx.lookup("Pythn") is None


def test_ed_fires_after_other_strategies_miss():
    # "Pythn" misses exact, norm (different normed form), and word_overlap
    # (word "pythn" not in word_index for "Python"). edit_distance catches it.
    idx = HashNormTitleIndex(
        ["Python"],
        strategies=("exact", "norm", "word_overlap_ordered", "edit_distance"),
    )
    assert idx.lookup("Pythn") == "Python"


def test_ed_import_error_without_rapidfuzz(monkeypatch):
    import sys
    idx = HashNormTitleIndex(["Python"], strategies=("edit_distance",))
    # Temporarily hide rapidfuzz from the import system.
    monkeypatch.setitem(sys.modules, "rapidfuzz", None)
    monkeypatch.setitem(sys.modules, "rapidfuzz.distance", None)
    monkeypatch.setitem(sys.modules, "rapidfuzz.process", None)
    with pytest.raises(ImportError, match="rapidfuzz"):
        idx.lookup("Pythn")


# ── MarkdownPromptAnnotator integration ──────────────────────────────────────

def test_annotator_uses_title_index_on_corpus_miss():
    """When the model generates a case-variant title, title_index resolves it."""
    from unittest.mock import MagicMock
    from eval.link_annotator import MarkdownPromptAnnotator

    corpus = MagicMock()
    corpus.has_document.side_effect = lambda s: s == "Python (programming language)"
    corpus.get_document.return_value = iter([1, 2, 3])

    idx = HashNormTitleIndex(["Python (programming language)"])
    ann = MarkdownPromptAnnotator(
        corpus=corpus,
        title_index=idx,
        link_retrieval_mode="corpus_only",
    )

    aux_lists, aux_ids, fired = ann._fetch_aux(
        model=MagicMock(), target_str="python (programming language)", device="cpu"
    )
    assert fired is True
    assert aux_ids == ["Python (programming language)"]


def test_annotator_without_title_index_uses_verbatim():
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

    _, _, fired_miss = ann._fetch_aux(
        model=MagicMock(), target_str="python (programming language)", device="cpu"
    )
    assert fired_miss is False

    _, _, fired_hit = ann._fetch_aux(
        model=MagicMock(), target_str="Python (programming language)", device="cpu"
    )
    assert fired_hit is True


def test_annotator_ordered_overlap_via_title_index():
    """word_overlap_ordered resolves a truncated title for the annotator."""
    from unittest.mock import MagicMock
    from eval.link_annotator import MarkdownPromptAnnotator

    corpus = MagicMock()
    corpus.has_document.side_effect = lambda s: s == "Russian Civil War"
    corpus.get_document.return_value = iter([10, 20, 30])

    idx = HashNormTitleIndex(
        ["Russian Civil War"],
        strategies=("exact", "norm", "word_overlap_ordered"),
    )
    ann = MarkdownPromptAnnotator(
        corpus=corpus,
        title_index=idx,
        link_retrieval_mode="corpus_only",
    )

    aux_lists, aux_ids, fired = ann._fetch_aux(
        model=MagicMock(), target_str="Russian Civil", device="cpu"
    )
    assert fired is True
    assert aux_ids == ["Russian Civil War"]
