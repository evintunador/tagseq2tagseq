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


# ── TrieTitleIndex ────────────────────────────────────────────────────────────

class _FakeTok:
    """Minimal tokenizer for trie tests: maps each character to ord(char)."""
    def encode(self, s: str):
        return [ord(c) for c in s]
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _make_trie(titles, **kwargs):
    from eval.link_annotator import TrieTitleIndex
    return TrieTitleIndex(titles, _FakeTok(), **kwargs)


def _make_model_for_trie(token_sequence, vocab_size=200):
    """Model whose forward_inference always returns high logit for the next expected token."""
    import torch
    from unittest.mock import MagicMock

    model = MagicMock()
    call_count = [0]

    def _fwd(tokens, doc_spans, mask_type=None, **kw):
        step = call_count[0]
        call_count[0] += 1
        T = tokens.shape[1]
        logits = torch.zeros(1, T, vocab_size)
        if step < len(token_sequence):
            logits[0, -1, token_sequence[step]] = 10.0
        return logits

    model.forward_inference.side_effect = _fwd
    return model


class TestTrieTitleIndex:

    def test_satisfies_title_index_protocol(self):
        idx = _make_trie(["Python"])
        assert isinstance(idx, TitleIndex)

    def test_lookup_no_fallback_returns_none(self):
        idx = _make_trie(["Python"])
        assert idx.lookup("Python") is None
        assert idx.lookup("anything") is None

    def test_lookup_delegates_to_fallback(self):
        fallback = HashNormTitleIndex(["Python"])
        idx = _make_trie(["Python"], fallback_index=fallback)
        assert idx.lookup("python") == "Python"
        assert idx.lookup("missing") is None

    def test_generate_title_exact_match(self):
        """Model always predicts the correct next token → committed leaf."""
        title = "AB"
        token_seq = [ord("A"), ord("B")]
        idx = _make_trie([title])
        model = _make_model_for_trie(token_seq)
        prefix = [1, 2, 3]
        result = idx.generate_title(model, prefix, "cpu")
        assert result is not None
        raw, toks = result
        assert raw == title
        assert toks == token_seq

    def test_generate_title_forces_onto_valid_path(self):
        """Even when the model prefers an off-trie token it is masked out and the
        trie path is still followed — constrained generation commits to the corpus."""
        # Trie has only "AB"; model always prefers "Z" (off-trie).
        # The masking forces the model onto A then B regardless.
        idx = _make_trie(["AB"])
        model = _make_model_for_trie([ord("Z"), ord("Z")])  # model wants Z both times
        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        raw, toks = result
        assert raw == "AB"

    def test_generate_title_interior_leaf_commits_on_close_paren(self):
        """'New York' is a prefix of 'New York City'; model assigns P(")") > P("C")."""
        import torch
        from unittest.mock import MagicMock

        titles = ["New York", "New York City"]
        idx = _make_trie(titles)

        # close_paren_id for _FakeTok: ord(")")
        close_id = ord(")")
        # Token IDs for the two titles
        ny_ids   = [ord(c) for c in "New York"]
        nyc_ids  = [ord(c) for c in "New York City"]
        # "C" is the next child token after "New York " (space is part of "City")
        next_child = ord(" ")  # " " from " City"

        call_count = [0]

        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            step = call_count[0]
            call_count[0] += 1
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            # Steps 0..len(ny_ids)-1: steer model along "New York"
            if step < len(ny_ids):
                logits[0, -1, ny_ids[step]] = 10.0
            else:
                # At the interior-leaf check: make P(")") > P(next_child)
                logits[0, -1, close_id] = 10.0
                logits[0, -1, next_child] = 5.0
            return logits

        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        raw, toks = result
        assert raw == "New York"

    def test_generate_title_interior_leaf_continues_when_child_wins(self):
        """'New York' interior leaf, but P(child) > P(")") → continues to 'New York City'."""
        import torch
        from unittest.mock import MagicMock

        titles = ["New York", "New York City"]
        idx = _make_trie(titles)

        close_id   = ord(")")
        ny_ids     = [ord(c) for c in "New York"]
        space_id   = ord(" ")
        city_ids   = [ord(c) for c in "City"]
        nyc_extra  = [space_id] + city_ids  # tokens after "New York"

        call_count = [0]

        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            step = call_count[0]
            call_count[0] += 1
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            all_ids = ny_ids + nyc_extra
            if step < len(ny_ids):
                # Steer along "New York"
                logits[0, -1, ny_ids[step]] = 10.0
            elif step == len(ny_ids):
                # Interior-leaf check: child wins over ")"
                logits[0, -1, close_id] = 3.0
                logits[0, -1, space_id] = 10.0
            elif step < len(all_ids):
                logits[0, -1, all_ids[step]] = 10.0
            return logits

        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        raw, toks = result
        assert raw == "New York City"

    def test_generate_title_threshold_abort(self):
        """Joint log-prob below threshold → None (with threshold set)."""
        import torch
        from unittest.mock import MagicMock

        title = "AB"
        idx = _make_trie([title], min_joint_logprob=-0.01)  # very tight threshold

        # Model assigns low probability to the correct token so joint logprob is low.
        call_count = [0]
        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            step = call_count[0]
            call_count[0] += 1
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            if step == 0:
                # Tiny logit so softmax P is near uniform → log P very negative.
                logits[0, -1, ord("A")] = 0.001
            return logits
        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is None

    def test_generate_title_threshold_none_never_aborts(self):
        """With threshold=None a low-prob path still commits on reaching a leaf."""
        title = "A"
        token_seq = [ord("A")]
        idx = _make_trie([title], min_joint_logprob=None)
        import torch
        from unittest.mock import MagicMock

        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            logits[0, -1, ord("A")] = 0.001  # tiny but present
            return logits

        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        raw, toks = result
        assert raw == "A"

    def test_collision_first_inserted_wins(self):
        """Two titles tokenizing identically — first-inserted raw_identifier is returned."""
        # With _FakeTok, encode("A") = [65]; use two different strings that share tokens.
        # Simplest: both are the same string — only one can be first.
        from eval.link_annotator import TrieTitleIndex
        tok = _FakeTok()
        idx = TrieTitleIndex(["Alpha", "Alpha"], tok)
        # Internal: both have same token path; second insert is a no-op.
        token_seq = [ord(c) for c in "Alpha"]
        model = _make_model_for_trie(token_seq)
        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        assert result[0] == "Alpha"

    def test_empty_corpus_returns_none(self):
        idx = _make_trie([])
        model = _make_model_for_trie([65])
        result = idx.generate_title(model, [1], "cpu")
        assert result is None


class TestTrieTitleIndexBeam:
    """Tests for beam_width > 1 behaviour."""

    def test_beam_width_1_same_as_greedy(self):
        """beam_width=1 should produce the same result as default (greedy) behaviour."""
        import torch
        from unittest.mock import MagicMock

        # Corpus: only "AB". Model always picks A then B.
        idx1 = _make_trie(["AB"], beam_width=1)
        idx2 = _make_trie(["AB"])  # default beam_width=1

        for idx in (idx1, idx2):
            model = _make_model_for_trie([ord("A"), ord("B")])
            result = idx.generate_title(model, [1], "cpu")
            assert result is not None
            assert result[0] == "AB"

    def test_beam_selects_higher_total_logprob(self):
        """beam_width=2 picks "AB" over "CD" when greedy would pick "CD".

        Corpus: "AB" and "CD" (both 2 tokens, no shared prefix).
        P("C")=0.6 > P("A")=0.3 at step 0 → greedy picks C → "CD".
        But P("D"|C)=0.1 is low, so logprob("CD")=log(0.6)+log(0.1)≈-2.81.
        P("B"|A)=0.999, so logprob("AB")=log(0.3)+log(0.999)≈-1.20.
        beam=2 explores both paths and returns "AB" (higher total logprob).
        """
        import torch
        import math
        from unittest.mock import MagicMock

        idx = _make_trie(["AB", "CD"], beam_width=2)
        close_id = ord(")")

        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            title_so_far = tokens[0, len([1]):].tolist()  # strip the single prefix token
            if len(title_so_far) == 0:
                # Step 0: P("C")=0.6 > P("A")=0.3 — greedy picks "C"
                logits[0, -1, ord("C")] = math.log(0.6 / 0.4)   # ~0.405
                logits[0, -1, ord("A")] = math.log(0.3 / 0.4)   # ~-0.288
            elif title_so_far[-1] == ord("A"):
                # After "A": P("B")=0.999
                logits[0, -1, ord("B")] = 10.0
            elif title_so_far[-1] == ord("C"):
                # After "C": P("D")=0.1, everything else near uniform
                logits[0, -1, ord("D")] = math.log(0.1 / 0.9)   # ~-2.197
            # close paren always low
            logits[0, -1, close_id] = -10.0
            return logits

        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        raw, toks = result
        # Beam should pick "AB" (higher total logprob) over "CD" (greedy choice)
        assert raw == "AB", f"Expected 'AB' but got {raw!r}"

    def test_beam_width_1_greedy_picks_wrong_branch(self):
        """Sanity check: beam_width=1 (greedy) picks 'CD' in the same scenario."""
        import torch
        import math
        from unittest.mock import MagicMock

        idx = _make_trie(["AB", "CD"], beam_width=1)
        close_id = ord(")")

        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            title_so_far = tokens[0, len([1]):].tolist()
            if len(title_so_far) == 0:
                logits[0, -1, ord("C")] = math.log(0.6 / 0.4)
                logits[0, -1, ord("A")] = math.log(0.3 / 0.4)
            elif title_so_far[-1] == ord("A"):
                logits[0, -1, ord("B")] = 10.0
            elif title_so_far[-1] == ord("C"):
                logits[0, -1, ord("D")] = math.log(0.1 / 0.9)
            logits[0, -1, close_id] = -10.0
            return logits

        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        raw, toks = result
        # Greedy picks C first → commits to "CD"
        assert raw == "CD", f"Expected greedy to pick 'CD' but got {raw!r}"

    def test_beam_all_paths_pruned_returns_none(self):
        """All beam paths pruned by min_joint_logprob → None."""
        import torch
        from unittest.mock import MagicMock

        idx = _make_trie(["AB", "CD"], beam_width=2, min_joint_logprob=-0.001)

        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            # All valid tokens get low logit → low prob → joint logprob drops fast
            logits[0, -1, ord("A")] = 0.001
            logits[0, -1, ord("C")] = 0.001
            logits[0, -1, ord(")")] = -10.0
            return logits

        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is None

    def test_beam_multiple_candidates_best_wins(self):
        """Both paths complete; the one with higher total logprob is returned.

        Corpus: "A" (1 token) and "BC" (2 tokens). beam_width=2 keeps both paths.
        "A" commits after 1 step with joint_logprob = log(P("A")).
        "BC" commits after 2 steps with log(P("B")) + log(P("C"|B)).
        Since every extra factor multiplies by something < 1, "A" wins on total
        joint logprob. This test confirms the best candidate is selected, not merely
        the last one found.
        """
        import torch
        import math
        from unittest.mock import MagicMock

        idx = _make_trie(["A", "BC"], beam_width=2)
        close_id = ord(")")

        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            title_so_far = tokens[0, 1:].tolist()
            if len(title_so_far) == 0:
                # P("B") slightly > P("A"); both kept by beam=2
                logits[0, -1, ord("B")] = math.log(0.5 / 0.1)
                logits[0, -1, ord("A")] = math.log(0.4 / 0.1)
                logits[0, -1, close_id] = -10.0
            elif title_so_far == [ord("A")]:
                logits[0, -1, close_id] = 10.0  # pure leaf → commit
            elif title_so_far[-1] == ord("B"):
                logits[0, -1, ord("C")] = math.log(0.9 / 0.1)
                logits[0, -1, close_id] = -10.0
            elif title_so_far[-1] == ord("C"):
                logits[0, -1, close_id] = 10.0  # pure leaf → commit
            return logits

        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        raw, toks = result
        # "A" has higher joint logprob (1 factor) than "BC" (2 factors, each <1).
        assert raw == "A", f"Expected 'A' (higher joint logprob) but got {raw!r}"

    def test_length_penalty_favors_longer_title(self):
        """With length_penalty=1.0, per-token mean log-prob is used.

        Corpus: "A" (1 token) and "BC" (2 tokens), same setup as
        test_beam_multiple_candidates_best_wins. Without length penalty "A" wins
        because it has fewer factors. With length_penalty=1.0 we compare
        log P("A") / 1  vs  (log P("B") + log P("C"|B)) / 2.
        If P("B")=0.5, P("C"|B)=0.9: mean = (log0.5 + log0.9)/2 ≈ -0.399
        P("A")=0.4: score = log0.4/1 ≈ -0.916
        "BC" wins on length-normalized score.
        """
        import torch
        import math
        from unittest.mock import MagicMock

        idx = _make_trie(["A", "BC"], beam_width=2, length_penalty=1.0)
        close_id = ord(")")

        def _fwd(tokens, doc_spans, mask_type=None, **kw):
            T = tokens.shape[1]
            logits = torch.zeros(1, T, 200)
            title_so_far = tokens[0, 1:].tolist()
            if len(title_so_far) == 0:
                logits[0, -1, ord("B")] = math.log(0.5 / 0.1)
                logits[0, -1, ord("A")] = math.log(0.4 / 0.1)
                logits[0, -1, close_id] = -10.0
            elif title_so_far == [ord("A")]:
                logits[0, -1, close_id] = 10.0
            elif title_so_far[-1] == ord("B"):
                logits[0, -1, ord("C")] = math.log(0.9 / 0.1)
                logits[0, -1, close_id] = -10.0
            elif title_so_far[-1] == ord("C"):
                logits[0, -1, close_id] = 10.0
            return logits

        model = MagicMock()
        model.forward_inference.side_effect = _fwd

        result = idx.generate_title(model, [1], "cpu")
        assert result is not None
        raw, toks = result
        assert raw == "BC", f"Expected 'BC' (better length-normalized score) but got {raw!r}"
