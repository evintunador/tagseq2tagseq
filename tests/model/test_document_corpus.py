"""
tests/model/test_document_corpus.py — unit tests for the dataset-agnostic
PretokCorpus index-building logic.

These exercise the pure `_build_indexes` helper directly with synthetic node
dicts, so no on-disk shards / GraphIndex are required. `index_doc_span` (the
match-key function) never touches the tokenizer, so real detectors are used with
a trivial decode_fn.
"""
import pytest

from eval.title_index import HashNormTitleIndex
from model.document_corpus import _build_indexes, _NodeSpan, _resolve_target
from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
from model.graph_traversal.arxiv_cite_detector import ArxivCiteDetector
from model.graph_traversal.python_import_detector import PythonImportDetector


def _decode(_toks):
    # index_doc_span does not decode tokens; a stub keeps detector construction cheap.
    return ""


def _node(raw, normed):
    return {"raw_identifier": raw, "normed_identifier": normed}


def _nodes(*pairs):
    """Build a normed_identifier -> node dict mapping, as GraphIndex.nodes stores."""
    return {normed: _node(raw, normed) for raw, normed in pairs}


# --- resolution helper: call the real PretokCorpus._resolve implementation ---
# _resolve_target is the exact single source of truth used by PretokCorpus, so
# tests exercise real resolution ordering without on-disk shards.

def _resolve(exact, key, target, has_detector, title_index=None):
    return _resolve_target(
        target, exact, key, has_detector=has_detector, title_index=title_index
    )


# ─── wiki / markdown: exact match, detector-key mirrors it ───────────────────

def test_markdown_exact_match_unchanged():
    det = MarkdownLinkDetector(decode_fn=_decode)
    nodes = _nodes(("Python (programming language)", "python_a1"), ("France", "france_b2"))
    exact, key = _build_indexes(nodes, det)
    # markdown index_doc_span returns raw_identifier verbatim → key mirrors exact.
    assert exact == {"Python (programming language)": "python_a1", "France": "france_b2"}
    assert key == exact
    assert _resolve(exact, key, "France", True) == "france_b2"
    assert _resolve(exact, key, "Nonexistent Title", True) is None


# ─── arxiv: titles with spaces/punctuation, exact match ──────────────────────

def test_arxiv_exact_match_unchanged():
    det = ArxivCiteDetector(decode_fn=_decode)
    nodes = _nodes(
        ("Statistics of Turbulence from Spectral-Line Data Cubes", "arx_1"),
        ("On the inverse Compton scattering model of radio pulsars", "arx_2"),
    )
    exact, key = _build_indexes(nodes, det)
    assert key == exact
    assert _resolve(exact, key, "On the inverse Compton scattering model of radio pulsars", True) == "arx_2"


# ─── code / python: single-repo bare-path hit via detector key ───────────────

def test_python_single_repo_bare_path_hit():
    det = PythonImportDetector(decode_fn=_decode)
    nodes = _nodes(
        ("owner/repo:utils/helpers.py", "h1"),
        ("owner/repo:setup.py", "s1"),
    )
    exact, key = _build_indexes(nodes, det)
    # Detector key strips the repo prefix → bare path resolves.
    assert key == {"utils/helpers.py": "h1", "setup.py": "s1"}
    assert _resolve(exact, key, "utils/helpers.py", True) == "h1"
    # Full repo-qualified identifier still resolves via the exact index.
    assert _resolve(exact, key, "owner/repo:setup.py", True) == "s1"


def test_node_span_shim_exposes_raw_identifier_only():
    span = _NodeSpan("owner/repo:a/b.py")
    assert span.raw_identifier == "owner/repo:a/b.py"
    det = PythonImportDetector(decode_fn=_decode)
    assert det.index_doc_span(span) == "a/b.py"


# ─── detector=None: exact-only, no detector-key index ────────────────────────

def test_detector_none_exact_only():
    nodes = _nodes(("owner/repo:utils/helpers.py", "h1"))
    exact, key = _build_indexes(nodes, None)
    assert key == {}
    # Full raw identifier resolves; bare path does not (no detector-key fallback).
    assert _resolve(exact, key, "owner/repo:utils/helpers.py", False) == "h1"
    assert _resolve(exact, key, "utils/helpers.py", False) is None


# ─── robustness ──────────────────────────────────────────────────────────────

def test_missing_fields_skipped():
    nodes = {
        "n1": {"raw_identifier": "A", "normed_identifier": "n1"},
        "n2": {"raw_identifier": "B"},           # missing normed_identifier
        "n3": {"normed_identifier": "n3"},        # missing raw_identifier
    }
    det = MarkdownLinkDetector(decode_fn=_decode)
    exact, key = _build_indexes(nodes, det)
    assert exact == {"A": "n1"}
    assert key == {"A": "n1"}


def test_detector_key_first_wins_on_collision():
    """When two docs map to the same detector key, the first-seen wins."""
    det = PythonImportDetector(decode_fn=_decode)
    # Two repos both have setup.py → same bare-path key. (This is why code corpora
    # must be single-repo; here we only assert deterministic first-wins.)
    nodes = {}
    nodes["a1"] = _node("repoA:setup.py", "a1")
    nodes["b1"] = _node("repoB:setup.py", "b1")
    exact, key = _build_indexes(nodes, det)
    assert key["setup.py"] == "a1"                # first inserted wins
    assert exact["repoA:setup.py"] == "a1" and exact["repoB:setup.py"] == "b1"


# ─── fuzzy tier: near-miss recovery via HashNormTitleIndex ───────────────────
# The fuzzy tier fires ONLY after both exact indexes miss, so it can only add
# resolutions (near-miss titles) — never override or change an exact hit.

def _wiki_indexes():
    det = MarkdownLinkDetector(decode_fn=_decode)
    nodes = _nodes(
        ("Russian Civil War", "rcw"),
        ("Python (programming language)", "py"),
        ("France", "fr"),
    )
    return _build_indexes(nodes, det)


def test_fuzzy_off_by_default_near_miss_misses():
    """Without a title_index, a near-miss target still returns None (unchanged)."""
    exact, key = _wiki_indexes()
    # Casing/punctuation variant: not an exact or detector-key hit.
    assert _resolve(exact, key, "russian civil war", True) is None
    assert _resolve(exact, key, "Python programming language", True) is None


def test_fuzzy_norm_recovers_casing_variant():
    exact, key = _wiki_indexes()
    idx = HashNormTitleIndex(exact.keys(), strategies=("exact", "norm"))
    # "norm" strategy normalizes casing/punctuation → resolves to the corpus doc.
    assert _resolve(exact, key, "russian civil war", True, title_index=idx) == "rcw"


def test_fuzzy_word_overlap_recovers_truncated_title():
    exact, key = _wiki_indexes()
    idx = HashNormTitleIndex(
        exact.keys(), strategies=("exact", "norm", "word_overlap_ordered")
    )
    # Model emitted only a prefix of the real title.
    assert _resolve(exact, key, "Russian Civil", True, title_index=idx) == "rcw"


def test_fuzzy_edit_distance_recovers_typo():
    exact, key = _wiki_indexes()
    idx = HashNormTitleIndex(
        exact.keys(),
        strategies=("exact", "norm", "edit_distance"),
        edit_distance_threshold=0.2,
    )
    # Single-char typo in a long title → within edit-distance threshold.
    assert _resolve(exact, key, "Russian Civil Waer", True, title_index=idx) == "rcw"


def test_fuzzy_exact_hit_short_circuits_before_fuzzy():
    """An exact/detector-key hit never consults the fuzzy index."""
    exact, key = _wiki_indexes()
    # A title_index that would (wrongly) map everything to 'fr' if consulted.
    class _Trap:
        def lookup(self, _s):
            raise AssertionError("fuzzy index consulted despite an exact hit")
    assert _resolve(exact, key, "France", True, title_index=_Trap()) == "fr"


def test_fuzzy_hallucinated_title_still_misses():
    """A target with no plausible corpus match returns None even with fuzzy on."""
    exact, key = _wiki_indexes()
    idx = HashNormTitleIndex(
        exact.keys(), strategies=("exact", "norm", "word_overlap_ordered", "edit_distance")
    )
    assert _resolve(exact, key, "Completely Fabricated Nonsense Topic", True,
                    title_index=idx) is None
