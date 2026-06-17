"""
Tests for data/arxiv_graph_extractor/extract.py — body rehydration and dual-path
citation resolution. Uses synthetic records (no corpus dependency).
"""
import importlib

import pytest

extract = importlib.import_module("data.arxiv_graph_extractor.extract")


@pytest.fixture(autouse=True)
def _reset_globals():
    """Populate the module globals the rehydration logic reads, then clear after."""
    extract._ARXIV_TO_TITLE = {
        "2401.00002": "Paper Two Title",
        "2401.00003": "Paper Three Title",
    }
    extract._ARXIV_TO_NORMED = {
        "2401.00002": "2401_00002_aaa",
        "2401.00003": "2401_00003_bbb",
    }
    extract._OA_TO_ARXIV = {
        "W123": "2401.00003",   # OpenAlex id that maps to an in-corpus paper
        "W999": "9999.99999",   # maps to an out-of-corpus paper
    }
    yield
    extract._ARXIV_TO_TITLE = {}
    extract._ARXIV_TO_NORMED = {}
    extract._OA_TO_ARXIV = {}


# ---------------------------------------------------------------------------
# _resolve_bib_to_arxiv
# ---------------------------------------------------------------------------

class TestResolveBib:
    def test_direct_contained_arxiv_id(self):
        be = {"contained_arXiv_ids": [{"id": "2401.00002"}], "ids": {}}
        assert extract._resolve_bib_to_arxiv(be) == "2401.00002"

    def test_direct_version_canonicalized(self):
        be = {"contained_arXiv_ids": [{"id": "2401.00002v3"}], "ids": {}}
        assert extract._resolve_bib_to_arxiv(be) == "2401.00002"

    def test_direct_ids_arxiv_id(self):
        be = {"contained_arXiv_ids": [], "ids": {"arxiv_id": "2401.00003"}}
        assert extract._resolve_bib_to_arxiv(be) == "2401.00003"

    def test_openalex_fallback(self):
        # No direct id, but OpenAlex id maps to an in-corpus paper.
        be = {"contained_arXiv_ids": [], "ids": {"open_alex_id": "https://openalex.org/W123"}}
        assert extract._resolve_bib_to_arxiv(be) == "2401.00003"

    def test_direct_preferred_over_openalex(self):
        be = {"contained_arXiv_ids": [{"id": "2401.00002"}],
              "ids": {"open_alex_id": "https://openalex.org/W123"}}
        assert extract._resolve_bib_to_arxiv(be) == "2401.00002"

    def test_out_of_corpus_direct_returns_none(self):
        be = {"contained_arXiv_ids": [{"id": "9999.99999"}], "ids": {}}
        assert extract._resolve_bib_to_arxiv(be) is None

    def test_out_of_corpus_openalex_returns_none(self):
        be = {"contained_arXiv_ids": [], "ids": {"open_alex_id": "https://openalex.org/W999"}}
        assert extract._resolve_bib_to_arxiv(be) is None

    def test_no_ids_returns_none(self):
        assert extract._resolve_bib_to_arxiv({"contained_arXiv_ids": [], "ids": {}}) is None


# ---------------------------------------------------------------------------
# _rehydrate_body
# ---------------------------------------------------------------------------

def _record(text, bib_entries=None, ref_entries=None):
    return {
        "sections": {"S1": {"text": text}},
        "bib_entries": bib_entries or {},
        "ref_entries": ref_entries or {},
    }


# Realistic hex keys: ref_entries uuids are hex-with-dashes; cite keys are SHA1 hex.
_F1 = "f5e8543d-c971-4ef2-9f0d-46bfeacad7c2"
_FIG1 = "84547980-693d-429b-9bd2-5cc3c95d8813"
_T1 = "fc72947b-5dc0-46b9-afee-19c8fec3a561"
_K1 = "b1c06b34d06c7653e80ab0839d5dfa8930fb80f9"
_K2 = "af2293d1c79fd5889a6e2ea5ee52db4c61070451"


class TestRehydrate:
    def test_formula_inlined_as_latex(self):
        rec = _record(
            f"The loss {{{{formula:{_F1}}}}} is minimized.",
            ref_entries={_F1: {"latex": "\\mathcal{L}", "type": "formula"}},
        )
        body, outgoing = extract._rehydrate_body(rec)
        assert "$\\mathcal{L}$" in body
        assert outgoing == []

    def test_figure_with_caption(self):
        rec = _record(
            f"See {{{{figure:{_FIG1}}}}}.",
            ref_entries={_FIG1: {"caption": "Architecture diagram", "type": "figure"}},
        )
        body, _ = extract._rehydrate_body(rec)
        assert "\\ref{figure} (Architecture diagram)" in body

    def test_figure_no_caption(self):
        rec = _record(
            f"See {{{{table:{_T1}}}}}.",
            ref_entries={_T1: {"caption": "NO_CAPTION", "type": "table"}},
        )
        body, _ = extract._rehydrate_body(rec)
        assert "\\ref{table}" in body
        assert "NO_CAPTION" not in body

    def test_incorpus_cite_rewritten_to_title(self):
        rec = _record(
            f"As in {{{{cite:{_K1}}}}} we observe...",
            bib_entries={_K1: {"contained_arXiv_ids": [{"id": "2401.00002"}], "ids": {}}},
        )
        body, outgoing = extract._rehydrate_body(rec)
        assert "\\cite{Paper Two Title}" in body
        assert outgoing == ["2401_00002_aaa"]

    def test_openalex_cite_rewritten(self):
        rec = _record(
            f"Prior work {{{{cite:{_K1}}}}}.",
            bib_entries={_K1: {"contained_arXiv_ids": [],
                               "ids": {"open_alex_id": "https://openalex.org/W123"}}},
        )
        body, outgoing = extract._rehydrate_body(rec)
        assert "\\cite{Paper Three Title}" in body
        assert outgoing == ["2401_00003_bbb"]

    def test_out_of_corpus_cite_dropped(self):
        rec = _record(
            f"Unrelated {{{{cite:{_K1}}}}} reference.",
            bib_entries={_K1: {"contained_arXiv_ids": [{"id": "9999.99999"}], "ids": {}}},
        )
        body, outgoing = extract._rehydrate_body(rec)
        assert "cite" not in body.lower()  # marker fully removed
        assert outgoing == []
        assert "Unrelated  reference." in body  # marker excised, text preserved

    def test_xref_to_ref(self):
        rec = _record("As shown in (REF ) above.")
        body, _ = extract._rehydrate_body(rec)
        assert "\\ref{}" in body
        assert "(REF )" not in body

    def test_outgoing_deduped_and_sorted(self):
        rec = _record(
            f"{{{{cite:{_K1}}}}} and again {{{{cite:{_K2}}}}} and {{{{cite:{_K1}}}}}.",
            bib_entries={
                _K1: {"contained_arXiv_ids": [{"id": "2401.00003"}], "ids": {}},
                _K2: {"contained_arXiv_ids": [{"id": "2401.00002"}], "ids": {}},
            },
        )
        _, outgoing = extract._rehydrate_body(rec)
        assert outgoing == ["2401_00002_aaa", "2401_00003_bbb"]  # sorted, deduped

    def test_multiple_sections_joined(self):
        rec = {
            "sections": {"A": {"text": "First."}, "B": {"text": "Second."}},
            "bib_entries": {}, "ref_entries": {},
        }
        body, _ = extract._rehydrate_body(rec)
        assert body == "First.\n\nSecond."
