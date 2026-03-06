"""
Tests for DocumentContext (Stage 1: single-document subset).

All tests run on CPU — no CUDA required.
"""
import numpy as np
import pytest
import torch

from data.layout import DocLayoutInfo
from model.document_context import DocumentContext, _DocEntry
from model.identifier_utils import create_normed_identifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_context(**overrides):
    kwargs = dict(
        max_context_length=512,
        max_auxiliary_documents=6,
        eviction_policy="drop_oldest",
        device="cpu",
    )
    kwargs.update(overrides)
    return DocumentContext(**kwargs)


# ---------------------------------------------------------------------------
# add_root
# ---------------------------------------------------------------------------

def test_add_root_stores_prompt_tokens():
    ctx = make_context()
    prompt = [1, 2, 3]
    entry = ctx.add_root(raw_identifier="", prompt_tokens=prompt)

    assert entry.tokens == [1, 2, 3]
    assert entry.is_root is True
    assert entry.source == "generated"
    assert entry.depth == 0
    assert entry.done is False
    assert entry.truncated is False
    assert entry.parent_raw_identifier is None
    assert entry.raw_identifier == ""
    assert entry.normed_identifier == ""


def test_add_root_assigns_doc_id_zero():
    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[10])
    assert entry.doc_id == 0


def test_add_root_only_once():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[1])
    with pytest.raises(AssertionError):
        ctx.add_root(raw_identifier="", prompt_tokens=[2])


def test_add_root_with_layout_policy():
    """Layout policy prefix tokens are prepended to the prompt."""
    class FakePolicy:
        def prefix_length(self, info: DocLayoutInfo): return 1
        def suffix_length(self, info: DocLayoutInfo): return 1
        def prefix_tokens(self, info: DocLayoutInfo): return [999]
        def suffix_tokens(self, info: DocLayoutInfo): return [888]

    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2], layout_policy=FakePolicy())
    assert entry.tokens == [999, 1, 2]


# ---------------------------------------------------------------------------
# append_token
# ---------------------------------------------------------------------------

def test_append_token_grows_list():
    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2])
    ctx.append_token(entry, 3)
    ctx.append_token(entry, 4)
    assert entry.tokens == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# mark_done
# ---------------------------------------------------------------------------

def test_mark_done_sets_flag():
    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    assert entry.done is False
    ctx.mark_done(entry)
    assert entry.done is True


def test_mark_done_stores_suffix_separately():
    class FakePolicy:
        def prefix_length(self, info: DocLayoutInfo): return 0
        def suffix_length(self, info: DocLayoutInfo): return 1
        def prefix_tokens(self, info: DocLayoutInfo): return []
        def suffix_tokens(self, info: DocLayoutInfo): return [777]

    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2])
    ctx.mark_done(entry, layout_policy=FakePolicy())
    # Suffix stored separately — body tokens unchanged
    assert entry.tokens == [1, 2]
    assert entry.suffix_tokens == [777]
    assert entry.done is True


def test_build_sequence_includes_suffix():
    class FakePolicy:
        def prefix_length(self, info: DocLayoutInfo): return 0
        def suffix_length(self, info: DocLayoutInfo): return 1
        def prefix_tokens(self, info: DocLayoutInfo): return []
        def suffix_tokens(self, info: DocLayoutInfo): return [777]

    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2])
    ctx.mark_done(entry, layout_policy=FakePolicy())
    tokens, doc_spans = ctx.build_sequence()

    assert tokens.shape == (1, 3)
    assert tokens[0].tolist() == [1, 2, 777]
    assert doc_spans[0].end == 3


def test_total_tokens_includes_suffix():
    class FakePolicy:
        def prefix_length(self, info: DocLayoutInfo): return 0
        def suffix_length(self, info: DocLayoutInfo): return 1
        def prefix_tokens(self, info: DocLayoutInfo): return []
        def suffix_tokens(self, info: DocLayoutInfo): return [777]

    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2])
    assert ctx.total_tokens == 2
    ctx.mark_done(entry, layout_policy=FakePolicy())
    assert ctx.total_tokens == 3


# ---------------------------------------------------------------------------
# total_tokens / num_aux_docs
# ---------------------------------------------------------------------------

def test_total_tokens_counts_all():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[1, 2, 3])
    assert ctx.total_tokens == 3


def test_total_tokens_after_appending():
    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2])
    ctx.append_token(entry, 3)
    assert ctx.total_tokens == 3


def test_num_aux_docs_zero_for_root_only():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[1])
    assert ctx.num_aux_docs == 0


# ---------------------------------------------------------------------------
# build_sequence
# ---------------------------------------------------------------------------

def test_build_sequence_shape():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[10, 20, 30])
    tokens, doc_spans = ctx.build_sequence()

    assert tokens.shape == (1, 3)
    assert tokens.dtype == torch.long
    assert len(doc_spans) == 1


def test_build_sequence_correct_span():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[10, 20, 30])
    tokens, doc_spans = ctx.build_sequence()

    span = doc_spans[0]
    assert span.start == 0
    assert span.end == 3
    assert span.doc_id == 0
    assert span.raw_identifier == ""
    assert span.truncated is False


def test_build_sequence_updates_after_append():
    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2])
    ctx.append_token(entry, 3)
    tokens, doc_spans = ctx.build_sequence()

    assert tokens.shape == (1, 3)
    assert doc_spans[0].end == 3


def test_build_sequence_token_values():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[5, 10, 15])
    tokens, _ = ctx.build_sequence()
    assert tokens[0].tolist() == [5, 10, 15]


def test_build_sequence_device():
    ctx = make_context(device="cpu")
    ctx.add_root(raw_identifier="", prompt_tokens=[1, 2])
    tokens, _ = ctx.build_sequence()
    assert tokens.device.type == "cpu"


# ---------------------------------------------------------------------------
# get_all_documents
# ---------------------------------------------------------------------------

def test_get_all_documents_single_root():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[1, 2, 3])
    docs = ctx.get_all_documents()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.is_root is True
    assert doc.source == "generated"
    assert doc.tokens is not None
    assert isinstance(doc.tokens, np.ndarray)
    assert doc.tokens.tolist() == [1, 2, 3]
    assert doc.text is None  # not decoded yet


def test_get_all_documents_root_is_first():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[99])
    docs = ctx.get_all_documents()
    assert docs[0].is_root is True


def test_get_all_documents_fields():
    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.append_token(entry, 2)
    entry.truncated = True
    ctx.mark_done(entry)

    docs = ctx.get_all_documents()
    doc = docs[0]
    assert doc.depth == 0
    assert doc.truncated is True
    assert doc.parent_raw_identifier is None


# ---------------------------------------------------------------------------
# Layout policy receives correct identifiers (not doc_id counter)
# ---------------------------------------------------------------------------

def test_add_root_layout_policy_receives_identifier():
    """prefix_tokens receives a DocLayoutInfo with correct identifier fields."""
    received = []

    class SpyPolicy:
        def prefix_length(self, info: DocLayoutInfo): return 0
        def suffix_length(self, info: DocLayoutInfo): return 0
        def prefix_tokens(self, info: DocLayoutInfo):
            received.append(info)
            return []
        def suffix_tokens(self, info: DocLayoutInfo): return []

    ctx = make_context()
    ctx.add_root(
        raw_identifier="Python (programming language)",
        prompt_tokens=[1],
        layout_policy=SpyPolicy(),
    )
    assert len(received) == 1
    info = received[0]
    assert info.raw_identifier == "Python (programming language)"
    assert info.normed_identifier == create_normed_identifier("Python (programming language)")
    assert info.body_tokens == [1]  # prompt tokens available at prefix time


def test_mark_done_layout_policy_receives_identifier():
    """suffix_tokens receives a DocLayoutInfo with the document's body tokens."""
    received = []

    class SpyPolicy:
        def prefix_length(self, info: DocLayoutInfo): return 0
        def suffix_length(self, info: DocLayoutInfo): return 0
        def prefix_tokens(self, info: DocLayoutInfo): return []
        def suffix_tokens(self, info: DocLayoutInfo):
            received.append(info)
            return []

    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.append_token(entry, 2)
    ctx.mark_done(entry, layout_policy=SpyPolicy())
    assert len(received) == 1
    info = received[0]
    assert info.raw_identifier == ""
    assert info.normed_identifier == ""
    assert info.body_tokens == [1, 2]  # full body (prefix + generated) at suffix time
