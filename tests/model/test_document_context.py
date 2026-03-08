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

    assert entry.prefix_tokens == []
    assert entry.tokens == [1, 2, 3]   # body only
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
    """Layout policy prefix tokens are stored in prefix_tokens; body stays separate."""
    class FakePolicy:
        def prefix_length(self, info: DocLayoutInfo): return 1
        def suffix_length(self, info: DocLayoutInfo): return 1
        def prefix_tokens(self, info: DocLayoutInfo): return [999]
        def suffix_tokens(self, info: DocLayoutInfo): return [888]

    ctx = make_context()
    entry = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2], layout_policy=FakePolicy())
    assert entry.prefix_tokens == [999]
    assert entry.tokens == [1, 2]   # body only
    assert entry.suffix_tokens == []  # suffix applied only at mark_done


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


# ---------------------------------------------------------------------------
# add_corpus_doc
# ---------------------------------------------------------------------------

def test_add_corpus_doc_stores_tokens():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    entry = ctx.add_corpus_doc(
        raw_identifier="Python",
        corpus_tokens=[10, 20, 30],
        layout_policy=None,
        parent_raw_identifier="",
        depth=1,
        before_entry=root,
    )
    assert entry.tokens == [10, 20, 30]
    assert entry.suffix_tokens == []


def test_add_corpus_doc_metadata():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    entry = ctx.add_corpus_doc(
        raw_identifier="Python",
        corpus_tokens=[10],
        layout_policy=None,
        parent_raw_identifier="",
        depth=1,
        before_entry=root,
    )
    assert entry.source == "corpus"
    assert entry.is_root is False
    assert entry.done is True
    assert entry.truncated is False
    assert entry.raw_identifier == "Python"
    assert entry.parent_raw_identifier == ""
    assert entry.depth == 1


def test_add_corpus_doc_inserted_before_root():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_corpus_doc(
        raw_identifier="Python",
        corpus_tokens=[10, 20],
        layout_policy=None,
        parent_raw_identifier="",
        depth=1,
        before_entry=root,
    )
    # Topological order: aux before root
    assert ctx._docs[0].raw_identifier == "Python"
    assert ctx._docs[1].is_root is True


def test_add_corpus_doc_with_layout_prefix():
    class FakePolicy:
        def prefix_length(self, info): return 2
        def suffix_length(self, info): return 0
        def prefix_tokens(self, info): return [777, 888]
        def suffix_tokens(self, info): return []

    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    entry = ctx.add_corpus_doc(
        raw_identifier="Python",
        corpus_tokens=[10, 20],
        layout_policy=FakePolicy(),
        parent_raw_identifier="",
        depth=1,
        before_entry=root,
    )
    assert entry.prefix_tokens == [777, 888]
    assert entry.tokens == [10, 20]   # body only
    assert entry.suffix_tokens == []


def test_add_corpus_doc_with_layout_suffix():
    class FakePolicy:
        def prefix_length(self, info): return 0
        def suffix_length(self, info): return 1
        def prefix_tokens(self, info): return []
        def suffix_tokens(self, info): return [999]

    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    entry = ctx.add_corpus_doc(
        raw_identifier="Python",
        corpus_tokens=[10, 20],
        layout_policy=FakePolicy(),
        parent_raw_identifier="",
        depth=1,
        before_entry=root,
    )
    assert entry.prefix_tokens == []
    assert entry.tokens == [10, 20]
    assert entry.suffix_tokens == [999]


def test_add_corpus_doc_increments_num_aux():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    assert ctx.num_aux_docs == 0
    ctx.add_corpus_doc("A", [1], None, "", 1, root)
    assert ctx.num_aux_docs == 1
    ctx.add_corpus_doc("B", [2], None, "", 1, root)
    assert ctx.num_aux_docs == 2


# ---------------------------------------------------------------------------
# add_generated_doc
# ---------------------------------------------------------------------------

def test_add_generated_doc_starts_empty():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    entry = ctx.add_generated_doc(
        raw_identifier="Go",
        layout_policy=None,
        parent_raw_identifier="",
        depth=1,
        before_entry=root,
    )
    assert entry.tokens == []
    assert entry.done is False
    assert entry.source == "generated"
    assert entry.is_root is False


def test_add_generated_doc_with_layout_prefix():
    class FakePolicy:
        def prefix_length(self, info): return 1
        def suffix_length(self, info): return 0
        def prefix_tokens(self, info): return [555]
        def suffix_tokens(self, info): return []

    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    entry = ctx.add_generated_doc("Go", FakePolicy(), "", 1, root)
    assert entry.prefix_tokens == [555]
    assert entry.tokens == []   # body starts empty


def test_add_generated_doc_inserted_before_root():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_generated_doc("Go", None, "", 1, root)
    assert ctx._docs[0].raw_identifier == "Go"
    assert ctx._docs[1].is_root is True


def test_add_generated_doc_tokens_accumulate():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    entry = ctx.add_generated_doc("Go", None, "", 1, root)
    ctx.append_token(entry, 10)
    ctx.append_token(entry, 20)
    assert entry.tokens == [10, 20]


# ---------------------------------------------------------------------------
# has_identifier
# ---------------------------------------------------------------------------

def test_has_identifier_true_for_active():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_corpus_doc("Python", [10], None, "", 1, root)
    assert ctx.has_identifier("Python") is True


def test_has_identifier_false_for_absent():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[1])
    assert ctx.has_identifier("Python") is False


def test_has_identifier_false_for_root_empty_string():
    # Root uses "" which should never collide with real link targets
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[1])
    assert ctx.has_identifier("") is True   # root itself is found by ""
    assert ctx.has_identifier("anything") is False


# ---------------------------------------------------------------------------
# can_add_document / make_room / evict_oldest_aux
# ---------------------------------------------------------------------------

def test_can_add_document_true_when_under_limits():
    ctx = make_context(max_context_length=100, max_auxiliary_documents=6)
    ctx.add_root(raw_identifier="", prompt_tokens=[1, 2, 3])  # 3 tokens
    assert ctx.can_add_document(10) is True


def test_can_add_document_false_when_token_limit_exceeded():
    ctx = make_context(max_context_length=10)
    ctx.add_root(raw_identifier="", prompt_tokens=[1, 2, 3])  # 3 tokens used
    assert ctx.can_add_document(8) is False  # 3 + 8 = 11 > 10


def test_can_add_document_false_when_aux_limit_reached():
    ctx = make_context(max_context_length=1000, max_auxiliary_documents=2)
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_corpus_doc("A", [1], None, "", 1, root)
    ctx.add_corpus_doc("B", [2], None, "", 1, root)
    assert ctx.can_add_document(1) is False  # 2 aux docs == max


def test_evict_oldest_aux_removes_first_non_root():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_corpus_doc("A", [10], None, "", 1, root)
    ctx.add_corpus_doc("B", [20], None, "", 1, root)
    evicted = ctx.evict_oldest_aux()
    assert evicted.raw_identifier == "A"
    assert ctx.num_aux_docs == 1
    assert ctx._docs[0].raw_identifier == "B"


def test_evict_oldest_aux_appends_to_evicted_list():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_corpus_doc("A", [10], None, "", 1, root)
    ctx.evict_oldest_aux()
    assert len(ctx._evicted) == 1
    assert ctx._evicted[0].raw_identifier == "A"


def test_evict_oldest_aux_raises_when_no_aux():
    ctx = make_context()
    ctx.add_root(raw_identifier="", prompt_tokens=[1])
    with pytest.raises(RuntimeError):
        ctx.evict_oldest_aux()


def test_make_room_evicts_until_space():
    # 3-token root; max_context_length=10; add two 3-token aux docs
    ctx = make_context(max_context_length=10, max_auxiliary_documents=6)
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2, 3])  # 3 tokens
    ctx.add_corpus_doc("A", [10, 11, 12], None, "", 1, root)         # +3 = 6
    ctx.add_corpus_doc("B", [20, 21, 22], None, "", 1, root)         # +3 = 9
    # Need 4 more tokens: 9+4=13 > 10; must evict A (3 tokens freed → 6+4=10 ok)
    success = ctx.make_room(4)
    assert success is True
    assert ctx.num_aux_docs == 1
    assert ctx._docs[0].raw_identifier == "B"


def test_make_room_returns_false_when_only_root():
    ctx = make_context(max_context_length=5)
    ctx.add_root(raw_identifier="", prompt_tokens=[1, 2, 3])  # 3 tokens
    # Need 10 more but max is 5; no aux docs to evict
    success = ctx.make_room(10)
    assert success is False


def test_make_room_no_op_when_already_fits():
    ctx = make_context(max_context_length=100)
    ctx.add_root(raw_identifier="", prompt_tokens=[1])
    assert ctx.make_room(5) is True
    assert ctx.num_aux_docs == 0  # nothing was evicted


# ---------------------------------------------------------------------------
# build_sequence with multiple documents
# ---------------------------------------------------------------------------

def test_build_sequence_two_docs_correct_offsets():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1, 2])         # 2 tokens
    ctx.add_corpus_doc("aux", [10, 20, 30], None, "", 1, root)           # 3 tokens before root
    tokens, doc_spans = ctx.build_sequence()

    assert tokens.shape == (1, 5)
    assert tokens[0].tolist() == [10, 20, 30, 1, 2]
    assert doc_spans[0].start == 0 and doc_spans[0].end == 3   # aux
    assert doc_spans[1].start == 3 and doc_spans[1].end == 5   # root


def test_build_sequence_offsets_after_append_to_root():
    """Appending to the root shifts nothing since root is last."""
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_corpus_doc("aux", [10, 20], None, "", 1, root)
    ctx.append_token(root, 2)
    _, doc_spans = ctx.build_sequence()
    # aux: [0, 2), root: [2, 4)
    assert doc_spans[0].end == 2
    assert doc_spans[1].start == 2
    assert doc_spans[1].end == 4


# ---------------------------------------------------------------------------
# get_all_documents includes evicted entries
# ---------------------------------------------------------------------------

def test_get_all_documents_includes_evicted():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_corpus_doc("A", [10], None, "", 1, root)
    ctx.evict_oldest_aux()
    docs = ctx.get_all_documents()
    identifiers = [d.raw_identifier for d in docs]
    assert "" in identifiers    # root
    assert "A" in identifiers   # evicted but still returned


def test_get_all_documents_root_first_then_active_then_evicted():
    ctx = make_context()
    root = ctx.add_root(raw_identifier="", prompt_tokens=[1])
    ctx.add_corpus_doc("A", [10], None, "", 1, root)
    ctx.add_corpus_doc("B", [20], None, "", 1, root)
    ctx.evict_oldest_aux()  # evicts A

    docs = ctx.get_all_documents()
    assert docs[0].is_root is True
    assert docs[1].raw_identifier == "B"   # active aux
    assert docs[2].raw_identifier == "A"   # evicted last
