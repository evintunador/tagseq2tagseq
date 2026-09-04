"""
tests/eval/test_scoring.py — unit tests for eval.scoring primitives.

All tests run on CPU with a mock model that returns uniform logits.
No CUDA required.
"""
import math
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from data.collate import DocSpan
from data.layout import EOSLayoutPolicy, NullLayoutPolicy
from eval.scoring import (
    score_completion, score_completions_batched,
    score_completions_independent_batched, score_completion_with_context_docs,
    score_completion_concat,
    score_doc, score_docs_batched, score_doc_with_context,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

VOCAB_SIZE = 256


def _make_mock_model(vocab_size: int = VOCAB_SIZE):
    """Mock TS2TSModel whose forward_inference returns zero logits (uniform distribution).

    Zero logits → softmax gives uniform p = 1/V for every token.
    Expected NLL = log(V).
    """
    model = MagicMock()

    def _forward(tokens, doc_spans, **kwargs):
        T = tokens.shape[1]
        return torch.zeros(1, T, vocab_size)

    model.forward_inference.side_effect = _forward

    # score_doc falls back to model.active_layout_policy when layout_policy=None
    model.active_layout_policy = NullLayoutPolicy()

    # score_completion reads device from model.backbone.parameters()
    dummy_param = nn.Parameter(torch.zeros(1))
    model.backbone.parameters.return_value = iter([dummy_param])

    return model


@pytest.fixture
def mock_model():
    return _make_mock_model()


@pytest.fixture
def null_policy():
    return NullLayoutPolicy()


@pytest.fixture
def eos_policy():
    return EOSLayoutPolicy(eos_token_id=1)


BODY_TOKENS = [10, 20, 30, 40, 50]


# ─── score_doc tests ──────────────────────────────────────────────────────────

def test_score_doc_returns_expected_keys(mock_model, null_policy):
    result = score_doc(mock_model, BODY_TOKENS, null_policy, device="cpu")
    assert set(result.keys()) == {"mean_nll", "num_tokens"}


def test_score_doc_uniform_logits_approx_log_V(mock_model, null_policy):
    result = score_doc(mock_model, BODY_TOKENS, null_policy, device="cpu")
    expected_nll = math.log(VOCAB_SIZE)
    assert result["num_tokens"] > 0
    assert abs(result["mean_nll"] - expected_nll) < 1e-4


def test_score_doc_excludes_prefix_suffix_from_num_tokens(mock_model, eos_policy):
    # EOSLayoutPolicy has no prefix. Because prefix_len==0, the first body token
    # has no preceding logit and is skipped. num_tokens = len(BODY_TOKENS) - 1.
    result = score_doc(mock_model, BODY_TOKENS, eos_policy, device="cpu")
    assert result["num_tokens"] == len(BODY_TOKENS) - 1


def test_score_doc_empty_body_returns_zero(mock_model, null_policy):
    result = score_doc(mock_model, [], null_policy, device="cpu")
    assert result == {"mean_nll": 0.0, "num_tokens": 0}


def test_score_doc_single_token_no_prefix(mock_model, eos_policy):
    # Single-token body with EOS suffix: [tok, EOS]. prefix_len==0 → the body
    # token has no preceding logit, so it is skipped. num_tokens == 0.
    result = score_doc(mock_model, [99], eos_policy, device="cpu")
    assert result["num_tokens"] == 0


def test_score_doc_calls_forward_inference_once(mock_model, null_policy):
    score_doc(mock_model, BODY_TOKENS, null_policy, device="cpu")
    assert mock_model.forward_inference.call_count == 1


def test_score_doc_passes_correct_token_length_to_forward(mock_model, eos_policy):
    # With EOSLayoutPolicy, total sequence = 5 body + 1 EOS = 6 tokens.
    score_doc(mock_model, BODY_TOKENS, eos_policy, device="cpu")
    call_args = mock_model.forward_inference.call_args
    tokens_arg = call_args[0][0]  # positional arg 0
    assert tokens_arg.shape == (1, len(BODY_TOKENS) + 1)  # body + EOS


# ─── score_completion tests ───────────────────────────────────────────────────

def test_score_completion_returns_float(mock_model):
    result = score_completion(mock_model, [1, 2, 3], [4, 5], device="cpu")
    assert isinstance(result, float)


def test_score_completion_uniform_logits_approx_log_V(mock_model):
    result = score_completion(mock_model, [1, 2, 3], [4, 5], device="cpu")
    assert abs(result - math.log(VOCAB_SIZE)) < 1e-4


def test_score_completion_empty_completion_returns_zero(mock_model):
    result = score_completion(mock_model, [1, 2, 3], [], device="cpu")
    assert result == 0.0


def test_score_completion_passes_full_sequence_to_forward(mock_model):
    context = [1, 2, 3]
    completion = [4, 5, 6, 7]
    score_completion(mock_model, context, completion, device="cpu")
    call_args = mock_model.forward_inference.call_args
    tokens_arg = call_args[0][0]
    assert tokens_arg.shape == (1, len(context) + len(completion))


def test_score_completion_prompt_preprocessor_is_called(mock_model):
    """Preprocessor appends a token; forward_inference should see the longer context."""
    context = [1, 2, 3]
    completion = [9]
    extra_token = [99]

    def preprocessor(tokens):
        return tokens + extra_token

    score_completion(mock_model, context, completion,
                     prompt_preprocessor=preprocessor, device="cpu")
    call_args = mock_model.forward_inference.call_args
    tokens_arg = call_args[0][0]
    expected_len = len(context) + len(extra_token) + len(completion)
    assert tokens_arg.shape == (1, expected_len)


def test_score_completion_concat_returns_float(mock_model):
    result = score_completion_concat(
        mock_model, [[7, 8]], [1, 2, 3], [4, 5], device="cpu"
    )
    assert isinstance(result, float)
    assert abs(result - math.log(VOCAB_SIZE)) < 1e-4


def test_score_completion_concat_no_aux_returns_none(mock_model):
    # No non-empty aux span → nothing to concatenate → None.
    assert score_completion_concat(mock_model, [], [1, 2, 3], [4, 5], device="cpu") is None
    assert score_completion_concat(mock_model, [[]], [1, 2, 3], [4, 5], device="cpu") is None


def test_score_completion_concat_empty_completion_returns_zero(mock_model):
    assert score_completion_concat(mock_model, [[7]], [1, 2, 3], [], device="cpu") == 0.0


def test_score_completion_concat_empty_context_returns_none(mock_model):
    assert score_completion_concat(mock_model, [[7]], [], [4, 5], device="cpu") is None


def test_score_completion_concat_packs_aux_before_primary(mock_model):
    aux = [[7, 8, 9], [10]]        # 4 aux tokens total
    context = [1, 2, 3]
    completion = [4, 5]
    score_completion_concat(mock_model, aux, context, completion, device="cpu")
    call_args = mock_model.forward_inference.call_args
    tokens_arg = call_args[0][0]
    # Full sequence = aux(4) + context(3) + completion(2) = 9
    assert tokens_arg.shape == (1, 4 + len(context) + len(completion))


def test_score_completion_concat_propagates_mask_type(mock_model):
    score_completion_concat(
        mock_model, [[7]], [1, 2], [3], mask_type="doc_causal", device="cpu"
    )
    _, kwargs = mock_model.forward_inference.call_args
    assert kwargs.get("mask_type") == "doc_causal"


def test_score_completion_concat_shared_component_id_on_all_spans(mock_model):
    # doc_concatenated only merges spans that share a component_id; every packed
    # span must carry component_id=0 or the cell silently degrades to doc_causal.
    score_completion_concat(mock_model, [[7, 8]], [1, 2, 3], [4, 5], device="cpu")
    spans = mock_model.forward_inference.call_args[0][1]
    assert len(spans) == 2  # one aux span + primary
    assert all(s.component_id == 0 for s in spans)


def test_score_completion_preprocessor_none_unchanged(mock_model):
    """When prompt_preprocessor is None, context is not modified."""
    context = [1, 2, 3]
    completion = [4]
    score_completion(mock_model, context, completion,
                     prompt_preprocessor=None, device="cpu")
    call_args = mock_model.forward_inference.call_args
    tokens_arg = call_args[0][0]
    assert tokens_arg.shape == (1, len(context) + len(completion))


# ─── score_doc_with_context tests ────────────────────────────────────────────

def _make_linked_batch(T: int = 20, vocab_size: int = VOCAB_SIZE):
    """Batch where doc_0 (context-only) links to doc_1 (target).

    doc_0: positions [0, 10), outgoing_identifiers=['doc_1']  — context only
    doc_1: positions [10, 20), outgoing_identifiers=[]        — target (has incoming)
    """
    tokens = torch.zeros(1, T, dtype=torch.long)
    spans = [
        DocSpan(
            doc_id=0, normed_identifier="doc_0", raw_identifier="Doc 0",
            start=0, end=10, truncated=False, outgoing_identifiers=["doc_1"],
        ),
        DocSpan(
            doc_id=1, normed_identifier="doc_1", raw_identifier="Doc 1",
            start=10, end=20, truncated=False, outgoing_identifiers=[],
        ),
    ]
    batch = {"tokens": tokens, "doc_spans": spans}

    model = MagicMock()

    def _forward(tokens, doc_spans, **kwargs):
        return torch.zeros(1, tokens.shape[1], vocab_size)

    model.forward_inference.side_effect = _forward
    model.active_layout_policy = NullLayoutPolicy()
    return model, batch


def test_score_doc_with_context_returns_expected_keys():
    model, batch = _make_linked_batch()
    result = score_doc_with_context(model, batch, NullLayoutPolicy(), device="cpu")
    assert set(result.keys()) == {"mean_nll", "num_tokens"}


def test_score_doc_with_context_uniform_logits_approx_log_V():
    model, batch = _make_linked_batch()
    result = score_doc_with_context(model, batch, NullLayoutPolicy(), device="cpu")
    assert result["num_tokens"] > 0
    assert abs(result["mean_nll"] - math.log(VOCAB_SIZE)) < 1e-4


def test_score_doc_with_context_only_scores_target_doc():
    # doc_1 occupies positions [10, 20). NullLayoutPolicy: prefix=0, suffix=0.
    # body_start=10: logit at position 9 is valid → all 10 body tokens scored.
    model, batch = _make_linked_batch(T=20)
    result = score_doc_with_context(model, batch, NullLayoutPolicy(), device="cpu")
    assert result["num_tokens"] == 10


def test_score_doc_with_context_forward_sees_full_sequence():
    # forward_inference receives the full packed sequence (both docs), not just
    # the target doc — the context doc must be present for cross-doc grants.
    model, batch = _make_linked_batch(T=20)
    score_doc_with_context(model, batch, NullLayoutPolicy(), device="cpu")
    tokens_arg = model.forward_inference.call_args[0][0]
    assert tokens_arg.shape == (1, 20)


def test_score_doc_with_context_no_edges_returns_zero():
    # All spans have empty outgoing_identifiers → no cross-doc edges → skip.
    model = MagicMock()
    batch = {
        "tokens": torch.zeros(1, 20, dtype=torch.long),
        "doc_spans": [
            DocSpan(doc_id=0, normed_identifier="a", raw_identifier="A",
                    start=0, end=10, truncated=False, outgoing_identifiers=[]),
            DocSpan(doc_id=1, normed_identifier="b", raw_identifier="B",
                    start=10, end=20, truncated=False, outgoing_identifiers=[]),
        ],
    }
    result = score_doc_with_context(model, batch, NullLayoutPolicy(), device="cpu")
    assert result == {"mean_nll": 0.0, "num_tokens": 0}
    assert model.forward_inference.call_count == 0


def test_score_doc_with_context_calls_forward_once():
    model, batch = _make_linked_batch()
    score_doc_with_context(model, batch, NullLayoutPolicy(), device="cpu")
    assert model.forward_inference.call_count == 1


def test_score_doc_with_context_passes_mask_type_to_forward():
    model, batch = _make_linked_batch()
    score_doc_with_context(
        model, batch, NullLayoutPolicy(), device="cpu", mask_type="doc_causal"
    )
    call_kwargs = model.forward_inference.call_args[1]
    assert call_kwargs.get("mask_type") == "doc_causal"


# ─── score_completions_batched tests ─────────────────────────────────────────

CONTEXT = [1, 2, 3]
CHOICES = [[4, 5], [6, 7, 8], [9]]


def test_score_completions_batched_returns_list_of_floats():
    model = _make_mock_model()
    result = score_completions_batched(model, CONTEXT, CHOICES, device="cpu")
    assert isinstance(result, list)
    assert len(result) == len(CHOICES)
    assert all(isinstance(v, float) for v in result)


def test_score_completions_batched_uniform_logits_approx_log_V():
    model = _make_mock_model()
    result = score_completions_batched(model, CONTEXT, CHOICES, device="cpu")
    expected = math.log(VOCAB_SIZE)
    for nll in result:
        assert abs(nll - expected) < 1e-4


def test_score_completions_batched_empty_completion_returns_zero():
    model = _make_mock_model()
    result = score_completions_batched(model, CONTEXT, [[], [4, 5]], device="cpu")
    assert result[0] == 0.0
    assert abs(result[1] - math.log(VOCAB_SIZE)) < 1e-4


def test_score_completions_batched_empty_list_returns_empty():
    model = _make_mock_model()
    result = score_completions_batched(model, CONTEXT, [], device="cpu")
    assert result == []
    assert model.forward_inference.call_count == 0


def test_score_completions_batched_single_forward_call():
    model = _make_mock_model()
    score_completions_batched(model, CONTEXT, CHOICES, device="cpu")
    assert model.forward_inference.call_count == 1


def test_score_completions_batched_matches_individual_score_completion():
    """Batched NLL must equal individual score_completion for each choice.

    Uses position-dependent logits so results are non-trivially dependent on
    token positions — confirming correct slice indexing, not just log(V).
    """
    # Logits are position-index values broadcast across vocab:
    # logit[t, v] = t  →  log_softmax peaks at the last vocab position
    # but crucially NLL varies across token positions in a deterministic way.
    def _pos_dependent_forward(tokens, doc_spans, **kwargs):
        T = tokens.shape[1]
        # Shape [1, T, V]: each position t has logits = t for all vocab entries.
        # log_softmax of uniform-per-position → still uniform, but NLL = log(V).
        # To make results non-trivial, use arange across vocab dim instead:
        #   logits[0, t, v] = v  → softmax is always the same, so NLL is log(V).
        # Instead: logits[0, t, v] = t + v makes each (t, token_id) unique.
        t_idx = torch.arange(T, dtype=torch.float32).unsqueeze(1)   # [T, 1]
        v_idx = torch.arange(VOCAB_SIZE, dtype=torch.float32).unsqueeze(0)  # [1, V]
        logits = (t_idx + v_idx).unsqueeze(0)  # [1, T, V]
        return logits

    def _make_pos_model():
        model = MagicMock()
        model.forward_inference.side_effect = _pos_dependent_forward
        model.active_layout_policy = NullLayoutPolicy()
        dummy_param = nn.Parameter(torch.zeros(1))
        model.backbone.parameters.return_value = iter([dummy_param])
        return model

    ctx = [10, 20, 30]
    choices = [[40, 50], [60], [70, 80, 90]]

    # Batched call — single forward pass over the packed sequence.
    batched_model = _make_pos_model()
    batched_nlls = score_completions_batched(batched_model, ctx, choices, device="cpu")

    # Individual calls — each gets its own fresh forward pass.
    for i, (choice, batched_nll) in enumerate(zip(choices, batched_nlls)):
        indiv_model = _make_pos_model()
        indiv_nll = score_completion(indiv_model, ctx, choice, device="cpu")
        assert abs(batched_nll - indiv_nll) < 1e-4, (
            f"Choice {i}: batched={batched_nll:.6f} vs individual={indiv_nll:.6f}"
        )


# ─── score_completion_with_context_docs tests ─────────────────────────────────

from model.graph_traversal.link_detector import LinkInfo


def _make_cross_doc_model(vocab_size: int = VOCAB_SIZE):
    """Mock model that accepts link_to_target kwarg (cross_doc_link path)."""
    model = MagicMock()

    def _forward(tokens, doc_spans, **kwargs):
        T = tokens.shape[1]
        return torch.zeros(1, T, vocab_size)

    model.forward_inference.side_effect = _forward
    model.mask_type = "cross_doc_link"
    dummy_param = nn.Parameter(torch.zeros(1))
    model.backbone.parameters.return_value = iter([dummy_param])
    return model


def _make_detector(link_end_positions):
    """Mock LinkDetector reporting links at given absolute token positions."""
    detector = MagicMock()
    detector.detect_links.return_value = [
        LinkInfo(link_end_pos=p, target_str="xfile_0")
        for p in link_end_positions
    ]
    return detector


AUX_TOKS   = [1, 2, 3, 4, 5]        # 5 aux tokens
CTX_TOKS   = [10, 20, 30]            # 3 context tokens
COMP_TOKS  = [40, 50]                # 2 completion tokens


def test_cdoc_returns_float_when_link_in_primary():
    model    = _make_cross_doc_model()
    # Primary doc starts at position 5 (after aux); put link inside it.
    detector = _make_detector([len(AUX_TOKS) + 1])
    result   = score_completion_with_context_docs(
        model, [AUX_TOKS], CTX_TOKS, COMP_TOKS, detector, device="cpu"
    )
    assert isinstance(result, float)


def test_cdoc_returns_none_when_no_links():
    model    = _make_cross_doc_model()
    detector = _make_detector([])
    result   = score_completion_with_context_docs(
        model, [AUX_TOKS], CTX_TOKS, COMP_TOKS, detector, device="cpu"
    )
    assert result is None


def test_cdoc_returns_none_when_links_only_in_aux():
    model    = _make_cross_doc_model()
    # Link at position 1 — inside the aux span, not the primary doc.
    detector = _make_detector([1])
    result   = score_completion_with_context_docs(
        model, [AUX_TOKS], CTX_TOKS, COMP_TOKS, detector, device="cpu"
    )
    assert result is None


def test_cdoc_returns_none_when_no_aux():
    model    = _make_cross_doc_model()
    detector = _make_detector([1])
    result   = score_completion_with_context_docs(
        model, [], CTX_TOKS, COMP_TOKS, detector, device="cpu"
    )
    assert result is None


def test_cdoc_returns_none_when_context_empty():
    model    = _make_cross_doc_model()
    detector = _make_detector([len(AUX_TOKS) + 1])
    result   = score_completion_with_context_docs(
        model, [AUX_TOKS], [], COMP_TOKS, detector, device="cpu"
    )
    assert result is None


def test_cdoc_passes_link_to_target_to_forward():
    model    = _make_cross_doc_model()
    link_pos = len(AUX_TOKS) + 1
    detector = _make_detector([link_pos])

    captured_kwargs = {}

    def _forward_capture(tokens, doc_spans, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.zeros(1, tokens.shape[1], VOCAB_SIZE)

    model.forward_inference.side_effect = _forward_capture

    score_completion_with_context_docs(
        model, [AUX_TOKS], CTX_TOKS, COMP_TOKS, detector, device="cpu"
    )
    assert "link_to_target" in captured_kwargs
    ltt = captured_kwargs["link_to_target"]
    assert link_pos in ltt
    # Aux doc_id 0 should be in the grant list.
    assert 0 in ltt[link_pos]


def test_cdoc_aux_spans_precede_primary():
    model    = _make_cross_doc_model()
    link_pos = len(AUX_TOKS) + 1
    detector = _make_detector([link_pos])

    captured_spans = {}

    def _forward_capture(tokens, doc_spans, **kwargs):
        captured_spans["spans"] = doc_spans
        return torch.zeros(1, tokens.shape[1], VOCAB_SIZE)

    model.forward_inference.side_effect = _forward_capture

    score_completion_with_context_docs(
        model, [AUX_TOKS], CTX_TOKS, COMP_TOKS, detector, device="cpu"
    )
    spans = captured_spans["spans"]
    primary = spans[-1]
    for aux_span in spans[:-1]:
        assert aux_span.end <= primary.start, (
            f"Aux span end={aux_span.end} must be <= primary start={primary.start}"
        )


def test_cdoc_uses_cross_doc_link_mask():
    model    = _make_cross_doc_model()
    link_pos = len(AUX_TOKS) + 1
    detector = _make_detector([link_pos])

    captured_kwargs = {}

    def _forward_capture(tokens, doc_spans, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.zeros(1, tokens.shape[1], VOCAB_SIZE)

    model.forward_inference.side_effect = _forward_capture

    score_completion_with_context_docs(
        model, [AUX_TOKS], CTX_TOKS, COMP_TOKS, detector, device="cpu"
    )
    assert captured_kwargs.get("mask_type") == "cross_doc_link"


def test_cdoc_uses_last_link_position_when_multiple():
    """With multiple import links, the grant should use the LAST link_end_pos."""
    model    = _make_cross_doc_model()
    primary_start = len(AUX_TOKS)
    link_pos_early = primary_start + 1
    link_pos_late  = primary_start + 2

    detector = MagicMock()
    detector.detect_links.return_value = [
        LinkInfo(link_end_pos=link_pos_early, target_str="a"),
        LinkInfo(link_end_pos=link_pos_late,  target_str="b"),
    ]

    captured_kwargs = {}

    def _forward_capture(tokens, doc_spans, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.zeros(1, tokens.shape[1], VOCAB_SIZE)

    model.forward_inference.side_effect = _forward_capture

    score_completion_with_context_docs(
        model, [AUX_TOKS], CTX_TOKS, COMP_TOKS, detector, device="cpu"
    )
    ltt = captured_kwargs["link_to_target"]
    # Only the last position should be the key.
    assert link_pos_late in ltt
    assert link_pos_early not in ltt


# ─── score_docs_batched tests ────────────────────────────────────────────────

def _pos_dependent_forward(tokens, doc_spans, **kwargs):
    """logits[0, t, v] = t + v — makes NLL depend on absolute position so the
    parity check catches any offset/slicing bug, not just log(V) coincidences."""
    T = tokens.shape[1]
    t_idx = torch.arange(T, dtype=torch.float32).unsqueeze(1)          # [T, 1]
    v_idx = torch.arange(VOCAB_SIZE, dtype=torch.float32).unsqueeze(0)  # [1, V]
    return (t_idx + v_idx).unsqueeze(0)                                # [1, T, V]


def _make_batched_model(forward_fn, max_seq_len=None):
    model = MagicMock()
    model.forward_inference.side_effect = forward_fn
    model.active_layout_policy = NullLayoutPolicy()
    # backbone.max_seq_len drives the pack budget; None disables truncation/cap.
    model.backbone.max_seq_len = max_seq_len
    return model


DOCS = [
    ([10, 20, 30, 40, 50], "Doc A", "a"),
    ([60, 70], "Doc B", "b"),
    ([80, 90, 100], "Doc C", "c"),
]


def test_score_docs_batched_returns_one_result_per_doc():
    model = _make_batched_model(lambda t, s, **k: torch.zeros(1, t.shape[1], VOCAB_SIZE))
    results = score_docs_batched(model, DOCS, NullLayoutPolicy(), device="cpu")
    assert len(results) == len(DOCS)
    assert all(set(r) == {"mean_nll", "num_tokens"} for r in results)


def test_score_docs_batched_empty_input():
    model = _make_batched_model(lambda t, s, **k: torch.zeros(1, t.shape[1], VOCAB_SIZE))
    assert score_docs_batched(model, [], NullLayoutPolicy(), device="cpu") == []
    assert model.forward_inference.call_count == 0


def test_score_docs_batched_empty_body_yields_zero():
    model = _make_batched_model(lambda t, s, **k: torch.zeros(1, t.shape[1], VOCAB_SIZE))
    docs = [([], "empty", "e"), ([10, 20, 30], "ok", "o")]
    results = score_docs_batched(model, docs, NullLayoutPolicy(), device="cpu")
    assert results[0] == {"mean_nll": 0.0, "num_tokens": 0}
    assert results[1]["num_tokens"] > 0


@pytest.mark.parametrize("policy", [NullLayoutPolicy(), EOSLayoutPolicy(eos_token_id=1)])
def test_score_docs_batched_matches_score_doc(policy):
    """Each batched per-doc result must equal an individual score_doc call.

    Runs under both NullLayoutPolicy and EOSLayoutPolicy (the latter has
    prefix_len==0, exercising the skip-first-body-token path)."""
    batched_model = _make_batched_model(_pos_dependent_forward)
    batched = score_docs_batched(batched_model, DOCS, policy, device="cpu")

    for (body, raw_id, normed_id), b in zip(DOCS, batched):
        indiv_model = _make_batched_model(_pos_dependent_forward)
        indiv = score_doc(
            indiv_model, body, policy,
            raw_identifier=raw_id, normed_identifier=normed_id, device="cpu",
        )
        assert b["num_tokens"] == indiv["num_tokens"]
        assert abs(b["mean_nll"] - indiv["mean_nll"]) < 1e-4


def test_score_docs_batched_packs_into_fewer_forwards():
    """With a budget large enough to hold all docs, one forward covers them all."""
    model = _make_batched_model(_pos_dependent_forward, max_seq_len=1024)
    score_docs_batched(model, DOCS, NullLayoutPolicy(), device="cpu")
    assert model.forward_inference.call_count == 1


def test_score_docs_batched_respects_max_seq_len_budget():
    """A tight budget forces multiple packs; results still match score_doc."""
    # Each NullLayoutPolicy doc seq len == its body len (5, 2, 3). Budget 5 forces
    # doc A alone, then B+? — verify multiple forwards AND correctness.
    model = _make_batched_model(_pos_dependent_forward, max_seq_len=5)
    batched = score_docs_batched(model, DOCS, NullLayoutPolicy(), device="cpu")
    assert model.forward_inference.call_count >= 2
    for (body, raw_id, normed_id), b in zip(DOCS, batched):
        indiv_model = _make_batched_model(_pos_dependent_forward, max_seq_len=5)
        indiv = score_doc(
            indiv_model, body, NullLayoutPolicy(),
            raw_identifier=raw_id, normed_identifier=normed_id, device="cpu",
        )
        assert abs(b["mean_nll"] - indiv["mean_nll"]) < 1e-4


def test_score_docs_batched_body_truncated_to_budget():
    """A doc longer than the budget is head-truncated exactly like score_doc."""
    long_body = list(range(10, 10 + 20))          # 20 tokens
    docs = [(long_body, "Long", "l")]
    model = _make_batched_model(_pos_dependent_forward, max_seq_len=8)
    b = score_docs_batched(model, docs, NullLayoutPolicy(), device="cpu")[0]
    indiv_model = _make_batched_model(_pos_dependent_forward, max_seq_len=8)
    indiv = score_doc(indiv_model, long_body, NullLayoutPolicy(),
                      raw_identifier="Long", normed_identifier="l", device="cpu")
    assert b["num_tokens"] == indiv["num_tokens"]
    assert abs(b["mean_nll"] - indiv["mean_nll"]) < 1e-4


def test_score_docs_batched_uses_doc_causal_mask():
    captured = {}

    def _capture(tokens, doc_spans, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, tokens.shape[1], VOCAB_SIZE)

    model = _make_batched_model(_capture, max_seq_len=1024)
    score_docs_batched(model, DOCS, NullLayoutPolicy(), device="cpu")
    assert captured.get("mask_type") == "doc_causal"


# ─── score_completions_independent_batched tests ─────────────────────────────

PAIRS = [([10, 20, 30], [40, 50]), ([60, 70], [80]), ([90], [100, 110, 120])]


def _make_indep_model(forward_fn, max_seq_len=None):
    model = MagicMock()
    model.forward_inference.side_effect = forward_fn
    model.backbone.max_seq_len = max_seq_len
    dummy_param = nn.Parameter(torch.zeros(1))
    model.backbone.parameters.return_value = iter([dummy_param])
    return model


def test_indep_batched_returns_list_of_floats():
    model = _make_indep_model(lambda t, s, **k: torch.zeros(1, t.shape[1], VOCAB_SIZE))
    result = score_completions_independent_batched(model, PAIRS, device="cpu")
    assert isinstance(result, list) and len(result) == len(PAIRS)
    assert all(isinstance(v, float) for v in result)


def test_indep_batched_empty_input():
    model = _make_indep_model(lambda t, s, **k: torch.zeros(1, t.shape[1], VOCAB_SIZE))
    assert score_completions_independent_batched(model, [], device="cpu") == []
    assert model.forward_inference.call_count == 0


def test_indep_batched_empty_completion_scores_zero():
    model = _make_indep_model(lambda t, s, **k: torch.zeros(1, t.shape[1], VOCAB_SIZE))
    result = score_completions_independent_batched(
        model, [([1, 2, 3], []), ([4, 5], [6])], device="cpu"
    )
    assert result[0] == 0.0
    assert abs(result[1] - math.log(VOCAB_SIZE)) < 1e-4


@pytest.mark.parametrize("max_seq_len", [None, 1024, 4])
def test_indep_batched_matches_score_completion(max_seq_len):
    """Each batched pair result must equal an individual score_completion call.

    max_seq_len=4 forces multiple packs (and context head-truncation on the
    longer pairs), exercising the pack-splitting + truncation paths against the
    same truncation applied to individual calls."""
    batched_model = _make_indep_model(_pos_dependent_forward, max_seq_len=max_seq_len)
    batched = score_completions_independent_batched(batched_model, PAIRS, device="cpu")

    for (ctx, comp), b in zip(PAIRS, batched):
        # Apply the same head-truncation score_completions_independent_batched
        # would, so the individual reference matches when a pair overflows.
        c = list(ctx)
        if isinstance(max_seq_len, int):
            overflow = len(c) + len(comp) - max_seq_len
            if overflow > 0:
                c = c[overflow:] if overflow < len(c) else []
        indiv_model = _make_indep_model(_pos_dependent_forward, max_seq_len=max_seq_len)
        indiv = score_completion(indiv_model, c, comp, device="cpu")
        assert abs(b - indiv) < 1e-4, f"batched={b:.6f} vs individual={indiv:.6f}"


def test_indep_batched_packs_into_one_forward_when_budget_allows():
    model = _make_indep_model(_pos_dependent_forward, max_seq_len=1024)
    score_completions_independent_batched(model, PAIRS, device="cpu")
    assert model.forward_inference.call_count == 1


def test_indep_batched_uses_doc_causal_mask():
    captured = {}

    def _capture(tokens, doc_spans, **kwargs):
        captured.update(kwargs)
        return torch.zeros(1, tokens.shape[1], VOCAB_SIZE)

    model = _make_indep_model(_capture, max_seq_len=1024)
    score_completions_independent_batched(model, PAIRS, device="cpu")
    assert captured.get("mask_type") == "doc_causal"
