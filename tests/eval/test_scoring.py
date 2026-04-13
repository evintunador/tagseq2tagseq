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

from data.layout import EOSLayoutPolicy, NullLayoutPolicy
from eval.scoring import score_completion, score_doc


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


def test_score_completion_preprocessor_none_unchanged(mock_model):
    """When prompt_preprocessor is None, context is not modified."""
    context = [1, 2, 3]
    completion = [4]
    score_completion(mock_model, context, completion,
                     prompt_preprocessor=None, device="cpu")
    call_args = mock_model.forward_inference.call_args
    tokens_arg = call_args[0][0]
    assert tokens_arg.shape == (1, len(context) + len(completion))
