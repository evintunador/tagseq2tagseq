"""
tests/eval/test_link_annotator.py — unit tests for eval.link_annotator and the
run_benchmark_annotated driver in eval.nlp_benchmarks.

All tests run CPU-only with mock models and synthetic data — no GPU, no network.
"""
import math
import pytest
import torch
import torch.nn as nn
from typing import List
from unittest.mock import MagicMock, patch

from eval.link_annotator import AnnotatedPrompt, MarkdownPromptAnnotator, PromptAnnotator

VOCAB_SIZE = 17000  # must cover GPT-2 token IDs up to 16151 ('](')


# ─── Mock model helpers ──────────────────────────────────────────────────────

def _make_logits_tensor(seq_len: int, hot_pos: int, hot_tok: int, value: float = 10.0):
    """Return [1, seq_len, VOCAB_SIZE] logits with a spike at (hot_pos, hot_tok)."""
    logits = torch.zeros(1, seq_len, VOCAB_SIZE)
    logits[0, hot_pos, hot_tok] = value
    return logits


def _make_mock_model(
    opener_hot_pos: int = 3,
    opener_hot_tok: int = 685,   # ' ['
    mid_hot_pos: int = 6,
    mid_hot_tok: int = 16151,    # ']('
    title_tok: int = 65,         # 'A' — used as the generated title character
    close_paren_tok: int = 8,    # ')'
):
    """Mock TS2TSModel whose forward_inference returns deterministic logits.

    - Position opener_hot_pos has high prob for opener_hot_tok ('[' or ' [')
    - Position mid_hot_pos has high prob for mid_hot_tok ('](')
    - All other positions emit title_tok with high prob (for title generation)
    """
    model = MagicMock()
    model.mask_type = "cross_doc_link"
    model.backbone.parameters.return_value = iter([nn.Parameter(torch.zeros(1))])

    enc = MagicMock()
    enc.encode.side_effect = lambda s, **kw: [ord(c) % 256 for c in s]
    enc.decode.side_effect = lambda toks: "".join(chr(t % 256) for t in toks)
    model.tokenizer = enc

    def _fwd(tokens, doc_spans, mask_type=None, **kwargs):
        T = tokens.shape[1]
        logits = torch.zeros(1, T, VOCAB_SIZE)
        # link opener spike
        if opener_hot_pos < T:
            logits[0, opener_hot_pos, opener_hot_tok] = 10.0
        # link mid spike (in pass-2 sequence which is one longer)
        if mid_hot_pos < T:
            logits[0, mid_hot_pos, mid_hot_tok] = 10.0
        # title generation: ')' has high prob everywhere else
        logits[0, :, close_paren_tok] = 5.0
        # but make title_tok slightly higher so one title token is generated
        if T > mid_hot_pos + 1:
            logits[0, mid_hot_pos + 1, title_tok] = 8.0
        return logits

    model.forward_inference.side_effect = _fwd
    return model


def _make_annotator(corpus=None, mode="no_op", decay_factor=0.95):
    return MarkdownPromptAnnotator(
        corpus=corpus,
        link_retrieval_mode=mode,
        max_display_tokens=100,
        decay_factor=decay_factor,
        max_title_tokens=5,
        link_opener_token_ids=(58, 685),
        link_mid_token_id=16151,
        eos_token_id=50256,
    )


# ─── PromptAnnotator protocol ────────────────────────────────────────────────

def test_markdown_annotator_satisfies_protocol():
    ann = _make_annotator()
    assert isinstance(ann, PromptAnnotator)


# ─── scan_prob ───────────────────────────────────────────────────────────────

def test_scan_prob_returns_float_in_unit_interval():
    model = _make_mock_model(opener_hot_pos=2)
    ann = _make_annotator()
    p = ann.scan_prob(model, [1, 2, 3, 4, 5], device="cpu")
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_scan_prob_empty_tokens_returns_zero():
    model = _make_mock_model()
    ann = _make_annotator()
    p = ann.scan_prob(model, [], device="cpu")
    assert p == 0.0


# ─── link opener position ────────────────────────────────────────────────────

def test_link_opener_inserted_at_max_prob_position():
    model = _make_mock_model(opener_hot_pos=2, opener_hot_tok=685)
    ann = _make_annotator(mode="no_op")
    ctx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = ann.annotate(model, ctx, device="cpu")
    # The inserted opener should be at index link_opener_pos in context_tokens
    assert result.context_tokens[result.link_opener_pos] in (58, 685)


def test_link_opener_prob_matches_scan_prob():
    model = _make_mock_model(opener_hot_pos=3)
    ann = _make_annotator(mode="no_op")
    ctx = [1, 2, 3, 4, 5]
    scan = ann.scan_prob(model, ctx, device="cpu")
    result = ann.annotate(model, ctx, device="cpu")
    assert abs(result.link_opener_prob - scan) < 1e-5


# ─── link mid position ───────────────────────────────────────────────────────

def test_link_mid_within_display_window():
    model = _make_mock_model(opener_hot_pos=1, mid_hot_pos=4)
    ann = _make_annotator(mode="no_op", decay_factor=0.99)
    ctx = list(range(20))
    result = ann.annotate(model, ctx, device="cpu")
    distance = result.link_mid_pos - result.link_opener_pos
    assert 0 < distance <= ann.max_display_tokens + 1   # +1 for insertion shift


def test_distance_decay_prefers_nearby_link_mid():
    """When raw probs are equal at j=i+2 and j=i+20, strong decay picks j=i+2."""
    def _fwd_equal(tokens, doc_spans, mask_type=None, **kwargs):
        T = tokens.shape[1]
        logits = torch.zeros(1, T, VOCAB_SIZE)
        # opener at pos 0
        logits[0, 0, 685] = 10.0
        # equal P('](' at pos 2 and pos 20)
        for j in (2, 20):
            if j < T:
                logits[0, j, 16151] = 5.0
        # ')' everywhere for title stop
        logits[0, :, 8] = 8.0
        return logits

    model = MagicMock()
    model.forward_inference.side_effect = _fwd_equal
    enc = MagicMock()
    enc.decode.side_effect = lambda toks: "".join(chr(t % 256) for t in toks)
    enc.encode.side_effect = lambda s, **kw: [ord(c) % 256 for c in s]
    model.tokenizer = enc

    # Strong decay: 0.5^18 = negligible, so pos 2 wins
    ann = MarkdownPromptAnnotator(
        link_retrieval_mode="no_op",
        max_display_tokens=100,
        decay_factor=0.5,
        max_title_tokens=3,
        link_opener_token_ids=(58, 685),
        link_mid_token_id=16151,
        eos_token_id=50256,
    )
    ctx = list(range(30))
    result = ann.annotate(model, ctx, device="cpu")
    # After inserting '[' at pos 0, mid should be closer to pos 2 than pos 20
    assert result.link_mid_pos < 15


# ─── link retrieval modes ────────────────────────────────────────────────────

def test_no_op_mode_returns_empty_aux():
    model = _make_mock_model()
    ann = _make_annotator(mode="no_op")
    result = ann.annotate(model, list(range(15)), device="cpu")
    assert result.aux_token_lists == []
    assert result.aux_raw_identifiers == []
    assert result.link_fired is False


def test_corpus_hit_returns_aux_tokens():
    corpus = MagicMock()
    corpus.has_document.return_value = True
    corpus.get_document.return_value = iter([10, 20, 30])

    model = _make_mock_model()
    ann = _make_annotator(corpus=corpus, mode="corpus_only")
    result = ann.annotate(model, list(range(15)), device="cpu")

    assert result.link_fired is True
    assert len(result.aux_token_lists) == 1
    assert result.aux_token_lists[0] == [10, 20, 30]
    assert len(result.aux_raw_identifiers) == 1


def test_corpus_miss_in_corpus_only_mode():
    corpus = MagicMock()
    corpus.has_document.return_value = False

    model = _make_mock_model()
    ann = _make_annotator(corpus=corpus, mode="corpus_only")
    result = ann.annotate(model, list(range(15)), device="cpu")

    assert result.link_fired is False
    assert result.aux_token_lists == []


def test_corpus_miss_in_corpus_then_generate_falls_back(monkeypatch):
    corpus = MagicMock()
    corpus.has_document.return_value = False

    generated_tokens = [1, 2, 3, 4, 5]

    def _fake_run_generation(model, prompt_tokens, corpus, config, link_detector,
                              tokenizer_decode, layout_policy, root_identifier=""):
        doc = MagicMock()
        doc.tokens = torch.tensor(generated_tokens)
        result = MagicMock()
        result.root_document = doc
        return result

    monkeypatch.setattr("model.generation_loop.run_generation", _fake_run_generation)

    model = _make_mock_model()
    ann = _make_annotator(corpus=corpus, mode="corpus_then_generate")
    result = ann.annotate(model, list(range(15)), device="cpu")

    assert result.link_fired is True
    assert result.aux_token_lists == [generated_tokens]


# ─── context_tokens structure ────────────────────────────────────────────────

def test_annotated_context_longer_than_original():
    model = _make_mock_model()
    ann = _make_annotator(mode="no_op")
    ctx = list(range(10))
    result = ann.annotate(model, ctx, device="cpu")
    # At minimum we inserted '[' and '](' — at least 2 new tokens
    assert len(result.context_tokens) > len(ctx)


def test_empty_context_returns_safely():
    model = _make_mock_model()
    ann = _make_annotator(mode="no_op")
    result = ann.annotate(model, [], device="cpu")
    assert result.link_fired is False
    assert result.context_tokens == []


# ─── run_benchmark_annotated — threshold calibration ─────────────────────────

def test_threshold_calibration_splits_correctly():
    """4 items with probs [0.1, 0.3, 0.6, 0.9]; t=p50 should annotate ~2 of 4."""
    import eval.nlp_benchmarks as _bench

    enc = MagicMock()
    enc.encode.side_effect = lambda s, **kw: [1, 2, 3]
    model = MagicMock()
    model.tokenizer = enc

    # Each item gets an AnnotatedPrompt with a distinct link_opener_prob.
    # probs [0.1, 0.3, 0.6, 0.9] → sorted p50 = sorted[2] = 0.6
    # → items with prob >= 0.6: {0.6, 0.9} → 2 items annotated.
    annotated_results = [
        AnnotatedPrompt(
            context_tokens=[1, 2, 3, 100, 200, 300],
            aux_token_lists=[[5, 6, 7]],
            aux_raw_identifiers=["Target"],
            target_str="Target",
            link_opener_pos=1,
            link_mid_pos=3,
            link_opener_prob=p,
            link_fired=True,
        )
        for p in [0.1, 0.3, 0.6, 0.9]
    ]
    ann = MagicMock(spec=MarkdownPromptAnnotator)
    ann.annotate.side_effect = annotated_results

    items = [
        {"type": "fitb", "context_tokens": [1, 2, 3], "completion_tokens": [4]},
        {"type": "fitb", "context_tokens": [1, 2, 3], "completion_tokens": [4]},
        {"type": "fitb", "context_tokens": [1, 2, 3], "completion_tokens": [4]},
        {"type": "fitb", "context_tokens": [1, 2, 3], "completion_tokens": [4]},
    ]

    model.link_detector = MagicMock()
    model.link_detector.__class__ = MagicMock

    with patch.object(_bench, "_load_benchmark_items", return_value=items), \
         patch.object(_bench, "score_completion", return_value=1.0), \
         patch.object(_bench, "score_completion_with_context_docs", return_value=0.8):
        result = _bench.run_benchmark_annotated(
            model=model,
            benchmark_name="lambada",
            annotator=ann,
            max_examples=4,
            device="cpu",
        )

    # p50 should annotate roughly the top 50% (2 out of 4 items)
    p50_stats = result.get("t=p50", {})
    assert p50_stats.get("n_annotated", -1) == 2


def test_annotated_result_has_all_threshold_keys():
    import eval.nlp_benchmarks as _bench

    enc = MagicMock()
    enc.encode.side_effect = lambda s, **kw: [1, 2]
    model = MagicMock()
    model.tokenizer = enc
    model.link_detector = MagicMock()

    ann = MagicMock(spec=MarkdownPromptAnnotator)
    ann.scan_prob.return_value = 0.5
    ann.annotate.return_value = AnnotatedPrompt(
        context_tokens=[1, 2, 3],
        aux_token_lists=[],
        aux_raw_identifiers=[],
        target_str="",
        link_opener_pos=0,
        link_mid_pos=1,
        link_opener_prob=0.5,
        link_fired=False,
    )

    items = [{"type": "fitb", "context_tokens": [1, 2], "completion_tokens": [3]}]

    with patch.object(_bench, "_load_benchmark_items", return_value=items), \
         patch.object(_bench, "score_completion", return_value=1.0), \
         patch.object(_bench, "score_completion_with_context_docs", return_value=0.9):
        result = _bench.run_benchmark_annotated(
            model=model,
            benchmark_name="lambada",
            annotator=ann,
            max_examples=1,
            device="cpu",
        )

    for key in ("baseline_flat", "t=0.0", "t=p25", "t=p50", "t=p75", "threshold_values"):
        assert key in result, f"Missing key: {key!r}"

    for t_key in ("t=0.0", "t=p25", "t=p50", "t=p75"):
        stats = result[t_key]
        assert "n_annotated" in stats
        assert "n_link_fired" in stats
        assert "total_examples" in stats

    tv = result["threshold_values"]
    for k in ("p25", "p50", "p75"):
        assert k in tv


def test_mc_benchmark_shared_context_not_re_annotated():
    """Annotator.annotate is called once per item, not once per choice."""
    import eval.nlp_benchmarks as _bench

    enc = MagicMock()
    enc.encode.side_effect = lambda s, **kw: [1, 2]
    model = MagicMock()
    model.tokenizer = enc
    model.link_detector = MagicMock()

    ann = MagicMock(spec=MarkdownPromptAnnotator)
    ann.scan_prob.return_value = 0.9
    ann.annotate.return_value = AnnotatedPrompt(
        context_tokens=[1, 2, 100, 200],
        aux_token_lists=[[5, 6]],
        aux_raw_identifiers=["Target"],
        target_str="Target",
        link_opener_pos=0,
        link_mid_pos=2,
        link_opener_prob=0.9,
        link_fired=True,
    )

    # 2 MC items, each with 3 choices
    items = [
        {"type": "mc", "context_tokens": [1, 2],
         "completion_token_lists": [[3], [4], [5]], "label": 0},
        {"type": "mc", "context_tokens": [1, 2],
         "completion_token_lists": [[3], [4], [5]], "label": 1},
    ]

    with patch.object(_bench, "_load_benchmark_items", return_value=items), \
         patch.object(_bench, "score_completions_batched", return_value=[0.5, 0.8, 1.0]), \
         patch.object(_bench, "score_completion_with_context_docs", return_value=0.4):
        result = _bench.run_benchmark_annotated(
            model=model,
            benchmark_name="hellaswag",
            annotator=ann,
            max_examples=2,
            device="cpu",
        )

    # annotate was called once per item (2 items), not once per choice (6 calls)
    assert ann.annotate.call_count == 2


def test_unknown_benchmark_raises():
    import eval.nlp_benchmarks as _bench
    ann = _make_annotator()
    model = MagicMock()
    model.tokenizer = MagicMock()
    with pytest.raises(ValueError, match="not in ANNOTATABLE_BENCHMARKS"):
        _bench.run_benchmark_annotated(model, "mmlu", ann, device="cpu")
