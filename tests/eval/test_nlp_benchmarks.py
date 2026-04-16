"""
tests/eval/test_nlp_benchmarks.py — unit tests for eval.nlp_benchmarks.

Covers run_hellaswag, run_wiki_qa, run_arc, run_lambada.
All tests run on CPU with mock models and synthetic data — no HuggingFace
network access required.
"""
import pytest
import torch
import torch.nn as nn
from typing import List
from unittest.mock import MagicMock

import eval.nlp_benchmarks as _bench
from eval.nlp_benchmarks import run_hellaswag, run_wiki_qa, run_arc, run_lambada

VOCAB_SIZE = 256


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _make_mock_model(tokenizer=True):
    model = MagicMock()

    def _forward(tokens, doc_spans, **kwargs):
        return torch.zeros(1, tokens.shape[1], VOCAB_SIZE)

    model.forward_inference.side_effect = _forward
    model.backbone.parameters.return_value = iter([nn.Parameter(torch.zeros(1))])
    if tokenizer:
        model.tokenizer = MagicMock()
        model.tokenizer.encode.side_effect = lambda s: [ord(c) % 256 for c in s]
    else:
        model.tokenizer = None
    return model


def _mc_item(context, choices, label):
    try:
        from tunalab.evaluations.multiple_choice import MultipleChoiceItem
    except ImportError:
        pytest.skip("tunalab NLP catalog not installed")
    return MultipleChoiceItem(context=context, choices=choices, label=label)


def _fitb_item(prompt, answer):
    try:
        from tunalab.evaluations.fill_in_the_blank import FillInTheBlankItem
    except ImportError:
        pytest.skip("tunalab NLP catalog not installed")
    return FillInTheBlankItem(prompt=prompt, answer=answer)


def _patch_mc_dataset(monkeypatch, dataset_class_path, items):
    """Patch a dataset class (by dotted path) to yield synthetic items."""
    module_path, cls_name = dataset_class_path.rsplit(".", 1)
    try:
        import importlib
        cls = getattr(importlib.import_module(module_path), cls_name)
    except (ImportError, AttributeError):
        pytest.skip("tunalab NLP catalog not installed")
    monkeypatch.setattr(cls, "__init__", lambda self, **kw: setattr(self, "data", list(items)) or None)
    monkeypatch.setattr(cls, "__len__",  lambda self: len(self.data))
    monkeypatch.setattr(cls, "__getitem__", lambda self, i: self.data[i])


def _patch_fitb_dataset(monkeypatch, dataset_class_path, items):
    _patch_mc_dataset(monkeypatch, dataset_class_path, items)


# ─── Tokenizer guards ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn,kwargs", [
    (run_hellaswag, {}),
    (run_wiki_qa,   {}),
    (run_arc,       {"config": "easy"}),
    (run_lambada,   {}),
])
def test_requires_tokenizer(fn, kwargs):
    model = _make_mock_model(tokenizer=False)
    with pytest.raises(ValueError, match="model.tokenizer"):
        fn(model=model, device="cpu", **kwargs)


# ─── ARC config validation ────────────────────────────────────────────────────

def test_run_arc_invalid_config_raises():
    with pytest.raises(ValueError, match="config must be"):
        run_arc(model=_make_mock_model(), config="medium", device="cpu")  # type: ignore[arg-type]


# ─── HellaSwag ────────────────────────────────────────────────────────────────

def test_hellaswag_calls_batched_once_per_item(monkeypatch):
    items = [_mc_item("The cat sat on the", ["floor", "mat", "ceiling", "moon"], 1)] * 2
    _patch_mc_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.multiple_choice.hellaswag.HellaSwagDataset", items)
    call_count = {"n": 0}
    def _fake(model, ctx, choices, device=None):
        call_count["n"] += 1
        return [0.5] * len(choices)
    monkeypatch.setattr(_bench, "score_completions_batched", _fake)
    run_hellaswag(model=_make_mock_model(), max_examples=2, device="cpu")
    assert call_count["n"] == 2


def test_hellaswag_picks_lowest_nll(monkeypatch):
    items = [_mc_item("hello", ["a", "bb", "ccc", "dddd"], label=2)]
    _patch_mc_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.multiple_choice.hellaswag.HellaSwagDataset", items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [1.0, 0.8, 0.3, 0.9])
    result = run_hellaswag(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["total_examples"] == 1


# ─── WikiQA ───────────────────────────────────────────────────────────────────

def test_wiki_qa_calls_batched_once_per_item(monkeypatch):
    items = [_mc_item("What is the capital of France?", ["Paris", "London", "Berlin"], 0)] * 2
    _patch_mc_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.multiple_choice.wiki_qa.WikiQADataset", items)
    call_count = {"n": 0}
    def _fake(model, ctx, choices, device=None):
        call_count["n"] += 1
        return [0.5] * len(choices)
    monkeypatch.setattr(_bench, "score_completions_batched", _fake)
    run_wiki_qa(model=_make_mock_model(), max_examples=2, device="cpu")
    assert call_count["n"] == 2


def test_wiki_qa_picks_lowest_nll(monkeypatch):
    items = [_mc_item("What is water made of?",
                      ["sand and rock", "hydrogen and oxygen", "iron and carbon"], label=1)]
    _patch_mc_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.multiple_choice.wiki_qa.WikiQADataset", items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [1.0, 0.2, 0.9])
    result = run_wiki_qa(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["total_examples"] == 1


def test_wiki_qa_variable_choice_count(monkeypatch):
    items = [_mc_item("Q1", ["a", "b"], 0), _mc_item("Q2", ["x", "y", "z", "w"], 3)]
    _patch_mc_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.multiple_choice.wiki_qa.WikiQADataset", items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)), [float(i) for i in range(len(choices))])[1])
    run_wiki_qa(model=_make_mock_model(), max_examples=2, device="cpu")
    assert counts == [2, 4]


# ─── ARC ──────────────────────────────────────────────────────────────────────

def test_arc_calls_batched_once_per_item(monkeypatch):
    items = [_mc_item("What is H2O?", ["water", "air", "fire", "earth"], 0)] * 2
    _patch_mc_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.multiple_choice.arc.ARCDataset", items)
    call_count = {"n": 0}
    def _fake(model, ctx, choices, device=None):
        call_count["n"] += 1
        return [0.5] * len(choices)
    monkeypatch.setattr(_bench, "score_completions_batched", _fake)
    run_arc(model=_make_mock_model(), config="challenge", max_examples=2, device="cpu")
    assert call_count["n"] == 2


def test_arc_picks_lowest_nll(monkeypatch):
    items = [_mc_item("What gas do plants absorb?",
                      ["oxygen", "nitrogen", "carbon dioxide", "hydrogen"], label=2)]
    _patch_mc_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.multiple_choice.arc.ARCDataset", items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [1.0, 0.9, 0.2, 0.8])
    result = run_arc(model=_make_mock_model(), config="easy", max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["total_examples"] == 1


def test_arc_easy_and_challenge_both_work(monkeypatch):
    items = [_mc_item("Q?", ["A", "B", "C", "D"], 0)]
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda *a, **kw: [float(i) for i in range(len(a[2]))])
    for cfg in ("easy", "challenge"):
        _patch_mc_dataset(monkeypatch,
            "tunalab.data_sources.evaluations.multiple_choice.arc.ARCDataset", items)
        result = run_arc(model=_make_mock_model(), config=cfg, max_examples=1, device="cpu")
        assert "accuracy" in result
        assert result["total_examples"] == 1


# ─── LAMBADA ──────────────────────────────────────────────────────────────────

def test_lambada_calls_score_completion_once_per_item(monkeypatch):
    items = [_fitb_item("The cat sat on the", "mat"), _fitb_item("She opened the", "door")]
    _patch_fitb_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.fill_in_the_blank.lambada.LambadaDataset", items)
    call_count = {"n": 0}
    def _fake(m, ctx, completion, **kw):
        call_count["n"] += 1
        return 1.5
    monkeypatch.setattr(_bench, "score_completion", _fake)
    run_lambada(model=_make_mock_model(), max_examples=2, device="cpu")
    assert call_count["n"] == 2


def test_lambada_returns_perplexity_key(monkeypatch):
    items = [_fitb_item("The sky is", "blue")]
    _patch_fitb_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.fill_in_the_blank.lambada.LambadaDataset", items)
    monkeypatch.setattr(_bench, "score_completion", lambda *a, **kw: 1.0)
    result = run_lambada(model=_make_mock_model(), max_examples=1, device="cpu")
    assert "perplexity" in result
    assert result["total_examples"] == 1


def test_lambada_prepends_space_to_answer(monkeypatch):
    items = [_fitb_item("The answer is", "yes")]
    _patch_fitb_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.fill_in_the_blank.lambada.LambadaDataset", items)
    encoded_calls = []
    model = _make_mock_model()
    orig = model.tokenizer.encode.side_effect
    model.tokenizer.encode.side_effect = lambda s: (encoded_calls.append(s), orig(s))[1]
    monkeypatch.setattr(_bench, "score_completion", lambda *a, **kw: 1.0)
    run_lambada(model=model, max_examples=1, device="cpu")
    assert any(c.startswith(" ") for c in encoded_calls)
