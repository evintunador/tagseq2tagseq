"""
tests/eval/test_nlp_benchmarks.py — unit tests for eval.nlp_benchmarks.

Covers run_hellaswag, run_wiki_qa, run_arc, run_lambada,
run_winogrande, run_piqa, run_boolq, run_commonsense_qa, run_copa,
run_openbookqa, run_sciq, run_codexglue_line_completion,
run_mmlu, run_mathqa, run_math, run_codexglue_code_to_text,
run_repobench, run_repobench_cross_doc, run_humaneval_buggy,
run_hotpotqa, run_hotpotqa_cross_doc.
All tests run on CPU with mock models and synthetic data — no HuggingFace
network access required.
"""
import pytest
import torch
import torch.nn as nn
from typing import List
from unittest.mock import MagicMock, patch

import eval.nlp_benchmarks as _bench
from eval.nlp_benchmarks import (
    run_hellaswag, run_wiki_qa, run_arc, run_lambada,
    run_winogrande, run_piqa, run_boolq, run_commonsense_qa, run_copa,
    run_openbookqa, run_sciq, run_codexglue_line_completion,
    run_mmlu, run_mathqa, run_math,
    run_codexglue_code_to_text, run_repobench, run_repobench_cross_doc,
    run_humaneval_buggy,
    run_hotpotqa, run_hotpotqa_cross_doc,
)

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
        model.tokenizer.encode.side_effect = lambda s, **kw: [ord(c) % 256 for c in s]
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
    (run_hellaswag,             {}),
    (run_wiki_qa,               {}),
    (run_arc,                   {"config": "easy"}),
    (run_lambada,               {}),
    (run_winogrande,            {}),
    (run_piqa,                  {}),
    (run_boolq,                 {}),
    (run_commonsense_qa,        {}),
    (run_copa,                  {}),
    (run_openbookqa,            {}),
    (run_sciq,                  {}),
    (run_codexglue_line_completion, {}),
    (run_mmlu,                  {}),
    (run_mathqa,                {}),
    (run_math,                  {}),
    (run_codexglue_code_to_text, {}),
    (run_repobench,             {}),
    (run_repobench_cross_doc,   {}),
    (run_humaneval_buggy,       {}),
    (run_hotpotqa,              {}),
    (run_hotpotqa_cross_doc,    {}),
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

def test_lambada_scores_every_item(monkeypatch):
    items = [_fitb_item("The cat sat on the", "mat"), _fitb_item("She opened the", "door")]
    _patch_fitb_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.fill_in_the_blank.lambada.LambadaDataset", items)
    seen = {"pairs": 0}
    def _fake(m, pairs, **kw):
        seen["pairs"] += len(pairs)
        return [1.5] * len(pairs)
    monkeypatch.setattr(_bench, "score_completions_independent_batched", _fake)
    run_lambada(model=_make_mock_model(), max_examples=2, device="cpu")
    assert seen["pairs"] == 2


def test_lambada_returns_perplexity_key(monkeypatch):
    items = [_fitb_item("The sky is", "blue")]
    _patch_fitb_dataset(monkeypatch,
        "tunalab.data_sources.evaluations.fill_in_the_blank.lambada.LambadaDataset", items)
    monkeypatch.setattr(_bench, "score_completions_independent_batched",
                        lambda m, pairs, **kw: [1.0] * len(pairs))
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
    model.tokenizer.encode.side_effect = lambda s, **kw: (encoded_calls.append(s), orig(s))[1]
    monkeypatch.setattr(_bench, "score_completions_independent_batched",
                        lambda m, pairs, **kw: [1.0] * len(pairs))
    run_lambada(model=model, max_examples=1, device="cpu")
    assert any(c.startswith(" ") for c in encoded_calls)


# ─── WinoGrande ───────────────────────────────────────────────────────────────

_WINOGRANDE_DS = "tunalab.data_sources.evaluations.multiple_choice.winogrande.WinoGrandeDataset"


def test_winogrande_picks_lowest_nll(monkeypatch):
    items = [_mc_item("Sarah was a better surgeon so _ got the easy cases.",
                      ["Sarah got the easy cases.", "Maria got the easy cases."], label=0)]
    _patch_mc_dataset(monkeypatch, _WINOGRANDE_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [0.3, 0.9])
    result = run_winogrande(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["total_examples"] == 1


def test_winogrande_two_choices(monkeypatch):
    items = [_mc_item("ctx", ["A", "B"], 1)] * 3
    _patch_mc_dataset(monkeypatch, _WINOGRANDE_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)), [float(i) for i in range(len(choices))])[1])
    run_winogrande(model=_make_mock_model(), max_examples=3, device="cpu")
    assert all(c == 2 for c in counts)


# ─── PIQA ─────────────────────────────────────────────────────────────────────

_PIQA_DS = "tunalab.data_sources.evaluations.multiple_choice.piqa.PIQADataset"


def test_piqa_picks_lowest_nll(monkeypatch):
    items = [_mc_item("How do you soften butter?",
                      ["Leave it at room temperature.", "Freeze it overnight."], label=0)]
    _patch_mc_dataset(monkeypatch, _PIQA_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [0.2, 1.5])
    result = run_piqa(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)


def test_piqa_two_choices(monkeypatch):
    items = [_mc_item("goal", ["s1", "s2"], 0)] * 2
    _patch_mc_dataset(monkeypatch, _PIQA_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)), [0.0] * len(choices))[1])
    run_piqa(model=_make_mock_model(), max_examples=2, device="cpu")
    assert all(c == 2 for c in counts)


# ─── BoolQ ────────────────────────────────────────────────────────────────────

_BOOLQ_DS = "tunalab.data_sources.evaluations.multiple_choice.boolq.BoolQDataset"


def test_boolq_picks_lowest_nll(monkeypatch):
    items = [_mc_item("Water is H2O.\n\nQuestion: is water made of hydrogen?",
                      ["Yes", "No"], label=0)]
    _patch_mc_dataset(monkeypatch, _BOOLQ_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [0.1, 2.0])
    result = run_boolq(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)


def test_boolq_two_choices(monkeypatch):
    items = [_mc_item("ctx", ["Yes", "No"], 1)] * 4
    _patch_mc_dataset(monkeypatch, _BOOLQ_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)), [1.0, 0.0])[1])
    run_boolq(model=_make_mock_model(), max_examples=4, device="cpu")
    assert all(c == 2 for c in counts)


# ─── CommonsenseQA ────────────────────────────────────────────────────────────

_CSQA_DS = "tunalab.data_sources.evaluations.multiple_choice.commonsense_qa.CommonsenseQADataset"


def test_commonsense_qa_picks_lowest_nll(monkeypatch):
    items = [_mc_item("Where would you find a revolving door?",
                      ["bank", "library", "park", "forest", "ocean"], label=0)]
    _patch_mc_dataset(monkeypatch, _CSQA_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda *a, **kw: [0.1, 0.9, 1.0, 1.1, 1.2])
    result = run_commonsense_qa(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)


def test_commonsense_qa_five_choices(monkeypatch):
    items = [_mc_item("Q?", ["a", "b", "c", "d", "e"], 2)] * 2
    _patch_mc_dataset(monkeypatch, _CSQA_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)), [float(i) for i in range(len(choices))])[1])
    run_commonsense_qa(model=_make_mock_model(), max_examples=2, device="cpu")
    assert all(c == 5 for c in counts)


# ─── COPA ─────────────────────────────────────────────────────────────────────

_COPA_DS = "tunalab.data_sources.evaluations.multiple_choice.copa.COPADataset"


def test_copa_picks_lowest_nll(monkeypatch):
    items = [_mc_item("The man turned on the faucet. What happened as a result?",
                      ["The toilet filled with water.", "Water flowed from the spout."], label=1)]
    _patch_mc_dataset(monkeypatch, _COPA_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [1.5, 0.2])
    result = run_copa(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)


def test_copa_two_choices(monkeypatch):
    items = [_mc_item("ctx", ["c1", "c2"], 0)] * 3
    _patch_mc_dataset(monkeypatch, _COPA_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)), [0.0, 1.0])[1])
    run_copa(model=_make_mock_model(), max_examples=3, device="cpu")
    assert all(c == 2 for c in counts)


# ─── OpenBookQA ───────────────────────────────────────────────────────────────

_OBQA_DS = "tunalab.data_sources.evaluations.multiple_choice.openbookqa.OpenBookQADataset"


def test_openbookqa_picks_lowest_nll(monkeypatch):
    items = [_mc_item("Frilled sharks live far beneath the surface, known as",
                      ["deep sea animals", "fish", "Long Sea Fish", "Far Sea Animals"], label=0)]
    _patch_mc_dataset(monkeypatch, _OBQA_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda *a, **kw: [0.1, 0.8, 0.9, 1.0])
    result = run_openbookqa(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)


# ─── SciQ ─────────────────────────────────────────────────────────────────────

_SCIQ_DS = "tunalab.data_sources.evaluations.multiple_choice.sciq.SciQDataset"


def test_sciq_picks_lowest_nll(monkeypatch):
    items = [_mc_item("Who proposed natural selection?",
                      ["Linnaeus", "Scopes", "Shaw", "Darwin"], label=3)]
    _patch_mc_dataset(monkeypatch, _SCIQ_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda *a, **kw: [1.0, 0.9, 0.8, 0.1])
    result = run_sciq(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)


def test_sciq_four_choices(monkeypatch):
    items = [_mc_item("Q?", ["a", "b", "c", "d"], 0)] * 2
    _patch_mc_dataset(monkeypatch, _SCIQ_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)), [float(i) for i in range(len(choices))])[1])
    run_sciq(model=_make_mock_model(), max_examples=2, device="cpu")
    assert all(c == 4 for c in counts)


# ─── CodeXGLUE line completion ────────────────────────────────────────────────

_CODEXGLUE_DS = (
    "tunalab.data_sources.evaluations.fill_in_the_blank"
    ".codexglue_line_completion.CodeXGLUELineCompletionDataset"
)


def test_codexglue_scores_every_item(monkeypatch):
    items = [
        _fitb_item("import os\nfrom pathlib import Path", "result = os.path.join(base, name)"),
        _fitb_item("def greet(name):", "    return f'Hello, {name}'"),
    ]
    _patch_fitb_dataset(monkeypatch, _CODEXGLUE_DS, items)
    seen = {"pairs": 0}
    def _fake(m, pairs, **kw):
        seen["pairs"] += len(pairs)
        return [1.0] * len(pairs)
    monkeypatch.setattr(_bench, "score_completions_independent_batched", _fake)
    run_codexglue_line_completion(model=_make_mock_model(), max_examples=2, device="cpu")
    assert seen["pairs"] == 2


def test_codexglue_returns_perplexity_key(monkeypatch):
    items = [_fitb_item("x = 1\ny = 2", "z = x + y")]
    _patch_fitb_dataset(monkeypatch, _CODEXGLUE_DS, items)
    monkeypatch.setattr(_bench, "score_completions_independent_batched",
                        lambda m, pairs, **kw: [2.0] * len(pairs))
    result = run_codexglue_line_completion(model=_make_mock_model(), max_examples=1, device="cpu")
    assert "perplexity" in result
    assert result["total_examples"] == 1


def test_codexglue_prepends_newline_to_answer(monkeypatch):
    items = [_fitb_item("x = 1", "    return x")]
    _patch_fitb_dataset(monkeypatch, _CODEXGLUE_DS, items)
    encoded_calls = []
    model = _make_mock_model()
    orig = model.tokenizer.encode.side_effect
    model.tokenizer.encode.side_effect = lambda s, **kw: (encoded_calls.append(s), orig(s))[1]
    monkeypatch.setattr(_bench, "score_completions_independent_batched",
                        lambda m, pairs, **kw: [1.0] * len(pairs))
    run_codexglue_line_completion(model=model, max_examples=1, device="cpu")
    assert any(c.startswith("\n") for c in encoded_calls)


_MMLU_DS      = "tunalab.data_sources.evaluations.multiple_choice.mmlu.MMLUDataset"
_MATHQA_DS    = "tunalab.data_sources.evaluations.multiple_choice.mathqa.MathQADataset"
_HUMANEVAL_DS = "tunalab.data_sources.evaluations.multiple_choice.humaneval_buggy.HumanEvalBuggyDataset"
_MATH_DS      = "tunalab.data_sources.evaluations.fill_in_the_blank.math_competition.MATHDataset"
_CODE2TXT_DS  = "tunalab.data_sources.evaluations.fill_in_the_blank.codexglue_code_to_text.CodeXGLUECodeToTextDataset"
_REPOBENCH_DS = "tunalab.data_sources.evaluations.fill_in_the_blank.repobench.RepoBenchDataset"


# ─── MMLU ─────────────────────────────────────────────────────────────────────

def test_mmlu_picks_lowest_nll(monkeypatch):
    items = [_mc_item("What is 2+2?", ["3", "4", "5", "6"], 1)]
    _patch_mc_dataset(monkeypatch, _MMLU_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [1.0, 0.1, 0.9, 0.8])
    result = run_mmlu(model=_make_mock_model(), subject="college_mathematics",
                      max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["total_examples"] == 1


def test_mmlu_four_choices(monkeypatch):
    items = [_mc_item("Q?", ["A", "B", "C", "D"], 0)] * 3
    _patch_mc_dataset(monkeypatch, _MMLU_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)),
                                              [float(i) for i in range(len(choices))])[1])
    run_mmlu(model=_make_mock_model(), max_examples=3, device="cpu")
    assert all(c == 4 for c in counts)


def test_mmlu_wrong_answer(monkeypatch):
    items = [_mc_item("Q?", ["A", "B", "C", "D"], 2)]
    _patch_mc_dataset(monkeypatch, _MMLU_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [0.1, 0.2, 0.9, 0.3])
    result = run_mmlu(model=_make_mock_model(), device="cpu")
    assert result["accuracy"] == pytest.approx(0.0)


# ─── MathQA ───────────────────────────────────────────────────────────────────

def test_mathqa_picks_lowest_nll(monkeypatch):
    items = [_mc_item("How many ways to arrange 3 items?", ["3", "6", "9", "12", "24"], 1)]
    _patch_mc_dataset(monkeypatch, _MATHQA_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched",
                        lambda *a, **kw: [1.0, 0.2, 0.8, 0.9, 1.1])
    result = run_mathqa(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["total_examples"] == 1


def test_mathqa_five_choices(monkeypatch):
    items = [_mc_item("Q?", ["1", "2", "3", "4", "5"], 0)] * 2
    _patch_mc_dataset(monkeypatch, _MATHQA_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)),
                                              [float(i) for i in range(len(choices))])[1])
    run_mathqa(model=_make_mock_model(), max_examples=2, device="cpu")
    assert all(c == 5 for c in counts)


def test_mathqa_loads_from_local_file(tmp_path):
    import json
    rows = [
        {"Problem": "Q1", "options": "a ) 1 , b ) 2 , c ) 3 , d ) 4 , e ) 5", "correct": "a"},
        {"Problem": "Q2", "options": "a ) 6 , b ) 7 , c ) 8 , d ) 9 , e ) 10", "correct": "b"},
    ]
    (tmp_path / "test.json").write_text(json.dumps(rows))
    call_count = {"n": 0}
    def _fake_batched(m, ctx, choices, device=None):
        call_count["n"] += 1
        return [float(i) for i in range(len(choices))]
    monkeypatch_obj = type("MP", (), {"setattr": staticmethod(lambda *a: None)})()
    import eval.nlp_benchmarks as _b
    orig = _b.score_completions_batched
    _b.score_completions_batched = _fake_batched
    try:
        result = run_mathqa(model=_make_mock_model(), max_examples=2,
                            data_dir=str(tmp_path), device="cpu")
    finally:
        _b.score_completions_batched = orig
    assert result["total_examples"] == 2
    assert call_count["n"] == 2


# ─── MATH competition ─────────────────────────────────────────────────────────

def test_math_returns_perplexity(monkeypatch):
    items = [_fitb_item("Find $x$ if $x^2=4$.", "\n$x=2$.")]
    _patch_fitb_dataset(monkeypatch, _MATH_DS, items)
    monkeypatch.setattr(_bench, "score_completions_independent_batched",
                        lambda m, pairs, **kw: [2.0] * len(pairs))
    result = run_math(model=_make_mock_model(), subject="algebra", max_examples=1, device="cpu")
    assert "perplexity" in result
    assert result["total_examples"] == 1


def test_math_scores_every_item(monkeypatch):
    items = [_fitb_item("P1", "\nS1"), _fitb_item("P2", "\nS2")]
    _patch_fitb_dataset(monkeypatch, _MATH_DS, items)
    seen = {"pairs": 0}
    def _fake(m, pairs, **kw):
        seen["pairs"] += len(pairs)
        return [1.5] * len(pairs)
    monkeypatch.setattr(_bench, "score_completions_independent_batched", _fake)
    run_math(model=_make_mock_model(), max_examples=2, device="cpu")
    assert seen["pairs"] == 2


# ─── CodeXGLUE code-to-text ───────────────────────────────────────────────────

def test_codexglue_code_to_text_returns_perplexity(monkeypatch):
    items = [_fitb_item("def add(a, b):\n    return a + b", "\nAdd two numbers.")]
    _patch_fitb_dataset(monkeypatch, _CODE2TXT_DS, items)
    monkeypatch.setattr(_bench, "score_completions_independent_batched",
                        lambda m, pairs, **kw: [1.5] * len(pairs))
    result = run_codexglue_code_to_text(model=_make_mock_model(), max_examples=1, device="cpu")
    assert "perplexity" in result
    assert result["total_examples"] == 1


def test_codexglue_code_to_text_scores_every_item(monkeypatch):
    items = [_fitb_item("def f(): pass", "\nDoes nothing."),
             _fitb_item("def g(x): return x", "\nIdentity.")]
    _patch_fitb_dataset(monkeypatch, _CODE2TXT_DS, items)
    seen = {"pairs": 0}
    def _fake(m, pairs, **kw):
        seen["pairs"] += len(pairs)
        return [1.0] * len(pairs)
    monkeypatch.setattr(_bench, "score_completions_independent_batched", _fake)
    run_codexglue_code_to_text(model=_make_mock_model(), max_examples=2, device="cpu")
    assert seen["pairs"] == 2


# ─── RepoBench ────────────────────────────────────────────────────────────────

def test_repobench_invalid_split_raises():
    with pytest.raises(ValueError, match="split must be one of"):
        run_repobench(model=_make_mock_model(), split="bad_split", device="cpu")


def test_repobench_returns_perplexity(monkeypatch):
    items = [_fitb_item("import os\n\ndef main():\n    x = os.path.join", "\n    return x")]
    _patch_fitb_dataset(monkeypatch, _REPOBENCH_DS, items)
    monkeypatch.setattr(_bench, "score_completions_independent_batched",
                        lambda m, pairs, **kw: [1.2] * len(pairs))
    result = run_repobench(model=_make_mock_model(), split="cross_file_first",
                           max_examples=1, device="cpu")
    assert "perplexity" in result
    assert result["total_examples"] == 1


def test_repobench_scores_every_item(monkeypatch):
    items = [_fitb_item("ctx1\n\nf1", "\nline1"),
             _fitb_item("f2", "\nline2")]
    _patch_fitb_dataset(monkeypatch, _REPOBENCH_DS, items)
    seen = {"pairs": 0}
    def _fake(m, pairs, **kw):
        seen["pairs"] += len(pairs)
        return [1.0] * len(pairs)
    monkeypatch.setattr(_bench, "score_completions_independent_batched", _fake)
    run_repobench(model=_make_mock_model(), split="cross_file_random",
                  max_examples=2, device="cpu")
    assert seen["pairs"] == 2


# ─── RepoBench cross-doc-link variant ─────────────────────────────────────────

def _make_cross_doc_model():
    """Mock cross_doc_link model with PythonImportDetector stub and tokenizer."""
    from model.graph_traversal.python_import_detector import PythonImportDetector
    model = _make_mock_model(tokenizer=True)
    model.mask_type = "cross_doc_link"
    # Use a real PythonImportDetector so the isinstance guard in
    # run_repobench_cross_doc passes; encode/decode functions don't matter for
    # tests that patch score_completion_with_context_docs anyway.
    model.link_detector = PythonImportDetector(decode_fn=lambda toks: "")
    return model


def _raw_repobench_examples():
    """Minimal fake HF dataset rows for repobench_cross_doc tests."""
    return [
        {
            "next_line": "    return result",
            "cross_file_context": ["def helper():\n    pass\n"],
            "file_context": "import helper\n\ndef main():\n    result = helper()\n",
        },
        {
            "next_line": "    pass",
            "cross_file_context": [],
            "file_context": "def noop():\n",
        },
        {
            "next_line": "",   # empty — should be skipped
            "cross_file_context": [],
            "file_context": "x = 1\n",
        },
    ]


def test_repobench_cross_doc_requires_cross_doc_link_model():
    model = _make_mock_model()
    model.mask_type = "doc_causal"
    model.link_detector = MagicMock()
    with pytest.raises(ValueError, match="cross_doc_link"):
        run_repobench_cross_doc(model=model, device="cpu")


def test_repobench_cross_doc_requires_link_detector():
    model = _make_mock_model()
    model.mask_type = "cross_doc_link"
    model.link_detector = None
    with pytest.raises(ValueError, match="link_detector"):
        run_repobench_cross_doc(model=model, device="cpu")


def test_repobench_cross_doc_link_found_increments_n_link_found(monkeypatch):
    model = _make_cross_doc_model()
    examples = _raw_repobench_examples()[:2]  # 2 valid examples

    flat_calls = {"pairs": 0}
    def _fake_flat_batched(m, pairs, **kw):
        flat_calls["pairs"] += len(pairs)
        return [2.0] * len(pairs)

    with patch("datasets.load_dataset", return_value=_FakeHFDataset(examples)), \
         patch.object(_bench, "score_completion_with_context_docs", return_value=1.5), \
         patch.object(_bench, "score_completions_independent_batched", side_effect=_fake_flat_batched):
        result = run_repobench_cross_doc(model=model, max_examples=2, device="cpu")

    assert result["n_link_found"] == 2
    assert result["n_link_not_found"] == 0
    assert result["total_examples"] == 2
    # Flat NLL is computed (batched) once per example — grant-found or fallback.
    assert flat_calls["pairs"] == 2
    assert "perplexity_flat_linked_only" in result
    assert "average_nll_flat_linked_only" in result


def test_repobench_cross_doc_no_link_falls_back_to_flat(monkeypatch):
    model = _make_cross_doc_model()
    examples = _raw_repobench_examples()[:1]

    flat_called = {"pairs": 0}

    def _fake_flat_batched(m, pairs, **kw):
        flat_called["pairs"] += len(pairs)
        return [2.0] * len(pairs)

    with patch("datasets.load_dataset", return_value=_FakeHFDataset(examples)), \
         patch.object(_bench, "score_completion_with_context_docs", return_value=None), \
         patch.object(_bench, "score_completions_independent_batched", side_effect=_fake_flat_batched):
        result = run_repobench_cross_doc(model=model, max_examples=1, device="cpu")

    assert result["n_link_not_found"] == 1
    assert result["n_link_found"] == 0
    assert flat_called["pairs"] == 1


def test_repobench_cross_doc_reports_both_metric_keys(monkeypatch):
    model = _make_cross_doc_model()
    examples = _raw_repobench_examples()[:2]

    with patch("datasets.load_dataset", return_value=_FakeHFDataset(examples)), \
         patch.object(_bench, "score_completion_with_context_docs", return_value=1.0), \
         patch.object(_bench, "score_completion", return_value=2.0):
        result = run_repobench_cross_doc(model=model, max_examples=2, device="cpu")

    for key in ("perplexity_cross_doc_only", "perplexity_with_fallback",
                "average_nll_cross_doc_only", "average_nll_with_fallback",
                "perplexity_flat_linked_only", "average_nll_flat_linked_only",
                "n_cross_doc", "total_examples", "n_link_found", "n_link_not_found"):
        assert key in result, f"Missing key: {key}"


def test_repobench_cross_doc_skips_empty_next_line(monkeypatch):
    model = _make_cross_doc_model()
    # Only the empty-next_line example
    examples = [_raw_repobench_examples()[2]]

    with patch("datasets.load_dataset", return_value=_FakeHFDataset(examples)), \
         patch.object(_bench, "score_completion_with_context_docs", return_value=1.0), \
         patch.object(_bench, "score_completion", return_value=2.0):
        result = run_repobench_cross_doc(model=model, max_examples=1, device="cpu")

    assert result["total_examples"] == 0


class _FakeHFDataset:
    """Minimal fake HuggingFace dataset that supports len() and select()."""

    def __init__(self, rows):
        self._rows = list(rows)

    def __len__(self):
        return len(self._rows)

    def select(self, indices):
        return [self._rows[i] for i in indices]

    def __iter__(self):
        return iter(self._rows)


# ─── RepoBench cross-doc: multi-language port ─────────────────────────────────

def _make_java_cross_doc_model():
    """Mock cross_doc_link model backed by a real JavaImportDetector."""
    from model.graph_traversal.java_import_detector import JavaImportDetector
    model = _make_mock_model(tokenizer=True)
    model.mask_type = "cross_doc_link"
    model.link_detector = JavaImportDetector(decode_fn=lambda toks: "")
    return model


def test_repobench_cross_doc_invalid_language_raises():
    model = _make_cross_doc_model()
    with pytest.raises(ValueError, match="language must be one of"):
        run_repobench_cross_doc(model=model, language="cobol", device="cpu")


def test_repobench_cross_doc_java_requires_java_detector():
    # A Python detector must be rejected when language='java'.
    model = _make_cross_doc_model()  # PythonImportDetector
    with pytest.raises(ValueError, match="JavaImportDetector"):
        run_repobench_cross_doc(model=model, language="java", device="cpu")


def test_repobench_cross_doc_python_requires_python_detector():
    # A Java detector must be rejected when language='python' (the default).
    model = _make_java_cross_doc_model()
    with pytest.raises(ValueError, match="PythonImportDetector"):
        run_repobench_cross_doc(model=model, language="python", device="cpu")


def test_strip_java_source_root():
    strip = _bench._strip_java_source_root
    assert strip("service-core/x/src/main/java/fun/isite/entity/SystemMenu.java") == \
        "fun/isite/entity/SystemMenu.java"
    assert strip("backend/src/test/java/com/foo/BarTest.java") == "com/foo/BarTest.java"
    assert strip("app/src/main/kotlin/com/foo/Baz.kt") == "com/foo/Baz.kt"
    # No known source root → returned unchanged (falls back to flat scoring).
    assert strip("com/foo/Bar.java") == "com/foo/Bar.java"


def test_repobench_java_aux_identifier_roundtrips_to_import_fqn():
    """The Java aux identifier must dotify (via index_doc_span) to the import FQN.

    This is the crux of the Java port: RepoBench-Java imports are dotted FQNs
    (com.foo.Bar) but snippet paths carry a build source root. The aux
    identifier must be shaped so JavaImportDetector.index_doc_span yields the
    exact FQN the detector emits for the import, or no grant fires.
    """
    from model.graph_traversal.java_import_detector import JavaImportDetector

    class _Span:
        def __init__(self, rid):
            self.raw_identifier = rid

    det = JavaImportDetector(decode_fn=lambda toks: "")
    rid = _bench._repobench_aux_identifier(
        "java", "myrepo",
        "service/src/main/java/fun/isite/entity/SystemMenu.java",
        "public class SystemMenu {}",
    )
    # index_doc_span strips the repo prefix and dotifies the FQN path.
    assert det.index_doc_span(_Span(rid)) == "fun.isite.entity.SystemMenu"


def test_repobench_python_aux_identifier_uses_path_verbatim():
    rid = _bench._repobench_aux_identifier(
        "python", "myrepo", "pkg/sub/module.py", "def f(): pass",
    )
    assert rid == "myrepo:pkg/sub/module.py"


def test_repobench_cross_doc_java_loads_java_repo(monkeypatch):
    """language='java' must load the java HF repo, not the python one."""
    model = _make_java_cross_doc_model()
    examples = [{
        "next_line": "    List<X> f();",
        "context": [{"identifier": "X",
                     "path": "svc/src/main/java/com/foo/X.java",
                     "snippet": "public class X {}"}],
        "import_statement": "import com.foo.X;",
        "cropped_code": "class Y {\n",
        "file_path": "svc/src/main/java/com/foo/Y.java",
        "repo_name": "myrepo",
    }]
    seen = {}

    def _fake_load(repo, **kw):
        seen["repo"] = repo
        seen["verification_mode"] = kw.get("verification_mode")
        return _FakeHFDataset(examples)

    with patch("datasets.load_dataset", side_effect=_fake_load), \
         patch.object(_bench, "score_completion_with_context_docs", return_value=1.5), \
         patch.object(_bench, "score_completions_independent_batched",
                      side_effect=lambda m, pairs, **kw: [2.0] * len(pairs)):
        result = run_repobench_cross_doc(model=model, language="java",
                                         max_examples=1, device="cpu")

    assert seen["repo"] == "tianyang/repobench_java_v1.1"
    assert seen["verification_mode"] == "no_checks"
    assert result["n_link_found"] == 1


# ─── HumanEvalPack canonical-vs-buggy ─────────────────────────────────────────

def test_humaneval_buggy_invalid_language_raises():
    with pytest.raises(ValueError, match="language must be one of"):
        run_humaneval_buggy(model=_make_mock_model(), language="brainfuck", device="cpu")


def test_humaneval_buggy_picks_canonical(monkeypatch):
    items = [_mc_item("def add(a, b):\n    \"\"\"Add.\"\"\"\n",
                      ["    return a + b\n", "    return a - b\n"], 0)]
    _patch_mc_dataset(monkeypatch, _HUMANEVAL_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [0.5, 1.5])
    result = run_humaneval_buggy(model=_make_mock_model(), language="python",
                                 max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["total_examples"] == 1


def test_humaneval_buggy_fails_on_buggy_preferred(monkeypatch):
    items = [_mc_item("def f():\n    pass\n",
                      ["    return 1\n", "    return 2\n"], 0)]
    _patch_mc_dataset(monkeypatch, _HUMANEVAL_DS, items)
    monkeypatch.setattr(_bench, "score_completions_batched", lambda *a, **kw: [1.5, 0.3])
    result = run_humaneval_buggy(model=_make_mock_model(), language="python",
                                 max_examples=1, device="cpu")
    assert result["accuracy"] == pytest.approx(0.0)


def test_humaneval_buggy_two_choices(monkeypatch):
    items = [_mc_item("p1", ["c1", "b1"], 0),
             _mc_item("p2", ["c2", "b2"], 0)]
    _patch_mc_dataset(monkeypatch, _HUMANEVAL_DS, items)
    counts = []
    monkeypatch.setattr(_bench, "score_completions_batched",
        lambda m, ctx, choices, device=None: (counts.append(len(choices)), [0.0, 1.0])[1])
    run_humaneval_buggy(model=_make_mock_model(), max_examples=2, device="cpu")
    assert all(c == 2 for c in counts)


# ─── HotpotQA helpers ─────────────────────────────────────────────────────────

def _make_hotpotqa_model():
    """Mock cross_doc_link model with MarkdownLinkDetector and tokenizer."""
    from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
    model = _make_mock_model(tokenizer=True)
    model.mask_type = "cross_doc_link"
    model.link_detector = MarkdownLinkDetector(decode_fn=lambda toks: "")
    return model


# Corpus returned by monkeypatched _load_hotpotqa_corpus.
# Article A's sentence already contains the markdown link to Article B, simulating
# what _html_links_to_markdown produces from a real <a href> anchor tag.
_FAKE_HOTPOTQA_CORPUS = {
    "article a": ['Article A mentions [Article B](Article B) here.'],
    "article b": ['Article B text about the topic.'],
    "article c": ['Article C has no link to Article D.'],
    "article d": ['Article D text.'],
}

_FAKE_HOTPOTQA_BRIDGE = [
    {
        "question": "Who founded X?",
        "answer": "John",
        "supporting_facts": {"title": ["Article A", "Article B"], "sent_id": [0, 0]},
        "type": "bridge",
    },
    {
        "question": "Comparison Q",
        "answer": "Yes",
        "supporting_facts": {"title": ["Article A", "Article B"], "sent_id": [0, 0]},
        "type": "comparison",  # must be filtered out
    },
]

_FAKE_HOTPOTQA_NO_LINK = [
    {
        "question": "Q without link?",
        "answer": "No",
        "supporting_facts": {"title": ["Article C", "Article D"], "sent_id": [0, 0]},
        "type": "bridge",
    },
]


# ─── run_hotpotqa ─────────────────────────────────────────────────────────────

def test_hotpotqa_scores_all_question_types(monkeypatch):
    # Both bridge and comparison examples should be scored (no type filter in flat).
    monkeypatch.setattr(_bench, "_hotpotqa_examples",
                        lambda max_examples, cache_dir, bridge_only=False: _FAKE_HOTPOTQA_BRIDGE)
    monkeypatch.setattr(_bench, "_load_hotpotqa_corpus",
                        lambda **kw: _FAKE_HOTPOTQA_CORPUS)
    monkeypatch.setattr(_bench, "_hotpotqa_titles",
                        lambda ex: ("Article A", "Article B"))
    monkeypatch.setattr(_bench, "_hotpotqa_supporting_sent_ids",
                        lambda ex, title: [0])
    monkeypatch.setattr(_bench, "score_completion", lambda *a, **kw: 1.0)
    result = run_hotpotqa(model=_make_mock_model(), max_examples=5, device="cpu")
    # Both bridge and comparison examples counted
    assert result["total_examples"] == 2
    assert result["n_bridge"] == 1
    assert result["n_comparison"] == 1


def test_hotpotqa_returns_perplexity_keys(monkeypatch):
    monkeypatch.setattr(_bench, "_hotpotqa_examples",
                        lambda max_examples, cache_dir, bridge_only=False: [_FAKE_HOTPOTQA_BRIDGE[0]])
    monkeypatch.setattr(_bench, "_load_hotpotqa_corpus",
                        lambda **kw: _FAKE_HOTPOTQA_CORPUS)
    monkeypatch.setattr(_bench, "_hotpotqa_titles",
                        lambda ex: ("Article A", "Article B"))
    monkeypatch.setattr(_bench, "_hotpotqa_supporting_sent_ids",
                        lambda ex, title: [0])
    monkeypatch.setattr(_bench, "score_completion", lambda *a, **kw: 1.5)
    result = run_hotpotqa(model=_make_mock_model(), max_examples=1, device="cpu")
    for key in ("perplexity", "average_nll", "total_examples",
                "n_skipped_no_corpus", "n_bridge", "n_comparison"):
        assert key in result, f"Missing key: {key}"
    assert result["total_examples"] == 1


def test_hotpotqa_skips_missing_corpus(monkeypatch):
    monkeypatch.setattr(_bench, "_hotpotqa_examples",
                        lambda max_examples, cache_dir, bridge_only=False: [_FAKE_HOTPOTQA_BRIDGE[0]])
    monkeypatch.setattr(_bench, "_load_hotpotqa_corpus",
                        lambda **kw: {})  # empty corpus — both articles missing
    monkeypatch.setattr(_bench, "_hotpotqa_titles",
                        lambda ex: ("Article A", "Article B"))
    monkeypatch.setattr(_bench, "_hotpotqa_supporting_sent_ids",
                        lambda ex, title: [0])
    monkeypatch.setattr(_bench, "score_completion", lambda *a, **kw: 1.0)
    result = run_hotpotqa(model=_make_mock_model(), max_examples=1, device="cpu")
    assert result["total_examples"] == 0
    assert result["n_skipped_no_corpus"] == 1


# ─── run_hotpotqa_cross_doc ───────────────────────────────────────────────────

def test_hotpotqa_cross_doc_requires_cross_doc_link_model():
    model = _make_mock_model()
    model.mask_type = "doc_causal"
    model.link_detector = MagicMock()
    with pytest.raises(ValueError, match="cross_doc_link"):
        run_hotpotqa_cross_doc(model=model, device="cpu")


def test_hotpotqa_cross_doc_requires_markdown_link_detector():
    from model.graph_traversal.python_import_detector import PythonImportDetector
    model = _make_mock_model()
    model.mask_type = "cross_doc_link"
    model.link_detector = PythonImportDetector(decode_fn=lambda toks: "")
    with pytest.raises(ValueError, match="MarkdownLinkDetector"):
        run_hotpotqa_cross_doc(model=model, device="cpu")


def test_hotpotqa_cross_doc_requires_link_detector_not_none():
    model = _make_mock_model()
    model.mask_type = "cross_doc_link"
    model.link_detector = None
    with pytest.raises(ValueError, match="link_detector"):
        run_hotpotqa_cross_doc(model=model, device="cpu")


def test_hotpotqa_cross_doc_link_found_increments_n_link_found(monkeypatch):
    monkeypatch.setattr(_bench, "_load_hotpotqa_corpus",
                        lambda **kw: _FAKE_HOTPOTQA_CORPUS)
    monkeypatch.setattr(_bench, "_hotpotqa_bridge_examples",
                        lambda max_examples, cache_dir: [_FAKE_HOTPOTQA_BRIDGE[0]])
    monkeypatch.setattr(_bench, "_hotpotqa_titles",
                        lambda ex: ("Article A", "Article B"))
    monkeypatch.setattr(_bench, "_hotpotqa_supporting_sent_ids",
                        lambda ex, title: [0])
    monkeypatch.setattr(_bench, "score_completion_with_context_docs",
                        lambda *a, **kw: 1.5)
    flat_calls = {"pairs": 0}
    def _fake_flat_batched(m, pairs, **kw):
        flat_calls["pairs"] += len(pairs)
        return [2.0] * len(pairs)
    monkeypatch.setattr(_bench, "score_completions_independent_batched", _fake_flat_batched)
    result = run_hotpotqa_cross_doc(model=_make_hotpotqa_model(), max_examples=1,
                                    device="cpu")
    assert result["n_link_found"] == 1
    assert result["n_link_not_found"] == 0
    assert result["total_examples"] == 1
    # Flat NLL is computed (batched) for the grant-found example
    assert flat_calls["pairs"] == 1
    assert "perplexity_flat_linked_only" in result
    assert "average_nll_flat_linked_only" in result


def test_hotpotqa_cross_doc_no_grant_falls_back(monkeypatch):
    monkeypatch.setattr(_bench, "_load_hotpotqa_corpus",
                        lambda **kw: _FAKE_HOTPOTQA_CORPUS)
    monkeypatch.setattr(_bench, "_hotpotqa_bridge_examples",
                        lambda max_examples, cache_dir: [_FAKE_HOTPOTQA_BRIDGE[0]])
    monkeypatch.setattr(_bench, "_hotpotqa_titles",
                        lambda ex: ("Article A", "Article B"))
    monkeypatch.setattr(_bench, "_hotpotqa_supporting_sent_ids",
                        lambda ex, title: [0])
    monkeypatch.setattr(_bench, "score_completion_with_context_docs",
                        lambda *a, **kw: None)
    flat_called = {"pairs": 0}
    def _fake_flat_batched(m, pairs, **kw):
        flat_called["pairs"] += len(pairs)
        return [2.0] * len(pairs)
    monkeypatch.setattr(_bench, "score_completions_independent_batched", _fake_flat_batched)
    result = run_hotpotqa_cross_doc(model=_make_hotpotqa_model(), max_examples=1,
                                    device="cpu")
    assert result["n_link_not_found"] == 1
    assert result["n_link_found"] == 0
    assert flat_called["pairs"] == 1


def test_hotpotqa_cross_doc_reports_all_metric_keys(monkeypatch):
    monkeypatch.setattr(_bench, "_load_hotpotqa_corpus",
                        lambda **kw: _FAKE_HOTPOTQA_CORPUS)
    monkeypatch.setattr(_bench, "_hotpotqa_bridge_examples",
                        lambda max_examples, cache_dir: [_FAKE_HOTPOTQA_BRIDGE[0]])
    monkeypatch.setattr(_bench, "_hotpotqa_titles",
                        lambda ex: ("Article A", "Article B"))
    monkeypatch.setattr(_bench, "_hotpotqa_supporting_sent_ids",
                        lambda ex, title: [0])
    monkeypatch.setattr(_bench, "score_completion_with_context_docs",
                        lambda *a, **kw: 1.0)
    monkeypatch.setattr(_bench, "score_completion", lambda *a, **kw: 2.0)
    result = run_hotpotqa_cross_doc(model=_make_hotpotqa_model(), max_examples=1,
                                    device="cpu")
    for key in (
        "perplexity_cross_doc_only", "average_nll_cross_doc_only",
        "perplexity_flat_linked_only", "average_nll_flat_linked_only",
        "n_cross_doc",
        "perplexity_with_fallback", "average_nll_with_fallback", "total_examples",
        "n_link_found", "n_link_not_found", "n_skipped_no_corpus", "n_skipped_no_link",
    ):
        assert key in result, f"Missing key: {key}"


def test_hotpotqa_cross_doc_skips_missing_corpus_article(monkeypatch):
    # Empty corpus so both title lookups fail
    monkeypatch.setattr(_bench, "_load_hotpotqa_corpus", lambda **kw: {})
    monkeypatch.setattr(_bench, "_hotpotqa_bridge_examples",
                        lambda max_examples, cache_dir: [_FAKE_HOTPOTQA_BRIDGE[0]])
    monkeypatch.setattr(_bench, "_hotpotqa_titles",
                        lambda ex: ("Article A", "Article B"))
    monkeypatch.setattr(_bench, "_hotpotqa_supporting_sent_ids",
                        lambda ex, title: [0])
    result = run_hotpotqa_cross_doc(model=_make_hotpotqa_model(), max_examples=1,
                                    device="cpu")
    assert result["total_examples"] == 0
    assert result["n_skipped_no_corpus"] == 1


def test_hotpotqa_cross_doc_skips_no_link_in_a_sentences(monkeypatch):
    # Article C has no markdown link to Article D — pre-filter must catch this.
    monkeypatch.setattr(_bench, "_load_hotpotqa_corpus",
                        lambda **kw: _FAKE_HOTPOTQA_CORPUS)
    monkeypatch.setattr(_bench, "_hotpotqa_bridge_examples",
                        lambda max_examples, cache_dir: _FAKE_HOTPOTQA_NO_LINK)
    monkeypatch.setattr(_bench, "_hotpotqa_titles",
                        lambda ex: ("Article C", "Article D"))
    monkeypatch.setattr(_bench, "_hotpotqa_supporting_sent_ids",
                        lambda ex, title: [0])
    result = run_hotpotqa_cross_doc(model=_make_hotpotqa_model(), max_examples=1,
                                    device="cpu")
    assert result["total_examples"] == 0
    assert result["n_skipped_no_link"] == 1
