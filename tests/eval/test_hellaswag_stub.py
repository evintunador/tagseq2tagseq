"""
tests/eval/test_hellaswag_stub.py — unit tests for eval.hellaswag.run_hellaswag.

All tests run on CPU with mock models and synthetic data. No HuggingFace
network access is required.
"""
import math
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from eval.hellaswag import run_hellaswag


VOCAB_SIZE = 256


def _make_mock_model(tokenizer=True):
    """Mock TS2TSModel sufficient for HellaSwag evaluation."""
    model = MagicMock()

    # forward_inference returns uniform logits
    def _forward(tokens, doc_spans, **kwargs):
        T = tokens.shape[1]
        return torch.zeros(1, T, VOCAB_SIZE)

    model.forward_inference.side_effect = _forward
    model.backbone.parameters.return_value = iter([nn.Parameter(torch.zeros(1))])

    if tokenizer:
        model.tokenizer = MagicMock()
        # Simple char-level tokenizer: ord(c) % 256
        model.tokenizer.encode.side_effect = lambda s: [ord(c) % 256 for c in s]
    else:
        model.tokenizer = None

    return model


def _synthetic_raw_item():
    """One raw HellaSwag dict with a known correct answer (label=1)."""
    return _raw_item(
        context="The cat sat on the",
        choices=["floor", "mat", "ceiling", "moon"],
        label=1,
    )


# ─── Import guard ─────────────────────────────────────────────────────────────

def test_module_imports_without_datasets_package(monkeypatch):
    """eval.hellaswag can be imported even if the 'datasets' HuggingFace
    package is unavailable, because its imports are deferred inside the function."""
    import sys
    import importlib

    original = sys.modules.get("datasets")
    sys.modules["datasets"] = None  # type: ignore[assignment]

    try:
        if "eval.hellaswag" in sys.modules:
            del sys.modules["eval.hellaswag"]
        import eval.hellaswag  # noqa: F401  — should not raise
    finally:
        if original is None:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = original


# ─── Tokenizer guard ──────────────────────────────────────────────────────────

def test_run_hellaswag_requires_tokenizer():
    """run_hellaswag raises ValueError when model.tokenizer is None."""
    model = _make_mock_model(tokenizer=False)
    with pytest.raises(ValueError, match="model.tokenizer"):
        run_hellaswag(model=model, device="cpu")


# ─── Scoring behaviour ────────────────────────────────────────────────────────

def _patch_dataset(monkeypatch, raw_dicts):
    """Replace HellaSwagDataset.__init__ so it loads synthetic items without HF.

    raw_dicts must be a list of dicts with keys 'ctx', 'endings', 'label' —
    matching the format HellaSwagDataset.__getitem__ expects.
    """
    try:
        from tunalab.data_sources.evaluations.multiple_choice.hellaswag import HellaSwagDataset
    except ImportError:
        pytest.skip("tunalab NLP catalog not installed")

    def _fake_init(self, split=None, cache_dir=None, streaming=False, limit=None):
        self.data = list(raw_dicts)
        self.streaming = False
        self.limit = limit

    monkeypatch.setattr(HellaSwagDataset, "__init__", _fake_init)


def _raw_item(context, choices, label):
    """Build a raw HellaSwag dict as HellaSwagDataset.__getitem__ expects."""
    return {"ctx": context, "endings": choices, "label": label}


def test_run_hellaswag_calls_score_completions_batched_once_per_item(monkeypatch):
    """score_completions_batched is called exactly once per HellaSwag item."""
    item = _synthetic_raw_item()
    _patch_dataset(monkeypatch, [item, item])

    call_count = {"n": 0}

    def _fake_batched(model, ctx, choices, device=None):
        call_count["n"] += 1
        return [0.5] * len(choices)

    import eval.scoring as _scoring_mod
    monkeypatch.setattr(_scoring_mod, "score_completions_batched", _fake_batched)

    model = _make_mock_model()
    run_hellaswag(model=model, max_examples=2, device="cpu")

    assert call_count["n"] == 2


def test_run_hellaswag_picks_lowest_nll_choice(monkeypatch):
    """run_hellaswag returns accuracy=1.0 when the model assigns lowest NLL to the correct choice."""
    # label=2 — correct choice is index 2
    item = _raw_item(context="hello", choices=["a", "bb", "ccc", "dddd"], label=2)
    _patch_dataset(monkeypatch, [item])

    def _fake_batched(mdl, ctx, choices, device=None):
        # nlls: [1.0, 0.8, 0.3, 0.9]  → index 2 wins → matches label=2
        return [1.0, 0.8, 0.3, 0.9]

    import eval.scoring as _scoring_mod
    monkeypatch.setattr(_scoring_mod, "score_completions_batched", _fake_batched)

    model = _make_mock_model()
    results = run_hellaswag(model=model, max_examples=1, device="cpu")

    assert results["accuracy"] == pytest.approx(1.0)
    assert results["total_examples"] == 1
