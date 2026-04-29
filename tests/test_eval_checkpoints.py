"""
tests/test_eval_checkpoints.py — unit tests for eval_checkpoints.py.

Covers:
  - _KNOWN_BENCHMARKS / _SINGLE_DOC_BENCHMARKS consistency
  - _BUILTIN_CONDITIONS structure
  - Condition skip logic: requires_cross_doc_link, single-doc experimental skip
  - doceval runs on all model types
  - benchmark dispatch calls the right function
  - per-benchmark error isolation (one failure doesn't crash the run)
  - CLI _bench_extras wiring (mmlu-subject, math-subject, repobench-split,
    humaneval-language forwarded into spec dicts)

All tests run on CPU with mock models — no CUDA, no HuggingFace network access.
"""
import pytest
from unittest.mock import MagicMock, patch, call
import torch
import torch.nn as nn

import eval_checkpoints as ec
from eval_checkpoints import (
    _KNOWN_BENCHMARKS,
    _SINGLE_DOC_BENCHMARKS,
    _BUILTIN_CONDITIONS,
    run_benchmarks_on_model,
)
from eval.nlp_benchmarks import MMLU_STEM_SUBJECTS, MATH_SUBJECTS


# ─── Mock model helpers ───────────────────────────────────────────────────────

def _make_model(mask_type="doc_causal"):
    model = MagicMock()
    model.mask_type = mask_type
    model.inference_layout_policy = MagicMock()
    model.training_layout_policy = MagicMock()
    model.inference_layout_policy.eos_token_id = 50256
    model.tokenizer = MagicMock()
    model.tokenizer.encode.side_effect = lambda s: [ord(c) % 256 for c in s]
    model.backbone = MagicMock()
    model.backbone.parameters.return_value = iter([nn.Parameter(torch.zeros(1))])
    model.forward_inference.side_effect = lambda t, s, **kw: torch.zeros(1, t.shape[1], 256)
    return model


def _fake_benchmark_result(bname):
    if bname in ("held_out_perplexity", "pack_contrastive_perplexity"):
        return {"mean_nll": 1.0, "num_docs": 1, "perplexity": 2.718}
    return {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 10}


# ─── Registry consistency ─────────────────────────────────────────────────────

def test_all_single_doc_benchmarks_are_known():
    unknown = _SINGLE_DOC_BENCHMARKS - set(_KNOWN_BENCHMARKS)
    assert not unknown, f"In _SINGLE_DOC_BENCHMARKS but not _KNOWN_BENCHMARKS: {unknown}"


def test_pack_contrastive_not_in_single_doc():
    assert "pack_contrastive_perplexity" not in _SINGLE_DOC_BENCHMARKS


def test_builtin_conditions_have_required_keys():
    for name, cond in _BUILTIN_CONDITIONS.items():
        if cond.get("_is_annotated"):
            continue  # annotated is a sentinel, not a model-config condition
        assert "mask_type" in cond, f"Condition {name!r} missing 'mask_type'"
        assert "layout_policy" in cond, f"Condition {name!r} missing 'layout_policy'"


def test_doceval_has_no_requires_cross_doc_link():
    assert not _BUILTIN_CONDITIONS["doceval"].get("requires_cross_doc_link")


def test_baseline_requires_cross_doc_link():
    assert _BUILTIN_CONDITIONS["baseline"].get("requires_cross_doc_link")


def test_experimental_mask_type_is_none():
    assert _BUILTIN_CONDITIONS["experimental"]["mask_type"] is None


# ─── Condition skip: requires_cross_doc_link ─────────────────────────────────

def test_baseline_skipped_for_doc_causal_model():
    model = _make_model(mask_type="doc_causal")
    cfg = {"benchmarks": [{"name": "piqa", "conditions": ["baseline"]}], "max_docs": 1}
    with patch.object(ec, "run_piqa", return_value={"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 1}) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_not_called()
    assert "piqa/baseline" not in results


def test_baseline_runs_for_cross_doc_link_model():
    model = _make_model(mask_type="cross_doc_link")
    cfg = {"benchmarks": [{"name": "piqa", "conditions": ["baseline"]}], "max_docs": 1}
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 1}
    with patch.object(ec, "run_piqa", return_value=ret), \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    assert "piqa/baseline" in results


# ─── Condition skip: experimental on cross_doc_link + single-doc ─────────────

def test_experimental_skipped_for_cross_doc_link_on_single_doc_benchmark():
    model = _make_model(mask_type="cross_doc_link")
    cfg = {"benchmarks": [{"name": "piqa", "conditions": ["experimental"]}], "max_docs": 1}
    with patch.object(ec, "run_piqa", return_value={"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 1}) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_not_called()
    assert "piqa/experimental" not in results


def test_experimental_runs_for_doc_causal_on_single_doc_benchmark():
    model = _make_model(mask_type="doc_causal")
    cfg = {"benchmarks": [{"name": "piqa", "conditions": ["experimental"]}], "max_docs": 1}
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 1}
    with patch.object(ec, "run_piqa", return_value=ret), \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    assert "piqa/experimental" in results


# ─── doceval runs on all model types ─────────────────────────────────────────

@pytest.mark.parametrize("mask_type", ["doc_causal", "cross_doc_link"])
def test_doceval_runs_regardless_of_mask_type(mask_type):
    model = _make_model(mask_type=mask_type)
    cfg = {"benchmarks": [{"name": "piqa", "conditions": ["doceval"]}], "max_docs": 1}
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 1}
    with patch.object(ec, "run_piqa", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_called_once()
    assert "piqa/doceval" in results


def test_doceval_uses_doc_causal_mask():
    model = _make_model(mask_type="cross_doc_link")
    cfg = {"benchmarks": [{"name": "piqa", "conditions": ["doceval"]}], "max_docs": 1}
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 1}
    captured_mask = {}
    def _fake_piqa(model, max_examples, device):
        return ret
    with patch.object(ec, "run_piqa", side_effect=_fake_piqa), \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()) as mock_layout:
        run_benchmarks_on_model(model, "/fake/dataset",
                                eval_cfg={"benchmarks": [{"name": "piqa", "conditions": ["doceval"]}], "max_docs": 1},
                                device="cpu")
    # doceval resolves with "eos" layout_policy string
    mock_layout.assert_called_once_with("eos", model)


# ─── Dispatch: each benchmark name calls the right function ──────────────────

@pytest.mark.parametrize("bname,fn_attr,extra_spec", [
    ("hellaswag",              "run_hellaswag",              {}),
    ("wiki_qa",                "run_wiki_qa",                {}),
    ("lambada",                "run_lambada",                {}),
    ("arc_easy",               "run_arc",                    {}),
    ("arc_challenge",          "run_arc",                    {}),
    ("winogrande",             "run_winogrande",             {}),
    ("piqa",                   "run_piqa",                   {}),
    ("boolq",                  "run_boolq",                  {}),
    ("commonsense_qa",         "run_commonsense_qa",         {}),
    ("copa",                   "run_copa",                   {}),
    ("openbookqa",             "run_openbookqa",             {}),
    ("sciq",                   "run_sciq",                   {}),
    ("codexglue_line_completion", "run_codexglue_line_completion", {}),
    ("mathqa",                 "run_mathqa",                 {}),
    ("codexglue_code_to_text", "run_codexglue_code_to_text", {}),
    ("humaneval_buggy",        "run_humaneval_buggy",        {"language": "go"}),
])
def test_dispatch_calls_correct_function(bname, fn_attr, extra_spec):
    model = _make_model(mask_type="doc_causal")
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 5}
    spec = {"name": bname, "conditions": ["doceval"], **extra_spec}
    cfg = {"benchmarks": [spec], "max_docs": 5}
    with patch.object(ec, fn_attr, return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_called_once()
    assert f"{bname}/doceval" in results


def test_arc_easy_passes_config_easy():
    model = _make_model(mask_type="doc_causal")
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 5}
    cfg = {"benchmarks": [{"name": "arc_easy", "conditions": ["doceval"]}], "max_docs": 5}
    with patch.object(ec, "run_arc", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    _, kwargs = mock_fn.call_args
    assert kwargs["config"] == "easy"


def test_arc_challenge_passes_config_challenge():
    model = _make_model(mask_type="doc_causal")
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 5}
    cfg = {"benchmarks": [{"name": "arc_challenge", "conditions": ["doceval"]}], "max_docs": 5}
    with patch.object(ec, "run_arc", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    _, kwargs = mock_fn.call_args
    assert kwargs["config"] == "challenge"


def test_mmlu_runs_all_stem_subjects():
    model = _make_model(mask_type="doc_causal")
    ret = {"accuracy": 0.3, "accuracy_ci": (0.1, 0.5), "total_examples": 5}
    cfg = {"benchmarks": [{"name": "mmlu", "conditions": ["doceval"]}], "max_docs": 5}
    with patch.object(ec, "run_mmlu", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    assert mock_fn.call_count == len(MMLU_STEM_SUBJECTS)
    called_subjects = {kw["subject"] for _, kw in mock_fn.call_args_list}
    assert called_subjects == set(MMLU_STEM_SUBJECTS)
    for subject in MMLU_STEM_SUBJECTS:
        assert f"mmlu/{subject}/doceval" in results


def test_mmlu_emits_per_subject_keys_not_flat_key():
    model = _make_model(mask_type="doc_causal")
    ret = {"accuracy": 0.3, "accuracy_ci": (0.1, 0.5), "total_examples": 5}
    cfg = {"benchmarks": [{"name": "mmlu", "conditions": ["doceval"]}], "max_docs": 5}
    with patch.object(ec, "run_mmlu", return_value=ret), \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    assert "mmlu/doceval" not in results


def test_mmlu_subject_failure_isolated():
    model = _make_model(mask_type="doc_causal")
    good = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 5}
    call_count = 0

    def _side_effect(model, subject, max_examples, device):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("dataset unavailable")
        return good

    cfg = {"benchmarks": [{"name": "mmlu", "conditions": ["doceval"]}], "max_docs": 5}
    with patch.object(ec, "run_mmlu", side_effect=_side_effect), \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    failed_key = f"mmlu/{MMLU_STEM_SUBJECTS[0]}/doceval"
    assert "error" in results[failed_key]
    assert results[f"mmlu/{MMLU_STEM_SUBJECTS[1]}/doceval"]["accuracy"] == pytest.approx(0.5)


def test_math_runs_all_subjects():
    model = _make_model(mask_type="doc_causal")
    ret = {"perplexity": 10.0, "average_nll": 2.3, "total_examples": 5,
           "exact_match_accuracy": 0.0, "perplexity_ci_low": 8.0, "perplexity_ci_high": 12.0,
           "nll_ci_low": 2.0, "nll_ci_high": 2.6}
    cfg = {"benchmarks": [{"name": "math", "conditions": ["doceval"]}], "max_docs": 5}
    with patch.object(ec, "run_math", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    assert mock_fn.call_count == len(MATH_SUBJECTS)
    called_subjects = {kw["subject"] for _, kw in mock_fn.call_args_list}
    assert called_subjects == set(MATH_SUBJECTS)
    for subject in MATH_SUBJECTS:
        assert f"math/{subject}/doceval" in results


def test_math_emits_per_subject_keys_not_flat_key():
    model = _make_model(mask_type="doc_causal")
    ret = {"perplexity": 10.0, "average_nll": 2.3, "total_examples": 5,
           "exact_match_accuracy": 0.0, "perplexity_ci_low": 8.0, "perplexity_ci_high": 12.0,
           "nll_ci_low": 2.0, "nll_ci_high": 2.6}
    cfg = {"benchmarks": [{"name": "math", "conditions": ["doceval"]}], "max_docs": 5}
    with patch.object(ec, "run_math", return_value=ret), \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    assert "math/doceval" not in results


def test_repobench_forwards_split_from_spec():
    model = _make_model(mask_type="doc_causal")
    ret = {"perplexity": 10.0, "average_nll": 2.3, "total_examples": 5, "exact_match_accuracy": 0.0,
           "perplexity_ci_low": 8.0, "perplexity_ci_high": 12.0, "nll_ci_low": 2.0, "nll_ci_high": 2.6}
    spec = {"name": "repobench", "conditions": ["doceval"], "split": "in_file"}
    cfg = {"benchmarks": [spec], "max_docs": 5}
    with patch.object(ec, "run_repobench", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    _, kwargs = mock_fn.call_args
    assert kwargs["split"] == "in_file"


def test_humaneval_buggy_forwards_language_from_spec():
    model = _make_model(mask_type="doc_causal")
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 5}
    spec = {"name": "humaneval_buggy", "conditions": ["doceval"], "language": "rust"}
    cfg = {"benchmarks": [spec], "max_docs": 5}
    with patch.object(ec, "run_humaneval_buggy", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    _, kwargs = mock_fn.call_args
    assert kwargs["language"] == "rust"


# ─── Error isolation ──────────────────────────────────────────────────────────

def test_failed_benchmark_doesnt_crash_run():
    model = _make_model(mask_type="doc_causal")
    cfg = {
        "benchmarks": [
            {"name": "piqa",      "conditions": ["doceval"]},
            {"name": "hellaswag", "conditions": ["doceval"]},
        ],
        "max_docs": 5,
    }
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 5}
    with patch.object(ec, "run_piqa", side_effect=RuntimeError("dataset unavailable")), \
         patch.object(ec, "run_hellaswag", return_value=ret), \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    assert "error" in results["piqa/doceval"]
    assert results["hellaswag/doceval"]["accuracy"] == pytest.approx(0.5)


def test_failed_benchmark_stores_error_string():
    model = _make_model(mask_type="doc_causal")
    cfg = {"benchmarks": [{"name": "piqa", "conditions": ["doceval"]}], "max_docs": 5}
    with patch.object(ec, "run_piqa", side_effect=RuntimeError("some error")), \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    assert "some error" in results["piqa/doceval"]["error"]


# ─── pack_contrastive_perplexity skip logic ───────────────────────────────────

def test_pack_contrastive_skipped_for_doc_causal():
    model = _make_model(mask_type="doc_causal")
    cfg = {
        "benchmarks": [{"name": "pack_contrastive_perplexity", "conditions": ["experimental"]}],
        "epoch_dirs": ["/fake/epoch"],
        "max_docs": 5,
    }
    with patch.object(ec, "run_pack_contrastive_perplexity") as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_not_called()
    assert "pack_contrastive_perplexity/experimental" not in results


def test_pack_contrastive_skipped_without_epoch_dirs():
    model = _make_model(mask_type="cross_doc_link")
    cfg = {
        "benchmarks": [{"name": "pack_contrastive_perplexity", "conditions": ["experimental"]}],
        "epoch_dirs": [],
        "max_docs": 5,
    }
    with patch.object(ec, "run_pack_contrastive_perplexity") as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_not_called()


# ─── Unknown benchmark / condition validation ─────────────────────────────────

def test_unknown_benchmark_raises():
    model = _make_model()
    cfg = {"benchmarks": [{"name": "nonexistent_bench", "conditions": ["doceval"]}]}
    with pytest.raises(ValueError, match="Unknown benchmarks"):
        run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")


def test_unknown_condition_raises():
    model = _make_model()
    cfg = {"benchmarks": [{"name": "piqa", "conditions": ["nonexistent_cond"]}]}
    with pytest.raises(ValueError, match="Unknown condition"):
        with patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
            run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")


# ─── String shorthand benchmark spec ─────────────────────────────────────────

def test_string_shorthand_uses_experimental_condition():
    model = _make_model(mask_type="doc_causal")
    ret = {"accuracy": 0.5, "accuracy_ci": (0.3, 0.7), "total_examples": 5}
    cfg = {"benchmarks": ["piqa"], "max_docs": 5}
    with patch.object(ec, "run_piqa", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_called_once()
    assert "piqa/experimental" in results


# ─── repobench_cross_doc registry + dispatch ─────────────────────────────────

def test_repobench_cross_doc_in_known_benchmarks():
    assert "repobench_cross_doc" in _KNOWN_BENCHMARKS


def test_repobench_cross_doc_not_in_single_doc_benchmarks():
    assert "repobench_cross_doc" not in _SINGLE_DOC_BENCHMARKS


def test_repobench_cross_doc_dispatch_calls_correct_function():
    model = _make_model(mask_type="cross_doc_link")
    ret = {
        "perplexity_cross_doc_only": 10.0, "average_nll_cross_doc_only": 2.3,
        "n_cross_doc": 5, "perplexity_with_fallback": 11.0,
        "average_nll_with_fallback": 2.4, "total_examples": 8,
        "n_link_found": 5, "n_link_not_found": 3,
    }
    cfg = {
        "benchmarks": [{"name": "repobench_cross_doc", "conditions": ["experimental"]}],
        "max_docs": 10,
    }
    with patch.object(ec, "run_repobench_cross_doc", return_value=ret) as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_called_once()
    assert "repobench_cross_doc/experimental" in results


def test_repobench_cross_doc_skipped_for_doc_causal_model():
    model = _make_model(mask_type="doc_causal")
    cfg = {
        "benchmarks": [{"name": "repobench_cross_doc", "conditions": ["experimental"]}],
        "max_docs": 10,
    }
    with patch.object(ec, "run_repobench_cross_doc") as mock_fn, \
         patch.object(ec, "_resolve_layout_policy", return_value=MagicMock()):
        results = run_benchmarks_on_model(model, "/fake/dataset", eval_cfg=cfg, device="cpu")
    mock_fn.assert_not_called()
    assert "repobench_cross_doc/experimental" not in results
