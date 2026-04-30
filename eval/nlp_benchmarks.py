"""
eval/nlp_benchmarks.py — external NLP benchmark adapters for TS2TS.

Entry points (NLP — commonsense / general):
  run_hellaswag(model, max_examples, cache_dir, device) -> Dict[str, Any]
  run_wiki_qa(model, max_examples, cache_dir, device)   -> Dict[str, Any]
  run_arc(model, config, max_examples, cache_dir, device) -> Dict[str, Any]
  run_lambada(model, max_examples, cache_dir, device)   -> Dict[str, Any]
  run_winogrande(model, max_examples, cache_dir, device) -> Dict[str, Any]
  run_piqa(model, max_examples, cache_dir, device)       -> Dict[str, Any]
  run_boolq(model, max_examples, cache_dir, device)      -> Dict[str, Any]
  run_commonsense_qa(model, max_examples, cache_dir, device) -> Dict[str, Any]
  run_copa(model, max_examples, cache_dir, device)       -> Dict[str, Any]
  run_openbookqa(model, max_examples, cache_dir, device) -> Dict[str, Any]
  run_sciq(model, max_examples, cache_dir, device)       -> Dict[str, Any]

Entry points (STEM / math):
  run_mmlu(model, subject, max_examples, cache_dir, device)  -> Dict[str, Any]
  run_mathqa(model, max_examples, cache_dir, device)         -> Dict[str, Any]
  run_math(model, subject, max_examples, cache_dir, device)  -> Dict[str, Any]

Entry points (code):
  run_codexglue_line_completion(model, max_examples, cache_dir, device) -> Dict[str, Any]
  run_codexglue_code_to_text(model, max_examples, cache_dir, device)    -> Dict[str, Any]
  run_repobench(model, split, max_examples, cache_dir, device)          -> Dict[str, Any]
  run_repobench_cross_doc(model, max_examples, cache_dir, device)       -> Dict[str, Any]
  run_humaneval_buggy(model, language, max_examples, cache_dir, device) -> Dict[str, Any]

All adapters share the same design:
  - No layout decoration: external benchmarks are presented as raw text.
  - Scoring delegates to eval.scoring primitives (score_completions_batched
    for MC benchmarks, score_completion for fill-in-the-blank).
  - Single-doc benchmarks — cross-doc grants can never fire, so
    forward_inference always uses backend='flex' regardless of the model's
    trained mask_type.

Multiple-choice benchmarks:
  - K choices per item packed as K DocSpans in one forward pass (~K× faster).
  - Prediction: choice with minimum mean NLL.
  - Returns: {"accuracy": float, "accuracy_ci": ..., "total_examples": int}

Fill-in-the-blank benchmarks (lambada, codexglue_*, repobench, math):
  - NLL-only scoring: adapter returns ("", nll) — no greedy decoding.
  - Returns: {"perplexity": float, "average_nll": float, "total_examples": int, ...}
  - exact_match_accuracy is always ~0.0 and should be ignored.

Benchmarks backed by tunalab catalog adapters (hellaswag … codexglue_line_completion)
and benchmarks that load directly via the `datasets` library (mmlu, mathqa, math,
codexglue_code_to_text, repobench, repobench_cross_doc, humaneval_buggy) are all
imported from tunalab at module load time.
"""

import logging
import math as _math_module
import numpy as _np_module
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    from tunalab.evaluations.multiple_choice import MultipleChoiceEvaluation, MultipleChoiceItem
    from tunalab.evaluations.fill_in_the_blank import FillInTheBlankEvaluation, FillInTheBlankItem
    from tunalab.data_sources.evaluations.multiple_choice.hellaswag import HellaSwagDataset
    from tunalab.data_sources.evaluations.multiple_choice.hellaswag import Split as HellaSwagSplit
    from tunalab.data_sources.evaluations.multiple_choice.wiki_qa import WikiQADataset
    from tunalab.data_sources.evaluations.multiple_choice.wiki_qa import Split as WikiQASplit
    from tunalab.data_sources.evaluations.multiple_choice.arc import ARCDataset, Config as ARCConfig
    from tunalab.data_sources.evaluations.multiple_choice.arc import Split as ARCSplit
    from tunalab.data_sources.evaluations.multiple_choice.winogrande import WinoGrandeDataset
    from tunalab.data_sources.evaluations.multiple_choice.winogrande import Config as WinoGrandeConfig
    from tunalab.data_sources.evaluations.multiple_choice.winogrande import Split as WinoGrandeSplit
    from tunalab.data_sources.evaluations.multiple_choice.piqa import PIQADataset
    from tunalab.data_sources.evaluations.multiple_choice.piqa import Split as PIQASplit
    from tunalab.data_sources.evaluations.multiple_choice.boolq import BoolQDataset
    from tunalab.data_sources.evaluations.multiple_choice.boolq import Split as BoolQSplit
    from tunalab.data_sources.evaluations.multiple_choice.commonsense_qa import CommonsenseQADataset
    from tunalab.data_sources.evaluations.multiple_choice.commonsense_qa import Split as CommonsenseQASplit
    from tunalab.data_sources.evaluations.multiple_choice.copa import COPADataset
    from tunalab.data_sources.evaluations.multiple_choice.copa import Split as COPASplit
    from tunalab.data_sources.evaluations.multiple_choice.openbookqa import OpenBookQADataset
    from tunalab.data_sources.evaluations.multiple_choice.openbookqa import Split as OpenBookQASplit
    from tunalab.data_sources.evaluations.multiple_choice.sciq import SciQDataset
    from tunalab.data_sources.evaluations.multiple_choice.sciq import Split as SciQSplit
    from tunalab.data_sources.evaluations.multiple_choice.mmlu import MMLUDataset
    from tunalab.data_sources.evaluations.multiple_choice.mmlu import Split as MMLUSplit
    from tunalab.data_sources.evaluations.multiple_choice.mathqa import MathQADataset
    from tunalab.data_sources.evaluations.multiple_choice.mathqa import Split as MathQASplit
    from tunalab.data_sources.evaluations.multiple_choice.humaneval_buggy import HumanEvalBuggyDataset
    from tunalab.data_sources.evaluations.multiple_choice.humaneval_buggy import Language as HumanEvalLanguage
    from tunalab.data_sources.evaluations.fill_in_the_blank.lambada import LambadaDataset
    from tunalab.data_sources.evaluations.fill_in_the_blank.codexglue_line_completion import (
        CodeXGLUELineCompletionDataset,
        Language as CodeXGLUELanguage,
        Split as CodeXGLUESplit,
    )
    from tunalab.data_sources.evaluations.fill_in_the_blank.math_competition import MATHDataset
    from tunalab.data_sources.evaluations.fill_in_the_blank.math_competition import Subject as MATHSubject
    from tunalab.data_sources.evaluations.fill_in_the_blank.math_competition import Split as MATHSplit
    from tunalab.data_sources.evaluations.fill_in_the_blank.codexglue_code_to_text import (
        CodeXGLUECodeToTextDataset,
        Language as CodeToTextLanguage,
        Split as CodeToTextSplit,
    )
    from tunalab.data_sources.evaluations.fill_in_the_blank.repobench import RepoBenchDataset
    from tunalab.data_sources.evaluations.fill_in_the_blank.repobench import Split as RepoBenchSplit
    from tunalab.evaluation import register_handler
except ImportError as _e:
    raise ImportError(
        "tunalab NLP catalog is required for eval.nlp_benchmarks. "
        "Install it from https://github.com/evintunador/tunalab "
        "(catalogs/nlp, editable install)."
    ) from _e

from eval.scoring import score_completions_batched, score_completion, score_completion_with_context_docs
from eval.link_annotator import MarkdownPromptAnnotator, AnnotatedPrompt


def _make_encoder(tokenizer):
    """Return an encode fn that treats special-token strings as literal text.

    Code datasets (RepoBench, CodeXGLUE) frequently contain literal
    '<|endoftext|>' in source snippets; tiktoken raises by default.
    """
    return lambda text: tokenizer.encode(text, disallowed_special=())


def _require_tokenizer(model, benchmark: str) -> None:
    if not hasattr(model, "tokenizer") or model.tokenizer is None:
        raise ValueError(
            f"model.tokenizer is required for {benchmark} eval. "
            "Call to_inference_model(tokenizer=...) before evaluating."
        )


# ─── HellaSwag ────────────────────────────────────────────────────────────────

def run_hellaswag(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on HellaSwag commonsense sentence completion (multiple choice).

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full validation set).
        cache_dir: Cache directory for downloaded data. Defaults to
            data/.cache/hellaswag/.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "HellaSwag")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = HellaSwagDataset(split=HellaSwagSplit.VAL, cache_dir=cache_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── WikiQA ───────────────────────────────────────────────────────────────────

def run_wiki_qa(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on WikiQA open-domain question answering (multiple choice).

    Each question is paired with a variable number of candidate Wikipedia
    sentence answers; the model picks the answer with the lowest NLL.
    Particularly relevant for Wikipedia-trained models.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of questions (None = full validation set).
        cache_dir: Cache directory for downloaded data. Defaults to
            data/.cache/wiki_qa/.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "WikiQA")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = WikiQADataset(split=WikiQASplit.VAL, cache_dir=cache_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── ARC ──────────────────────────────────────────────────────────────────────

def run_arc(
    model,
    config: Literal["easy", "challenge"] = "challenge",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on ARC multiple-choice science reasoning.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        config: ``'easy'`` for ARC-Easy or ``'challenge'`` for ARC-Challenge.
        max_examples: Limit number of examples (None = full test split).
        cache_dir: Cache directory for downloaded data. Defaults to
            data/.cache/arc/.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    if config not in ("easy", "challenge"):
        raise ValueError(f"config must be 'easy' or 'challenge', got {config!r}")
    _require_tokenizer(model, "ARC")
    enc = _make_encoder(model.tokenizer)
    arc_config = ARCConfig.EASY if config == "easy" else ARCConfig.CHALLENGE

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = ARCDataset(config=arc_config, split=ARCSplit.TEST, cache_dir=cache_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── LAMBADA ──────────────────────────────────────────────────────────────────

def run_lambada(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on LAMBADA last-word prediction (fill-in-the-blank).

    Reports perplexity over the final word of each passage. exact_match_accuracy
    is always ~0.0 (NLL-only adapter, no greedy decoding) and should be ignored.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full test set, ~5153).
        cache_dir: Cache directory for downloaded data. Defaults to
            data/.cache/lambada/.
        device: Device for token tensors.

    Returns:
        {"perplexity": float, "average_nll": float, "total_examples": int,
         "exact_match_accuracy": float, "perplexity_ci": ..., ...}
    """
    _require_tokenizer(model, "LAMBADA")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("fill_in_the_blank")
        def handle_batch(self_, batch: List[FillInTheBlankItem]):
            outputs = []
            for item in batch:
                nll = score_completion(model, enc(item.prompt), enc(" " + item.answer), device=device)
                outputs.append(("", nll))
            return outputs

    dataset = LambadaDataset(cache_dir=cache_dir, limit=max_examples)
    return FillInTheBlankEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── WinoGrande ───────────────────────────────────────────────────────────────

def run_winogrande(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on WinoGrande commonsense pronoun resolution (2-choice MC).

    Each item is a sentence with a blank; the model picks which of two noun
    phrases correctly fills the blank. Uses the XL training split (40k) with
    the standard validation set for scoring.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full validation set).
        cache_dir: Cache directory for downloaded data.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "WinoGrande")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = WinoGrandeDataset(
        config=WinoGrandeConfig.XL,
        split=WinoGrandeSplit.VAL,
        cache_dir=cache_dir,
        limit=max_examples,
    )
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── PIQA ─────────────────────────────────────────────────────────────────────

def run_piqa(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on PIQA physical intuition question answering (2-choice MC).

    Each item presents a goal and two solution strings; the model picks the
    physically plausible solution.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full validation set).
        cache_dir: Cache directory for downloaded data.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "PIQA")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = PIQADataset(split=PIQASplit.VAL, cache_dir=cache_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── BoolQ ────────────────────────────────────────────────────────────────────

def run_boolq(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on BoolQ boolean yes/no question answering (2-choice MC).

    Context is the Wikipedia passage + question; choices are "Yes" and "No".

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full validation set).
        cache_dir: Cache directory for downloaded data.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "BoolQ")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = BoolQDataset(split=BoolQSplit.VAL, cache_dir=cache_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── CommonsenseQA ────────────────────────────────────────────────────────────

def run_commonsense_qa(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on CommonsenseQA 5-way multiple-choice commonsense reasoning.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full validation set).
        cache_dir: Cache directory for downloaded data.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "CommonsenseQA")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = CommonsenseQADataset(
        split=CommonsenseQASplit.VAL, cache_dir=cache_dir, limit=max_examples
    )
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── COPA ─────────────────────────────────────────────────────────────────────

def run_copa(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on COPA causal plausibility (2-choice MC, from SuperGLUE).

    Each item gives a premise and asks for the more plausible cause or effect.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full 100-item val set).
        cache_dir: Cache directory for downloaded data.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "COPA")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = COPADataset(split=COPASplit.VAL, cache_dir=cache_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── OpenBookQA ───────────────────────────────────────────────────────────────

def run_openbookqa(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on OpenBookQA elementary science reasoning (4-choice MC).

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full validation set).
        cache_dir: Cache directory for downloaded data.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "OpenBookQA")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = OpenBookQADataset(
        split=OpenBookQASplit.VAL, cache_dir=cache_dir, limit=max_examples
    )
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── SciQ ─────────────────────────────────────────────────────────────────────

def run_sciq(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on SciQ science exam questions (4-choice MC).

    Questions cover physics, chemistry, and biology. Distractors are shuffled
    seeded-randomly so the correct answer appears at a random position.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full validation set).
        cache_dir: Cache directory for downloaded data.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "SciQ")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = SciQDataset(split=SciQSplit.VAL, cache_dir=cache_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── CodeXGLUE line completion ────────────────────────────────────────────────

def run_codexglue_line_completion(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on CodeXGLUE next-line code completion (fill-in-the-blank).

    Each item is a Python source file; the prompt is all lines except the last
    non-trivial line; the answer is that final line. NLL is computed over the
    answer tokens. Tests code understanding without execution.

    Primary metric: perplexity over last-line tokens (lower = better).
    exact_match_accuracy is always ~0.0 (NLL-only adapter) and should be ignored.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full test set, ~50k).
        cache_dir: Cache directory for downloaded data.
        device: Device for token tensors.

    Returns:
        {"perplexity": float, "average_nll": float, "total_examples": int,
         "exact_match_accuracy": float, "perplexity_ci": ..., ...}
    """
    _require_tokenizer(model, "CodeXGLUE line completion")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("fill_in_the_blank")
        def handle_batch(self_, batch: List[FillInTheBlankItem]):
            outputs = []
            for item in batch:
                nll = score_completion(
                    model, enc(item.prompt), enc("\n" + item.answer), device=device
                )
                outputs.append(("", nll))
            return outputs

    dataset = CodeXGLUELineCompletionDataset(
        language=CodeXGLUELanguage.PYTHON,
        split=CodeXGLUESplit.TEST,
        cache_dir=cache_dir,
        limit=max_examples,
    )
    return FillInTheBlankEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── MMLU (STEM subsets, 4-way MC) ────────────────────────────────────────────

#: Canonical MMLU STEM subjects supported by run_mmlu.
MMLU_STEM_SUBJECTS = (
    "college_mathematics",
    "high_school_mathematics",
    "high_school_physics",
    "college_physics",
    "high_school_chemistry",
    "college_chemistry",
    "high_school_biology",
    "high_school_statistics",
    "college_computer_science",
    "high_school_computer_science",
    "machine_learning",
    "abstract_algebra",
    "conceptual_physics",
)


def run_mmlu(
    model,
    subject: str = "college_mathematics",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on an MMLU subject (4-way multiple choice).

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        subject: MMLU subject name (default: ``'college_mathematics'``).
            Any cais/mmlu subset name is valid; see MMLU_STEM_SUBJECTS for the
            recommended STEM subjects.
        max_examples: Limit number of test examples (None = full test set).
        cache_dir: HuggingFace cache directory.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, f"MMLU/{subject}")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = MMLUDataset(subject=subject, split=MMLUSplit.TEST,
                          cache_dir=cache_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── MathQA (5-way MC math word problems) ─────────────────────────────────────

def run_mathqa(
    model,
    max_examples: Optional[int] = None,
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on MathQA (5-way multiple-choice math word problems).

    Loads from raw JSON files (``data/.cache/mathqa/test.json`` by default).
    Download with:
        wget https://math-qa.github.io/math-QA/data/MathQA.zip
        unzip MathQA.zip -d data/.cache/mathqa/

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of test examples (None = full test set, ~2985).
        data_dir: Directory containing ``test.json``. Defaults to
            ``data/.cache/mathqa/`` relative to the project root.
        cache_dir: Unused; kept for API consistency.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    _require_tokenizer(model, "MathQA")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(" " + c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = MathQADataset(split=MathQASplit.TEST, data_dir=data_dir, limit=max_examples)
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── MATH competition (LaTeX fill-in-the-blank) ───────────────────────────────

#: Canonical MATH dataset subjects.
MATH_SUBJECTS = tuple(s.value for s in MATHSubject)


def run_math(
    model,
    subject: str = "algebra",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on the Hendrycks MATH dataset (LaTeX fill-in-the-blank).

    Scores NLL of the full solution given the problem. Results are not
    comparable to published solve-rate numbers; report as "perplexity on
    canonical solution".

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        subject: One of MATH_SUBJECTS (default: ``'algebra'``).
        max_examples: Limit number of test examples (None = full split).
        cache_dir: HuggingFace cache directory.
        device: Device for token tensors.

    Returns:
        {"perplexity": float, "average_nll": float, "total_examples": int, ...}
    """
    _require_tokenizer(model, f"MATH/{subject}")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("fill_in_the_blank")
        def handle_batch(self_, batch: List[FillInTheBlankItem]):
            outputs = []
            for item in batch:
                nll = score_completion(model, enc(item.prompt), enc(item.answer), device=device)
                outputs.append(("", nll))
            return outputs

    dataset = MATHDataset(
        subject=MATHSubject(subject),
        split=MATHSplit.TEST,
        cache_dir=cache_dir,
        limit=max_examples,
    )
    return FillInTheBlankEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── CodeXGLUE code-to-text (code → docstring) ────────────────────────────────

def run_codexglue_code_to_text(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on CodeXGLUE code-to-text (Python function → docstring).

    Directly relevant for Stack-trained models: tests whether the model has
    learned the association between code structure and natural-language
    summaries. Cleaner signal than line completion (docstrings are semantically
    rich, not just "predict the next syntactic line").

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Limit number of test examples (None = full test set, 14,918).
        cache_dir: HuggingFace cache directory.
        device: Device for token tensors.

    Returns:
        {"perplexity": float, "average_nll": float, "total_examples": int, ...}
    """
    _require_tokenizer(model, "CodeXGLUE code-to-text")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("fill_in_the_blank")
        def handle_batch(self_, batch: List[FillInTheBlankItem]):
            outputs = []
            for item in batch:
                nll = score_completion(model, enc(item.prompt), enc(item.answer), device=device)
                outputs.append(("", nll))
            return outputs

    dataset = CodeXGLUECodeToTextDataset(
        language=CodeToTextLanguage.PYTHON,
        split=CodeToTextSplit.TEST,
        cache_dir=cache_dir,
        limit=max_examples,
    )
    return FillInTheBlankEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── RepoBench-C (cross-file next-line, repo context) ─────────────────────────

#: Valid RepoBench split names.
REPOBENCH_SPLITS = tuple(s.value for s in RepoBenchSplit)


def run_repobench(
    model,
    split: str = "cross_file_first",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on RepoBench-C Python cross-file next-line completion (flat baseline).

    All cross-file snippets are text-concatenated before the primary file prefix
    under doc_causal masking. For the cross_doc_link-aware variant that packs
    snippets as proper aux DocSpans, use run_repobench_cross_doc instead.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        split: One of REPOBENCH_SPLITS (default: ``'cross_file_first'``).
        max_examples: Limit number of examples (None = full split).
        cache_dir: HuggingFace cache directory.
        device: Device for token tensors.

    Returns:
        {"perplexity": float, "average_nll": float, "total_examples": int, ...}
    """
    if split not in REPOBENCH_SPLITS:
        raise ValueError(f"split must be one of {REPOBENCH_SPLITS}, got {split!r}")
    _require_tokenizer(model, f"RepoBench/{split}")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("fill_in_the_blank")
        def handle_batch(self_, batch: List[FillInTheBlankItem]):
            outputs = []
            for item in batch:
                nll = score_completion(model, enc(item.prompt), enc(item.answer), device=device)
                outputs.append(("", nll))
            return outputs

    dataset = RepoBenchDataset(
        split=RepoBenchSplit(split),
        cache_dir=cache_dir,
        limit=max_examples,
    )
    return FillInTheBlankEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── RepoBench-C cross-doc-link variant ──────────────────────────────────────

def run_repobench_cross_doc(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """RepoBench-C cross_file_first scored with cross-doc-link attention.

    Packs each example's cross-file snippets as aux DocSpans preceding the
    primary file context. Uses PythonImportDetector to match each import
    statement in the primary doc to its specific snippet via the snippet's
    file path (precise matching — each import grants attention only to the
    snippet it actually imports). Requires a cross_doc_link model with a
    link_detector set.

    Reports two perplexity figures for transparency:
      - ``cross_doc_only``: only examples where an import link was detected
        (true cross-doc evaluation).
      - ``with_fallback``: all examples; those without a detected import fall
        back to flat doc_causal scoring (same as run_repobench).

    Args:
        model: TS2TSModel in eval mode. Must have mask_type='cross_doc_link'
            and link_detector set. Must have model.tokenizer set.
        max_examples: Limit number of examples (None = full split).
        cache_dir: HuggingFace cache directory.
        device: Device for token tensors.

    Returns:
        {
            "perplexity_cross_doc_only":  float,
            "average_nll_cross_doc_only": float,
            "n_cross_doc":                int,
            "perplexity_with_fallback":   float,
            "average_nll_with_fallback":  float,
            "total_examples":             int,
            "n_link_found":               int,
            "n_link_not_found":           int,
        }
    """
    _require_tokenizer(model, "RepoBench/cross_file_first (cross-doc)")
    if not hasattr(model, "mask_type") or model.mask_type != "cross_doc_link":
        raise ValueError(
            "run_repobench_cross_doc requires a cross_doc_link model. "
            f"Got mask_type={getattr(model, 'mask_type', None)!r}."
        )
    if not hasattr(model, "link_detector") or model.link_detector is None:
        raise ValueError(
            "run_repobench_cross_doc requires model.link_detector to be set. "
            "Use to_inference_model(link_detector=...) or TS2TSModel.__init__."
        )
    from model.graph_traversal.python_import_detector import PythonImportDetector
    if not isinstance(model.link_detector, PythonImportDetector):
        raise ValueError(
            f"run_repobench_cross_doc requires a PythonImportDetector "
            f"(got {type(model.link_detector).__name__!r}). "
            "RepoBench is a Python dataset; other link detectors cannot match "
            "Python import statements to cross-file snippets. When support for "
            "additional programming languages is added, consider splitting this "
            "benchmark by language and matching each to its <Language>ImportDetector."
        )

    from datasets import load_dataset as _load_dataset
    raw = _load_dataset(
        "tianyang/repobench_python_v1.1",
        split="cross_file_first",
        cache_dir=cache_dir or "data/.cache/repobench",
    )

    enc = _make_encoder(model.tokenizer)
    link_detector = model.link_detector
    repo_name = None  # populated per-example below

    cross_doc_nlls: List[float] = []
    paired_flat_nlls: List[float] = []
    all_nlls: List[float] = []
    n_link_found = 0
    n_link_not_found = 0

    limit = max_examples if max_examples is not None else len(raw)
    for ex in raw.select(range(min(limit, len(raw)))):
        next_line = ex.get("next_line", "")
        if not next_line.strip():
            continue

        # context is list of {'identifier', 'path', 'snippet'} dicts.
        context_items = ex.get("context", [])
        repo = ex.get("repo_name", "repo")

        # raw_identifier "repo:path/to/file.py" — PythonImportDetector.index_doc_span
        # strips the repo prefix, so candidate paths produced by the import detector
        # (e.g. "pkg/module.py") match the path component of the identifier.
        aux_token_lists: List[List[int]] = []
        aux_raw_identifiers: List[str] = []
        for item in context_items:
            snippet = item.get("snippet", "")
            path    = item.get("path", "")
            if not snippet.strip():
                continue
            aux_token_lists.append(enc(snippet))
            aux_raw_identifiers.append(f"{repo}:{path}")

        # import_statement contains the imports that reference the cross-file snippets;
        # cropped_code is the file body starting after those imports. Concatenate so
        # PythonImportDetector can detect the relevant import positions in the sequence.
        import_stmt    = ex.get("import_statement", "")
        context_tokens = enc(import_stmt + "\n" + ex.get("cropped_code", ""))
        completion_tokens = enc("\n" + next_line)

        nll = score_completion_with_context_docs(
            model,
            aux_token_lists=aux_token_lists,
            context_tokens=context_tokens,
            completion_tokens=completion_tokens,
            link_detector=link_detector,
            aux_raw_identifiers=aux_raw_identifiers,
            source_file_path=ex.get("file_path", ""),
            device=device,
        )

        if nll is not None:
            cross_doc_nlls.append(nll)
            all_nlls.append(nll)
            n_link_found += 1
            flat_nll = score_completion(model, context_tokens, completion_tokens, device=device)
            paired_flat_nlls.append(flat_nll)
        else:
            flat_nll = score_completion(model, context_tokens, completion_tokens, device=device)
            all_nlls.append(flat_nll)
            n_link_not_found += 1

    def _ppl(nlls: List[float]) -> float:
        import math as _math
        return _math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")

    def _mean(nlls: List[float]) -> float:
        return sum(nlls) / len(nlls) if nlls else float("nan")

    return {
        "perplexity_cross_doc_only":    _ppl(cross_doc_nlls),
        "average_nll_cross_doc_only":   _mean(cross_doc_nlls),
        "perplexity_flat_linked_only":  _ppl(paired_flat_nlls),
        "average_nll_flat_linked_only": _mean(paired_flat_nlls),
        "n_cross_doc":                  len(cross_doc_nlls),
        "perplexity_with_fallback":     _ppl(all_nlls),
        "average_nll_with_fallback":    _mean(all_nlls),
        "total_examples":               len(all_nlls),
        "n_link_found":                 n_link_found,
        "n_link_not_found":             n_link_not_found,
    }


# ─── HotpotQA bridge QA ──────────────────────────────────────────────────────

_HOTPOTQA_CORPUS_URL_ABSTRACTS = (
    "https://nlp.stanford.edu/projects/hotpotqa/"
    "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
)
_HOTPOTQA_CORPUS_URL_FULL = (
    "https://nlp.stanford.edu/projects/hotpotqa/"
    "enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2"
)

# Module-level cache so both run_hotpotqa and run_hotpotqa_cross_doc share
# the same loaded dict within a single eval session.
_HOTPOTQA_CORPUS_CACHE: Optional[dict] = None


def _html_links_to_markdown(sentence: str) -> str:
    """Convert HotpotQA inline HTML links to [text](Title) markdown.

    HotpotQA corpus stores links as <a href="url%20encoded%20title">anchor</a>.
    After conversion, MarkdownLinkDetector fires on the ]( bigram and extracts
    the URL-decoded title as target_str — identical to how our wikitext pipeline
    produces [text](Article Title) from [[Article Title]] during training.
    """
    import re as _re
    import urllib.parse as _up
    return _re.sub(
        r'<a href="([^"]*)">(.*?)</a>',
        lambda m: f'[{m.group(2)}]({_up.unquote(m.group(1))})',
        sentence,
    )


def _strip_html_links(sentence: str) -> str:
    """Strip HTML anchor tags, keeping only the anchor text."""
    import re as _re
    return _re.sub(r'<a href="[^"]*">(.*?)</a>', r'\1', sentence)


def _load_hotpotqa_corpus(
    cache_dir: Optional[str] = None,
    use_full: bool = False,
) -> dict:
    """Return title.lower() -> List[str] of sentences with <a href> links intact.

    Downloads the HotpotQA Wikipedia corpus on first call and caches a pickled
    dict to data/.cache/hotpotqa/. Subsequent calls load from pickle.

    The abstracts corpus (~1.55 GB) covers introductory paragraphs for all
    English Wikipedia articles (2017 dump). Most HotpotQA supporting facts
    come from intro paragraphs, so this is usually sufficient. If link match
    rate is too low in practice, pass use_full=True to use the full corpus
    (~7.4 GB).

    Leakage note: the comparison is always contrastive (cross_doc vs doc_causal
    on the same text), so memorisation cancels out. The 2017 HotpotQA dump
    predates our 2025-2026 training dumps and covers the same Wikipedia articles
    -- being in-distribution is a feature, not a confound.
    """
    global _HOTPOTQA_CORPUS_CACHE
    if _HOTPOTQA_CORPUS_CACHE is not None:
        return _HOTPOTQA_CORPUS_CACHE

    import os as _os
    import bz2 as _bz2
    import json as _json
    import pickle as _pickle
    import tarfile as _tarfile
    import urllib.request as _req

    cache_root = cache_dir or "data/.cache/hotpotqa"
    suffix = "full" if use_full else "abstracts"
    pkl_path = _os.path.join(cache_root, f"corpus_{suffix}.pkl")

    if _os.path.exists(pkl_path):
        logger.info("Loading HotpotQA corpus from cache: %s", pkl_path)
        with open(pkl_path, "rb") as f:
            _HOTPOTQA_CORPUS_CACHE = _pickle.load(f)
        return _HOTPOTQA_CORPUS_CACHE

    url = _HOTPOTQA_CORPUS_URL_FULL if use_full else _HOTPOTQA_CORPUS_URL_ABSTRACTS
    tar_path = _os.path.join(cache_root, _os.path.basename(url))
    _os.makedirs(cache_root, exist_ok=True)

    if not _os.path.exists(tar_path):
        logger.info("Downloading HotpotQA corpus (~%.1f GB): %s",
                    7.4 if use_full else 1.55, url)
        try:
            from tqdm.auto import tqdm as _tqdm

            class _TqdmHook:
                def __init__(self):
                    self._t = None
                def __call__(self, b, bsize, total):
                    if self._t is None:
                        self._t = _tqdm(total=total, unit="B", unit_scale=True,
                                        desc="hotpotqa corpus")
                    self._t.update(b * bsize - self._t.n)
                def __del__(self):
                    if self._t is not None:
                        self._t.close()

            _req.urlretrieve(url, tar_path, reporthook=_TqdmHook())
        except Exception:
            _req.urlretrieve(url, tar_path)

    logger.info("Extracting HotpotQA corpus...")
    corpus: dict = {}
    with _tarfile.open(tar_path, "r:bz2") as tf:
        for member in tf:   # streaming iteration — avoids building full member list
            if not member.isfile():
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            # Each member is itself bz2-compressed (double compression).
            try:
                raw = _bz2.decompress(f.read())
            except Exception:
                continue
            for line in raw.split(b"\n"):
                if not line.strip():
                    continue
                try:
                    art = _json.loads(line)
                except (_json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(art, dict):
                    continue
                title = art.get("title", "")
                if not title:
                    continue
                # text_with_links is a flat List[str] (one sentence per element)
                # covering the article's introductory paragraph, with inline
                # <a href="url%20encoded%20title">anchor text</a> links.
                # The full corpus uses text (same structure, also has <a href>).
                sents_with_links = art.get("text_with_links") or art.get("text")
                if not sents_with_links or not isinstance(sents_with_links, list):
                    continue
                corpus[title.lower()] = [
                    s for s in sents_with_links if isinstance(s, str)
                ]

    logger.info("HotpotQA corpus loaded: %d articles", len(corpus))
    with open(pkl_path, "wb") as f:
        _pickle.dump(corpus, f, protocol=_pickle.HIGHEST_PROTOCOL)

    _HOTPOTQA_CORPUS_CACHE = corpus
    return corpus


def _hotpotqa_examples(
    max_examples: Optional[int],
    cache_dir: Optional[str],
    bridge_only: bool = False,
) -> List[dict]:
    """Load HotpotQA fullwiki validation split.

    Args:
        max_examples: Cap on number of returned examples (applied after
            optional bridge filter).
        cache_dir: HuggingFace cache directory.
        bridge_only: If True, return only bridge-type questions (~5918/7405).
            Bridge questions have a natural hyperlink from article A to article
            B, which is required for cross-doc attention grants to fire.
            For the flat benchmark (run_hotpotqa) this filter is not applied;
            all 7405 validation examples are used for full benchmark coverage.
    """
    from datasets import load_dataset as _load_dataset
    raw = _load_dataset(
        "hotpotqa/hotpot_qa",
        "fullwiki",
        split="validation",
        cache_dir=cache_dir or "data/.cache/hotpotqa",
    )
    examples = [ex for ex in raw if ex.get("type") == "bridge"] if bridge_only else list(raw)
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


# Keep old name as alias so monkeypatching in tests still works during transition.
def _hotpotqa_bridge_examples(max_examples, cache_dir):
    return _hotpotqa_examples(max_examples, cache_dir, bridge_only=True)


def _hotpotqa_titles(ex: dict):
    """Return (a_title, b_title) — the two distinct supporting article titles.

    supporting_facts["title"] is a flat list of titles (one per supporting
    sentence, so titles repeat when multiple sentences come from the same
    article). We need the first two *distinct* titles in appearance order.
    """
    seen = []
    for t in ex["supporting_facts"]["title"]:
        if t not in seen:
            seen.append(t)
        if len(seen) == 2:
            break
    return (seen[0], seen[1]) if len(seen) == 2 else (None, None)


def _hotpotqa_context_sents(ex: dict, title: str) -> List[str]:
    """Return the plain-text context sentences for a given article title.

    Uses ex["context"], which is a dict of parallel lists {"title": [...],
    "sentences": [[sent, ...], ...]}. Returns the sentence list for the
    matching title, or [] if not found.
    """
    ctx = ex.get("context", {})
    titles = ctx.get("title", [])
    sentences = ctx.get("sentences", [])
    for i, t in enumerate(titles):
        if t == title and i < len(sentences):
            return list(sentences[i])
    return []


def _hotpotqa_supporting_sent_ids(ex: dict, title: str) -> List[int]:
    """Return the sent_ids listed in supporting_facts for a given title."""
    sf = ex.get("supporting_facts", {})
    sf_titles = sf.get("title", [])
    sf_sent_ids = sf.get("sent_id", [])
    return [sf_sent_ids[i] for i, t in enumerate(sf_titles) if t == title]


def run_hotpotqa(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on HotpotQA (all question types, flat concat, doc_causal baseline).

    Scores all validation examples (bridge + comparison, ~7405 total) using
    the gold supporting sentences from the downloaded Wikipedia corpus as
    plain-text context. This is the full-benchmark version for comparison
    against other models and published numbers.

    For the paired cross-doc comparison (bridge-only, same examples as
    run_hotpotqa_cross_doc), see the ``average_nll_flat_linked_only`` key
    returned by run_hotpotqa_cross_doc — that is the preferred metric for
    measuring the benefit of cross-doc attention.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        max_examples: Cap on examples evaluated (None = all ~7405).
        cache_dir: Cache directory for HF dataset + corpus download.
        device: Device for token tensors.

    Returns:
        {"perplexity": float, "average_nll": float, "total_examples": int,
         "n_skipped_no_corpus": int, "n_bridge": int, "n_comparison": int}
    """
    import math as _math
    _require_tokenizer(model, "HotpotQA")
    enc = _make_encoder(model.tokenizer)

    corpus = _load_hotpotqa_corpus(cache_dir=cache_dir)
    examples = _hotpotqa_examples(max_examples, cache_dir, bridge_only=False)

    nlls: List[float] = []
    n_skipped_no_corpus = 0
    n_bridge = 0
    n_comparison = 0

    for ex in examples:
        a_title, b_title = _hotpotqa_titles(ex)
        if a_title is None:
            continue

        a_sents_raw = corpus.get(a_title.lower())
        b_sents_raw = corpus.get(b_title.lower())
        if a_sents_raw is None or b_sents_raw is None:
            n_skipped_no_corpus += 1
            continue

        a_ids = _hotpotqa_supporting_sent_ids(ex, a_title)
        b_ids = _hotpotqa_supporting_sent_ids(ex, b_title)

        def _pick_plain(sents, ids):
            picked = [_strip_html_links(sents[i]) for i in ids if i < len(sents)]
            return picked if picked else [_strip_html_links(sents[0])]

        a_text = " ".join(_pick_plain(a_sents_raw, a_ids))
        b_text = " ".join(_pick_plain(b_sents_raw, b_ids))
        context = a_text + " " + b_text + "\nQuestion: " + ex["question"] + "\nAnswer: "

        nll = score_completion(model, enc(context), enc(ex["answer"]), device=device)
        nlls.append(nll)
        if ex.get("type") == "bridge":
            n_bridge += 1
        else:
            n_comparison += 1

    mean_nll = sum(nlls) / len(nlls) if nlls else float("nan")
    try:
        ppl = _math.exp(mean_nll)
    except OverflowError:
        ppl = float("inf")

    return {
        "perplexity":            ppl,
        "average_nll":           mean_nll,
        "total_examples":        len(nlls),
        "n_skipped_no_corpus":   n_skipped_no_corpus,
        "n_bridge":              n_bridge,
        "n_comparison":          n_comparison,
    }


def run_hotpotqa_cross_doc(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """HotpotQA bridge questions scored with cross-doc-link attention.

    Packs article B's supporting sentences as an aux DocSpan preceding article
    A's supporting sentences (the primary doc). MarkdownLinkDetector fires on
    the [text](B title) link that is naturally present in article A's text —
    the same mechanism that fires during training on Wikipedia hyperlinks.
    Requires a cross_doc_link model with a MarkdownLinkDetector.

    Only bridge-type questions are used. Within those, examples are further
    filtered to cases where at least one article A supporting sentence contains
    a rendered [text](B title) markdown link after HTML-to-markdown conversion.
    This ensures the grant fires naturally rather than being forced.

    Leakage note: the comparison is always contrastive (cross_doc vs doc_causal
    on the same text), so memorisation cancels out. The 2017 HotpotQA dump
    predates our 2025-2026 training dumps and covers the same Wikipedia articles
    -- being in-distribution is a feature, not a confound.

    Reports three perplexity figures:
      - ``cross_doc_only`` / ``flat_linked_only``: the primary paired comparison.
        Same N examples (those where a grant actually fired), scored under
        cross-doc attention vs plain doc_causal. This is the cleanest measure
        of cross-doc benefit — identical questions, identical tokenizations,
        differing only in whether grants are active. Use this for the headline
        delta.
      - ``with_fallback``: all examples with both articles in corpus; those
        where the grant didn't fire (n_link_not_found) use flat scoring.

    Note on ``n_link_not_found``: these are examples where the pre-check
    found the marker text but MarkdownLinkDetector still didn't fire a matched
    grant after tokenization. Two structural causes — both consistent with
    training distribution and not fixable without changing the detector:
      - Title contains parentheses, e.g. "Alien (film)": the detector stops at
        the first ')' and extracts "Alien (film" instead of "Alien (film)".
        During training [[Alien (film)]] produces the same mismatch, so these
        links never fired grants on the training corpus either.
      - Title is quoted in context, e.g. '"[Animorphs](Animorphs)"': the '"['
        tokenizes differently from bare '[', so the backwards scan for the
        link-open token fails.
    These 26/200 fallbacks are correct; forcing grants would give the model
    cross-doc attention it was never trained to use.

    Args:
        model: TS2TSModel in eval mode. Must have mask_type='cross_doc_link',
            link_detector set to a MarkdownLinkDetector, and tokenizer set.
        max_examples: Limit number of bridge examples checked (None = all ~5.9k).
        cache_dir: Cache directory for HF dataset + corpus download.
        device: Device for token tensors.

    Returns:
        {
            "perplexity_cross_doc_only":   float,
            "average_nll_cross_doc_only":  float,
            "perplexity_flat_linked_only": float,   # paired baseline, same N
            "average_nll_flat_linked_only": float,
            "n_cross_doc":                 int,
            "perplexity_with_fallback":    float,
            "average_nll_with_fallback":   float,
            "total_examples":              int,
            "n_link_found":                int,
            "n_link_not_found":            int,
            "n_skipped_no_corpus":         int,
            "n_skipped_no_link":           int,
        }
    """
    import math as _math
    _require_tokenizer(model, "HotpotQA (cross-doc)")
    if not hasattr(model, "mask_type") or model.mask_type != "cross_doc_link":
        raise ValueError(
            "run_hotpotqa_cross_doc requires a cross_doc_link model. "
            f"Got mask_type={getattr(model, 'mask_type', None)!r}."
        )
    if not hasattr(model, "link_detector") or model.link_detector is None:
        raise ValueError(
            "run_hotpotqa_cross_doc requires model.link_detector to be set. "
            "Use to_inference_model(link_detector=...) or TS2TSModel.__init__."
        )
    from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
    if not isinstance(model.link_detector, MarkdownLinkDetector):
        raise ValueError(
            f"run_hotpotqa_cross_doc requires a MarkdownLinkDetector "
            f"(got {type(model.link_detector).__name__!r}). "
            "HotpotQA is a Wikipedia dataset; other link detectors cannot match "
            "Wikipedia hyperlinks to supporting articles."
        )

    enc = _make_encoder(model.tokenizer)
    link_detector = model.link_detector

    corpus = _load_hotpotqa_corpus(cache_dir=cache_dir)
    examples = _hotpotqa_bridge_examples(max_examples, cache_dir)

    cross_doc_nlls: List[float] = []
    paired_flat_nlls: List[float] = []   # doc_causal on same examples as cross_doc_nlls
    all_nlls: List[float] = []
    n_link_found = 0
    n_link_not_found = 0
    n_skipped_no_corpus = 0
    n_skipped_no_link = 0

    for ex in examples:
        a_title, b_title = _hotpotqa_titles(ex)
        if a_title is None:
            continue
        a_sent_ids = _hotpotqa_supporting_sent_ids(ex, a_title)
        b_sent_ids = _hotpotqa_supporting_sent_ids(ex, b_title)

        a_sents_raw = corpus.get(a_title.lower())
        b_sents_raw = corpus.get(b_title.lower())
        if a_sents_raw is None or b_sents_raw is None:
            n_skipped_no_corpus += 1
            continue

        def _pick_raw(sents, ids):
            picked = [sents[i] for i in ids if i < len(sents)]
            return picked if picked else [sents[0]]

        # Article A: convert HTML links to [text](Title) markdown so
        # MarkdownLinkDetector fires on the ]( bigram naturally.
        a_sents_md = [_html_links_to_markdown(s) for s in _pick_raw(a_sents_raw, a_sent_ids)]
        # Article B: plain text as aux doc (no links needed in it).
        b_sents_plain = [_strip_html_links(s) for s in _pick_raw(b_sents_raw, b_sent_ids)]

        # Pre-filter: at least one A sentence must contain ](B_title) after
        # conversion; otherwise the grant can never fire and the result would
        # be identical to the flat baseline.
        marker = f"]({b_title})"
        if not any(marker in s for s in a_sents_md):
            n_skipped_no_link += 1
            continue

        a_text_md = " ".join(a_sents_md)
        b_text_plain = " ".join(b_sents_plain)

        aux_tokens = enc(b_text_plain)
        context_tokens = enc(a_text_md + "\nQuestion: " + ex["question"] + "\nAnswer: ")
        completion_tokens = enc(ex["answer"])

        nll = score_completion_with_context_docs(
            model,
            aux_token_lists=[aux_tokens],
            context_tokens=context_tokens,
            completion_tokens=completion_tokens,
            link_detector=link_detector,
            aux_raw_identifiers=[b_title],
            device=device,
        )

        if nll is not None:
            cross_doc_nlls.append(nll)
            all_nlls.append(nll)
            n_link_found += 1
            # Paired flat baseline: same example, same tokenization, doc_causal only.
            # Computed here so both scores share identical context_tokens /
            # completion_tokens — no re-running of link detection or filtering.
            flat_nll = score_completion(model, context_tokens, completion_tokens, device=device)
            paired_flat_nlls.append(flat_nll)
        else:
            flat_nll = score_completion(model, context_tokens, completion_tokens, device=device)
            all_nlls.append(flat_nll)
            n_link_not_found += 1

    def _ppl(nlls: List[float]) -> float:
        if not nlls:
            return float("nan")
        try:
            return _math.exp(sum(nlls) / len(nlls))
        except OverflowError:
            return float("inf")

    def _mean(nlls: List[float]) -> float:
        return sum(nlls) / len(nlls) if nlls else float("nan")

    return {
        "perplexity_cross_doc_only":    _ppl(cross_doc_nlls),
        "average_nll_cross_doc_only":   _mean(cross_doc_nlls),
        "perplexity_flat_linked_only":  _ppl(paired_flat_nlls),
        "average_nll_flat_linked_only": _mean(paired_flat_nlls),
        "n_cross_doc":                  len(cross_doc_nlls),
        "perplexity_with_fallback":     _ppl(all_nlls),
        "average_nll_with_fallback":    _mean(all_nlls),
        "total_examples":               len(all_nlls),
        "n_link_found":                 n_link_found,
        "n_link_not_found":             n_link_not_found,
        "n_skipped_no_corpus":          n_skipped_no_corpus,
        "n_skipped_no_link":            n_skipped_no_link,
    }


# ─── HumanEvalPack canonical-vs-buggy (2-way MC, no execution) ────────────────

#: Languages available in HumanEvalPack.
HUMANEVAL_LANGUAGES = tuple(l.value for l in HumanEvalLanguage)


def run_humaneval_buggy(
    model,
    language: str = "python",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on HumanEvalPack canonical-vs-buggy (2-way MC, no execution).

    The model picks whichever solution has lower NLL. 50% random baseline;
    even modest deltas above chance are meaningful.

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set.
        language: One of HUMANEVAL_LANGUAGES (default: ``'python'``).
        max_examples: Limit number of problems (None = full set, 164).
        cache_dir: HuggingFace cache directory.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}
    """
    if language not in HUMANEVAL_LANGUAGES:
        raise ValueError(
            f"language must be one of {HUMANEVAL_LANGUAGES}, got {language!r}"
        )
    _require_tokenizer(model, f"HumanEvalPack/{language}")
    enc = _make_encoder(model.tokenizer)

    class _Adapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                nlls = score_completions_batched(
                    model, enc(item.context),
                    [enc(c) for c in item.choices],
                    device=device,
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    dataset = HumanEvalBuggyDataset(
        language=HumanEvalLanguage(language),
        cache_dir=cache_dir,
        limit=max_examples,
    )
    return MultipleChoiceEvaluation(_Adapter()).run(dataset, batch_size=1, limit=max_examples)


# ─── Annotated benchmark driver ───────────────────────────────────────────────

#: Benchmarks supported by run_benchmark_annotated.
ANNOTATABLE_BENCHMARKS = frozenset({
    "hellaswag", "wiki_qa", "arc_easy", "arc_challenge", "lambada",
    "winogrande", "piqa", "boolq", "commonsense_qa", "copa",
    "openbookqa", "sciq", "hotpotqa",
})


def _load_benchmark_items(
    benchmark_name: str,
    enc,
    max_examples: Optional[int],
    cache_dir: Optional[str],
) -> List[Dict[str, Any]]:
    """Load items for a benchmark as plain token lists (no model calls).

    Returns a list of dicts with keys:
      - For MC benchmarks:
        {"type": "mc", "context_tokens": List[int],
         "completion_token_lists": List[List[int]], "label": int}
      - For fill-in-the-blank:
        {"type": "fitb", "context_tokens": List[int],
         "completion_tokens": List[int]}
      - For hotpotqa (no tunalab dataset class; loaded directly):
        same fitb structure

    Benchmarks that need bridge-only filtering (hotpotqa) handle it internally.
    """
    items: List[Dict[str, Any]] = []

    # ── multiple-choice benchmarks ──────────────────────────────────────
    _MC_CONFIGS = {
        "hellaswag": (
            "tunalab.data_sources.evaluations.multiple_choice.hellaswag.HellaSwagDataset",
            {"split": "val"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "wiki_qa": (
            "tunalab.data_sources.evaluations.multiple_choice.wiki_qa.WikiQADataset",
            {"split": "val"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "arc_easy": (
            "tunalab.data_sources.evaluations.multiple_choice.arc.ARCDataset",
            {"config": "easy", "split": "test"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "arc_challenge": (
            "tunalab.data_sources.evaluations.multiple_choice.arc.ARCDataset",
            {"config": "challenge", "split": "test"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "winogrande": (
            "tunalab.data_sources.evaluations.multiple_choice.winogrande.WinoGrandeDataset",
            {"config": "xl", "split": "val"},
            lambda c: enc(c),
            lambda ch: enc(ch),
        ),
        "piqa": (
            "tunalab.data_sources.evaluations.multiple_choice.piqa.PIQADataset",
            {"split": "val"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "boolq": (
            "tunalab.data_sources.evaluations.multiple_choice.boolq.BoolQDataset",
            {"split": "val"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "commonsense_qa": (
            "tunalab.data_sources.evaluations.multiple_choice.commonsense_qa.CommonsenseQADataset",
            {"split": "val"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "copa": (
            "tunalab.data_sources.evaluations.multiple_choice.copa.COPADataset",
            {"split": "val"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "openbookqa": (
            "tunalab.data_sources.evaluations.multiple_choice.openbookqa.OpenBookQADataset",
            {"split": "val"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
        "sciq": (
            "tunalab.data_sources.evaluations.multiple_choice.sciq.SciQDataset",
            {"split": "val"},
            lambda c: enc(c),
            lambda ch: enc(" " + ch),
        ),
    }

    if benchmark_name in _MC_CONFIGS:
        cls_path, extra_kwargs, ctx_enc, ch_enc = _MC_CONFIGS[benchmark_name]
        # Dynamically import the dataset class
        import importlib
        module_path, cls_name = cls_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError) as exc:
            raise ImportError(f"Could not load {cls_path}: {exc}") from exc

        # Map our kwargs to Enum values as the dataset class expects
        init_kwargs: Dict[str, Any] = {"cache_dir": cache_dir, "limit": max_examples}
        # split → Split enum
        if "split" in extra_kwargs:
            _split_map = {"val": "VAL", "test": "TEST"}
            split_str = _split_map.get(extra_kwargs["split"], extra_kwargs["split"].upper())
            split_mod = importlib.import_module(module_path)
            split_cls_name = "Split"
            if hasattr(split_mod, split_cls_name):
                try:
                    init_kwargs["split"] = getattr(split_mod, split_cls_name)[split_str]
                except KeyError:
                    init_kwargs["split"] = getattr(split_mod, split_cls_name).VAL
        # config → Config enum (arc)
        if "config" in extra_kwargs:
            cfg_mod = importlib.import_module(module_path)
            if hasattr(cfg_mod, "Config"):
                cfg_val = extra_kwargs["config"].upper()
                try:
                    init_kwargs["config"] = getattr(cfg_mod, "Config")[cfg_val]
                except KeyError:
                    init_kwargs["config"] = getattr(cfg_mod, "Config").CHALLENGE
            if hasattr(cfg_mod, "ARCConfig"):
                cfg_val = extra_kwargs["config"].upper()
                try:
                    init_kwargs["config"] = getattr(cfg_mod, "ARCConfig")[cfg_val]
                except KeyError:
                    pass
        # winogrande config
        if benchmark_name == "winogrande":
            wg_mod = importlib.import_module(module_path)
            if hasattr(wg_mod, "Config"):
                init_kwargs["config"] = wg_mod.Config.XL

        try:
            dataset = cls(**init_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to construct {cls_name}: {exc}") from exc

        lim = max_examples if max_examples is not None else len(dataset)
        for i in range(min(lim, len(dataset))):
            item = dataset[i]
            items.append({
                "type": "mc",
                "context_tokens": ctx_enc(item.context),
                "completion_token_lists": [ch_enc(ch) for ch in item.choices],
                "label": item.label,
            })
        return items

    # ── lambada (fill-in-the-blank) ─────────────────────────────────────
    if benchmark_name == "lambada":
        from tunalab.data_sources.evaluations.fill_in_the_blank.lambada import LambadaDataset
        dataset = LambadaDataset(cache_dir=cache_dir, limit=max_examples)
        lim = max_examples if max_examples is not None else len(dataset)
        for i in range(min(lim, len(dataset))):
            item = dataset[i]
            items.append({
                "type": "fitb",
                "context_tokens": enc(item.prompt),
                "completion_tokens": enc(" " + item.answer),
            })
        return items

    # ── hotpotqa (fill-in-the-blank, bridge only) ───────────────────────
    if benchmark_name == "hotpotqa":
        import math as _m
        corpus = _load_hotpotqa_corpus(cache_dir=cache_dir)
        examples = _hotpotqa_examples(max_examples, cache_dir, bridge_only=False)
        for ex in examples:
            a_title, b_title = _hotpotqa_titles(ex)
            if a_title is None:
                continue
            a_sents_raw = corpus.get(a_title.lower())
            b_sents_raw = corpus.get(b_title.lower())
            if a_sents_raw is None or b_sents_raw is None:
                continue
            a_ids = _hotpotqa_supporting_sent_ids(ex, a_title)
            b_ids = _hotpotqa_supporting_sent_ids(ex, b_title)

            def _pick_plain(sents, ids):
                picked = [_strip_html_links(sents[i]) for i in ids if i < len(sents)]
                return picked if picked else [_strip_html_links(sents[0])]

            a_text = " ".join(_pick_plain(a_sents_raw, a_ids))
            b_text = " ".join(_pick_plain(b_sents_raw, b_ids))
            context = a_text + " " + b_text + "\nQuestion: " + ex["question"] + "\nAnswer: "
            items.append({
                "type": "fitb",
                "context_tokens": enc(context),
                "completion_tokens": enc(ex["answer"]),
            })
        return items

    raise ValueError(
        f"Benchmark {benchmark_name!r} is not supported by run_benchmark_annotated. "
        f"Supported: {sorted(ANNOTATABLE_BENCHMARKS)}"
    )


def run_benchmark_annotated(
    model,
    benchmark_name: str,
    annotator: "MarkdownPromptAnnotator",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Run a benchmark under annotated cross-doc conditions.

    Injects a Wikipedia-style link into each example's context using the
    annotator, then scores under four link-probability thresholds and a flat
    baseline. The threshold values are calibrated from the distribution of
    max P('[') values observed across all examples in a cheap phase-1 scan.

    Only supports ANNOTATABLE_BENCHMARKS (benchmarks where the context is
    contiguous natural text that a markdown link can meaningfully augment).
    Requires a cross_doc_link model with MarkdownLinkDetector.

    Args:
        model: TS2TSModel in eval mode. Must have tokenizer set.
        benchmark_name: One of ANNOTATABLE_BENCHMARKS.
        annotator: MarkdownPromptAnnotator instance.
        max_examples: Cap on examples. None = full benchmark.
        cache_dir: Cache directory for dataset downloads.
        device: Device for tensor ops.

    Returns:
        Dict with keys:
          "baseline_flat"  — flat score on original context (no annotation)
          "t=0.0"          — all examples annotated
          "t=p25"          — 75% annotated (threshold = p25 of prob distribution)
          "t=p50"          — 50% annotated
          "t=p75"          — 25% annotated
          "threshold_values" — {"p25": float, "p50": float, "p75": float}
        Each threshold dict has either "accuracy" (MC) or "perplexity"/"average_nll"
        (fill-in-the-blank) plus "n_annotated", "n_link_fired", "total_examples".
    """
    if benchmark_name not in ANNOTATABLE_BENCHMARKS:
        raise ValueError(
            f"Benchmark {benchmark_name!r} not in ANNOTATABLE_BENCHMARKS. "
            f"Supported: {sorted(ANNOTATABLE_BENCHMARKS)}"
        )
    _require_tokenizer(model, f"run_benchmark_annotated/{benchmark_name}")
    enc = _make_encoder(model.tokenizer)

    logger.info(
        "run_benchmark_annotated: loading %s items for %s ...",
        max_examples or "all", benchmark_name,
    )
    items = _load_benchmark_items(benchmark_name, enc, max_examples, cache_dir)
    if not items:
        logger.warning("run_benchmark_annotated: no items loaded for %s", benchmark_name)
        return {}

    is_mc = items[0]["type"] == "mc"

    # ── Annotate all items — link_opener_prob used for threshold calibration ──
    # We annotate first (which includes the forward pass that would have been
    # phase-1's scan_prob), then derive percentile thresholds from the recorded
    # link_opener_prob values. This avoids a redundant forward pass per item.
    logger.info(
        "run_benchmark_annotated: annotating %d items ...", len(items)
    )
    annotated_cache: List[Optional[AnnotatedPrompt]] = []
    for item in items:
        try:
            ann = annotator.annotate(model, item["context_tokens"], device)
        except Exception as exc:
            logger.warning("Annotation failed: %s", exc)
            ann = None
        annotated_cache.append(ann)

    probs: List[float] = [
        ann.link_opener_prob if ann is not None else 0.0
        for ann in annotated_cache
    ]

    sorted_probs = sorted(probs)
    n = len(sorted_probs)
    # Threshold at the k-th percentile: items whose prob is >= this value are annotated.
    # p25 threshold → ~75% of items annotated (prob >= 25th-percentile value)
    # p50 threshold → ~50% of items annotated (prob >= 50th-percentile value)
    # p75 threshold → ~25% of items annotated (prob >= 75th-percentile value)
    p25 = sorted_probs[min(n - 1, int(0.25 * n))]
    p50 = sorted_probs[min(n - 1, int(0.50 * n))]
    p75 = sorted_probs[min(n - 1, int(0.75 * n))]
    threshold_values = {"p25": p25, "p50": p50, "p75": p75}
    threshold_specs = [
        ("t=0.0", 0.0),
        ("t=p25", p25),
        ("t=p50", p50),
        ("t=p75", p75),
    ]
    logger.info(
        "run_benchmark_annotated: thresholds p25=%.4f p50=%.4f p75=%.4f",
        p25, p50, p75,
    )

    # ── Scoring helpers ─────────────────────────────────────────────────
    def _score_item_flat(item: Dict[str, Any]) -> Tuple[float, Optional[int]]:
        """Score under doc_causal on original context. Returns (nll_or_acc_signal, pred_label)."""
        if is_mc:
            nlls = score_completions_batched(
                model, item["context_tokens"], item["completion_token_lists"], device=device,
            )
            pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
            return float(nlls[pred]), pred
        else:
            nll = score_completion(model, item["context_tokens"], item["completion_tokens"], device=device)
            return nll, None

    def _score_item_annotated(
        item: Dict[str, Any], ann: AnnotatedPrompt
    ) -> Tuple[float, Optional[int]]:
        """Score under cross-doc (or no-op link) using the annotated prompt."""
        if ann.link_fired and ann.aux_token_lists:
            # Cross-doc scoring
            if is_mc:
                nlls_cross = []
                for comp_toks in item["completion_token_lists"]:
                    nll = score_completion_with_context_docs(
                        model,
                        aux_token_lists=ann.aux_token_lists,
                        context_tokens=ann.context_tokens,
                        completion_tokens=comp_toks,
                        link_detector=model.link_detector,
                        aux_raw_identifiers=ann.aux_raw_identifiers,
                        device=device,
                    )
                    # Fall back to flat if cross-doc returns None for this choice
                    if nll is None:
                        nll = score_completion(
                            model, ann.context_tokens, comp_toks, device=device
                        )
                    nlls_cross.append(nll)
                pred = int(min(range(len(nlls_cross)), key=lambda i: nlls_cross[i]))
                return float(nlls_cross[pred]), pred
            else:
                nll = score_completion_with_context_docs(
                    model,
                    aux_token_lists=ann.aux_token_lists,
                    context_tokens=ann.context_tokens,
                    completion_tokens=item["completion_tokens"],
                    link_detector=model.link_detector,
                    aux_raw_identifiers=ann.aux_raw_identifiers,
                    device=device,
                )
                if nll is None:
                    nll = score_completion(
                        model, ann.context_tokens, item["completion_tokens"], device=device
                    )
                return nll, None
        else:
            # Link injected but no aux doc (no_op or corpus miss) — score on
            # annotated context_tokens under doc_causal
            if is_mc:
                nlls = score_completions_batched(
                    model, ann.context_tokens, item["completion_token_lists"], device=device,
                )
                pred = int(min(range(len(nlls)), key=lambda i: nlls[i]))
                return float(nlls[pred]), pred
            else:
                nll = score_completion(
                    model, ann.context_tokens, item["completion_tokens"], device=device
                )
                return nll, None

    def _aggregate(
        nll_or_signal_list: List[float],
        pred_labels: List[Optional[int]],
        true_labels: List[Optional[int]],
        n_annotated: int,
        n_link_fired: int,
    ) -> Dict[str, Any]:
        total = len(nll_or_signal_list)
        if is_mc:
            correct = sum(
                1 for p, t in zip(pred_labels, true_labels)
                if p is not None and t is not None and p == t
            )
            acc = correct / total if total > 0 else float("nan")
            try:
                from tunalab.stats_funcs import calculate_bootstrap_ci
                acc_vals = [1.0 if p == t else 0.0
                            for p, t in zip(pred_labels, true_labels)
                            if p is not None and t is not None]
                ci = calculate_bootstrap_ci(acc_vals) if acc_vals else (float("nan"), float("nan"))
            except Exception:
                ci = (float("nan"), float("nan"))
            return {
                "accuracy": acc,
                "accuracy_ci": list(ci),
                "total_examples": total,
                "n_annotated": n_annotated,
                "n_link_fired": n_link_fired,
            }
        else:
            mean_nll = float(_np_module.mean(nll_or_signal_list)) if nll_or_signal_list else float("nan")
            try:
                ppl = _math_module.exp(mean_nll)
            except OverflowError:
                ppl = float("inf")
            return {
                "perplexity": ppl,
                "average_nll": mean_nll,
                "total_examples": total,
                "n_annotated": n_annotated,
                "n_link_fired": n_link_fired,
            }

    # ── Baseline flat pass ──────────────────────────────────────────────
    logger.info("run_benchmark_annotated: scoring baseline_flat ...")
    flat_signals: List[float] = []
    flat_preds: List[Optional[int]] = []
    flat_labels: List[Optional[int]] = []
    for item in items:
        sig, pred = _score_item_flat(item)
        flat_signals.append(sig)
        flat_preds.append(pred)
        flat_labels.append(item.get("label"))

    results: Dict[str, Any] = {
        "baseline_flat": _aggregate(flat_signals, flat_preds, flat_labels, 0, 0),
        "threshold_values": threshold_values,
    }
    # Remove n_annotated/n_link_fired from baseline_flat (not meaningful there)
    results["baseline_flat"].pop("n_annotated", None)
    results["baseline_flat"].pop("n_link_fired", None)

    # ── Threshold passes ────────────────────────────────────────────────
    for t_label, threshold in threshold_specs:
        logger.info(
            "run_benchmark_annotated: scoring %s (threshold=%.4f) ...",
            t_label, threshold,
        )
        signals: List[float] = []
        preds: List[Optional[int]] = []
        labels: List[Optional[int]] = []
        n_annotated = 0
        n_link_fired = 0

        for item, prob, ann in zip(items, probs, annotated_cache):
            if prob >= threshold and ann is not None:
                sig, pred = _score_item_annotated(item, ann)
                n_annotated += 1
                if ann.link_fired:
                    n_link_fired += 1
            else:
                sig, pred = _score_item_flat(item)
            signals.append(sig)
            preds.append(pred)
            labels.append(item.get("label"))

        results[t_label] = _aggregate(signals, preds, labels, n_annotated, n_link_fired)

    return results
