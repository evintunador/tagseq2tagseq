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

Tunalab-backed benchmarks (hellaswag … codexglue_line_completion) use the
tunalab NLP catalog. The newer HF-direct benchmarks (mmlu, mathqa, math,
codexglue_code_to_text, repobench, humaneval_buggy) load via the `datasets`
library directly and do not require tunalab adapters.
"""

from typing import Any, Dict, List, Literal, Optional

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

from eval.scoring import score_completions_batched, score_completion


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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode
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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
    """Evaluate on RepoBench-C Python cross-file next-line completion.

    The ``cross_file_first`` split is the most scientifically interesting for
    cross_doc_link vs doc_causal comparison: cross-file context is directly
    analogous to auxiliary documents in cross_doc_link attention.

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
    enc = model.tokenizer.encode

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
    enc = model.tokenizer.encode

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
