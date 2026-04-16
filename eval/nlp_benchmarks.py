"""
eval/nlp_benchmarks.py — external NLP benchmark adapters for TS2TS.

Entry points:

  run_hellaswag(model, max_examples, cache_dir, device) -> Dict[str, Any]
  run_wiki_qa(model, max_examples, cache_dir, device)   -> Dict[str, Any]
  run_arc(model, config, max_examples, cache_dir, device) -> Dict[str, Any]
  run_lambada(model, max_examples, cache_dir, device)   -> Dict[str, Any]

All four adapters share the same design:
  - No layout decoration: external benchmarks are presented as raw text.
  - Scoring delegates to eval.scoring primitives (score_completions_batched
    for MC benchmarks, score_completion for LAMBADA).
  - Single-doc benchmarks — cross-doc grants can never fire, so
    forward_inference always uses backend='flex' regardless of the model's
    trained mask_type.

Multiple-choice benchmarks (hellaswag, wiki_qa, arc):
  - K choices per item packed as K DocSpans in one forward pass (~K× faster).
  - Prediction: choice with minimum mean NLL.
  - Returns: {"accuracy": float, "accuracy_ci": ..., "total_examples": int}

Fill-in-the-blank benchmark (lambada):
  - NLL-only scoring: adapter returns ("", nll) — no greedy decoding.
  - Primary metric is perplexity over the last word; exact_match_accuracy is
    always ~0.0 and should be ignored.
  - Returns: {"perplexity": float, "average_nll": float, "total_examples": int, ...}
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
    from tunalab.data_sources.evaluations.fill_in_the_blank.lambada import LambadaDataset
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
