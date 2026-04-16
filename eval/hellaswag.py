"""
eval/hellaswag.py — HellaSwag multiple-choice adapter for TS2TS.

Entry point:

  run_hellaswag(model, max_examples, cache_dir, device) -> Dict[str, Any]

Design notes:
  - Uses tunalab's MultipleChoiceEvaluation runner and HellaSwagDataset.
  - Scoring is done via score_completions_batched() from eval.scoring.
  - K choices per item are packed as K separate DocSpans into a single
    forward pass. With doc_causal masking each span is independent —
    equivalent to K separate runs but ~K× faster.
  - NullLayoutPolicy is used throughout (HellaSwag is out-of-distribution;
    no layout decoration applied).
  - All tunalab imports are deferred inside the function body so this module
    is importable even when the 'datasets' HuggingFace package is unavailable.
"""

from typing import Any, Dict, List, Optional


def run_hellaswag(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate on HellaSwag commonsense sentence completion (multiple choice).

    Args:
        model: TS2TSModel in eval mode. Must have model.tokenizer set (call
            to_inference_model(tokenizer=...) before evaluating).
        max_examples: Limit number of examples (None = full validation set,
            capped at 1024 by HellaSwagDataset).
        cache_dir: Directory to cache downloaded HellaSwag data. Defaults to
            data/.cache/hellaswag/ relative to the project root.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}

    Raises:
        ImportError: If tunalab NLP catalog is not installed.
        ValueError: If model.tokenizer is None.
    """
    try:
        from tunalab.evaluations.multiple_choice import (
            MultipleChoiceEvaluation,
            MultipleChoiceItem,
        )
        from tunalab.data_sources.evaluations.multiple_choice.hellaswag import (
            HellaSwagDataset,
            Split,
        )
        from tunalab.evaluation import register_handler
    except ImportError as e:
        raise ImportError(
            "tunalab NLP catalog must be installed for HellaSwag eval. "
            "Check that /fss/evin_t/tunalab/catalogs/nlp is on the Python path."
        ) from e

    if not hasattr(model, "tokenizer") or model.tokenizer is None:
        raise ValueError(
            "model.tokenizer is required for HellaSwag eval. "
            "Call to_inference_model(tokenizer=...) before evaluating."
        )

    from eval.scoring import score_completions_batched

    enc = model.tokenizer.encode

    class _HellaSwagAdapter:
        @register_handler("multiple_choice")
        def handle_batch(self_, batch: List[MultipleChoiceItem]) -> List[int]:
            predictions = []
            for item in batch:
                ctx_tokens   = enc(item.context)
                # GPT-2 convention: prepend a space before each choice.
                choice_lists = [enc(" " + c) for c in item.choices]
                nlls = score_completions_batched(
                    model, ctx_tokens, choice_lists, device=device
                )
                predictions.append(int(min(range(len(nlls)), key=lambda i: nlls[i])))
            return predictions

    adapter = _HellaSwagAdapter()
    runner  = MultipleChoiceEvaluation(adapter)
    dataset = HellaSwagDataset(split=Split.VAL, cache_dir=cache_dir, limit=max_examples)
    return runner.run(dataset, batch_size=1, limit=max_examples)
