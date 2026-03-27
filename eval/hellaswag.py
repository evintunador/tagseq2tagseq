"""
eval/hellaswag.py — HellaSwag multiple-choice adapter for TS2TS.

STUB: raises NotImplementedError until NL datasets (Wikipedia/fineweb) come
online. The full implementation is sketched in comments below for when it
needs to be activated.

Entry point:

  run_hellaswag(model, max_examples, cache_dir, device) -> Dict[str, float]

Design notes for the full implementation:
  - Uses tunalab's MultipleChoiceEvaluation runner and HellaSwagDataset.
  - Scoring is done via score_completion() from eval.scoring.
  - For efficiency, K choices per item are packed as K separate DocSpans into
    a single [1, K*N, V] forward pass. With doc_causal masking each span is
    independent — equivalent to K separate runs but ~4x faster.
  - NullLayoutPolicy is used throughout (HellaSwag is out-of-distribution;
    no layout decoration applied).
"""

from typing import Dict, Optional


def run_hellaswag(
    model,
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """Evaluate on HellaSwag commonsense sentence completion (multiple choice).

    Deferred until NL datasets are available. Activate by removing the
    NotImplementedError and uncommenting the implementation below.

    Args:
        model: TS2TSModel in eval mode.
        max_examples: Limit number of examples (None = full validation set).
        cache_dir: Directory to cache downloaded HellaSwag data.
        device: Device for token tensors.

    Returns:
        {"accuracy": float, "accuracy_ci": (float, float), "total_examples": int}

    Raises:
        NotImplementedError: Always, until NL datasets are online.
    """
    raise NotImplementedError(
        "HellaSwag eval deferred: NL datasets (Wikipedia / fineweb) not yet "
        "online. Remove this raise and uncomment the implementation in "
        "eval/hellaswag.py when ready."
    )

    # ── Full implementation (activate when NL data is available) ──────────────
    #
    # try:
    #     from tunalab.evaluation import register_handler
    #     from tunalab.evaluations.multiple_choice import (
    #         MultipleChoiceEvaluation,
    #         MultipleChoiceItem,
    #     )
    #     from tunalab.data_sources.evaluations.multiple_choice.hellaswag import (
    #         HellaSwagDataset,
    #         Split,
    #     )
    # except ImportError as e:
    #     raise ImportError(
    #         "tunalab NLP catalog must be installed for HellaSwag eval."
    #     ) from e
    #
    # import torch
    # import torch.nn.functional as F
    # from data.collate import DocSpan
    # from eval.scoring import score_completion
    #
    # class _HellaSwagAdapter:
    #     """Thin wrapper that exposes a @register_handler for MultipleChoiceEvaluation."""
    #
    #     def __init__(self, model, device):
    #         self._model = model
    #         self._device = device
    #
    #     @register_handler("multiple_choice")
    #     def handle_batch(self, batch):
    #         """Score each MultipleChoiceItem; return list of predicted choice indices."""
    #         predictions = []
    #         for item in batch:
    #             pred = _score_mc_item(self._model, item, self._device)
    #             predictions.append(pred)
    #         return predictions
    #
    # def _score_mc_item(model, item: MultipleChoiceItem, device: str) -> int:
    #     """Return the index of the choice with the lowest NLL.
    #
    #     Efficient implementation: pack all K choices as K separate DocSpans
    #     into one forward pass. doc_causal mask ensures independence between
    #     spans (equivalent to K separate runs, ~K× faster).
    #     """
    #     enc = model.tokenizer.encode
    #     ctx_tokens = enc(item.context)
    #
    #     # Tokenize each choice (GPT-2 convention: prepend space)
    #     choice_token_lists = [enc(" " + c) for c in item.choices]
    #     K = len(choice_token_lists)
    #
    #     # ── Packed single-pass scoring ──────────────────────────────────────
    #     # Concatenate: [ctx + choice_0 | ctx + choice_1 | ... | ctx + choice_K-1]
    #     # Build one DocSpan per option so doc_causal attention isolates them.
    #
    #     all_tokens = []
    #     spans = []
    #     ctx_lengths = []
    #     choice_lengths = []
    #     offset = 0
    #
    #     for i, choice_toks in enumerate(choice_token_lists):
    #         seq = ctx_tokens + choice_toks
    #         spans.append(DocSpan(
    #             doc_id=i,
    #             normed_identifier="",
    #             start=offset,
    #             end=offset + len(seq),
    #             truncated=False,
    #             outgoing_identifiers=[],
    #             raw_identifier="",
    #         ))
    #         all_tokens.extend(seq)
    #         ctx_lengths.append(len(ctx_tokens))
    #         choice_lengths.append(len(choice_toks))
    #         offset += len(seq)
    #
    #     tokens_tensor = torch.tensor(
    #         all_tokens, dtype=torch.long, device=device
    #     ).unsqueeze(0)  # [1, total_T]
    #
    #     logits = model.forward_inference(tokens_tensor, spans)   # [1, total_T, V]
    #     log_probs = F.log_softmax(logits[0].float(), dim=-1)     # [total_T, V]
    #
    #     nlls = []
    #     abs_offset = 0
    #     for i in range(K):
    #         ctx_len = ctx_lengths[i]
    #         choice_len = choice_lengths[i]
    #         # Logit at (abs_offset + ctx_len - 1) predicts choice token 0
    #         logit_start = abs_offset + ctx_len - 1
    #         logit_end = logit_start + choice_len
    #         tgt_start = abs_offset + ctx_len
    #         tgt_end = tgt_start + choice_len
    #
    #         lp = log_probs[logit_start:logit_end]                  # [C, V]
    #         tgt = tokens_tensor[0, tgt_start:tgt_end]              # [C]
    #         nll = -lp[torch.arange(choice_len, device=device), tgt].mean().item()
    #         nlls.append(nll)
    #         abs_offset += ctx_lengths[i] + choice_lengths[i]
    #
    #     return int(min(range(K), key=lambda i: nlls[i]))
    #
    # dataset = HellaSwagDataset(split=Split.VAL, cache_dir=cache_dir, limit=max_examples)
    # adapter = _HellaSwagAdapter(model, device)
    # runner = MultipleChoiceEvaluation(adapter)
    # return runner.run(dataset, batch_size=1, limit=max_examples)
