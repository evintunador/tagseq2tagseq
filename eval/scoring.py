"""
eval/scoring.py — primitive scoring utilities for TS2TS models.

Provides two entry points:

  score_doc(model, tokens, layout_policy, raw_identifier, normed_identifier, device)
      -> {"mean_nll": float, "num_tokens": int}

      Single forward pass over [prefix | body | suffix]. Only body tokens
      contribute to the NLL; prefix/suffix are included in the sequence so
      the model sees the same context it saw during training, but they are
      excluded from the loss computation. Used for held-out perplexity.

  score_completion(model, context_tokens, completion_tokens,
                   layout_policy, prompt_preprocessor)
      -> float  (mean NLL over completion tokens only)

      Scores how well the model predicts completion_tokens given
      context_tokens. Used for multiple-choice and fill-in-the-blank
      benchmarks. Always uses NullLayoutPolicy (external benchmarks are
      presented as raw text).

      prompt_preprocessor: Optional[Callable[[List[int]], List[int]]]
          Hook for the future link-injection feature — if provided, it is
          called on context_tokens before packing, allowing a pre-processor
          to annotate the context with generated aux-doc links.
"""

import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from data.collate import DocSpan
from data.layout import DocLayoutInfo, DocLayoutPolicy, NullLayoutPolicy


def score_doc(
    model,
    tokens: List[int],
    layout_policy: DocLayoutPolicy,
    raw_identifier: str = "",
    normed_identifier: str = "",
    device: str = "cuda",
) -> Dict[str, float]:
    """Score a single document under its training layout policy.

    Args:
        model: TS2TSModel in eval mode (forward_inference is @no_grad).
        tokens: Raw body token IDs (no prefix/suffix decoration).
        layout_policy: The layout policy used during training for this
            checkpoint. Prefix/suffix are prepended/appended to the body
            before the forward pass but are excluded from the NLL.
        raw_identifier: Human-readable document identifier (e.g. filename).
        normed_identifier: Normalised + hashed identifier used as the corpus key.
        device: Target device for the token tensor.

    Returns:
        {"mean_nll": float, "num_tokens": int}
        Returns {"mean_nll": 0.0, "num_tokens": 0} for empty bodies.
    """
    if not tokens:
        return {"mean_nll": 0.0, "num_tokens": 0}

    info = DocLayoutInfo(
        raw_identifier=raw_identifier,
        normed_identifier=normed_identifier,
        body_tokens=tokens,
    )
    prefix = layout_policy.prefix_tokens(info)
    suffix = layout_policy.suffix_tokens(info)
    prefix_len = len(prefix)
    suffix_len = len(suffix)

    full_seq = prefix + tokens + suffix
    T = len(full_seq)

    # Body must have at least one token for there to be a prediction target.
    if len(tokens) < 1:
        return {"mean_nll": 0.0, "num_tokens": 0}

    tokens_tensor = torch.tensor(full_seq, dtype=torch.long, device=device).unsqueeze(0)

    span = DocSpan(
        doc_id=0,
        normed_identifier=normed_identifier,
        start=0,
        end=T,
        truncated=False,
        outgoing_identifiers=[],
        raw_identifier=raw_identifier,
    )

    # forward_inference is decorated with @torch.no_grad() — no extra context needed.
    logits = model.forward_inference(tokens_tensor, [span])  # [1, T, V]
    log_probs = F.log_softmax(logits[0].float(), dim=-1)     # [T, V]

    # Logit at position t predicts token t+1.
    # Body tokens occupy positions [prefix_len, prefix_len + len(tokens)).
    # To score body token at index i (absolute position prefix_len + i),
    # we read logit from position (prefix_len + i - 1) and target token at
    # position (prefix_len + i).
    body_start = prefix_len          # first body token index in full_seq
    body_end = prefix_len + len(tokens)  # exclusive

    logit_indices = list(range(body_start - 1, body_end - 1))  # length = len(tokens)
    target_indices = list(range(body_start, body_end))         # length = len(tokens)

    # If prefix_len == 0 the first body token has no preceding logit; skip it.
    if prefix_len == 0:
        logit_indices = logit_indices[1:]
        target_indices = target_indices[1:]

    if not logit_indices:
        return {"mean_nll": 0.0, "num_tokens": 0}

    lp_slice = log_probs[logit_indices, :]                            # [N, V]
    tgt = tokens_tensor[0, target_indices]                            # [N]
    nll_per_tok = -lp_slice[torch.arange(len(tgt), device=device), tgt]  # [N]
    mean_nll = nll_per_tok.mean().item()

    return {"mean_nll": mean_nll, "num_tokens": len(logit_indices)}


def score_completion(
    model,
    context_tokens: List[int],
    completion_tokens: List[int],
    layout_policy: Optional[DocLayoutPolicy] = None,
    prompt_preprocessor: Optional[Callable[[List[int]], List[int]]] = None,
    device: Optional[str] = None,
) -> float:
    """Score the NLL of completion_tokens given context_tokens.

    Used for multiple-choice and fill-in-the-blank benchmarks where we want
    the mean NLL over only the completion region.

    Args:
        model: TS2TSModel in eval mode.
        context_tokens: Prompt token IDs.
        completion_tokens: Continuation token IDs to score.
        layout_policy: Ignored (always uses NullLayoutPolicy — external
            benchmarks present bare text without layout decoration).
            Parameter kept for potential future use.
        prompt_preprocessor: Optional callable applied to context_tokens
            before packing. This is the hook for the deferred link-injection
            feature: a preprocessor can annotate the context with links and
            insert aux-doc tokens before scoring.
        device: Device string (e.g. "cuda", "cpu"). If None, inferred from
            model.backbone.parameters().

    Returns:
        Mean NLL over completion tokens as a float.
    """
    if prompt_preprocessor is not None:
        context_tokens = prompt_preprocessor(context_tokens)

    if not completion_tokens:
        return 0.0

    full_seq = context_tokens + completion_tokens
    T = len(full_seq)
    ctx_len = len(context_tokens)

    if device is None:
        device = next(model.backbone.parameters()).device
    tokens_tensor = torch.tensor(full_seq, dtype=torch.long, device=device).unsqueeze(0)

    span = DocSpan(
        doc_id=0,
        normed_identifier="",
        start=0,
        end=T,
        truncated=False,
        outgoing_identifiers=[],
        raw_identifier="",
    )

    logits = model.forward_inference(tokens_tensor, [span])  # [1, T, V]
    log_probs = F.log_softmax(logits[0].float(), dim=-1)     # [T, V]

    # Logit at position (ctx_len - 1) predicts the first completion token.
    # Logit at position (ctx_len + i - 1) predicts completion token i.
    logit_start = ctx_len - 1          # inclusive; predicts completion_tokens[0]
    logit_end = ctx_len + len(completion_tokens) - 1  # exclusive

    lp_slice = log_probs[logit_start:logit_end, :]     # [C, V]
    tgt = tokens_tensor[0, ctx_len:ctx_len + len(completion_tokens)]  # [C]
    nll_per_tok = -lp_slice[torch.arange(len(tgt), device=device), tgt]  # [C]

    return nll_per_tok.mean().item()
