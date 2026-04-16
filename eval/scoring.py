"""
eval/scoring.py — primitive scoring utilities for TS2TS models.

Provides four entry points:

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

  score_completions_batched(model, context_tokens, completion_token_lists, device)
      -> List[float]  (mean NLL per completion)

      Vectorised multiple-choice scoring: packs K (context + choice_k) sequences
      as K DocSpans into a single forward_inference call (~K× faster than K
      individual score_completion calls). doc_causal masking isolates each span
      so results are identical to calling score_completion K times.

  score_doc_with_context(model, batch, layout_policy, device, mask_type)
      -> {"mean_nll": float, "num_tokens": int}

      Scores body tokens of docs that have incoming cross-doc edges within
      the pack (as yielded by BucketedPackDataset). Context-only docs are
      excluded — their NLL is identical under both mask conditions and only
      dilutes the contrastive signal. Used by pack_contrastive_perplexity.
"""

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.collate import DocSpan
from data.layout import DocLayoutInfo, DocLayoutPolicy, NullLayoutPolicy


def score_doc(
    model,
    tokens: List[int],
    layout_policy: Optional[DocLayoutPolicy] = None,
    raw_identifier: str = "",
    normed_identifier: str = "",
    device: str = "cuda",
    mask_type: Optional[str] = None,
) -> Dict[str, float]:
    """Score a single document under its layout policy.

    Args:
        model: TS2TSModel in eval mode (forward_inference is @no_grad).
        tokens: Raw body token IDs (no prefix/suffix decoration).
        layout_policy: The layout policy for prefix/suffix decoration.
            Defaults to model.active_layout_policy if not provided.
        raw_identifier: Human-readable document identifier (e.g. filename).
        normed_identifier: Normalised + hashed identifier used as the corpus key.
        device: Target device for the token tensor.
        mask_type: Optional mask type override passed to forward_inference.
            None uses the model's default. Pass 'doc_causal' to explicitly
            disable cross-doc attention (e.g. for baseline comparisons).

    Returns:
        {"mean_nll": float, "num_tokens": int}
        Returns {"mean_nll": 0.0, "num_tokens": 0} for empty bodies.
    """
    if not tokens:
        return {"mean_nll": 0.0, "num_tokens": 0}

    if layout_policy is None:
        layout_policy = model.active_layout_policy

    info = DocLayoutInfo(
        raw_identifier=raw_identifier,
        normed_identifier=normed_identifier,
        body_tokens=tokens,
    )
    prefix = layout_policy.prefix_tokens(info)
    suffix = layout_policy.suffix_tokens(info)
    prefix_len = len(prefix)
    suffix_len = len(suffix)

    # Truncate body to fit within the model's max_seq_len.
    max_seq_len = getattr(getattr(model, "backbone", None), "max_seq_len", None)
    if isinstance(max_seq_len, int):
        max_body = max_seq_len - prefix_len - suffix_len
        if max_body <= 0:
            return {"mean_nll": 0.0, "num_tokens": 0}
        tokens = tokens[:max_body]

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

    logits = model.forward_inference(tokens_tensor, [span], mask_type=mask_type)  # [1, T, V]
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

    logits = model.forward_inference(tokens_tensor, [span], mask_type='doc_causal')  # [1, T, V]
    log_probs = F.log_softmax(logits[0].float(), dim=-1)     # [T, V]

    # Logit at position (ctx_len - 1) predicts the first completion token.
    # Logit at position (ctx_len + i - 1) predicts completion token i.
    logit_start = ctx_len - 1          # inclusive; predicts completion_tokens[0]
    logit_end = ctx_len + len(completion_tokens) - 1  # exclusive

    lp_slice = log_probs[logit_start:logit_end, :]     # [C, V]
    tgt = tokens_tensor[0, ctx_len:ctx_len + len(completion_tokens)]  # [C]
    nll_per_tok = -lp_slice[torch.arange(len(tgt), device=device), tgt]  # [C]

    return nll_per_tok.mean().item()


def score_completions_batched(
    model,
    context_tokens: List[int],
    completion_token_lists: List[List[int]],
    device: Optional[str] = None,
) -> List[float]:
    """Score K completions against a shared context in a single forward pass.

    Packs [ctx + choice_0 | ctx + choice_1 | ... | ctx + choice_{K-1}] as K
    DocSpans into one forward_inference call. doc_causal masking isolates each
    span, so results are identical to calling score_completion K times but
    ~K× faster.

    Args:
        model: TS2TSModel in eval mode.
        context_tokens: Shared prompt token IDs.
        completion_token_lists: List of K completion token ID lists.
        device: Device string. If None, inferred from model.backbone.parameters().

    Returns:
        List of K floats — mean NLL over each completion's tokens.
        An empty completion yields 0.0.
    """
    K = len(completion_token_lists)
    if K == 0:
        return []

    if device is None:
        device = next(model.backbone.parameters()).device

    all_tokens: List[int] = []
    spans: List[DocSpan] = []
    ctx_lengths: List[int] = []
    choice_lengths: List[int] = []
    offset = 0

    for i, choice_toks in enumerate(completion_token_lists):
        seq = context_tokens + choice_toks
        spans.append(DocSpan(
            doc_id=i,
            normed_identifier="",
            raw_identifier="",
            start=offset,
            end=offset + len(seq),
            truncated=False,
            outgoing_identifiers=[],
        ))
        all_tokens.extend(seq)
        ctx_lengths.append(len(context_tokens))
        choice_lengths.append(len(choice_toks))
        offset += len(seq)

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device).unsqueeze(0)
    logits = model.forward_inference(tokens_tensor, spans, mask_type='doc_causal')  # [1, total_T, V]
    log_probs = F.log_softmax(logits[0].float(), dim=-1)  # [total_T, V]

    nlls: List[float] = []
    abs_offset = 0
    for i in range(K):
        ctx_len = ctx_lengths[i]
        choice_len = choice_lengths[i]

        if choice_len == 0:
            nlls.append(0.0)
            abs_offset += ctx_len
            continue

        # Logit at (abs_offset + ctx_len - 1) predicts the first completion token.
        logit_start = abs_offset + ctx_len - 1
        logit_end   = logit_start + choice_len
        tgt_start   = abs_offset + ctx_len
        tgt_end     = tgt_start + choice_len

        lp  = log_probs[logit_start:logit_end, :]                         # [C, V]
        tgt = tokens_tensor[0, tgt_start:tgt_end]                         # [C]
        nll = -lp[torch.arange(choice_len, device=device), tgt].mean().item()
        nlls.append(nll)
        abs_offset += ctx_len + choice_len

    return nlls


def score_doc_with_context(
    model,
    batch: Dict[str, Any],
    layout_policy: Optional[DocLayoutPolicy] = None,
    device: str = "cuda",
    mask_type: Optional[str] = None,
) -> Dict[str, float]:
    """Score body tokens of docs that have incoming cross-doc edges within the pack.

    Runs a single forward pass over the full packed sequence, then computes NLL
    only for spans whose normed_identifier appears in another span's
    outgoing_identifiers. Context-only docs (no incoming edges in the pack)
    are excluded — their NLL is identical under both mask conditions and only
    dilutes the contrastive signal.

    Args:
        model: TS2TSModel in eval mode.
        batch: Batch dict as yielded by BucketedPackDataset. Expected keys:
            ``"tokens"`` (LongTensor [1, T]) and ``"doc_spans"`` (List[DocSpan]).
        layout_policy: Layout policy used to determine per-span prefix/suffix
            lengths. Defaults to model.active_layout_policy if not provided.
        device: Device string (e.g. "cuda", "cpu").
        mask_type: Optional mask type override passed to forward_inference.
            None uses the model's default. Pass 'doc_causal' for the
            baseline condition in contrastive evaluation.

    Returns:
        {"mean_nll": float, "num_tokens": int}
        Returns {"mean_nll": 0.0, "num_tokens": 0} if the pack has no
        cross-doc edges (no target docs to score) or if doc_spans is empty.
    """
    if layout_policy is None:
        layout_policy = model.active_layout_policy

    tokens_tensor = batch["tokens"].to(device)   # [1, T]
    doc_spans = batch["doc_spans"]

    if not doc_spans:
        return {"mean_nll": 0.0, "num_tokens": 0}

    # Identify target docs: spans whose normed_identifier is referenced by
    # at least one other span's outgoing_identifiers.
    pack_normed_ids = {span.normed_identifier for span in doc_spans}
    target_ids: set = set()
    for span in doc_spans:
        for oid in span.outgoing_identifiers:
            if oid in pack_normed_ids:
                target_ids.add(oid)

    if not target_ids:
        return {"mean_nll": 0.0, "num_tokens": 0}

    # Truncate to model's max_seq_len if the pack is longer.
    # In production the pack token budget equals max_seq_len; truncation is
    # only triggered when a smoke-test model with a short context scores packs
    # built for a longer context.
    max_seq_len = getattr(getattr(model, "backbone", None), "max_seq_len", None)
    if isinstance(max_seq_len, int) and tokens_tensor.shape[1] > max_seq_len:
        from data.collate import DocSpan as _DocSpan
        tokens_tensor = tokens_tensor[:, :max_seq_len]
        clipped = []
        for s in doc_spans:
            if s.start >= max_seq_len:
                continue
            if s.end > max_seq_len:
                s = _DocSpan(
                    doc_id=s.doc_id, normed_identifier=s.normed_identifier,
                    raw_identifier=s.raw_identifier,
                    start=s.start, end=max_seq_len,
                    truncated=True, outgoing_identifiers=s.outgoing_identifiers,
                )
            clipped.append(s)
        doc_spans = clipped

    if not doc_spans:
        return {"mean_nll": 0.0, "num_tokens": 0}

    # Single forward pass over the full packed sequence.
    logits = model.forward_inference(tokens_tensor, doc_spans, mask_type=mask_type)  # [1, T, V]
    log_probs = F.log_softmax(logits[0].float(), dim=-1)  # [T, V]

    nll_list: List[float] = []

    for span in doc_spans:
        # Only score target docs — those with at least one incoming pack edge.
        if span.normed_identifier not in target_ids:
            continue

        info = DocLayoutInfo(
            raw_identifier=span.raw_identifier,
            normed_identifier=span.normed_identifier,
        )
        prefix_len = layout_policy.prefix_length(info)
        suffix_len = layout_policy.suffix_length(info)

        body_start = span.start + prefix_len
        body_end = span.end - suffix_len

        if body_end <= body_start:
            continue

        # Logit at position t predicts token t+1. Body token at body_start is
        # predicted by logit at body_start - 1.
        logit_indices = list(range(body_start - 1, body_end - 1))
        target_indices = list(range(body_start, body_end))

        # The very first position in the full sequence has no preceding logit.
        # This only occurs for the first span when prefix_len == 0.
        if body_start == 0:
            logit_indices = logit_indices[1:]
            target_indices = target_indices[1:]

        if not logit_indices:
            continue

        tgt = tokens_tensor[0, target_indices]                                    # [N]
        lp_slice = log_probs[logit_indices, :]                                    # [N, V]
        nll_per_tok = -lp_slice[torch.arange(len(tgt), device=device), tgt]      # [N]
        nll_list.extend(nll_per_tok.tolist())

    if not nll_list:
        return {"mean_nll": 0.0, "num_tokens": 0}

    return {"mean_nll": float(np.mean(nll_list)), "num_tokens": len(nll_list)}
