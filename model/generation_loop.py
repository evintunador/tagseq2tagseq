"""
TS2TS autoregressive generation loop.

Stage 1: single-document generation only (no link detection or aux doc handling).
Link detection and multi-document logic are deferred to Stage 2.
"""
from typing import Callable, List, Optional

import torch

from model.document_context import DocumentContext, _DocEntry
from model.generation_config import GenerationConfig
from model.generation_result import GenerationResult
from model.sampling import sample_token


def run_generation(
    model,
    prompt_tokens: List[int],
    corpus,
    config: GenerationConfig,
    link_detector,
    tokenizer_decode: Optional[Callable],
    layout_policy,
) -> GenerationResult:
    """
    Run autoregressive generation and return a GenerationResult.

    Args:
        model: TS2TSModel with forward_inference implemented.
        prompt_tokens: Initial token IDs for the root document.
        corpus: Optional DocumentCorpus (unused in Stage 1).
        config: Generation configuration.
        link_detector: Link detector (unused in Stage 1).
        tokenizer_decode: Optional fn(List[int]) -> str for populating
            GeneratedDocument.text. If None, text fields are left as None.
        layout_policy: Layout policy for prefix/suffix tokens (None = no tokens).
    """
    context = DocumentContext(
        max_context_length=config.max_context_length,
        max_auxiliary_documents=config.max_auxiliary_documents,
        eviction_policy=config.eviction_policy,
        device=config.device,
    )
    root_entry = context.add_root(
        raw_identifier="",
        prompt_tokens=prompt_tokens,
        layout_policy=layout_policy,
    )

    _generate_doc(root_entry, context, model, config, layout_policy)

    docs = context.get_all_documents()
    if tokenizer_decode is not None:
        for doc in docs:
            if doc.tokens is not None:
                doc.text = tokenizer_decode(doc.tokens.tolist())

    return GenerationResult(
        root_document=docs[0],
        auxiliary_documents=docs[1:],
        generation_config=config.to_dict(),
    )


def _generate_doc(
    entry: _DocEntry,
    context: DocumentContext,
    model,
    config: GenerationConfig,
    layout_policy,
    depth: int = 0,
) -> None:
    """
    Generate tokens autoregressively for entry until a stopping condition is met.

    Stopping conditions (checked in order after each token):
        1. EOS token generated — document is complete.
        2. max_new_tokens reached — generation budget exhausted (truncated=False).
        3. max_tokens_per_document reached — hard structural limit (truncated=True).
    """
    tokens_generated = 0

    while not entry.done:
        tokens_tensor, doc_spans = context.build_sequence()
        logits = model.forward_inference(tokens_tensor, doc_spans)  # [1, T, V]
        next_token = sample_token(logits[0, -1, :], config.temperature, config.top_k, config.top_p)
        context.append_token(entry, next_token)
        tokens_generated += 1

        if next_token == config.eos_token_id:
            context.mark_done(entry, layout_policy)
        elif tokens_generated >= config.max_new_tokens:
            context.mark_done(entry, layout_policy)
        elif len(entry.tokens) >= config.max_tokens_per_document:
            entry.truncated = True
            context.mark_done(entry, layout_policy)
