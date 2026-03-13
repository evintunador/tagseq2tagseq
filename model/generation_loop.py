"""
TS2TS autoregressive generation loop.
"""
from typing import Callable, List, Optional

import torch

from data.layout import DocLayoutInfo
from model.document_context import DocumentContext, _DocEntry
from model.generation_config import GenerationConfig
from model.generation_result import GenerationResult
from model.identifier_utils import create_normed_identifier
from model.sampling import sample_token


def run_generation(
    model,
    prompt_tokens: List[int],
    corpus,
    config: GenerationConfig,
    link_detector,
    tokenizer_decode: Optional[Callable],
    layout_policy,
    root_identifier: str = "",
) -> GenerationResult:
    """
    Run autoregressive generation and return a GenerationResult.

    Args:
        model: TS2TSModel with forward_inference implemented.
        prompt_tokens: Initial token IDs for the root document.
        corpus: Optional DocumentCorpus for link resolution.
        config: Generation configuration.
        link_detector: Link detector for cross-doc link detection.
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
        raw_identifier=root_identifier,
        prompt_tokens=prompt_tokens,
        layout_policy=layout_policy,
    )

    _generate_doc(root_entry, context, model, link_detector, corpus, config, layout_policy, depth=0)

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
    link_detector,
    corpus,
    config: GenerationConfig,
    layout_policy,
    depth: int = 0,
) -> None:
    """
    Generate tokens autoregressively for entry until a stopping condition is met.

    After each token, runs link detection on the last max_recent_link_tokens of
    the active document. If a complete link is detected ending at the just-appended
    token, _handle_link() is called to fetch or generate the aux document.

    Stopping conditions (checked in order after link handling):
        1. EOS token generated — document is complete.
        2. max_new_tokens reached — generation budget exhausted (truncated=False).
        3. max_tokens_per_document reached — hard structural limit (truncated=True).
    """
    tokens_generated = 0

    while not entry.done:
        tokens_tensor, doc_spans = context.build_sequence()
        logits = model.forward_inference(tokens_tensor, doc_spans)  # [1, T, V]

        # Apply repetition penalty over tokens already in this document.
        # Positive logits are divided by the penalty; negative ones are multiplied,
        # so both directions push down the probability of repeated tokens.
        next_logits = logits[0, -1, :].clone()
        if config.repetition_penalty != 1.0 and entry.tokens:
            for tid in set(entry.tokens):
                if next_logits[tid] > 0:
                    next_logits[tid] /= config.repetition_penalty
                else:
                    next_logits[tid] *= config.repetition_penalty

        next_token = sample_token(next_logits, config.temperature, config.top_k, config.top_p)
        context.append_token(entry, next_token)
        tokens_generated += 1

        # Link detection: scan last max_recent_link_tokens of the active doc.
        # link_end_pos is relative to the `recent` window, so firing when
        # link_end_pos == len(recent) means the link closed at the just-appended token.
        if link_detector is not None:
            recent = torch.tensor(
                entry.tokens[-config.max_recent_link_tokens:], dtype=torch.long
            )
            links = link_detector.detect_links(recent)
            for link in links:
                if link.link_end_pos == len(recent):
                    _handle_link(
                        link, entry, context, model, link_detector,
                        corpus, config, layout_policy, depth,
                    )
                    break  # at most one new doc triggered per token step

        if next_token == config.eos_token_id:
            context.mark_done(entry, layout_policy)
        elif tokens_generated >= config.max_new_tokens:
            context.mark_done(entry, layout_policy)
        elif len(entry.tokens) >= config.max_tokens_per_document:
            entry.truncated = True
            context.mark_done(entry, layout_policy)


def _handle_link(
    link,
    active_entry: _DocEntry,
    context: DocumentContext,
    model,
    link_detector,
    corpus,
    config: GenerationConfig,
    layout_policy,
    depth: int,
) -> None:
    """
    Resolve a detected link by fetching or generating the target document.

    Decision tree (in order):
        1. Empty target → skip.
        2. Already in active window → skip (cross-doc mask handles attention).
        3. Previously evicted → restore (not yet implemented; see find_evicted).
        4. In corpus → fetch and insert before active_entry.
        5. allow_generation_fallback and depth < max_link_depth → recursively generate.
    """
    target = link.target_str
    if not target:
        return

    if context.has_identifier(target):
        return

    # Re-eviction: find_evicted returns None until restore_evicted is implemented.
    evicted = context.find_evicted(target)
    if evicted is not None:
        if config.eviction_policy == "drop_oldest":
            if not context.make_room(len(evicted.tokens) + len(evicted.suffix_tokens)):
                return
        elif not context.can_add_document(len(evicted.tokens) + len(evicted.suffix_tokens)):
            return
        context.restore_evicted(evicted, before_entry=active_entry)
        if depth + 1 <= config.max_link_depth:
            _process_existing_doc_links(
                evicted, context, model, link_detector,
                corpus, config, layout_policy, depth + 1,
            )
        return

    # Corpus fetch.
    if corpus is not None and corpus.has_document(target):
        corpus_tokens = list(corpus.get_document(target))
        normed_target = create_normed_identifier(target)
        if layout_policy is not None:
            info = DocLayoutInfo(
                raw_identifier=target,
                normed_identifier=normed_target,
                body_tokens=corpus_tokens,
            )
            expected_tokens = (
                layout_policy.prefix_length(info)
                + len(corpus_tokens)
                + layout_policy.suffix_length(info)
            )
        else:
            expected_tokens = len(corpus_tokens)

        if config.eviction_policy == "drop_oldest":
            if not context.make_room(expected_tokens):
                return
        elif not context.can_add_document(expected_tokens):
            return

        new_entry = context.add_corpus_doc(
            raw_identifier=target,
            corpus_tokens=corpus_tokens,
            layout_policy=layout_policy,
            parent_raw_identifier=active_entry.raw_identifier,
            depth=depth + 1,
            before_entry=active_entry,
        )
        if depth + 1 <= config.max_link_depth:
            _process_existing_doc_links(
                new_entry, context, model, link_detector,
                corpus, config, layout_policy, depth + 1,
            )
        return

    # Recursive generation fallback.
    if not config.allow_generation_fallback:
        return
    if depth >= config.max_link_depth:
        return

    # Room estimate uses max_tokens_per_document as a conservative upper bound.
    # For non-null layout policies this under-estimates slightly (omits prefix/suffix),
    # but is acceptable since over-eviction is preferable to OOM.
    if config.eviction_policy == "drop_oldest":
        if not context.make_room(config.max_tokens_per_document):
            return
    elif not context.can_add_document(config.max_tokens_per_document):
        return

    new_entry = context.add_generated_doc(
        raw_identifier=target,
        layout_policy=layout_policy,
        parent_raw_identifier=active_entry.raw_identifier,
        depth=depth + 1,
        before_entry=active_entry,
    )
    _generate_doc(
        new_entry, context, model, link_detector,
        corpus, config, layout_policy, depth + 1,
    )


def _process_existing_doc_links(
    entry: _DocEntry,
    context: DocumentContext,
    model,
    link_detector,
    corpus,
    config: GenerationConfig,
    layout_policy,
    depth: int,
) -> None:
    """
    Scan a completed document for links and handle each one.

    Used after inserting a corpus doc (or a restored-evicted doc) to
    recursively pull in its dependencies at depth+1.
    """
    if link_detector is None:
        return
    all_links = link_detector.detect_links(
        torch.tensor(entry.tokens, dtype=torch.long)
    )
    for link in all_links:
        _handle_link(
            link, entry, context, model, link_detector,
            corpus, config, layout_policy, depth,
        )
