"""
TS2TS autoregressive generation loop.
"""
import logging
from typing import Callable, List, Optional

import torch

from data.layout import DocLayoutInfo
from model.document_context import DocumentContext, _DocEntry
from model.generation_config import GenerationConfig
from model.generation_result import GeneratedDocument, GenerationResult, GenerationTrace
from model.identifier_utils import create_normed_identifier
from model.sampling import sample_token

logger = logging.getLogger(__name__)


def _layout_suffix_starts_with(layout_policy, entry: _DocEntry, token_id: int) -> bool:
    """True if the layout policy's suffix for this entry starts with token_id."""
    if layout_policy is None:
        return False
    info = DocLayoutInfo(
        raw_identifier=entry.raw_identifier,
        normed_identifier=entry.normed_identifier,
        body_tokens=list(entry.tokens),
    )
    suffix = layout_policy.suffix_tokens(info)
    return len(suffix) > 0 and suffix[0] == token_id


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

    trace = GenerationTrace() if config.record_trace else None

    if config.process_prompt_links and link_detector is not None:
        _process_existing_doc_links(
            root_entry, context, model, link_detector, corpus, config, layout_policy,
            depth=0, trace=trace,
        )

    _generate_doc(root_entry, context, model, link_detector, corpus, config, layout_policy,
                  depth=0, trace=trace)

    docs = context.get_all_documents()
    if tokenizer_decode is not None:
        for doc in docs:
            if doc.tokens is not None:
                doc.text = tokenizer_decode(doc.tokens.tolist())

    if trace is not None:
        trace.docs_evicted = context.eviction_count
        logger.info(
            "Generation complete: %d forward passes, %d tokens generated, "
            "%d docs generated, %d corpus fetches, %d links detected (%d resolved), "
            "%d docs evicted, max depth %d",
            trace.total_forward_passes, trace.total_tokens_generated,
            trace.docs_generated, trace.corpus_fetches,
            trace.links_detected, trace.links_resolved,
            trace.docs_evicted, trace.max_depth_reached,
        )

    return GenerationResult(
        root_document=docs[0],
        auxiliary_documents=docs[1:],
        generation_config=config.to_dict(),
        trace=trace,
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
    trace: Optional[GenerationTrace] = None,
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
        if trace is not None:
            trace.total_forward_passes += 1

        # Find the logit position corresponding to this entry's last token.
        # The entry may not be the last document in the packed sequence (e.g.
        # an aux doc inserted before root), so we look up its span end rather
        # than blindly using position -1.
        entry_end = None
        for span in doc_spans:
            if span.doc_id == entry.doc_id:
                entry_end = span.end
                break
        logit_pos = (entry_end - 1) if entry_end is not None else -1

        # Apply repetition penalty over tokens already in this document.
        # Positive logits are divided by the penalty; negative ones are multiplied,
        # so both directions push down the probability of repeated tokens.
        next_logits = logits[0, logit_pos, :].clone()
        if config.repetition_penalty != 1.0 and entry.tokens:
            for tid in set(entry.tokens):
                if next_logits[tid] > 0:
                    next_logits[tid] /= config.repetition_penalty
                else:
                    next_logits[tid] *= config.repetition_penalty

        next_token = sample_token(next_logits, config.temperature, config.top_k, config.top_p)

        # EOS handling: in training, the model predicts the layout suffix
        # token (e.g. EOS) as the next token after the last body token.
        # The suffix is NOT part of the body.  If we appended EOS to the
        # body AND mark_done added it again as suffix, the document would
        # contain a double EOS that never appeared during training.
        #
        # When the layout policy provides a suffix whose first token is
        # the EOS we just sampled, treat the generated EOS as the suffix
        # trigger — don't append it to the body.
        if next_token == config.eos_token_id and _layout_suffix_starts_with(
            layout_policy, entry, config.eos_token_id
        ):
            context.mark_done(entry, layout_policy)
            tokens_generated += 1
            if trace is not None:
                trace.total_tokens_generated += 1
            break

        context.append_token(entry, next_token)
        tokens_generated += 1
        if trace is not None:
            trace.total_tokens_generated += 1

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
                    if trace is not None:
                        trace.links_detected += 1
                    logger.debug(
                        "Link detected: '%s' at depth %d in doc '%s'",
                        link.target_str, depth, entry.raw_identifier,
                    )
                    _handle_link(
                        link, entry, context, model, link_detector,
                        corpus, config, layout_policy, depth, trace=trace,
                    )
                    break  # at most one new doc triggered per token step

        if next_token == config.eos_token_id:
            context.mark_done(entry, layout_policy)
        elif tokens_generated >= config.max_new_tokens:
            context.mark_done(entry, layout_policy)
        elif len(entry.prefix_tokens) + len(entry.tokens) >= config.max_tokens_per_document:
            # max_tokens_per_document: caps total document length (prefix + body).
            # Sets truncated=True — this is a hard structural limit.
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
    trace: Optional[GenerationTrace] = None,
) -> None:
    """
    Resolve a detected link by fetching or generating the target document.

    Decision tree (in order):
        1. Empty target → skip.
        2. depth >= max_link_depth → skip (enforces depth limit for all doc types).
        3. Already in active window → skip (cross-doc mask handles attention).
        4. Previously evicted → restore (potentially evicting another first).
        5. In corpus → fetch and insert before active_entry.
        6. allow_generation_fallback → recursively generate.
    """
    target = link.target_str
    if not target:
        return

    # Enforce max_link_depth for all doc types (corpus, generated, re-evicted).
    # depth=0 means the active doc is the root; new docs would be at depth+1.
    # With max_link_depth=0 this fires immediately, disabling all aux doc insertion.
    if depth >= config.max_link_depth:
        return

    if context.has_identifier(target):
        return

    # Re-eviction: restore a previously evicted doc if possible.
    evicted = context.find_evicted(target)
    if evicted is not None:
        exact_tokens = len(evicted.prefix_tokens) + len(evicted.tokens) + len(evicted.suffix_tokens)
        if config.eviction_policy == "drop_oldest":
            if not context.make_room(exact_tokens):
                return
        elif not context.can_add_document(exact_tokens):
            return
        context.restore_evicted(evicted, before_entry=active_entry)
        if trace is not None:
            trace.links_resolved += 1
            trace.max_depth_reached = max(trace.max_depth_reached, evicted.depth)
        logger.debug(
            "Re-eviction: restored '%s' (depth %d) before '%s'",
            target, evicted.depth, active_entry.raw_identifier,
        )
        if depth + 1 <= config.max_link_depth:
            _process_existing_doc_links(
                evicted, context, model, link_detector,
                corpus, config, layout_policy, depth + 1, trace=trace,
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
        if trace is not None:
            trace.links_resolved += 1
            trace.corpus_fetches += 1
            trace.max_depth_reached = max(trace.max_depth_reached, depth + 1)
        logger.debug(
            "Corpus fetch: '%s' (%d tokens) at depth %d",
            target, len(corpus_tokens), depth + 1,
        )
        if depth + 1 <= config.max_link_depth:
            _process_existing_doc_links(
                new_entry, context, model, link_detector,
                corpus, config, layout_policy, depth + 1, trace=trace,
            )
        return

    # Recursive generation fallback.
    if not config.allow_generation_fallback:
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
    if trace is not None:
        trace.links_resolved += 1
        trace.docs_generated += 1
        trace.max_depth_reached = max(trace.max_depth_reached, depth + 1)
    logger.debug("Generating aux doc: '%s' at depth %d", target, depth + 1)
    _generate_doc(
        new_entry, context, model, link_detector,
        corpus, config, layout_policy, depth + 1, trace=trace,
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
    trace: Optional[GenerationTrace] = None,
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
            corpus, config, layout_policy, depth, trace=trace,
        )
