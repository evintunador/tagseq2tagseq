"""
eval/scoring.py — primitive scoring utilities for TS2TS models.

Provides six entry points:

  score_doc(model, tokens, layout_policy, raw_identifier, normed_identifier, device)
      -> {"mean_nll": float, "num_tokens": int}

      Single forward pass over [prefix | body | suffix]. Only body tokens
      contribute to the NLL; prefix/suffix are included in the sequence so
      the model sees the same context it saw during training, but they are
      excluded from the loss computation. Used for held-out perplexity.

  score_docs_batched(model, docs, layout_policy, device, mask_type)
      -> List[{"mean_nll": float, "num_tokens": int}]

      Batched score_doc: packs multiple docs (each as a doc_causal-isolated
      DocSpan) into one forward per max_seq_len-sized pack. Per-doc results are
      identical to score_doc — a throughput win that removes the batch-1 forwards
      dominating held-out perplexity.

  score_completion(model, context_tokens, completion_tokens,
                   layout_policy, prompt_preprocessor)
      -> float  (mean NLL over completion tokens only)

      Scores how well the model predicts completion_tokens given
      context_tokens. Used for multiple-choice and fill-in-the-blank
      benchmarks. Always uses NullLayoutPolicy (external benchmarks are
      presented as raw text).

      prompt_preprocessor: Optional[Callable[[List[int]], List[int]]]
          Hook for the deferred link-injection eval feature — if provided,
          it is called on context_tokens before packing so a pre-processor
          can augment the context with generated aux-doc links.

  score_completions_batched(model, context_tokens, completion_token_lists, device)
      -> List[float]  (mean NLL per completion)

      Vectorised multiple-choice scoring: packs K (context + choice_k) sequences
      as K DocSpans into a single forward_inference call (~K× faster than K
      individual score_completion calls). doc_causal masking isolates each span
      so results are identical to calling score_completion K times.

  score_completion_with_context_docs(
      model, aux_token_lists, context_tokens, completion_tokens,
      link_detector, aux_raw_identifiers, device)
      -> Optional[float]  (mean NLL over completion tokens, or None)

      Cross-doc-link variant: packs aux snippets as earlier DocSpans and the
      primary (context+completion) doc last. Two modes: precise (pass
      aux_raw_identifiers so the detector matches each import to its specific
      snippet via path) or coarse (no identifiers: last import grants access
      to all aux spans). Returns None when no import is detected or no grants
      fire (caller decides: skip or fall back).

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


def score_docs_batched(
    model,
    docs: List[Any],
    layout_policy: Optional[DocLayoutPolicy] = None,
    device: str = "cuda",
    mask_type: Optional[str] = "doc_causal",
) -> List[Dict[str, float]]:
    """Score many documents with as few forward passes as possible.

    Batched equivalent of calling ``score_doc`` once per document: each doc's
    decorated ``[prefix | body | suffix]`` sequence is packed as one DocSpan, and
    multiple docs are concatenated into a single flat sequence (up to the model's
    ``max_seq_len``) scored in one ``forward_inference`` call. ``doc_causal``
    masking isolates each span, so every per-doc result is numerically identical
    to ``score_doc`` — this is purely a throughput optimisation that eliminates
    the per-doc batch-1 forward passes that dominate held-out perplexity.

    Args:
        model: TS2TSModel in eval mode.
        docs: List of documents, each a tuple/sequence
            ``(body_tokens, raw_identifier, normed_identifier)``. ``body_tokens``
            is the raw body (no prefix/suffix decoration), as passed to
            ``score_doc``.
        layout_policy: Layout policy for prefix/suffix decoration. Defaults to
            ``model.active_layout_policy``.
        device: Target device for token tensors.
        mask_type: Mask type passed to ``forward_inference``. Defaults to
            ``'doc_causal'`` (per-doc isolation — the only correct choice for
            scoring docs independently). Overridable for parity with
            ``score_doc``.

    Returns:
        A list of ``{"mean_nll": float, "num_tokens": int}`` dicts, one per input
        doc in the same order. Docs whose body cannot be scored (empty, or no
        target token after prefix handling) yield ``{"mean_nll": 0.0,
        "num_tokens": 0}`` — exactly as ``score_doc`` would.
    """
    n = len(docs)
    if n == 0:
        return []

    if layout_policy is None:
        layout_policy = model.active_layout_policy

    max_seq_len = getattr(getattr(model, "backbone", None), "max_seq_len", None)

    # ── Decorate each doc and precompute its body-scoring indices ─────────────
    # For each doc we build its full [prefix | body | suffix] sequence and record
    # the local logit/target index lists (relative to the doc's own start),
    # replicating score_doc's math exactly. Docs that produce no scoreable target
    # are marked and short-circuited to the zero result.
    prepared = []   # per-doc dict: seq, local_logit_idx, local_target_idx (or None)
    empty_result = {"mean_nll": 0.0, "num_tokens": 0}

    for doc in docs:
        body_tokens, raw_id, normed_id = doc[0], doc[1], doc[2]

        if not body_tokens:
            prepared.append(None)
            continue

        info = DocLayoutInfo(
            raw_identifier=raw_id,
            normed_identifier=normed_id,
            body_tokens=body_tokens,
        )
        prefix = layout_policy.prefix_tokens(info)
        suffix = layout_policy.suffix_tokens(info)
        prefix_len = len(prefix)
        suffix_len = len(suffix)

        body = body_tokens
        # Truncate body to fit within max_seq_len (head-first), matching score_doc.
        if isinstance(max_seq_len, int):
            max_body = max_seq_len - prefix_len - suffix_len
            if max_body <= 0:
                prepared.append(None)
                continue
            body = body[:max_body]

        if len(body) < 1:
            prepared.append(None)
            continue

        seq = list(prefix) + list(body) + list(suffix)

        # Body tokens occupy [prefix_len, prefix_len + len(body)) in seq.
        body_start = prefix_len
        body_end = prefix_len + len(body)
        logit_idx = list(range(body_start - 1, body_end - 1))
        target_idx = list(range(body_start, body_end))
        # If prefix_len == 0 the first body token has no preceding logit; skip it.
        if prefix_len == 0:
            logit_idx = logit_idx[1:]
            target_idx = target_idx[1:]

        if not logit_idx:
            prepared.append(None)
            continue

        prepared.append({
            "seq": seq,
            "logit_idx": logit_idx,
            "target_idx": target_idx,
            "normed_id": normed_id,
            "raw_id": raw_id,
        })

    # ── Bin-pack prepared docs into flat packs up to max_seq_len ──────────────
    # A single doc always fits (its body was truncated to the budget above), so
    # packs never exceed the limit. When no max_seq_len is known, pack greedily
    # without a length cap (one forward if it fits).
    budget = max_seq_len if isinstance(max_seq_len, int) else None
    packs: List[List[int]] = []   # each is a list of prepared-doc indices
    cur: List[int] = []
    cur_len = 0
    for i, p in enumerate(prepared):
        if p is None:
            continue
        seq_len = len(p["seq"])
        if budget is not None and cur and cur_len + seq_len > budget:
            packs.append(cur)
            cur = []
            cur_len = 0
        cur.append(i)
        cur_len += seq_len
    if cur:
        packs.append(cur)

    # ── One forward per pack; slice out each doc's NLL ────────────────────────
    results: List[Dict[str, float]] = [empty_result.copy() for _ in range(n)]

    for pack in packs:
        all_tokens: List[int] = []
        spans: List[DocSpan] = []
        offsets: List[int] = []
        for doc_pos, i in enumerate(pack):
            p = prepared[i]
            offset = len(all_tokens)
            offsets.append(offset)
            spans.append(DocSpan(
                doc_id=doc_pos,
                normed_identifier=p["normed_id"],
                raw_identifier=p["raw_id"],
                start=offset,
                end=offset + len(p["seq"]),
                truncated=False,
                outgoing_identifiers=[],
            ))
            all_tokens.extend(p["seq"])

        tokens_tensor = torch.tensor(
            all_tokens, dtype=torch.long, device=device
        ).unsqueeze(0)
        logits = model.forward_inference(tokens_tensor, spans, mask_type=mask_type)
        log_probs = F.log_softmax(logits[0].float(), dim=-1)   # [total_T, V]

        for doc_pos, i in enumerate(pack):
            p = prepared[i]
            offset = offsets[doc_pos]
            logit_indices = [offset + j for j in p["logit_idx"]]
            target_indices = [offset + j for j in p["target_idx"]]
            lp_slice = log_probs[logit_indices, :]                              # [N, V]
            tgt = tokens_tensor[0, target_indices]                             # [N]
            nll_per_tok = -lp_slice[torch.arange(len(tgt), device=device), tgt]  # [N]
            results[i] = {
                "mean_nll": nll_per_tok.mean().item(),
                "num_tokens": len(logit_indices),
            }

    return results


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
            before packing. Hook for the deferred link-injection eval feature
            (see NOTES.md): a preprocessor can augment the context with
            link-annotated aux-doc tokens before scoring.
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


def score_completion_with_context_docs(
    model,
    aux_token_lists: List[List[int]],
    context_tokens: List[int],
    completion_tokens: List[int],
    link_detector,
    aux_raw_identifiers: Optional[List[str]] = None,
    source_file_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Optional[float]:
    """Score a completion with cross-doc attention to auxiliary snippet documents.

    Packs aux snippets as earlier DocSpans (required by the DAG constraint:
    target docs must precede the link position) and the primary doc last.
    Runs link_detector on the full flat sequence to find import positions
    inside the primary doc, then builds link_to_target explicitly and passes
    it to forward_inference (bypassing CrossDocLinkMaskCreator's internal
    name-matching, which would re-detect on the full sequence anyway).

    Two matching modes:
      - Precise (aux_raw_identifiers provided): absolute imports matched via
        PythonImportDetector candidate paths; relative imports resolved using
        source_file_path when provided (eval-only — training pipeline unchanged).
        Each grant fires only for the specific snippet that import refers to.
      - Coarse (aux_raw_identifiers=None): last detected import in primary doc
        grants access to ALL aux spans simultaneously. Used when snippet paths
        are unavailable.

    Returns None when no grants can be established:
      - No non-empty aux snippets.
      - context_tokens is empty (no logit for first completion token).
      - No imports detected in the primary doc region.
      - Precise mode: no detected import matched any aux span.

    Args:
        model: TS2TSModel in eval mode with mask_type='cross_doc_link'.
        aux_token_lists: Pre-tokenized auxiliary snippet token lists, packed in
            order (aux_0 first). Empty lists are skipped.
        context_tokens: Primary-doc prefix token IDs.
        completion_tokens: Token IDs to score NLL over.
        link_detector: LinkDetector instance (e.g. PythonImportDetector).
        aux_raw_identifiers: Optional list of raw_identifier strings parallel
            to aux_token_lists (including empty-list entries). Format:
            "repo:path/to/file.py" — PythonImportDetector.index_doc_span
            strips the repo prefix. Enables precise per-import matching.
        source_file_path: Path of the primary doc within its repo (e.g.
            "pkg/subpkg/module.py"). Used to resolve relative imports
            (from . import X → pkg/subpkg/X.py) in precise mode. Ignored
            in coarse mode or when aux_raw_identifiers is None.
        device: Device string. If None, inferred from model.backbone.parameters().

    Returns:
        Mean NLL over completion_tokens as a float, or None.
    """
    if not completion_tokens:
        return 0.0

    if not context_tokens:
        return None

    if device is None:
        device = next(model.backbone.parameters()).device

    # Build flat sequence: non-empty aux docs first, primary doc last.
    all_tokens: List[int] = []
    aux_spans: List[DocSpan] = []
    offset = 0
    aux_idx = 0
    for i, aux_toks in enumerate(aux_token_lists):
        if not aux_toks:
            continue
        raw_id = (
            aux_raw_identifiers[i]
            if aux_raw_identifiers is not None and i < len(aux_raw_identifiers)
            else f"xfile_{aux_idx}"
        )
        aux_spans.append(DocSpan(
            doc_id=aux_idx,
            normed_identifier="",
            raw_identifier=raw_id,
            start=offset,
            end=offset + len(aux_toks),
            truncated=False,
            outgoing_identifiers=[],
        ))
        all_tokens.extend(aux_toks)
        offset += len(aux_toks)
        aux_idx += 1

    if not aux_spans:
        return None

    primary_start = offset
    primary_tokens = context_tokens + completion_tokens
    primary_doc_id = aux_idx
    primary_span = DocSpan(
        doc_id=primary_doc_id,
        normed_identifier="",
        raw_identifier="",
        start=primary_start,
        end=primary_start + len(primary_tokens),
        truncated=False,
        outgoing_identifiers=[],
    )
    all_tokens.extend(primary_tokens)
    all_spans = aux_spans + [primary_span]

    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long, device=device).unsqueeze(0)
    primary_end = primary_start + len(primary_tokens)

    # Detect links; keep only those whose link_end_pos falls within the primary doc.
    raw_links = link_detector.detect_links(tokens_tensor[0])
    primary_links = [lk for lk in raw_links if primary_start <= lk.link_end_pos <= primary_end]

    if not primary_links:
        return None

    if aux_raw_identifiers is not None:
        # Precise mode: match each detected import to its specific aux span.
        # Build a path → doc_id lookup using index_doc_span (strips repo prefix).
        path_to_doc_id: dict = {}
        for span in aux_spans:
            key = link_detector.index_doc_span(span)
            path_to_doc_id[key] = span.doc_id

        # Also handle relative imports using source_file_path.
        # The detector silently skips relative imports; we resolve them here
        # for eval purposes only (training pipeline unchanged).
        rel_grants: dict = {}  # link_end_pos → [doc_id, ...]
        if source_file_path:
            import re as _re
            # Decode the primary doc tokens to text for relative import parsing.
            primary_ids = tokens_tensor[0, primary_start:primary_end].tolist()
            try:
                primary_text = link_detector.decode_fn(primary_ids)
            except Exception:
                primary_text = ""
            src_parts = source_file_path.replace("\\", "/").split("/")
            for m in _re.finditer(
                r"^\s*from\s+(\.+)([\w.]*)\s+import\s+([^\n;(#]+)",
                primary_text, _re.MULTILINE,
            ):
                dots = len(m.group(1))
                rest = m.group(2).strip(".")
                names_str = m.group(3)
                names = [n.strip().split()[0] for n in names_str.split(",") if n.strip()]
                # Resolve base: go up `dots` dirs from source file's directory.
                base_parts = src_parts[:-dots] if dots <= len(src_parts) else []
                if rest:
                    base_parts = base_parts + rest.split(".")
                base = "/".join(base_parts)
                # Map char offset back to token position.
                char_end = m.end()
                cumulative = link_detector._build_char_to_token_index(primary_ids)
                link_end_pos = (primary_start +
                                link_detector._char_pos_to_token_pos(cumulative, char_end))
                for name in names if names else [""]:
                    candidates = (
                        [f"{base}/{name}.py", f"{base}/{name}/__init__.py", f"{base}.py"]
                        if name else [f"{base}.py", f"{base}/__init__.py"]
                    )
                    for cand in candidates:
                        if cand in path_to_doc_id:
                            rel_grants.setdefault(link_end_pos, [])
                            doc_id = path_to_doc_id[cand]
                            if doc_id not in rel_grants[link_end_pos]:
                                rel_grants[link_end_pos].append(doc_id)

        # Build link_to_target from absolute matches + relative matches.
        link_to_target: dict = {}
        for lk in primary_links:
            doc_id = path_to_doc_id.get(lk.target_str)
            if doc_id is not None:
                link_to_target.setdefault(lk.link_end_pos, [])
                if doc_id not in link_to_target[lk.link_end_pos]:
                    link_to_target[lk.link_end_pos].append(doc_id)
        for pos, doc_ids in rel_grants.items():
            link_to_target.setdefault(pos, [])
            for doc_id in doc_ids:
                if doc_id not in link_to_target[pos]:
                    link_to_target[pos].append(doc_id)

        if not link_to_target:
            return None
    else:
        # Coarse mode: last import grants access to all aux spans.
        last_link_end_pos = max(lk.link_end_pos for lk in primary_links)
        link_to_target = {last_link_end_pos: [span.doc_id for span in aux_spans]}

    logits = model.forward_inference(
        tokens_tensor, all_spans,
        mask_type='cross_doc_link',
        link_to_target=link_to_target,
    )  # [1, T, V]
    log_probs = F.log_softmax(logits[0].float(), dim=-1)  # [T, V]

    ctx_len = len(context_tokens)
    comp_len = len(completion_tokens)
    logit_start = primary_start + ctx_len - 1   # logit predicting completion_tokens[0]
    tgt_start   = primary_start + ctx_len

    lp  = log_probs[logit_start : logit_start + comp_len, :]  # [C, V]
    tgt = tokens_tensor[0, tgt_start : tgt_start + comp_len]  # [C]
    nll = -lp[torch.arange(comp_len, device=device), tgt].mean().item()
    return nll


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
