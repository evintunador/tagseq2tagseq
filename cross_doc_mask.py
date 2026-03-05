"""
Cross-Document Link Attention Mask

This module implements a custom attention mask that allows tokens to attend to
previously-linked documents in the batch. When a document contains a link to
another document (e.g. markdown [text](target), a Python import, a LaTeX \\cite),
all tokens appearing AFTER that link can attend to the target document if it
appears earlier in the batch.

Key Features:
- Link-aware attention: tokens after links gain access to linked documents
- Cumulative access: multiple links accumulate, granting access to multiple docs
- Causal by default: maintains causality within and across documents
- DAG-aware: only works for forward references (earlier docs in batch)
- Pluggable link detection: different datasets use different LinkDetector impls
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from typing import List, Any, Optional, NamedTuple, Callable, Dict, Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


class LinkInfo(NamedTuple):
    """Metadata about a detected link in the token sequence."""
    link_end_pos: int   # Token position just after the link's closing delimiter;
                        # attention to the target is granted from this position onward.
    target_str: str     # Decoded target identifier string (matched against DocSpan.clean_title)


@runtime_checkable
class LinkDetector(Protocol):
    """
    Protocol for dataset-specific link detection in packed token sequences.

    Implementations scan a 1D input_ids tensor and return structured information
    about each link they find. Different datasets need different implementations:
    - Wikipedia / Markdown:  [text](target) syntax          → MarkdownLinkDetector
    - Python / TheStack:     import statements              → PythonImportDetector
    - LaTeX / ArXiv:         \\cite{key} or \\ref{label}   → not yet implemented (see below)
    - Other languages:       Ruby require, JS/TS import,
                             Rust use, etc.                 → not yet implemented (see below)

    # TODO(@jamesljr): a LaTeX detector will be needed for the ArXiv dataset.
    # The exact abstraction is still TBD — this is a rough sketch, not a
    # confident design.  LaTeX cross-references can take many forms
    # (\\cite, \\citep, \\citet, \\ref, \\hyperref, \\input, \\include, and
    # others depending on the document class and packages used), and it is not
    # yet clear which of these should create attention links, what target_str
    # should look like, or how index_doc_span should map ArXiv identifiers.
    # Requires deeper understanding of the ArXiv graph structure before
    # committing to an implementation.

    # TODO(@jamesljr): additional programming languages (Ruby require, JS/TS
    # import, Rust use, Go import, etc.) will each need their own detector if
    # those datasets are added.  The right abstraction here is uncertain — the
    # examples below are illustrative guesses, not a confident design.  Open
    # questions include: one detector per language vs. a dispatch wrapper,
    # whether to identify language from the file extension in clean_title or
    # store it as metadata, and how import semantics differ enough across
    # languages to require fundamentally different logic vs. just different
    # regex patterns.  Revisit once a second code dataset is being added.

    The detector is responsible for all decoding; it returns already-decoded
    target strings rather than token spans, so CrossDocLinkMaskCreator has no
    dependency on a tokenizer.
    """

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """
        Detect links in a 1D token sequence.

        Args:
            input_ids: 1D tensor of token IDs, shape [seq_len]

        Returns:
            List of LinkInfo objects. Each describes the position in the sequence
            where attention to the target begins (link_end_pos) and the already-
            decoded target identifier string (target_str).
        """
        ...

    def index_doc_span(self, span: Any) -> str:
        """
        Return the lookup key for a DocSpan when building the target-matching index.

        Defaults to ``span.clean_title`` (exact match).  Detectors for datasets
        whose ``target_str`` is only a sub-component of ``clean_title`` (e.g.
        ``PythonImportDetector`` returns a bare file path while ``clean_title``
        includes a repo prefix) should override this to return the matching
        sub-component.
        """
        return span.clean_title


class MarkdownLinkDetector:
    """
    Detects markdown-style [text](target) links in GPT-2 tokenized sequences.

    Uses token-ID matching to locate the ]( delimiter, searches backwards for [,
    then decodes a growing window of tokens forward until ) appears in the decoded
    text.  This avoids relying on a single fixed token ID for ) — a bare ) shares
    a token with many other contexts — and makes the approach robust to BPE splits
    inside the target title.

    Args:
        decode_fn:             Function mapping List[int] -> str (e.g. tiktoken enc.decode).
        window_tokens:         Maximum tokens to scan forward when looking for the
                               closing ).  Default 50 is generous for Wikipedia titles.
        link_start_token_ids:  Token IDs that can represent '['.
                               GPT-2 defaults: 58 ('['), 685 (' [').
        link_mid_token_id:     Token ID for the '](' bigram.
                               GPT-2 default: 16151.
    """

    def __init__(
        self,
        decode_fn: Callable[[List[int]], str],
        window_tokens: int = 50,
        link_start_token_ids: Optional[List[int]] = None,
        link_mid_token_id: int = 16151,   # '](' in GPT-2
    ):
        if link_start_token_ids is None:
            link_start_token_ids = [58, 685]   # '[' and ' [' in GPT-2
        self.decode_fn = decode_fn
        self.window_tokens = window_tokens
        self.link_start_token_ids = set(link_start_token_ids)
        self.link_mid_token_id = link_mid_token_id

    def index_doc_span(self, span: Any) -> str:
        """Match against ``clean_title`` normalized to Wikipedia link format.

        Wikipedia internal links use underscores for spaces, e.g.
        ``[text](Sunshine_Coast,_Queensland)``.  We replace spaces with
        underscores so the key matches what ``detect_links`` returns.
        """
        return span.clean_title.replace(' ', '_')

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """
        Detect [text](target) links.

        Scans for ]( tokens, searches backwards for the opening [, then decodes
        a growing window forward until ) is found in the accumulated decoded text.
        The target string is taken as everything before the first ) in that window.
        """
        links = []
        seq_len = input_ids.shape[0]

        mid_positions = (input_ids == self.link_mid_token_id).nonzero(as_tuple=True)[0]

        for mid_pos_t in mid_positions:
            mid_pos = mid_pos_t.item()

            # Search backwards for the opening [ (up to 100 tokens back)
            link_start_pos = None
            for i in range(mid_pos - 1, max(-1, mid_pos - 101), -1):
                if input_ids[i].item() in self.link_start_token_ids:
                    link_start_pos = i
                    break

            if link_start_pos is None:
                continue

            # Decode a growing window of tokens after ]( until ) appears
            chunk_tokens: List[int] = []
            link_end_pos = None
            target_str = None

            for j in range(mid_pos + 1, min(mid_pos + 1 + self.window_tokens, seq_len)):
                chunk_tokens.append(input_ids[j].item())
                # Re-decode the whole chunk each time: required for BPE correctness
                # since individual tokens may not decode to valid UTF-8 in isolation.
                decoded = self.decode_fn(chunk_tokens)
                if ')' in decoded:
                    close_idx = decoded.index(')')
                    candidate = decoded[:close_idx].strip()
                    if candidate:
                        target_str = candidate
                        link_end_pos = j + 1   # exclusive: grant attention from here
                    break

            if link_end_pos is not None and target_str:
                links.append(LinkInfo(link_end_pos=link_end_pos, target_str=target_str))

        logger.debug(
            f"MarkdownLinkDetector: detected {len(links)} links "
            f"in sequence of length {seq_len}"
        )
        return links


class CrossDocLinkMaskCreator:
    """
    Creates attention masks that grant cross-document attention based on in-text links.

    This is a callable class that can be passed to TS2TSTrainingModule as the
    block_mask_creator parameter.

    Args:
        link_detector: A LinkDetector implementation appropriate for the dataset
                       (e.g. MarkdownLinkDetector for Wikipedia).

    Example:
        >>> import tiktoken
        >>> enc = tiktoken.get_encoding('gpt2')
        >>> detector = MarkdownLinkDetector(decode_fn=enc.decode)
        >>> mask_creator = CrossDocLinkMaskCreator(link_detector=detector)
        >>> model = TS2TSTrainingModule(
        ...     block_mask_creator=mask_creator,
        ...     vocab_size=50257,
        ...     ...
        ... )
    """

    def __init__(self, link_detector: LinkDetector):
        self.link_detector = link_detector
        logger.info(
            f"Initialized CrossDocLinkMaskCreator with detector: "
            f"{type(link_detector).__name__}"
        )

    def _match_links_to_docs(
        self,
        links: List[LinkInfo],
        doc_spans: List[Any],
    ) -> Dict[int, List[int]]:
        """
        Match detected links to documents in the batch.

        Uses ``self.link_detector.index_doc_span(span)`` to build the lookup
        key for each span.  This lets dataset-specific detectors (e.g.
        ``PythonImportDetector``) match on a sub-component of ``clean_title``
        (e.g. the bare file path) rather than the full identifier.

        Multiple ``LinkInfo`` objects can share the same ``link_end_pos`` when
        a single import generates several candidate file paths; all matches are
        collected in a list so every valid target receives cross-doc attention.

        Args:
            links:     List of LinkInfo objects from the detector.
            doc_spans: List of DocSpan objects for the current batch.

        Returns:
            Mapping ``link_end_pos -> [target_doc_id, ...]`` for valid links.
            Only includes links where the target document appears earlier in
            the batch (DAG property).
        """
        # Build detector-key -> (doc_id, start_pos) mapping
        index_to_doc: Dict[str, tuple] = {}
        for span in doc_spans:
            key = self.link_detector.index_doc_span(span)
            index_to_doc[key] = (span.doc_id, span.start)

        link_to_target: Dict[int, List[int]] = {}
        matched = 0

        for link in links:
            if link.target_str not in index_to_doc:
                logger.debug(
                    f"Link target '{link.target_str}' not found in batch, skipping"
                )
                continue

            target_doc_id, target_start_pos = index_to_doc[link.target_str]

            # Enforce DAG property: target must start before the link position
            if target_start_pos >= link.link_end_pos:
                logger.debug(
                    f"Link at {link.link_end_pos} to '{link.target_str}' violates DAG "
                    f"(target starts at {target_start_pos}), skipping"
                )
                continue

            link_to_target.setdefault(link.link_end_pos, []).append(target_doc_id)
            matched += 1
            logger.debug(
                f"Matched link at {link.link_end_pos} -> "
                f"doc {target_doc_id} ('{link.target_str}')"
            )

        logger.info(
            f"Matched {matched}/{len(links)} links to documents in batch"
        )
        return link_to_target

    def _build_cross_doc_mask(
        self,
        seq_len: int,
        doc_spans: List[Any],
        link_to_target: Dict[int, List[int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a 2D cross-document attention mask for link-based attention.

        Returns:
            cross_doc_mask: Tensor of shape [seq_len, seq_len] where
                            cross_doc_mask[q, kv] = True if position q can attend
                            to position kv via a link (not including same-doc attention).
        """
        cross_doc_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

        for link_pos, target_doc_ids in sorted(link_to_target.items()):
            for target_doc_id in target_doc_ids:
                # Find the source document containing this link
                link_doc_span = None
                for span in doc_spans:
                    if span.start <= link_pos < span.end:
                        link_doc_span = span
                        break

                if link_doc_span is None:
                    logger.warning(f"Link at position {link_pos} not in any document span")
                    continue

                # Find the target document span
                target_doc_span = None
                for span in doc_spans:
                    if span.doc_id == target_doc_id:
                        target_doc_span = span
                        break

                if target_doc_span is None:
                    logger.warning(f"Target doc {target_doc_id} not found in doc_spans")
                    continue

                # Grant access: positions from link_pos onward (within source doc)
                # can attend to all positions in the target doc
                grant_start = link_pos
                grant_end = min(seq_len, link_doc_span.end)
                target_start = max(0, target_doc_span.start)
                target_end = min(seq_len, target_doc_span.end)

                if grant_start < grant_end and target_start < target_end:
                    cross_doc_mask[grant_start:grant_end, target_start:target_end] = True
                    logger.debug(
                        f"Link at {link_pos}: positions [{grant_start}, {grant_end}) "
                        f"can attend to doc {target_doc_id} at [{target_start}, {target_end})"
                    )

        return cross_doc_mask

    def __call__(self, tokens: torch.Tensor, doc_spans: List[Any], **kwargs) -> BlockMask:
        """
        Create a cross-document link-aware attention mask.

        Args:
            tokens:    Tensor of shape [B, T] with token IDs (full sequence including target)
            doc_spans: List of DocSpan objects with start, end, doc_id, clean_title
            **kwargs:  Additional batch information (unused)

        Returns:
            BlockMask for FlexAttention
        """
        device = tokens.device
        # Input sequence length (model sees tokens[:-1])
        seq_len = tokens.shape[-1] - 1

        input_ids = tokens[0, :-1]  # [seq_len]

        links = self.link_detector.detect_links(input_ids)
        logger.info(f"Found {len(links)} links in batch")

        link_to_target = self._match_links_to_docs(links, doc_spans)

        cross_doc_mask = self._build_cross_doc_mask(
            seq_len, doc_spans, link_to_target, device
        )

        # Build document_ids tensor for base doc_causal mask
        document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for span in doc_spans:
            start = max(0, span.start)
            end = min(seq_len, span.end)
            if start < end:
                document_ids[start:end] = span.doc_id

        def cross_doc_link_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            same_doc = document_ids[q_idx] == document_ids[kv_idx]
            cross_doc_link = cross_doc_mask[q_idx, kv_idx]
            return causal & (same_doc | cross_doc_link)

        block_mask = create_block_mask(
            cross_doc_link_mod,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device
        )

        logger.info(f"Created cross-document link mask for sequence length {seq_len}")
        return block_mask

    def build_dense_mask_for_visualization(
        self,
        tokens: torch.Tensor,
        doc_spans: List[Any],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Build a dense 2D boolean mask for visualization.

        Uses the exact same logic as __call__ but returns a dense tensor instead
        of a FlexAttention BlockMask.
        """
        if device is None:
            device = tokens.device

        seq_len = tokens.shape[-1] - 1
        input_ids = tokens[0, :-1]

        links = self.link_detector.detect_links(input_ids)
        link_to_target = self._match_links_to_docs(links, doc_spans)
        cross_doc_mask = self._build_cross_doc_mask(
            seq_len, doc_spans, link_to_target, device
        )

        document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for span in doc_spans:
            start = max(0, span.start)
            end = min(seq_len, span.end)
            if start < end:
                document_ids[start:end] = span.doc_id

        q_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1]
        k_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]

        causal_mask = q_indices >= k_indices
        same_doc_mask = document_ids.unsqueeze(1) == document_ids.unsqueeze(0)

        dense_mask = causal_mask & (same_doc_mask | cross_doc_mask)
        return dense_mask
