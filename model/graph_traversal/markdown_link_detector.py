"""
MarkdownLinkDetector — link detection for Wikipedia-style [text](target) syntax.

Implements the LinkDetector protocol for datasets whose documents are linked via
markdown hyperlinks, primarily SimpleWiki and other Wikipedia-derived datasets.
"""
from __future__ import annotations

import logging
from typing import Callable, List, Optional, Any

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)


class MarkdownLinkDetector:
    """
    Detects markdown-style [text](target) links in GPT-2 tokenized sequences.

    Uses token-ID matching to locate the ]( delimiter, searches backwards for [,
    then decodes a growing window of tokens forward until ) appears in the decoded
    text.  This avoids relying on a single fixed token ID for ) — a bare ) shares
    a token with many other contexts — and makes the approach robust to BPE splits
    inside the target identifier.

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
        """Exact match against ``raw_identifier`` (the original article identifier with spaces)."""
        return span.raw_identifier

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
