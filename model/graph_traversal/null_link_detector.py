"""
NullLinkDetector — a no-op LinkDetector for edgeless corpora (e.g. FineWeb).

Datasets with no link structure (flat web text) have nothing to detect, so this
detector always returns zero links. Using it (rather than borrowing markdown /
python / arxiv) guarantees no spurious cross-doc grant can ever fire regardless
of what incidental ``[text](url)`` / ``import`` / ``\\cite{}`` syntax happens to
appear in the raw text, and documents intent: this corpus is a doc_causal-only
baseline.
"""
from __future__ import annotations

from typing import Any, Callable, List

import torch

from .link_detector import LinkInfo


class NullLinkDetector:
    """LinkDetector that never detects a link (for edgeless datasets)."""

    def __init__(self, decode_fn: Callable[[List[int]], str] | None = None):
        # decode_fn is accepted for a uniform constructor signature with the
        # other detectors (make_link_detector passes it); it is unused.
        self.decode_fn = decode_fn

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        return []

    def index_doc_span(self, span: Any) -> str:
        # Protocol default is not inherited (LinkDetector is a Protocol, not a
        # base class), and _match_links_to_docs indexes every doc span even when
        # detect_links returns nothing, so this must be defined explicitly.
        return span.raw_identifier
