"""
ArxivCiteDetector — link detection for LaTeX-native ``\\cite{Title}`` citations.

Implements the LinkDetector protocol for the ArXiv (unarXive) dataset. At extraction
time, every in-corpus citation placeholder is rewritten to ``\\cite{<paper title>}``
where the title is the cited paper's ``raw_identifier`` (see data/arxiv_graph_extractor).
Because the bibkey→title resolution already happened during extraction, this detector
is purely syntactic: it scans for ``\\cite{...}`` and returns the title verbatim. No
runtime bibliography resolution is needed.

Design mirrors PythonImportDetector: decode the full sequence once, regex-parse for
citations, and map regex character offsets back to token positions via a cumulative
per-token char-length index (exact for ASCII LaTeX; off-by-one at worst for the rare
multibyte token, which is acceptable for ``link_end_pos`` precision).
"""
from __future__ import annotations

import bisect
import logging
import re
from typing import Any, Callable, List

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)

# Match \cite{...}, \citep{...}, \citet{...} etc. and capture the brace contents.
# At extraction time we emit a bare \cite{Title}; the optional command-suffix and
# optional [pre/post] note groups are tolerated so hand-written or variant forms
# (\citep, \cite[p.5]{...}) still parse. Brace contents are captured greedily up to
# the first closing brace (titles never contain a literal '}').
_CITE_RE = re.compile(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])*\{([^}]*)\}")


class ArxivCiteDetector:
    """
    Detects ``\\cite{Title}`` citations in GPT-2 tokenized LaTeX sequences.

    The captured title is matched exactly against ``DocSpan.raw_identifier`` by
    ``CrossDocLinkMaskCreator``, so the string emitted at extraction time, the
    string the model learns to produce, and the cited document's ``raw_identifier``
    must all be byte-identical.

    Args:
        decode_fn: Callable mapping ``List[int]`` → ``str`` (e.g. ``tiktoken_enc.decode``).
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn

    # ------------------------------------------------------------------
    # LinkDetector protocol
    # ------------------------------------------------------------------

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """
        Detect ``\\cite{...}`` citations and emit one LinkInfo per citation.

        The full sequence is decoded once; a per-token character-length index maps
        each match's closing-brace character offset back to a token position. The
        target string is the (stripped) brace contents — empty captures (the
        out-of-corpus ``\\cite{}`` placeholders left during extraction) are skipped,
        since they intentionally match no document.
        """
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        links: List[LinkInfo] = []
        for m in _CITE_RE.finditer(full_text):
            target = m.group(1).strip()
            if not target:
                continue  # out-of-corpus placeholder \cite{} — matches nothing by design
            # link_end_pos: attention to the target is granted from the token just
            # after the closing brace, i.e. once the full citation has been emitted.
            link_end_pos = self._char_pos_to_token_pos(cumulative, m.end())
            links.append(LinkInfo(link_end_pos=link_end_pos, target_str=target))

        logger.debug(
            "ArxivCiteDetector: detected %d citations in sequence of length %d",
            len(links),
            len(tokens),
        )
        return links

    def index_doc_span(self, span: Any) -> str:
        """Exact match against ``raw_identifier`` (the paper title)."""
        return span.raw_identifier

    # ------------------------------------------------------------------
    # Internal helpers (identical approach to PythonImportDetector)
    # ------------------------------------------------------------------

    def _build_char_to_token_index(self, tokens: List[int]) -> List[int]:
        """
        Build a cumulative character-count index for O(log N) char→token lookup.

        ``cumulative[i]`` = number of characters in ``decode_fn(tokens[:i])``, with
        ``cumulative[0] = 0``. Each token is decoded individually to get its char
        contribution. Exact for ASCII (the common case for LaTeX source); for a rare
        multi-byte token the count may be off by a character, which is acceptable.
        """
        cumulative = [0] * (len(tokens) + 1)
        for i, tok in enumerate(tokens):
            try:
                char_len = len(self.decode_fn([tok]))
            except Exception:
                char_len = 1  # safe fallback: assume 1 char
            cumulative[i + 1] = cumulative[i] + char_len
        return cumulative

    def _char_pos_to_token_pos(self, cumulative: List[int], char_pos: int) -> int:
        """
        Return the smallest token index ``t`` such that ``cumulative[t] >= char_pos``.

        Uses ``bisect_left`` for O(log N) lookup; result clamped to ``[0, len(tokens)]``.
        """
        idx = bisect.bisect_left(cumulative, char_pos)
        return max(0, min(idx, len(cumulative) - 1))
