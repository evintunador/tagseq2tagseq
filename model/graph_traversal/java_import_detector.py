"""
Java Import Statement Link Detector

Detects Java ``import`` declarations in tokenized source and emits candidate
file-path links for the imported types.

Java maps onto the file-node model like Python: an import names a fully-qualified
type (``com.google.gson.internal.Excluder``) that maps deterministically to a file
path (``com/google/gson/internal/Excluder.java``) under a source root, and
``package`` == directory. So a Java NODE is a file, keyed by its
source-root-relative path.

Forms handled:
    import a.b.C;                 -> a/b/C.java
    import static a.b.C.m;        -> a/b/C.java     (enclosing type; member dropped)
    import a.b.*;                 -> (no file; a package, emitted as the dotted
                                      package so it can match a package-info or be
                                      dropped by resolution)

Unlike Python there is NO submodule-vs-symbol ambiguity: an import names exactly
one type. The only candidate nuance is static imports, where the last dotted
segment is a member (method/field), so we also emit the enclosing type's path.

``index_doc_span`` mirrors the Python detector: it strips a ``<repo>:`` prefix
from ``raw_identifier`` and returns the source-root-relative path with the
``.java`` suffix removed and slashes turned to dots, i.e. the dotted FQN — the
same space ``detect_links`` emits. Matching is then exact.

Detection mechanics mirror PythonImportDetector: decode once, regex the imports,
map char offsets to token positions via a cumulative per-token char index.
"""
from __future__ import annotations

import bisect
import logging
import re
from typing import Any, Callable, List, Tuple

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)

# import [static] a.b.C[.member] ;    or   import a.b.* ;
_IMPORT_RE = re.compile(
    r"^[ \t]*import[ \t]+(?:static[ \t]+)?"
    r"(?P<fqn>[A-Za-z_$][\w$]*(?:\s*\.\s*[A-Za-z_$][\w$]*)*)"
    r"(?P<star>\s*\.\s*\*)?[ \t]*;",
    re.MULTILINE,
)
_STATIC_RE = re.compile(r"^[ \t]*import[ \t]+static\b")


def _fqn_candidates(fqn: str, is_static: bool, is_star: bool) -> List[str]:
    """Candidate DOTTED type FQNs for an import.

    Emits the dotted FQN itself. For static imports the last segment is a member
    (method/field), so the enclosing type FQN is emitted too. For on-demand
    (``.*``) imports the FQN is a package with no single type/file, so nothing is
    emitted (package imports don't resolve to a file node).

    Returns dotted FQNs (e.g. ``com.google.gson.Gson``) — the SAME space
    ``index_doc_span`` returns, so resolution is exact string match (like Go).
    """
    parts = [p.strip() for p in fqn.split(".") if p.strip()]
    if not parts or is_star:
        return []
    cands = [".".join(parts)]
    if is_static and len(parts) >= 2:
        cands.append(".".join(parts[:-1]))
    return cands


def _parse_imports(text: str) -> List[Tuple[str, int]]:
    """Return (candidate_fqn, char_end) for every import in *text*."""
    results: List[Tuple[str, int]] = []
    for m in _IMPORT_RE.finditer(text):
        fqn = re.sub(r"\s+", "", m.group("fqn"))
        is_star = m.group("star") is not None
        is_static = _STATIC_RE.match(m.group(0)) is not None
        for cand in _fqn_candidates(fqn, is_static, is_star):
            results.append((cand, m.end()))
    return results


class JavaImportDetector:
    """Detects Java import declarations in tokenized source sequences.

    Implements the ``LinkDetector`` protocol. Emits candidate file-path links
    (dotted FQN -> ``a/b/C.java``). Args: ``decode_fn`` (List[int] -> str).
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)
        links: List[LinkInfo] = []
        for fqn, char_end in _parse_imports(full_text):
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=pos, target_str=fqn))
        logger.debug("JavaImportDetector: %d links from %d tokens",
                     len(links), len(tokens))
        return links

    def index_doc_span(self, span: Any) -> str:
        """Path component of raw_identifier as a dotted FQN key.

        ``"repo:src/main/java/com/google/gson/Gson.java"`` is not directly the
        FQN — the source root (``src/main/java/``) must be stripped for the dotted
        name to match an import. We can't know the source root in general here, so
        we return the bare path (post-':'), and the fixtures runner /
        build_java_file_nodes are responsible for keying nodes by the FQN derived
        with the known source root. For exact-match to work, node raw_identifiers
        should already encode the FQN-relative path (see build_java_file_nodes).
        """
        parts = span.raw_identifier.split(":", 1)
        path = parts[1] if len(parts) > 1 else span.raw_identifier
        if path.endswith(".java"):
            path = path[: -len(".java")]
        return path.replace("/", ".")

    def _build_char_to_token_index(self, tokens: List[int]) -> List[int]:
        cumulative = [0] * (len(tokens) + 1)
        for i, tok in enumerate(tokens):
            try:
                char_len = len(self.decode_fn([tok]))
            except Exception:
                char_len = 1
            cumulative[i + 1] = cumulative[i] + char_len
        return cumulative

    def _char_pos_to_token_pos(self, cumulative: List[int], char_pos: int) -> int:
        idx = bisect.bisect_left(cumulative, char_pos)
        return max(0, min(idx, len(cumulative) - 1))
