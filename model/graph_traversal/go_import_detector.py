"""
Go Import Statement Link Detector

Detects Go ``import`` declarations in tokenized source code sequences and emits
the imported package paths as link targets.

Design
------
Go import syntax is clean and unambiguous, which makes detection much simpler than
Python's (no candidate-path expansion — see below):

    import "github.com/owner/repo/pkg"          // single import
    import (
        "fmt"
        alias "github.com/owner/repo/pkg/sub"    // aliased
        _ "github.com/owner/repo/driver"         // blank (side-effect) import
    )

Every import path is a double-quoted (or, rarely, back-quoted) string literal that
is the package's FULL module-qualified path — globally unique by Go's design
(module mode, universal since ~2019; there are no relative imports). Because the
path is unambiguous, the detector emits it verbatim as ``target_str`` with NO
candidate expansion, unlike ``PythonImportDetector`` which must guess
submodule-vs-symbol and file-vs-package. Matching against a corpus node is exact
string equality on the import path (see ``index_doc_span``).

Node unit
---------
A Go corpus NODE is a PACKAGE (a directory of ``.go`` files), not a single file:
files in one directory share a ``package`` and never import each other, and every
intra-repo import references a directory under the module path. So the node's
``raw_identifier`` is its full import path (``"<module>/<pkgdir>"``), and
``index_doc_span`` returns it unchanged.

Detection mechanics mirror ``PythonImportDetector``: decode once, regex the import
paths, then map each match's character offset back to a token position via a
cumulative per-token char-length index (exact for ASCII, the overwhelming common
case for Go source).

Limitations
-----------
- Only detects paths that appear in genuine ``import`` declarations. A quoted
  string that merely looks like an import path elsewhere in the file is not
  matched, because we only scan within ``import`` statements / blocks.
- Standard-library imports (``"fmt"``, ``"net/http"``) are emitted like any other;
  they simply never resolve to a corpus node (they are not module-qualified), which
  the resolution layer handles by returning no match.
"""
from __future__ import annotations

import bisect
import logging
import re
from typing import Any, Callable, List, Tuple

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# A single import path token: an optional name/alias (identifier, '_' or '.')
# followed by a quoted path. Used to scan the CONTENTS of an import declaration.
# Group 'path' captures the quoted string's inner text (double- or back-quoted).
_IMPORT_SPEC_RE = re.compile(
    r"""(?:(?:[A-Za-z_.][\w]*)\s+)?          # optional alias / '_' / '.'
        (?:"(?P<dpath>[^"\n]+)"|`(?P<bpath>[^`\n]+)`)   # "path" or `path`
    """,
    re.VERBOSE,
)

# A grouped import block:  import ( ... )   — contents may span many lines.
_IMPORT_BLOCK_RE = re.compile(
    r"^[ \t]*import[ \t]*\((?P<body>[^)]*)\)",
    re.MULTILINE | re.DOTALL,
)

# A single-line import:  import "path"   or   import alias "path"
_IMPORT_SINGLE_RE = re.compile(
    r"^[ \t]*import[ \t]+"
    r"(?:(?:[A-Za-z_.][\w]*)[ \t]+)?"                 # optional alias
    r"(?:\"(?P<dpath>[^\"\n]+)\"|`(?P<bpath>[^`\n]+)`)",
    re.MULTILINE,
)


def _blank_comments(text: str) -> str:
    """Replace Go comments with equal-length spaces (preserving char offsets).

    Import declarations that appear inside ``//`` line comments or ``/* */`` block
    comments (e.g. usage examples in package doc comments) are NOT real code and
    must not be detected — tree-sitter correctly ignores them, so the token-space
    detector must too. We blank comments to spaces rather than deleting them so
    that every downstream char offset (and thus ``link_end_pos``) is unchanged.

    A hand-written scanner (not regex) is used so that comment markers INSIDE
    string literals (e.g. a URL ``"http://..."`` or a path) are not mistaken for
    comment starts, and quotes inside comments are not mistaken for strings.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # string literals: skip to matching close, honoring backslash escapes
        if c == '"' or c == "`":
            quote = c
            i += 1
            while i < n:
                if quote == '"' and text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        # rune literal 'x' (may contain an escape); keep simple — skip the quote
        if c == "'":
            i += 1
            while i < n and text[i] != "'":
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        # line comment
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        # block comment
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            # blank the closing */ too (if present)
            if i + 1 < n:
                out[i] = " "
                out[i + 1] = " "
                i += 2
            continue
        i += 1
    return "".join(out)


def _parse_imports(text: str) -> List[Tuple[str, int]]:
    """Find all Go imports in *text*.

    Returns a list of ``(import_path, char_end)`` tuples, where ``char_end`` is
    the character offset just past the import path's closing quote — used to place
    ``link_end_pos``.  Both grouped ``import ( ... )`` blocks and single-line
    ``import "x"`` forms are handled. Comments are blanked first so that import
    declarations inside doc comments are not matched.
    """
    text = _blank_comments(text)
    results: List[Tuple[str, int]] = []

    # --- grouped blocks: import ( ... ) ---
    for block in _IMPORT_BLOCK_RE.finditer(text):
        body = block.group("body")
        body_start = block.start("body")
        for m in _IMPORT_SPEC_RE.finditer(body):
            path = m.group("dpath") or m.group("bpath")
            if path:
                # char_end is just past the closing quote, in absolute coords.
                results.append((path, body_start + m.end()))

    # --- single-line: import "x" / import alias "x" ---
    for m in _IMPORT_SINGLE_RE.finditer(text):
        path = m.group("dpath") or m.group("bpath")
        if path:
            results.append((path, m.end()))

    return results


class GoImportDetector:
    """
    Detects Go import declarations in tokenized source sequences.

    Implements the ``LinkDetector`` protocol. Emits one ``LinkInfo`` per imported
    package path (no candidate expansion — Go paths are unambiguous). Matching
    against a ``DocSpan`` uses the full import path via ``index_doc_span``.

    Args:
        decode_fn: Callable mapping ``List[int]`` -> ``str`` (e.g.
            ``tiktoken_enc.decode``).
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn

    # ------------------------------------------------------------------
    # LinkDetector protocol
    # ------------------------------------------------------------------

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """Detect Go imports in *input_ids* and emit package-path links.

        Args:
            input_ids: 1-D token-ID tensor of shape ``[seq_len]``.

        Returns:
            List of ``LinkInfo``; one per imported path, ``link_end_pos`` being the
            exclusive token position just after the import path's closing quote.
        """
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        links: List[LinkInfo] = []
        for path, char_end in _parse_imports(full_text):
            link_end_pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=link_end_pos, target_str=path))

        logger.debug(
            "GoImportDetector: produced %d links from sequence of length %d",
            len(links), len(tokens),
        )
        return links

    def index_doc_span(self, span: Any) -> str:
        """Return the node's full import path for matching.

        A Go node's ``raw_identifier`` IS its full import path
        (``"github.com/owner/repo/pkg"``), which is exactly what ``detect_links``
        emits as ``target_str`` — so matching is exact string equality and this
        returns ``raw_identifier`` unchanged.
        """
        return span.raw_identifier

    # ------------------------------------------------------------------
    # Internal helpers (shared approach with PythonImportDetector)
    # ------------------------------------------------------------------

    def _build_char_to_token_index(self, tokens: List[int]) -> List[int]:
        """Cumulative per-token char-count index for O(log N) char→token lookup.

        ``cumulative[i]`` = number of chars in ``decode_fn(tokens[:i])``, with
        ``cumulative[0] = 0``. Exact for ASCII (the common case for Go source).
        """
        cumulative = [0] * (len(tokens) + 1)
        for i, tok in enumerate(tokens):
            try:
                char_len = len(self.decode_fn([tok]))
            except Exception:
                char_len = 1
            cumulative[i + 1] = cumulative[i] + char_len
        return cumulative

    def _char_pos_to_token_pos(self, cumulative: List[int], char_pos: int) -> int:
        """Smallest token index *t* with ``cumulative[t] >= char_pos`` (clamped)."""
        idx = bisect.bisect_left(cumulative, char_pos)
        return max(0, min(idx, len(cumulative) - 1))
