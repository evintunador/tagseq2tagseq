"""
Zig ``@import`` Link Detector.

Detects Zig ``@import("...")`` builtin calls in tokenized source and emits
candidate file-path links for the imported modules. Zig has the CLEANEST intra-repo
resolution of any code language here: an import is an EXPLICIT relative file path
(``@import("foo.zig")`` names the sibling ``foo.zig``), so there is no extension
inference and no directory/index-file candidate expansion (unlike TypeScript). One
``@import`` -> one resolved target.

Node unit
---------
A Zig NODE is a source FILE (like Python/Java/TypeScript), keyed by its
repo-relative path WITH the ``.zig`` extension kept (``src/util/helper.zig``). Only
path-like specifiers (ending in ``.zig``) are intra-repo and resolvable; bare
specifiers (``std``, ``builtin``, and package names wired via ``build.zig``) are
external stdlib/package deps that legitimately do NOT resolve to a corpus node.

Relative resolution (the Python/TS-like part)
---------------------------------------------
A relative specifier is resolved against the importing file's DIRECTORY, so
resolution needs per-document context (which file are we in?). So — like
``PythonImportDetector`` / ``TypeScriptImportDetector`` — this detector exposes
TWO methods:

* ``detect_links`` operates on a flat packed sequence with NO per-doc context, so
  it cannot resolve ``../``. It emits the normalized SPECIFIER-space detection key
  (``@import("sub/b.zig")`` -> ``sub/b``), used by the DETECTION-axis grader
  (run_detection) against the tree-sitter oracle.
* ``detect_links_for_doc`` receives a single document's tokens + its
  ``raw_identifier`` (encoding the file path), so it resolves ``./``/``../``
  against the importing file's dir and emits the full repo-relative path (WITH
  ``.zig``) — the RESOLUTION-axis key training/generation match via
  ``index_doc_span``. ``CrossDocLinkMaskCreator`` prefers this method (via
  ``hasattr``).

Zig resolution rules (deterministic, no filesystem):
    ``@import("foo.zig")``       from ``src/main.zig``  -> ``src/foo.zig``
    ``@import("lib/bar.zig")``   from ``src/main.zig``  -> ``src/lib/bar.zig``
    ``@import("../up/x.zig")``   from ``src/a/b.zig``   -> ``src/up/x.zig``
    ``@import("std")``                                  -> (external, no edge)

Detection mechanics mirror the other detectors: decode once, blank comments +
string literals (so an ``@import`` written inside a ``//`` comment or a string
literal is not matched, exactly as tree-sitter ignores it), regex the ``@import``
string arguments, map char offsets to token positions via a cumulative per-token
char index. Zig has NO block comments — only ``//`` line comments (``///`` and
``//!`` doc comments are still ``//``-prefixed and blanked the same way).
"""
from __future__ import annotations

import bisect
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)

# `@import ( "spec" )` — the whitespace-tolerant form. The string is double-quoted
# in Zig (no single-quoted string literals; `'x'` is a char literal). We match on
# comment/string-blanked text, so a real @import string survives (its body is kept
# — see _blank_comments_and_strings) while an @import inside a comment/other string
# is erased.
_IMPORT_RE = re.compile(
    r"@import\s*\(\s*\"(?P<spec>[^\"\n]*)\"",
)


def _spec_to_detection_key(spec: str) -> Optional[str]:
    """Relative specifier -> normalized detection key (specifier space).

    Returns None for a bare stdlib/package name (not in-corpus). Mirrors the frozen
    oracle (zig_spec._spec_to_detection_key) — they MUST agree, but neither imports
    the other. ``.zig`` stripped; ``.`` segments collapsed; ``..`` retained.
    """
    spec = spec.strip()
    if not spec.endswith(".zig"):
        return None
    stripped = spec[: -len(".zig")]
    out = []
    for seg in stripped.split("/"):
        if seg in ("", "."):
            continue
        out.append(seg)
    key = "/".join(out)
    return key or None


def _resolve_relative_spec(spec: str, source_file_path: str) -> Optional[str]:
    """Resolve a relative ``.zig`` specifier against the importing file's dir.

    Returns the full repo-relative path WITH ``.zig`` (a node key), or None for a
    bare specifier or one that escapes the repo root. ``source_file_path`` is the
    importing file's repo-relative path (with or without ``.zig``).
    """
    spec = spec.strip()
    if not spec.endswith(".zig"):
        return None  # bare stdlib/package import
    stripped = spec[: -len(".zig")]
    base_dir = source_file_path.replace("\\", "/").split("/")[:-1]
    cur = list(base_dir)
    for seg in stripped.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if cur:
                cur.pop()
            else:
                return None  # escapes repo root — unresolvable
        else:
            cur.append(seg)
    resolved = "/".join(cur)
    if not resolved:
        return None
    return f"{resolved}.zig"


def _blank_comments_and_strings(text: str) -> str:
    """Blank Zig comments to equal-length spaces; leave string bodies intact.

    Preserves char offsets so ``link_end_pos`` is unchanged. A hand scanner (not
    regex) so a ``//`` inside a string literal, or a ``"`` inside a comment, is not
    misread. Comments are blanked so an ``@import`` written inside a ``//`` comment
    (incl. ``///`` / ``//!`` doc comments) is NOT detected — tree-sitter ignores
    it, so we must too. Zig has NO block comments, so only ``//`` needs handling.

    Quoted-string bodies are NOT blanked — the ``@import`` specifier is itself a
    ``"..."`` string, so erasing string bodies would erase the very specifier we
    detect. A char literal ``'x'`` is skipped (it can contain an escaped ``"`` or
    ``//``). Multiline string literals (``\\\\...`` lines, common in codegen) ARE
    blanked to end-of-line: they routinely embed an ``@import("x.zig")`` STRING
    that tree-sitter treats as multiline-string text, not a real import, so
    blanking prevents that false positive. A ``\\\\`` at top level can only begin
    a multiline-string line (regular-string escapes are consumed inside the
    ``"..."`` branch), so this never eats real code.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # multiline string literal line: `\\....` to end of line. BLANK the body
        # (a codegen `\\const x = @import("y.zig")` is string text, not an import).
        if c == "\\" and i + 1 < n and text[i + 1] == "\\":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        # quoted string: skip to matching close (honor escapes); body left INTACT
        # (the import specifier is itself a "..." string).
        if c == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                if text[i] == "\n":
                    break  # unterminated string on this line — stop
                i += 1
            continue
        # char literal 'x' (may contain an escape). Skip so a `"` or `//` inside a
        # char literal isn't misread. Bounded to the current line.
        if c == "'":
            i += 1
            while i < n and text[i] not in ("'", "\n"):
                if text[i] == "\\":
                    i += 1
                i += 1
            if i < n and text[i] == "'":
                i += 1
            continue
        # line comment // (covers ///, //! doc comments)
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def _parse_import_specs(text: str) -> List[Tuple[str, int]]:
    """Find all ``@import("spec")`` specifiers in *text*.

    Returns ``(specifier, char_end)`` where ``char_end`` is just past the closing
    quote (used for ``link_end_pos``). Comments/char-literals are blanked first.
    """
    text = _blank_comments_and_strings(text)
    results: List[Tuple[str, int]] = []
    for m in _IMPORT_RE.finditer(text):
        spec = m.group("spec")
        if spec:
            results.append((spec, m.end()))
    return results


class ZigImportDetector:
    """Detects Zig ``@import`` builtins in tokenized source sequences.

    Implements the ``LinkDetector`` protocol (+ optional ``detect_links_for_doc``
    for relative-import resolution). Args: ``decode_fn`` (List[int] -> str).
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn

    # ------------------------------------------------------------------
    # LinkDetector protocol
    # ------------------------------------------------------------------

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """Flat-sequence detection (no per-doc context).

        Emits the SPECIFIER-space detection key for path-like specifiers only
        (bare stdlib/package imports emit nothing — they never resolve). Used by
        the detection grader. One ``@import`` -> at most one LinkInfo.
        """
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        links: List[LinkInfo] = []
        for spec, char_end in _parse_import_specs(full_text):
            key = _spec_to_detection_key(spec)
            if key is None:
                continue
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=pos, target_str=key))
        logger.debug("ZigImportDetector: %d links from %d tokens",
                     len(links), len(tokens))
        return links

    def detect_links_for_doc(
        self,
        span_tokens: torch.Tensor,
        raw_identifier: str,
    ) -> List[LinkInfo]:
        """Per-document detection that RESOLVES relative specifiers.

        The importing file's path is read from ``raw_identifier`` (post-``:``), so
        ``./``/``../`` resolve against its directory. Emits the full repo-relative
        path (WITH ``.zig``) with positions LOCAL to the span (caller offsets by
        ``span.start``). One ``@import`` -> at most one resolved candidate.
        """
        tokens = span_tokens.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        source_file_path = (
            raw_identifier.split(":", 1)[1] if ":" in raw_identifier else raw_identifier
        )

        links: List[LinkInfo] = []
        for spec, char_end in _parse_import_specs(full_text):
            tgt = _resolve_relative_spec(spec, source_file_path)
            if tgt is None:
                continue
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=pos, target_str=tgt))
        logger.debug("ZigImportDetector.detect_links_for_doc: %d links for %r",
                     len(links), raw_identifier)
        return links

    def index_doc_span(self, span: Any) -> str:
        """Repo-relative path (WITH ``.zig``) of a node's ``raw_identifier``.

        ``"repo:src/util/helper.zig"`` -> ``"src/util/helper.zig"``. Matches the
        resolved candidate ``detect_links_for_doc`` emits (a full ``.zig`` path).
        """
        parts = span.raw_identifier.split(":", 1)
        return parts[1] if len(parts) > 1 else span.raw_identifier

    # ------------------------------------------------------------------
    # Internal helpers (shared approach with the other code detectors)
    # ------------------------------------------------------------------

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
