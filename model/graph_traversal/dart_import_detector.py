"""
Dart Import Statement Link Detector

Detects Dart ``import`` / ``export`` directives in tokenized source and emits
candidate file-path links for the RELATIVE imports (the intra-repo graph signal).

Node unit
---------
A Dart NODE is a source FILE (like Python/Java/TypeScript), keyed by its
repo-relative path INCLUDING the ``.dart`` extension (``lib/models/user.dart``).
Dart requires the ``.dart`` extension explicitly on every import URI and has no
directory/``index`` convention, so a relative URI names exactly one file — no
candidate expansion (unlike TypeScript, cleaner).

Which URIs are intra-repo
-------------------------
A Dart import URI is intra-repo iff it has NO scheme:

    import 'foo.dart';            -> relative (same dir)              INTRA-REPO
    import 'src/api.dart';        -> relative (subdir)               INTRA-REPO
    import '../models/user.dart'; -> relative (up a dir)             INTRA-REPO
    import './widget.dart';       -> relative (explicit current dir) INTRA-REPO
    import 'dart:async';          -> Dart SDK      (scheme 'dart')   EXTERNAL
    import 'package:flutter/x.dart'; -> pub dep     (scheme 'package') EXTERNAL

ALL ``package:`` URIs are treated as external/unresolved: The Stack has no
``pubspec.yaml`` so the repo's own package name can't be inferred, and we prefer
undercounting a few intra-repo edges (that were written ``package:<own>/...``) over
mis-resolving. The relative imports carry the graph. Combinator clauses
(``show``/``hide``/``as prefix``/``deferred``) follow the URI and don't change the
edge target. ``part 'x.dart';`` is NOT an import (it splices a part-file into the
SAME library) and is not matched.

Relative resolution (the Python/TypeScript-like part)
-----------------------------------------------------
A relative URI resolves against the IMPORTING file's directory, needing per-doc
context. So — like ``PythonImportDetector`` / ``TypeScriptImportDetector`` — this
detector exposes TWO methods:

* ``detect_links`` operates on a flat packed sequence with NO per-doc context, so
  it cannot resolve ``../``. It emits the normalized relative URI key
  (``../models/user.dart`` -> ``../models/user.dart``; ``./foo.dart`` -> ``foo.dart``),
  used by the DETECTION grader against the tree-sitter oracle.
* ``detect_links_for_doc`` receives a single document's tokens + its
  ``raw_identifier`` (encoding the file path), so it resolves ``./``/``../`` against
  the importing file's dir and emits the repo-relative path (WITH ``.dart``) — the
  RESOLUTION-axis key training/generation match via ``index_doc_span``.
  ``CrossDocLinkMaskCreator`` prefers this method (detected via ``hasattr``).

Detection mechanics mirror the other detectors: decode once, blank comments +
multi-line (triple-quoted) strings (so an import-looking line inside a doc comment
or a code-gen template string is not matched — single-line string bodies are left
intact because the URI is itself a ``'...'`` / ``"..."`` string), regex the
directives, map char offsets to token positions via a cumulative per-token char
index.
"""
from __future__ import annotations

import bisect
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)


def _leading_scheme(uri: str) -> Optional[str]:
    """Return the URI scheme (``dart``/``package``/...) or None for a relative URI.

    Mirrors ``dart_spec._leading_scheme`` (they must agree; neither imports the
    other). A scheme matches ``^[A-Za-z][A-Za-z0-9+.-]*:``.
    """
    i, n = 0, len(uri)
    if i >= n or not uri[i].isalpha():
        return None
    while i < n and (uri[i].isalnum() or uri[i] in "+.-"):
        i += 1
    if i < n and uri[i] == ":":
        return uri[:i]
    return None


def _normalize_relative_uri(uri: str) -> Optional[str]:
    """Relative URI -> normalized detection key (or None if external/empty).

    Mirrors the frozen oracle (dart_spec._canonical_uri): drops externals, collapses
    ``.`` segments, RETAINS ``..`` segments, keeps the ``.dart`` extension.
    """
    u = uri.strip()
    if not u or _leading_scheme(u) is not None:
        return None
    out = []
    for seg in u.split("/"):
        if seg in ("", "."):
            continue
        out.append(seg)
    key = "/".join(out)
    return key or None


def _resolve_relative_uri(uri: str, source_file_path: str) -> Optional[str]:
    """Resolve a relative URI against the importing file's dir -> repo-relative key.

    ``source_file_path`` is the importing file's repo-relative path (WITH ``.dart``).
    Returns the resolved repo-relative path (WITH ``.dart``), or None for an
    external URI or one that escapes the repo root.
    """
    if _leading_scheme(uri) is not None:
        return None
    base_dir = source_file_path.replace("\\", "/").split("/")[:-1]
    cur = list(base_dir)
    for seg in uri.strip().split("/"):
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
    return resolved or None


# `import`/`export` directive followed by its first URI string literal. The first
# string after the keyword is the edge target; combinator clauses (show/hide/as/
# deferred) and a conditional `if (...) 'other'` follow and are ignored (matching
# the tree-sitter oracle, which captures only the primary configurable_uri).
# `part`/`part of` use no such keyword and are not matched.
#
# ANCHORED to statement position: the `import`/`export` keyword must be at the
# start of a line (after optional indentation), or right after a `;`. Real Dart
# directives are always top-level statements; this anchor rejects an `import "x"`
# that appears MID-line inside a single-quoted string literal (e.g. an analyzer
# test's ``AddContentOverlay('import "none.dart";')``), which tree-sitter sees as
# ONE string, not an import. We can't blank single-line string bodies (the URI is
# itself such a string), so the anchor is how we avoid that false positive.
_DIRECTIVE_RE = re.compile(
    r"(?m)(?:^|;)[ \t]*(?:import|export)\b[ \t]*(?P<q>[\"'])(?P<uri>[^\"'\n]*)(?P=q)",
)


def _blank_comments_and_multiline_strings(text: str) -> str:
    """Blank Dart comments + triple-quoted strings to spaces (offsets preserved).

    A hand scanner (not regex) so comment markers inside strings and quotes inside
    comments are not misread. Comments (``//``, ``///`` doc, ``/* */`` incl. Dart's
    nestable block comments) are blanked so an ``import`` written in a doc comment is
    NOT detected (tree-sitter ignores it, so must we). Triple-quoted strings
    (``'''...'''`` / ``\"\"\"...\"\"\"``) ARE blanked: a URI is never triple-quoted,
    but code-gen templates commonly embed an ``import '...'`` line inside one that is
    not a real import. Single-line ``'...'`` / ``"..."`` bodies are left INTACT — the
    import URI is itself such a string, so blanking them would erase the target.
    Raw strings (``r'...'``) are treated like normal single-line strings.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # triple-quoted string: blank the whole body
        if (c == '"' or c == "'") and i + 2 < n and text[i + 1] == c and text[i + 2] == c:
            quote = c
            out[i] = out[i + 1] = out[i + 2] = " "
            i += 3
            while i < n:
                if text[i] == "\\":
                    if text[i] != "\n":
                        out[i] = " "
                    i += 1
                    if i < n and text[i] != "\n":
                        out[i] = " "
                    i += 1
                    continue
                if text[i] == quote and i + 2 < n and text[i + 1] == quote and text[i + 2] == quote:
                    out[i] = out[i + 1] = out[i + 2] = " "
                    i += 3
                    break
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            continue
        # single-line string: skip to matching close (honor escapes), body INTACT
        if c == '"' or c == "'":
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote or text[i] == "\n":
                    if text[i] == quote:
                        i += 1
                    break
                i += 1
            continue
        # line comment // (covers /// doc comments)
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        # block comment /* */ (Dart block comments nest)
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            depth = 1
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and depth > 0:
                if text[i] == "/" and i + 1 < n and text[i + 1] == "*":
                    depth += 1
                    out[i] = out[i + 1] = " "
                    i += 2
                    continue
                if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    depth -= 1
                    out[i] = out[i + 1] = " "
                    i += 2
                    continue
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def _parse_import_uris(text: str) -> List[Tuple[str, int]]:
    """Find all import/export URIs in *text*.

    Returns ``(uri, char_end)`` where ``char_end`` is just past the closing quote
    (used for ``link_end_pos``). Comments + triple-quoted strings are blanked first.
    """
    text = _blank_comments_and_multiline_strings(text)
    results: List[Tuple[str, int]] = []
    for m in _DIRECTIVE_RE.finditer(text):
        results.append((m.group("uri"), m.end()))
    return results


class DartImportDetector:
    """Detects Dart import/export directives in tokenized source sequences.

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

        Emits the normalized relative-URI detection key for relative imports only
        (``dart:``/``package:`` externals emit nothing — they never resolve). Used by
        the detection grader.
        """
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        links: List[LinkInfo] = []
        for uri, char_end in _parse_import_uris(full_text):
            key = _normalize_relative_uri(uri)
            if key is None:
                continue
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=pos, target_str=key))
        logger.debug("DartImportDetector: %d links from %d tokens",
                     len(links), len(tokens))
        return links

    def detect_links_for_doc(
        self,
        span_tokens: torch.Tensor,
        raw_identifier: str,
    ) -> List[LinkInfo]:
        """Per-document detection that RESOLVES relative URIs.

        The importing file's path is read from ``raw_identifier`` (post-``:``), so
        ``./``/``../`` resolve against its directory. Emits the resolved repo-relative
        path (WITH ``.dart``) with positions LOCAL to the span (caller offsets by
        ``span.start``).
        """
        tokens = span_tokens.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        source_file_path = (
            raw_identifier.split(":", 1)[1] if ":" in raw_identifier else raw_identifier
        )

        links: List[LinkInfo] = []
        for uri, char_end in _parse_import_uris(full_text):
            tgt = _resolve_relative_uri(uri, source_file_path)
            if tgt is None:
                continue
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=pos, target_str=tgt))
        logger.debug("DartImportDetector.detect_links_for_doc: %d links for %r",
                     len(links), raw_identifier)
        return links

    def index_doc_span(self, span: Any) -> str:
        """Repo-relative path (WITH ``.dart``) of a node's ``raw_identifier``.

        ``"repo:lib/models/user.dart"`` -> ``"lib/models/user.dart"``. Matches the
        resolved candidates ``detect_links_for_doc`` emits. Node keys keep the
        ``.dart`` extension, so no stripping.
        """
        parts = span.raw_identifier.split(":", 1)
        return parts[1] if len(parts) > 1 else span.raw_identifier

    # ------------------------------------------------------------------
    # Internal helpers
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
