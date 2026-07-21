"""
Dart LanguageSpec for the conformance harness.

Dart maps onto the file-node model (like Python/Java/TypeScript): a NODE is one
``.dart`` source file, keyed by its repo-relative path *including* the ``.dart``
extension (``lib/models/user.dart``). Unlike Go/Java (globally-unique import
strings), Dart intra-repo imports are RELATIVE URIs (``import '../models/user.dart'``,
``import 'src/foo.dart'``) — path-relative and therefore ambiguous across repos, so
a Dart corpus is SINGLE-REPO per pack (like Python / TypeScript relative imports).

Dart URI schemes (the discriminator that decides in-corpus vs external):
  * ``dart:core`` / ``dart:async``  — Dart SDK, external, UNRESOLVED (excluded).
  * ``package:flutter/material.dart`` / ``package:myapp/foo.dart`` — pub / own-package
    URIs. We treat ALL ``package:`` URIs as external/UNRESOLVED (see below), so they
    are excluded from both precision and recall — exactly like ``dart:``.
  * a URI with NO scheme (``import 'foo.dart'``, ``'src/x.dart'``, ``'../y.dart'``,
    ``'./z.dart'``) — a RELATIVE reference resolved against the importing file's dir.
    This is the MAIN graph signal and the only in-corpus key.

Why ``package:`` is treated as external (documented decision): The Stack is
filtered to source extensions, so ``pubspec.yaml`` (the only place the repo's own
package name is declared) is NOT present. Without it we cannot reliably tell
``package:myapp/...`` (intra-repo) from ``package:flutter/...`` (pub dep). Rather
than guess, we treat every ``package:`` URI as external — the relative imports
carry the intra-repo graph. This UNDERCOUNTS intra-repo edges that were written as
``package:<own>/...`` instead of relative, but keeps precision honest (no
mis-resolution). Consistent with Go inferring a module prefix only where
unambiguous and Java having no such inference at all.

Detection path: SIMPLE (``oracle_query`` + ``canonical_import``). Because Dart
requires the ``.dart`` extension EXPLICITLY on every import URI (no extension
inference, no ``index`` directory convention like TypeScript), one relative URI
licenses exactly ONE key — no candidate expansion. This is cleaner than the TS RICH
path. The canonical DETECTION key for a relative URI is the URI as written with the
leading ``./`` collapsed and ``../`` segments retained (resolution against the
importing dir is the separate RESOLUTION axis). Non-relative (``dart:``/``package:``)
URIs map to None so they are neither scored as FP nor counted in the recall
denominator (design §6.2: recall counts only in-corpus targets).

Independence: this oracle walks the tree-sitter parse tree to find the STATEMENT
set (``library_import`` and ``library_export``), authored independently of the
regex/token-space runtime detector. The key normalization is written here
separately from the detector's (they must agree on the same key, but neither
imports the other). ``part 'x.dart';`` directives are NOT imports (they splice a
part-file into the SAME library, not a dependency edge) and are excluded on both
sides.

Grammar: ``tree_sitter_dart.language()``. Verified node types: ``library_import`` >
``import_specification`` > ``configurable_uri`` > ``uri`` > ``string_literal`` (the
URI is the string literal's text, quotes included); ``library_export`` >
``configurable_uri`` > ``uri`` likewise. ``show``/``hide``/``as``/``deferred``
combinator clauses follow the URI and do not change the edge target.
"""
from __future__ import annotations

from typing import Optional

from ..spec import LanguageSpec


def _load_grammar():
    import tree_sitter_dart
    from tree_sitter import Language
    return Language(tree_sitter_dart.language())


# Captures the URI node of every import AND export directive. The URI node's text
# is the quoted string literal (e.g. ``'../models/user.dart'``). ``part`` directives
# use a different ``part_directive`` node and are intentionally NOT captured.
DART_ORACLE_QUERY = r"""
(library_import (import_specification (configurable_uri (uri) @mod)))
(library_export (configurable_uri (uri) @mod))
"""


def _strip_quotes(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
        return raw[1:-1]
    return raw


def _leading_scheme(uri: str) -> Optional[str]:
    """Return the URI scheme (``dart``/``package``/...) or None for a relative URI.

    A Dart URI scheme matches ``^[A-Za-z][A-Za-z0-9+.-]*:``. A relative path such as
    ``../a.dart`` or ``src/b.dart`` has no such prefix (the first ``:``, if any,
    would appear only after a ``/``).
    """
    i = 0
    n = len(uri)
    if i >= n or not uri[i].isalpha():
        return None
    while i < n and (uri[i].isalnum() or uri[i] in "+.-"):
        i += 1
    if i < n and uri[i] == ":":
        return uri[:i]
    return None


def _canonical_uri(uri: str) -> Optional[str]:
    """Normalize a Dart import/export URI into the DETECTION key space.

    Returns None for a non-relative URI (any ``scheme:`` prefix — ``dart:``,
    ``package:``, ``http:``, ...): those are external and excluded from scoring.
    For a relative URI, collapses ``.`` segments (so ``./foo.dart`` == ``foo.dart``)
    and RETAINS ``..`` segments (resolved on the resolution axis, not here). The
    ``.dart`` extension is kept — node keys keep it, so the key spaces align.
    """
    u = uri.strip()
    if not u:
        return None
    if _leading_scheme(u) is not None:
        return None  # external scheme (dart:, package:, ...)
    out = []
    for seg in u.split("/"):
        if seg in ("", "."):
            continue  # drop empty and current-dir segments
        out.append(seg)  # keep real segments AND '..' (resolution-axis concern)
    key = "/".join(out)
    return key or None


def _canonical_import(raw: str) -> Optional[str]:
    """Oracle node text (a quoted URI string literal) -> detection key or None."""
    return _canonical_uri(_strip_quotes(raw))


def _canonical_target(target: str) -> Optional[str]:
    """Detector-emitted target_str -> detection key space.

    The detector emits the normalized relative key already; this is a pass-through
    that drops externals. Defensively strips surrounding quotes and re-normalizes in
    case a target arrives un-normalized.
    """
    return _canonical_uri(_strip_quotes(target))


DART_SPEC = LanguageSpec(
    name="dart",
    extensions=frozenset({"dart"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    oracle_query=DART_ORACLE_QUERY,
    canonical_import=_canonical_import,
)
