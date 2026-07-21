"""
Zig LanguageSpec for the conformance harness.

Zig maps onto the file-node model (like Python/Java/TypeScript): a NODE is one
``.zig`` source file, keyed by its repo-relative path WITH the ``.zig`` extension
kept (``src/util/helper.zig``). Zig intra-repo imports are EXPLICIT relative FILE
PATHS — the cleanest resolution of any language here: ``@import("foo.zig")``
literally names a sibling file, ``@import("lib/bar.zig")`` a file under a subdir,
``@import("../up/x.zig")`` a file up a directory. There is no extension inference,
no directory/index-file candidate expansion, no import-search-path — the path is
the target, resolved against the importing file's DIRECTORY (like a C ``#include``
with a literal relative path). Because paths are relative they collide across
repos, so a Zig corpus is SINGLE-REPO per pack (like Python/TS relative imports).

Standard-library / package imports are BARE names with no ``.zig`` extension:
``@import("std")``, ``@import("builtin")``, and (via ``build.zig``, which The
Stack filters out) ``@import("mypkg")``. These are external, do NOT resolve to a
corpus node, and are EXCLUDED from both precision and recall (design §6.2: recall
counts only in-corpus targets). The detector emits nothing for them, and this
oracle licenses nothing for them, so they never enter the scored key space.

Detection path: SIMPLE (``oracle_query`` + ``canonical_import``). Unlike TS,
ONE ``@import`` names exactly ONE target key — there is no candidate expansion
(no extension guessing, no index files), so the simple one-node-one-key path is
the correct, cleaner choice. The frozen tree-sitter query captures the string
argument of every ``@import(...)`` builtin (a ``builtin_function`` whose
``builtin_identifier`` is ``@import``); the ``#eq?`` predicate excludes other
builtins like ``@embedFile`` / ``@cImport``. Comments and string literals are
NOT a concern for the oracle — tree-sitter only matches a real ``@import`` builtin
node, so an ``@import("x.zig")`` written inside a ``//`` comment or a ``"..."``
string literal is never captured (it parses as ``comment`` / ``string_content``,
not a ``builtin_function``). The token-space detector must blank comments/strings
to match this.

Canonical DETECTION key space: the raw string argument of ``@import`` with the
``.zig`` extension STRIPPED, but only for path-like imports (those ending in
``.zig``); bare stdlib names (``std``, ``builtin``) map to None and are dropped.
So ``@import("foo.zig")`` -> ``foo``, ``@import("lib/bar.zig")`` ->
``lib/bar``, ``@import("../up/x.zig")`` -> ``../up/x`` (the ``../`` is retained;
resolving it against the importing file's dir to a full repo-relative node key is
the RESOLUTION axis — fixtures + ``detect_links_for_doc`` + build_zig_file_nodes).

Grammar: ``tree_sitter_zig.language()`` — node types verified: ``@import`` is a
``builtin_function`` with a ``builtin_identifier`` child and an ``arguments`` node
holding a ``string`` -> ``string_content``.
"""
from __future__ import annotations

from typing import Optional

from ..spec import LanguageSpec


def _load_grammar():
    import tree_sitter_zig
    from tree_sitter import Language
    return Language(tree_sitter_zig.language())


# Captures the string_content of every @import(...) builtin. The #eq? predicate
# on the builtin_identifier excludes @embedFile / @cImport / @cInclude etc., so
# only genuine module imports are ground truth. One @import -> one key.
ZIG_ORACLE_QUERY = r"""
(builtin_function
  (builtin_identifier) @_b
  (arguments (string (string_content) @mod))
  (#eq? @_b "@import"))
"""


def _spec_to_detection_key(spec: str) -> Optional[str]:
    """Normalize an @import string argument into the DETECTION key.

    Returns None for a bare stdlib/package name (``std``, ``builtin``, ``mypkg``)
    — those are external, never resolve, and are excluded from scoring. Path-like
    imports (ending in ``.zig``) have the extension stripped; ``../`` and ``./``
    prefixes are retained (resolved on the resolution axis). ``./`` current-dir
    segments are collapsed so ``./foo.zig`` == ``foo``.
    """
    spec = spec.strip()
    if not spec.endswith(".zig"):
        return None  # bare stdlib/package import — external, not in-corpus
    stripped = spec[: -len(".zig")]
    out = []
    for seg in stripped.split("/"):
        if seg in ("", "."):
            continue  # drop empty and current-dir segments
        out.append(seg)  # keep real segments AND '..' (resolution-axis concern)
    key = "/".join(out)
    return key or None


def _canonical_import(raw: str) -> Optional[str]:
    """Oracle node text (the @import string_content, unquoted) -> detection key."""
    return _spec_to_detection_key(raw)


def _canonical_target(target: str) -> Optional[str]:
    """Detector-emitted target_str -> detection key.

    ``ZigImportDetector.detect_links`` already emits normalized keys (``.zig``
    stripped, bare imports dropped), so this is a pass-through that defensively
    re-strips a trailing ``.zig`` and drops bare names, keeping both sides in the
    same key space.
    """
    t = target.strip()
    if not t:
        return None
    if t.endswith(".zig"):
        return _spec_to_detection_key(t)
    # already-normalized path key (no extension). A bare stdlib name that somehow
    # reaches here would have no '/' and no '.zig'; the detector never emits those,
    # so a plain key is treated as an in-corpus path key.
    out = []
    for seg in t.split("/"):
        if seg in ("", "."):
            continue
        out.append(seg)
    key = "/".join(out)
    return key or None


ZIG_SPEC = LanguageSpec(
    name="zig",
    extensions=frozenset({"zig"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    oracle_query=ZIG_ORACLE_QUERY,
    canonical_import=_canonical_import,
)
