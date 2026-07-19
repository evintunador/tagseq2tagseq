"""
Go LanguageSpec — the pilot language.

Detection path: SIMPLE (query + canonical_import). Go's import grammar is clean:
every import is an `import_spec` whose `path` is an interpreted string literal
holding the full module-qualified import path (`"github.com/x/y/pkg"`,
`"fmt"`, or a relative `"./local"`). One node -> one key.

Canonical key space: the import path string with quotes stripped, e.g.
`github.com/x/y/pkg`. The Go link detector (to be built) emits `target_str` in
this same space; single-repo resolution then strips the `go.mod` module prefix to
get a repo-relative package dir. NOTE this spec covers the DETECTION axis only —
whether the path resolves to a corpus node is the RESOLUTION axis, checked
separately (fixtures + `go list`/`go/packages` + invariants). See design doc §2.

There is no `import foo.bar` ambiguity in Go: an import names exactly one package,
so unlike Python one statement licenses exactly one key — hence the simple path.
"""
from __future__ import annotations

from typing import Optional

from ..spec import LanguageSpec


def _load_grammar():
    import tree_sitter_go
    from tree_sitter import Language
    return Language(tree_sitter_go.language())


# Captures the string-literal path of every import spec. Covers both the single
# `import "x"` form and the grouped `import ( ... )` form (both use import_spec).
GO_ORACLE_QUERY = r"""
(import_spec path: (interpreted_string_literal) @mod)
(import_spec path: (raw_string_literal) @mod)
"""


def _strip_quotes(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] in "\"`" and raw[-1] in "\"`":
        return raw[1:-1]
    return raw


def _canonical_import(raw: str) -> Optional[str]:
    """Oracle node text (a quoted string literal) -> import path key."""
    key = _strip_quotes(raw).strip()
    return key or None


def _canonical_target(target: str) -> Optional[str]:
    """Detector-emitted target_str -> import path key.

    Accepts either a bare import path or (defensively) a quoted one, so a
    detector that forgets to strip quotes still projects into the same space.
    """
    key = _strip_quotes(target).strip()
    return key or None


GO_SPEC = LanguageSpec(
    name="go",
    extensions=frozenset({"go"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    oracle_query=GO_ORACLE_QUERY,
    canonical_import=_canonical_import,
)
