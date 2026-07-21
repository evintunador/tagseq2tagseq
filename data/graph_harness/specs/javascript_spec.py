"""
JavaScript LanguageSpec for the conformance harness.

JavaScript maps onto the file-node model (like Python/Java/TypeScript): a NODE is
one ``.js``/``.jsx``/``.mjs``/``.cjs`` source file, keyed by its repo-relative path
WITHOUT extension (``src/util/helper``). Like TypeScript (and unlike Go/Java's
globally-unique import strings), JS intra-repo imports are RELATIVE (``./foo``,
``../x/y``) — path-relative and therefore ambiguous across repos, so a JS corpus is
SINGLE-REPO per pack (like Python's relative imports). Bare specifiers (``react``,
``lodash``) are external node_modules deps that legitimately DON'T resolve and are
EXCLUDED (design §6.2: recall counts only in-corpus targets).

Detection path: RICH (``extract_keys``). One relative specifier licenses MORE than
one legitimate detection key, because the extension is usually omitted and a
directory import resolves to an index file: ``./foo`` may name ``foo.js`` /
``foo.jsx`` OR ``foo/index.js``. So the oracle licenses BOTH ``foo`` and
``foo/index`` for ``./foo`` — exactly the candidate set the detector emits — so an
index-file candidate is NOT scored as a false positive. If the oracle licensed only
``foo``, the correct ``foo/index`` candidate would be a spurious FP.

Import FORMS covered (all with a literal-string source):
  * ES modules: ``import x from './foo'``, ``import {a} from '../bar'``,
    ``import * as ns from './baz'``, side-effect ``import './x'``.
  * re-export: ``export {q} from './q'`` / ``export * from './r'`` (an edge).
  * CommonJS ``require('./foo')`` — bare identifier ``require`` only, NOT
    ``foo.require(...)``. This is MORE common in JS than TS (much of the corpus is
    CommonJS), so require detection must be solid.
  * dynamic ``import('./x')`` with a literal string.
Non-literal ``require(dynamicVar)`` / ``import(varName)`` are dropped (no static
target). JavaScript has NO type-only imports (that is a TS-only construct).

Independence: this oracle walks the tree-sitter parse tree to find the STATEMENT
set, authored independently of the regex/token-space runtime detector. The key
normalization is written here separately from the detector's (they must agree on
the same specifier, but neither imports the other).

Canonical DETECTION key space: the normalized relative specifier with a leading
``./`` and a recognized extension stripped, ``../`` prefixes retained, plus the
``/index`` directory-import candidate — e.g. ``./sub/b.js`` -> {``sub/b``,
``sub/b/index``}; ``../x/y`` -> {``../x/y``, ``../x/y/index``}. (This DETECTION
space keys on the specifier as written; the separate RESOLUTION axis — fixtures +
``build_javascript_file_nodes`` — resolves ``../`` against the importing file's dir
to a full repo-relative node key.)

Grammar: ``tree_sitter_javascript.language()``. The single JS grammar parses
``.js``/``.jsx``/``.mjs``/``.cjs`` (it accepts JSX and the ESM/CJS forms uniformly),
so — unlike TypeScript, which needs the tsx dialect to parse JSX — no dialect switch
is required.
"""
from __future__ import annotations

from typing import Optional, Set

from ..spec import LanguageSpec

# Recognized module extensions, longest-first. ``.jsx``/``.mjs``/``.cjs`` handle the
# JS module variants; ``.js`` last so a longer suffix strips first.
_EXTS = (".jsx", ".mjs", ".cjs", ".js")


def _strip_ext(spec: str) -> str:
    for ext in _EXTS:
        if spec.endswith(ext):
            return spec[: -len(ext)]
    return spec


def _spec_to_detection_keys(spec: str) -> Set[str]:
    """Normalize a relative import specifier into the DETECTION key set.

    Returns {} for a non-relative (bare/external) specifier — those are not
    in-corpus targets and are excluded from both precision and recall. Collapses
    ``.`` segments (so ``././foo`` == ``./foo``) and RETAINS ``..`` segments (a
    leading-``..`` key like ``../x/y`` is resolved against the importing dir on the
    resolution axis, not here).
    """
    if not (spec.startswith("./") or spec.startswith("../")):
        return set()
    s = _strip_ext(spec)
    out = []
    for seg in s.split("/"):
        if seg in ("", "."):
            continue  # drop empty and current-dir segments
        out.append(seg)  # keep real segments AND '..' (resolution-axis concern)
    key = "/".join(out)
    if key == "":
        return {"index"}  # `import "./"` -> the current dir's index file
    keys = {key}
    if not key.endswith("/index"):
        keys.add(f"{key}/index")
    return keys


def _load_grammar():
    import tree_sitter_javascript
    from tree_sitter import Language
    # Single JS grammar parses .js/.jsx/.mjs/.cjs (incl. JSX) — no dialect switch.
    return Language(tree_sitter_javascript.language())


def _spec_text(string_node, src_bytes: bytes) -> Optional[str]:
    """Extract the specifier text (without quotes) from a `string` node."""
    if string_node is None or string_node.type != "string":
        return None
    for c in string_node.named_children:
        if c.type == "string_fragment":
            return src_bytes[c.start_byte:c.end_byte].decode("utf-8", "replace")
    # empty string literal ("" or '') has no string_fragment child
    return ""


def _first_string_arg(arguments_node):
    if arguments_node is None:
        return None
    for c in arguments_node.named_children:
        if c.type == "string":
            return c
        # a non-literal first argument (identifier, template, expr) -> drop
        return None
    return None


def _extract_keys(root, src_bytes: bytes) -> Set[str]:
    """Independent statement finder (tree-sitter) + candidate licensor.

    Walks every import-bearing construct and licenses the same candidate key set
    the detector emits for the specifier. Only RELATIVE specifiers are licensed;
    bare/external specifiers (``react``) license nothing.
    """
    keys: Set[str] = set()

    def license(spec: Optional[str]):
        if spec:
            keys.update(_spec_to_detection_keys(spec))

    def walk(node):
        t = node.type
        if t == "import_statement":
            # covers default / named `{a,b}` / namespace `* as ns` /
            # side-effect `import "./x"`. (JS has no type-only imports.)
            license(_spec_text(node.child_by_field_name("source"), src_bytes))
        elif t == "export_statement":
            # re-export: `export {q} from "./q"` / `export * from "./r"`.
            # A local `export const x = 1` has no `source` field -> skipped.
            license(_spec_text(node.child_by_field_name("source"), src_bytes))
        elif t == "call_expression":
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if fn is not None:
                fn_txt = src_bytes[fn.start_byte:fn.end_byte].decode("utf-8", "replace")
                # CommonJS require("./z") — bare identifier only, NOT foo.require()
                is_require = fn.type == "identifier" and fn_txt == "require"
                # dynamic import("./x") — the function node is the `import` keyword
                is_dyn_import = fn.type == "import"
                if is_require or is_dyn_import:
                    license(_spec_text(_first_string_arg(args), src_bytes))
        for c in node.children:
            walk(c)

    walk(root)
    return keys


def _canonical_target(target: str) -> Optional[str]:
    """Project a detector-emitted target_str into the DETECTION key space.

    ``JavaScriptImportDetector.detect_links`` already emits normalized detection
    keys (relative specifiers, ``./`` + extension stripped, ``/index`` candidate),
    so this is a pass-through that drops empties. Defensively re-strips a leading
    ``./`` / trailing extension in case a target arrives un-normalized.
    """
    t = target.strip()
    if not t:
        return None
    if t.startswith("./") or t.startswith("../"):
        ks = _spec_to_detection_keys(t)
        return next(iter(sorted(ks)), None) if ks else None
    t = _strip_ext(t).rstrip("/")
    return t or None


JAVASCRIPT_SPEC = LanguageSpec(
    name="javascript",
    extensions=frozenset({"js", "jsx", "mjs", "cjs"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    extract_keys=_extract_keys,
)
