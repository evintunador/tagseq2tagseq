"""
TypeScript LanguageSpec for the conformance harness.

TypeScript maps onto the file-node model (like Python/Java): a NODE is one
``.ts``/``.tsx`` source file, keyed by its repo-relative path WITHOUT extension
(``src/util/helper``). Unlike Go/Java (globally-unique import strings), TS
intra-repo imports are RELATIVE (``./foo``, ``../x/y``) — path-relative and
therefore ambiguous across repos, so a TS corpus is SINGLE-REPO per pack (like
Python's relative imports). Bare specifiers (``react``, ``lodash``) are external
node_modules deps that legitimately DON'T resolve and are EXCLUDED (design §6.2:
recall counts only in-corpus targets).

Detection path: RICH (``extract_keys``). One relative specifier licenses MORE than
one legitimate detection key, because the extension is usually omitted and a
directory import resolves to an index file: ``./foo`` may name ``foo.ts`` /
``foo.tsx`` OR ``foo/index.ts``. So the oracle licenses BOTH ``foo`` and
``foo/index`` for ``./foo`` — exactly the candidate set the detector emits — so an
index-file candidate is NOT scored as a false positive. If the oracle licensed only
``foo``, the correct ``foo/index`` candidate would be a spurious FP.

Independence: this oracle walks the tree-sitter parse tree to find the STATEMENT
set (``import``/``export ... from``, ``require(...)``, dynamic ``import(...)``),
which is authored independently of the regex/token-space runtime detector. The key
normalization is written here separately from the detector's (they must agree on
the same specifier, but neither imports the other).

Canonical DETECTION key space: the normalized relative specifier with a leading
``./`` and a recognized extension stripped, ``../`` prefixes retained, plus the
``/index`` directory-import candidate — e.g. ``./sub/b.ts`` -> {``sub/b``,
``sub/b/index``}; ``../x/y`` -> {``../x/y``, ``../x/y/index``}. (This DETECTION
space keys on the specifier as written; the separate RESOLUTION axis — fixtures +
``build_typescript_file_nodes`` — resolves ``../`` against the importing file's dir
to a full repo-relative node key.)

Grammar: ``tree_sitter_typescript.language_tsx()``. The tsx dialect is a strict
superset that parses BOTH ``.ts`` and ``.tsx`` — crucially it parses the JSX in
``.tsx`` files, which the plain ``language_typescript()`` grammar chokes on
(measured on 1500 real Stack files: the plain grammar produced parse errors on 112
files vs 17 for tsx, and missed imports — e.g. a ``require(...)`` inside a JSX
attribute — that tsx recovers; tsx found 4 keys the plain grammar missed and missed
none). Import syntax parses identically in both, so using tsx makes the oracle
strictly more correct as ground truth for both extensions.
"""
from __future__ import annotations

from typing import Optional, Set

from ..spec import LanguageSpec

# Recognized module extensions, longest-first so ``.d.ts`` is stripped before
# ``.ts``. ``.js``/``.jsx``/``.mjs``/``.cjs`` handle the TS convention of writing
# ``./foo.js`` to mean the sibling ``./foo.ts`` (allowJs / ESM-output aside).
_EXTS = (".d.ts", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs")


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
    import tree_sitter_typescript
    from tree_sitter import Language
    # tsx dialect: strict superset that parses .ts AND .tsx (incl. JSX) — see the
    # module docstring for why this is strictly better as the detection oracle.
    return Language(tree_sitter_typescript.language_tsx())


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
            # side-effect `import "./x"` / type-only `import type {...}`
            src_field = node.child_by_field_name("source")
            if src_field is not None:
                license(_spec_text(src_field, src_bytes))
            else:
                # TS import-equals-require: `import X = require("./y")`. The string
                # lives inside an import_require_clause, not the `source` field.
                for c in node.named_children:
                    if c.type == "import_require_clause":
                        for cc in c.named_children:
                            if cc.type == "string":
                                license(_spec_text(cc, src_bytes))
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

    ``TypeScriptImportDetector.detect_links`` already emits normalized detection
    keys (relative specifiers, ``./`` + extension stripped, ``/index`` candidate),
    so this is a pass-through that drops empties. Defensively re-strips a leading
    ``./`` / trailing extension in case a target arrives un-normalized.
    """
    t = target.strip()
    if not t:
        return None
    if t.startswith("./") or t.startswith("../"):
        ks = _spec_to_detection_keys(t)
        # a single canonical key is expected here; if the target was already a
        # bare normalized key it won't start with ./ or ../ and falls through.
        return next(iter(sorted(ks)), None) if ks else None
    t = _strip_ext(t).rstrip("/")
    return t or None


TYPESCRIPT_SPEC = LanguageSpec(
    name="typescript",
    extensions=frozenset({"ts", "tsx"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    extract_keys=_extract_keys,
)
