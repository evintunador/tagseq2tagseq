"""
Python LanguageSpec — the harness's REFERENCE conformance case.

Python is the known-good language: the existing PythonImportDetector +
github_graph_extractor already ship and are trusted. Grading them against this
spec both (a) validates the harness itself (a correct detector must score ~1.0)
and (b) makes Python a peer of the new languages rather than a special case.

Detection path: RICH (`extract_keys`). One Python import statement licenses
SEVERAL legitimate detection keys, because `from a.b import c` cannot be resolved
to a single file at detection time — `c` may be a submodule (`a/b/c`) or a symbol
in `a/b`. The detector emits candidates for both; so must the oracle, or those
legitimate candidates would count as false positives. We reuse the project's OWN
`module_path_to_file_paths` as the licensor and tree-sitter (independent) as the
statement finder — the two are combined here, but the STATEMENT SET comes from the
grammar, not from the detector's regexes, so the detection recall check is still
independent of the implementation under test.

Canonical key space: repo-relative file paths WITHOUT the `.py` / `/__init__.py`
suffix, i.e. the module directory-or-file stem (e.g. `a/b/c`). Both the oracle's
licensed paths and the detector's emitted `target_str` (a real file path like
`a/b/c.py`) project into this space, so the many-candidate expansion collapses
cleanly and `a/b/c.py` vs `a/b/c/__init__.py` are the same key.
"""
from __future__ import annotations

from typing import Optional, Set

from ..spec import LanguageSpec
from model.graph_traversal.python_import_detector import module_path_to_file_paths


def _load_grammar():
    import tree_sitter_python
    from tree_sitter import Language
    return Language(tree_sitter_python.language())


def _stem(path: str) -> Optional[str]:
    """Collapse a candidate file path to its module stem key.

    `a/b/c.py` -> `a/b/c`; `a/b/__init__.py` -> `a/b`. Anything not looking like
    a python module path returns None (won't be scored).
    """
    p = path.strip()
    if p.endswith("/__init__.py"):
        return p[: -len("/__init__.py")]
    if p.endswith(".py"):
        return p[: -len(".py")]
    if p.endswith("/__init__.pyi"):
        return p[: -len("/__init__.pyi")]
    if p.endswith(".pyi"):
        return p[: -len(".pyi")]
    return None


def _canonical_target(target: str) -> Optional[str]:
    """Project a detector-emitted target_str (a file path) into stem-key space."""
    return _stem(target)


# tree-sitter node types that carry an absolute (non-relative) module reference.
def _extract_keys(root, src_bytes: bytes) -> Set[str]:
    """Independent statement finder (tree-sitter) + project's candidate licensor.

    Walks the parsed tree for import statements, extracts the dotted module path
    (and, for from-imports, each imported name), and licenses the SAME candidate
    stems `module_path_to_file_paths` would produce. Relative imports are skipped
    to match `detect_links` (which cannot resolve them without per-doc context).
    """
    keys: Set[str] = set()

    def text(node) -> str:
        return src_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")

    def license_import(module_path: str, from_name: str):
        if module_path.startswith("."):
            return  # relative — skipped by detect_links
        for cand in module_path_to_file_paths(module_path, from_name):
            k = _stem(cand)
            if k is not None:
                keys.add(k)

    def walk(node):
        t = node.type
        if t == "future_import_statement":
            # `from __future__ import annotations` parses to a DISTINCT node type
            # in tree-sitter. It is a real import statement the detector will
            # find, so the oracle must license it too (it simply never resolves
            # to a corpus node — that's the resolution axis's concern, not
            # detection's). Names live as dotted_name children.
            for child in node.named_children:
                if child.type == "dotted_name":
                    license_import("__future__", text(child).split(".")[0])
            # also license the bare module so `__future__` (module candidate) matches
            license_import("__future__", "")
        elif t == "import_statement":
            # import a.b.c [as x], d.e
            for child in node.named_children:
                if child.type == "dotted_name":
                    license_import(text(child), "")
                elif child.type == "aliased_import":
                    name = child.child_by_field_name("name")
                    if name is not None and name.type == "dotted_name":
                        license_import(text(name), "")
        elif t == "import_from_statement":
            mod = node.child_by_field_name("module_name")
            if mod is not None and mod.type == "dotted_name":
                module_path = text(mod)
                # imported names: all named children after module_name, or a
                # wildcard. If we can't enumerate names, fall back to '*'.
                # NOTE: compare by byte span, not `is` — tree-sitter recreates
                # node wrapper objects, so identity checks against `mod` fail and
                # the module name would be misread as an imported name.
                mod_span = (mod.start_byte, mod.end_byte)
                names = []
                for child in node.named_children:
                    if (child.start_byte, child.end_byte) == mod_span:
                        continue
                    if child.type == "dotted_name":
                        # `from a.b import c` -> c is a dotted_name (single seg)
                        names.append(text(child).split(".")[0])
                    elif child.type == "aliased_import":
                        n = child.child_by_field_name("name")
                        if n is not None:
                            names.append(text(n).split(".")[0])
                    elif child.type == "wildcard_import":
                        names.append("*")
                if not names:
                    license_import(module_path, "*")
                for nm in names:
                    license_import(module_path, nm)
            # relative from-imports (module_name is relative_import) are skipped
        for child in node.children:
            walk(child)

    walk(root)
    return keys


PYTHON_SPEC = LanguageSpec(
    name="python",
    extensions=frozenset({"py", "pyw"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    extract_keys=_extract_keys,
)
