"""
Rust LanguageSpec for the conformance harness.

Rust maps onto a FILE/MODULE-node model, keyed by the file's MODULE PATH
(``crate::net::tcp``) -- the chain of ``mod`` declarations from the crate root
(``src/lib.rs`` / ``src/main.rs`` / ``src/bin/*.rs``) to the file. Resolution is
single-crate (``crate::`` is a fixed keyword root; module paths collide across
crates like Python's relative paths).

Detection path: RICH (``extract_keys``). Like Python's from-imports and Java's
static imports, one Rust ``use`` statement licenses MORE THAN ONE legitimate
detection key:

  * a grouped/nested use ``use crate::a::{b::C, d::E}`` names several leaves;
  * each leaf ``crate::a::b::C`` licenses BOTH the parent module ``crate::a::b``
    (``C`` is a symbol) AND the full path ``crate::a::b::C`` (``C`` may itself be a
    submodule) -- a submodule-vs-symbol ambiguity only RESOLUTION can settle, so
    the detector rightly emits both and the oracle must license both or the correct
    candidate would count as a false positive;
  * a glob ``use crate::foo::*`` names the target MODULE ``crate::foo``.

This walker is authored INDEPENDENTLY of the detector: it expands the use-tree
directly over tree-sitter's ``scoped_use_list`` / ``use_list`` / ``use_wildcard`` /
``use_as_clause`` node structure (the frozen ground truth), NOT via the detector's
string regex. The STATEMENT set comes from the grammar, so detection recall is
graded independently. The shared CANDIDATE-KEY projection (parent + full; glob ->
module) is the agreed key space both sides live in (analogous to Java's
member+enclosing-type rule and Python's ``module_path_to_file_paths``).

Canonical key space: crate-relative ``::``-joined module paths in RELATIVE form
(``self``/``super`` kept, ``crate::`` kept, external roots like ``std`` kept) --
exactly what ``RustImportDetector.detect_links`` (no per-doc context) emits. So
``canonical_target`` is essentially identity (whitespace-normalized). Absolute
rewriting of ``self``/``super`` against a module path is the RESOLUTION axis
(fixtures + ``detect_links_for_doc`` + build_rust_module_nodes).
"""
from __future__ import annotations

from typing import List, Optional, Set

from ..spec import LanguageSpec


def _load_grammar():
    import tree_sitter_rust
    from tree_sitter import Language
    return Language(tree_sitter_rust.language())


def _norm(path: str) -> str:
    """Whitespace-normalize a ``::``-joined path; strip a leading ``::``."""
    out = []
    for seg in path.replace("\n", "").split("::"):
        seg = seg.strip()
        if seg:
            out.append(seg)
    return "::".join(out)


def _leaf_candidates(leaf: str) -> List[str]:
    """Project ONE leaf path into candidate keys (parent module + full; glob->mod)."""
    leaf = _norm(leaf)
    if not leaf or leaf == "*":
        return []
    if leaf.endswith("::*"):
        mod = leaf[:-3]
        return [mod] if mod else []
    if leaf == "self":
        return []
    segs = leaf.split("::")
    if len(segs) >= 2:
        return ["::".join(segs[:-1]), leaf]
    return [leaf]


def _extract_keys(root, src_bytes: bytes) -> Set[str]:
    """Independent statement finder (tree-sitter) + candidate licensor.

    Walks ``use_declaration`` and file-backed ``mod_item`` nodes and licenses the
    same candidate key set the detector produces.
    """
    keys: Set[str] = set()

    def text(node) -> str:
        return src_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")

    def expand_tree(node) -> List[str]:
        """Return the list of leaf paths named by a use-tree node (relative form)."""
        t = node.type
        if t in ("scoped_identifier", "identifier", "crate", "self", "super",
                 "metavariable", "_reserved_identifier"):
            return [_norm(text(node))]
        if t == "use_as_clause":
            # `path as alias` -> the path only (alias dropped)
            path = node.child_by_field_name("path")
            return expand_tree(path) if path is not None else []
        if t == "use_wildcard":
            # <path> :: *   — the leading path child names the target module
            path_node = None
            for c in node.named_children:
                path_node = c  # single named child is the path
                break
            if path_node is not None:
                base = _norm(text(path_node))
                return [f"{base}::*"] if base else ["*"]
            return ["*"]
        if t == "scoped_use_list":
            path = node.child_by_field_name("path")
            prefix = _norm(text(path)) if path is not None else ""
            lst = node.child_by_field_name("list")
            leaves: List[str] = []
            if lst is not None:
                for entry in lst.named_children:
                    for sub in expand_tree(entry):
                        if sub == "self":
                            leaves.append(prefix)
                        elif prefix:
                            leaves.append(f"{prefix}::{sub}")
                        else:
                            leaves.append(sub)
            return leaves
        if t == "use_list":
            leaves = []
            for entry in node.named_children:
                leaves.extend(expand_tree(entry))
            return leaves
        # unknown / punctuation
        return []

    def walk(node):
        if node.type == "use_declaration":
            arg = None
            for c in node.named_children:
                if c.type == "visibility_modifier":
                    continue
                arg = c
                break
            if arg is not None:
                for leaf in expand_tree(arg):
                    for k in _leaf_candidates(leaf):
                        keys.add(k)
        elif node.type == "mod_item":
            # file-backed only: `mod foo;` has NO declaration_list child.
            has_body = any(c.type == "declaration_list" for c in node.children)
            if not has_body:
                name_node = None
                for c in node.named_children:
                    if c.type == "identifier":
                        name_node = c
                        break
                if name_node is not None:
                    keys.add(_norm(text(name_node)))
        for c in node.children:
            walk(c)

    walk(root)
    return keys


def _canonical_target(target: str) -> Optional[str]:
    """Detector-emitted target -> canonical key (identity + whitespace-normalize)."""
    key = _norm(target)
    return key or None


RUST_SPEC = LanguageSpec(
    name="rust",
    extensions=frozenset({"rs"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    extract_keys=_extract_keys,
)
