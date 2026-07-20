"""
Kotlin node model — shared by the fixtures runner (mirrors the extractor
data/kotlin_graph_extractor/build_kotlin_graph.py).

CRITICAL difference from Java: a Kotlin ``.kt`` file can declare MANY top-level
symbols, and an ``import`` names a DECLARATION (not a file). The filename does NOT
determine the symbol name. So we build a SYMBOL -> FILE model: for each file, parse
its ``package`` header + ALL top-level declaration names and emit ONE NODE PER
DECLARED FQN ``<package>.<Name>``, each keyed by that FQN and carrying the SAME
file content.

Why one node per FQN (not one per file): the frozen resolver
(model/document_corpus._build_indexes and cross_doc_mask._match_links_to_docs)
keys each document by exactly ONE string via ``index_doc_span``. A file exposing N
FQNs therefore needs N nodes so every importable FQN resolves by exact match. Both
``key`` and ``raw_identifier`` are the FQN, so KotlinImportDetector.index_doc_span
(returns raw_identifier unchanged) matches an emitted import FQN exactly.

This module reuses the tree-sitter declaration/import parsing from the extractor so
the fixture oracle and the production builder agree by construction.
"""
from __future__ import annotations

from typing import List, Set

_DECL_TYPES = frozenset({
    "class_declaration",
    "object_declaration",
    "function_declaration",
    "property_declaration",
    "type_alias",
})


def _parse_file(source: str):
    """Return (package, [declared_top_level_names]) via tree-sitter Kotlin."""
    import tree_sitter_kotlin
    from tree_sitter import Language, Parser

    lang = Language(tree_sitter_kotlin.language())
    parser = Parser(lang)
    src = source.encode("utf-8", errors="replace")
    root = parser.parse(src).root_node

    def text(n) -> str:
        return src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    package = ""
    names: List[str] = []
    for c in root.children:
        if c.type == "package_header":
            for cc in c.named_children:
                if cc.type in ("qualified_identifier", "identifier"):
                    package = text(cc).strip()
                    break
        elif c.type in _DECL_TYPES:
            name = None
            for cc in c.named_children:
                if cc.type == "identifier":
                    name = text(cc).strip().strip("`")
                    break
                if cc.type == "variable_declaration":
                    for v in cc.named_children:
                        if v.type == "identifier":
                            name = text(v).strip().strip("`")
                            break
                    break
            if name:
                names.append(name)
    return package, names


def build_kotlin_file_nodes(files, extensions: Set[str]):
    """One node PER TOP-LEVEL DECLARED FQN across all .kt files.

    A file declaring N top-level symbols yields N nodes (each keyed by its FQN,
    all sharing the file's content). Excludes .kts scripts. FQN collisions across
    files keep the first-seen declaration.
    """
    from .fixtures import _FixtureNode  # lazy to avoid import cycle

    nodes = []
    seen: Set[str] = set()
    for f in files:
        if not f.relpath.endswith(".kt") or f.relpath.endswith(".kts"):
            continue
        package, names = _parse_file(f.content)
        for name in names:
            fqn = f"{package}.{name}" if package else name
            if fqn in seen:
                continue
            seen.add(fqn)
            nodes.append(_FixtureNode(
                key=fqn, raw_identifier=fqn, normed_identifier=fqn,
                content=f.content, relpath=f.relpath,
            ))
    return nodes
