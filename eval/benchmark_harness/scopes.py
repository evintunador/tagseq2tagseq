"""Target-scope policies — WHERE in the primary file we score the completion.

Motivation (2026-07-25): RepoBench's next_line is the FIRST line that USES a
symbol imported from a cross-file doc — that use site, not the import line, is
where cross-doc attention should pay off. ASE/CCEval targets are arbitrary FIM
spans that may not touch any imported symbol, which is a candidate cause of the
Kotlin Tier-2 miss. So we re-anchor scoring at a genuine USE SITE and offer
three nested target widths (holding CONTEXT identical, varying only what is
scored — a clean ablation of where the cross-doc signal lives):

  * ``use_line``    — the single logical statement at the first use site.
  * ``use_block``   — use site → end of its enclosing syntactic block.
  * ``rest_of_doc`` — use site → end of the file.
  * ``native``      — the port's own target (no re-anchoring); the only scope
                      available when full_file / a tree-sitter spec is absent.

"Uses an imported symbol" is decided WITHOUT parsing import syntax: the aux docs
ARE the import-resolved files, so we take the set of top-level names DECLARED in
the granted aux docs (tree-sitter over each aux) and find the first line of the
completion region that references any of them. This sidesteps ``import *``
entirely (a star import's names are exactly the aux's declarations) and needs no
per-language import grammar — only a top-level-declaration query per language,
which the graph_harness LanguageSpec's grammar already supports.

A use site requires the symbol's DEFINITION to be in the aux set, so a matched
line is genuinely cross-file-dependent. Falls back to None (example dropped for
that scope) when no such line exists in the region.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from .schema import AuxDoc, CrossDocExample

logger = logging.getLogger(__name__)

SCOPES = ("native", "use_line", "use_block", "rest_of_doc")

# tree-sitter node types whose `name` child is a top-level declared symbol,
# per language spec name. Covers the languages with a spec + a port today;
# extend alongside new ports.
_DECL_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration",
             "method_declaration", "record_declaration"},
    "kotlin": {"class_declaration", "function_declaration", "object_declaration",
               "property_declaration"},
    "typescript": {"function_declaration", "class_declaration",
                   "interface_declaration", "lexical_declaration",
                   "variable_declaration", "enum_declaration",
                   "type_alias_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "struct_item", "enum_item", "trait_item",
             "type_item", "mod_item", "const_item", "static_item",
             "union_item", "macro_definition"},
    "javascript": {"function_declaration", "generator_function_declaration",
                   "class_declaration", "lexical_declaration",
                   "variable_declaration", "method_definition"},
    "dart": {"class_definition", "function_signature", "enum_declaration",
             "extension_declaration", "mixin_declaration", "type_alias"},
    "zig": {"variable_declaration", "function_declaration"},
}

# Block node types that count as an "enclosing syntactic block" boundary for
# use_block. The nearest ancestor of one of these types bounds the span.
_BLOCK_NODE_TYPES = {"block", "function_definition", "function_declaration",
                     "method_declaration", "class_body", "statement_block",
                     "if_statement", "for_statement", "while_statement",
                     "expression_statement", "call", "call_expression"}


@dataclass(frozen=True)
class ScopedTarget:
    context: str
    target: str
    use_site_line: int          # 0-based line index of the use site in full_file
    n_use_sites: int            # total distinct use lines in the completion region
    matched_symbols: Tuple[str, ...]


def _aux_declared_names(aux: Tuple[AuxDoc, ...], language: str) -> Set[str]:
    """Top-level symbol names declared across the granted aux docs."""
    from data.graph_harness.specs import get_spec
    spec = get_spec(language)
    from tree_sitter import Parser
    parser = Parser(spec.load_grammar())
    decl_types = _DECL_NODE_TYPES.get(language, set())
    names: Set[str] = set()

    def walk(node, src: bytes):
        if node.type in decl_types:
            for c in node.children_by_field_name("name") or []:
                names.add(src[c.start_byte:c.end_byte].decode("utf-8", "replace"))
            # some grammars expose the name as a plain 'identifier' child
            if not node.children_by_field_name("name"):
                for c in node.named_children:
                    if c.type in ("identifier", "type_identifier",
                                  "simple_identifier"):
                        names.add(src[c.start_byte:c.end_byte].decode(
                            "utf-8", "replace"))
                        break
        for c in node.children:
            walk(c, src)

    for doc in aux:
        if not doc.content.strip():
            continue
        src = doc.content.encode("utf-8", "replace")
        walk(parser.parse(src).root_node, src)
    names.discard("")
    return names


def _identifier_lines(text: str, language: str) -> List[Set[str]]:
    """Per-line set of identifier tokens in `text` (tree-sitter)."""
    from data.graph_harness.specs import get_spec
    spec = get_spec(language)
    from tree_sitter import Parser
    parser = Parser(spec.load_grammar())
    src = text.encode("utf-8", "replace")
    n_lines = text.count("\n") + 1
    per_line: List[Set[str]] = [set() for _ in range(n_lines)]
    id_types = {"identifier", "type_identifier", "simple_identifier",
                "field_identifier", "property_identifier"}

    def walk(node):
        if node.type in id_types:
            ln = node.start_point[0]
            if ln < n_lines:
                per_line[ln].add(src[node.start_byte:node.end_byte].decode(
                    "utf-8", "replace"))
        for c in node.children:
            walk(c)

    walk(parser.parse(src).root_node)
    return per_line


def _block_end_line(text: str, use_line: int, language: str) -> int:
    """End line (0-based, inclusive) of the smallest block enclosing use_line."""
    from data.graph_harness.specs import get_spec
    spec = get_spec(language)
    from tree_sitter import Parser
    parser = Parser(spec.load_grammar())
    src = text.encode("utf-8", "replace")
    root = parser.parse(src).root_node
    # Descend to the deepest node starting at use_line, then walk up to the
    # nearest block-type ancestor.
    target_byte = sum(len(l) + 1 for l in text.split("\n")[:use_line])

    def deepest(node):
        best = node
        for c in node.children:
            if c.start_byte <= target_byte < c.end_byte:
                best = deepest(c)
                break
        return best

    node = deepest(root)
    while node is not None:
        if node.type in _BLOCK_NODE_TYPES and node.end_point[0] > use_line:
            return node.end_point[0]
        node = node.parent
    return text.count("\n")  # fall back to end of file


def scope_example(
    ex: CrossDocExample,
    language: str,
    scope: str,
) -> Optional[ScopedTarget]:
    """Re-carve `ex` to the requested target scope. None → drop for this scope.

    For non-native scopes the context is REBUILT as the full-file prefix up to
    the use site, so all three use-scopes share identical context and differ
    only in target width.
    """
    if scope == "native":
        return ScopedTarget(context=ex.context, target=ex.target,
                            use_site_line=-1, n_use_sites=0, matched_symbols=())
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; valid: {SCOPES}")
    if ex.full_file is None:
        return None  # can't re-scope without the full file

    names = _aux_declared_names(ex.aux, language)
    if not names:
        return None

    lines = ex.full_file.split("\n")
    # The completion region is everything AFTER the native context prefix, so
    # we never "use site" inside the given context. Locate the region start.
    ctx_lines = ex.context.count("\n")
    per_line_ids = _identifier_lines(ex.full_file, language)

    use_sites = [i for i in range(ctx_lines, len(per_line_ids))
                 if per_line_ids[i] & names]
    if not use_sites:
        return None
    first = use_sites[0]
    matched = tuple(sorted(per_line_ids[first] & names))

    context = "\n".join(lines[:first])
    if not context.strip():
        return None  # need a non-empty prefix to score the first target token

    if scope == "use_line":
        target = lines[first]
    elif scope == "use_block":
        end = _block_end_line(ex.full_file, first, language)
        target = "\n".join(lines[first:end + 1])
    else:  # rest_of_doc
        target = "\n".join(lines[first:])

    if not target.strip():
        return None
    return ScopedTarget(context=context, target="\n" + target,
                        use_site_line=first, n_use_sites=len(use_sites),
                        matched_symbols=matched)
