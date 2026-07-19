"""
TreeSitterOracle — the INDEPENDENT ground-truth import extractor.

Given source text and a LanguageSpec, it parses with the language's tree-sitter
grammar, runs the spec's frozen query, and returns the set of canonical import
keys the source actually contains. This is the detection oracle: no regex the
implementer wrote can bias it, because it comes from the maintained grammar plus a
query the harness (not the implementer) freezes.

The oracle deliberately knows NOTHING about the corpus — it answers only "what
does this file import?" (detection), never "which node does that resolve to?"
(resolution). Resolution is validated separately (fixtures + toolchain +
invariants); see docs/multilang_code_datasets_DESIGN.md §2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .spec import LanguageSpec


@dataclass(frozen=True)
class OracleImport:
    """One import the oracle found: its canonical key + raw captured text."""
    key: str
    raw: str
    start_byte: int
    end_byte: int


class TreeSitterOracle:
    """Extracts ground-truth canonical import keys from source via tree-sitter."""

    def __init__(self, spec: LanguageSpec):
        # Imported here so the harness package imports without tree_sitter until
        # an oracle is actually constructed.
        from tree_sitter import Parser

        self._spec = spec
        self._lang = spec.load_grammar()
        self._parser = Parser(self._lang)
        # Only the simple path needs a compiled Query.
        self._query = None
        self._QueryCursor = None
        if spec.oracle_query is not None:
            from tree_sitter import Query, QueryCursor
            self._query = Query(self._lang, spec.oracle_query)
            self._QueryCursor = QueryCursor

    def extract(self, source: str) -> List[OracleImport]:
        """Return the OracleImports found in `source` (deduped by canonical key).

        Only meaningful on the simple (query + canonical_import) path — it exposes
        node byte spans. On the rich (extract_keys) path there is no single node
        per key, so use `import_keys` instead; `extract` raises there.
        """
        if self._query is None:
            raise NotImplementedError(
                f"{self._spec.name!r} uses the rich extract_keys path; "
                "byte-span OracleImports are unavailable. Use import_keys()."
            )
        src_bytes = source.encode("utf-8", errors="replace")
        tree = self._parser.parse(src_bytes)
        cursor = self._QueryCursor(self._query)
        captures = cursor.captures(tree.root_node)

        seen_keys: set[str] = set()
        out: List[OracleImport] = []
        # captures: {capture_name: [nodes...]}. We only defined @mod captures.
        for nodes in captures.values():
            for node in nodes:
                raw = src_bytes[node.start_byte:node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                key = self._spec.canonical_import(raw)
                if key is None or key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append(
                    OracleImport(
                        key=key,
                        raw=raw,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                    )
                )
        return out

    def import_keys(self, source: str) -> set[str]:
        """Return the set of canonical import keys the oracle finds in `source`.

        Works on both paths: the rich `extract_keys` walker (many keys per
        statement) or the simple per-node query.
        """
        src_bytes = source.encode("utf-8", errors="replace")
        tree = self._parser.parse(src_bytes)
        if self._spec.extract_keys is not None:
            return set(self._spec.extract_keys(tree.root_node, src_bytes))
        return {imp.key for imp in self.extract(source)}
