"""
LanguageSpec — the per-language adapter a language implementer supplies.

This is the ONLY extension point a language agent authors for the detection axis.
It bundles:

  * the tree-sitter grammar (via a zero-arg loader, so importing the spec module
    does not require the grammar wheel until the oracle actually runs);
  * a FROZEN tree-sitter query that captures every import/use/require node
    (authored independently of the implementation — this is the ground truth);
  * a `canonical_import` normalizer that maps ONE oracle-captured import node's
    text to a canonical import key (e.g. Go: strip quotes; Python: the dotted
    module path); and
  * a `canonical_target` normalizer that maps ONE detector-emitted `target_str`
    back to that same canonical key space.

Why two normalizers, not one: the detector legitimately emits MULTIPLE candidate
targets per import (Python `import foo.bar` -> `foo/bar.py` AND
`foo/bar/__init__.py`), and the extractor emits resolved file paths, while the
oracle emits the raw module reference. Scoring compares SETS of canonical keys, so
each side needs its own projection into the shared key space. `canonical_target`
returns None for a target that cannot be a real import (lets the harness ignore
noise a detector may emit rather than counting it against precision incorrectly —
but only if the spec author deliberately maps it to None; by default everything
counts).

A spec is a frozen dataclass of callables — no per-language scoring logic lives
here, only projections into the canonical key space the frozen scorer consumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Set


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    # File extensions (no dot) that belong to this language, e.g. {"go"}.
    extensions: frozenset[str]
    # Zero-arg loader returning a tree_sitter.Language. Lazy so `import spec`
    # doesn't hard-require the grammar wheel.
    grammar_loader: Callable[[], "object"]
    # Map one detector/extractor-emitted target string -> canonical import key,
    # or None if it should not be scored.
    canonical_target: Callable[[str], Optional[str]]

    # --- Detection oracle: exactly ONE of the two paths below must be set. ---
    #
    # SIMPLE PATH (Go, most languages): a frozen tree-sitter query capturing
    # every import node as @mod, plus a per-node canonical projection. One node
    # -> one key.
    oracle_query: Optional[str] = None
    canonical_import: Optional[Callable[[str], Optional[str]]] = None
    #
    # RICH PATH (Python and any language where one import statement licenses
    # SEVERAL legitimate detection keys — e.g. `from a.b import c` licenses both
    # `a/b` (c is a symbol) and `a/b/c` (c is a submodule), a distinction only
    # RESOLUTION can settle). A tree-walker over the parsed root emitting the
    # full set of licensed canonical keys. This keeps precision fair: the oracle
    # over-generates in exactly the controlled way the detector legitimately
    # does, so a real from-import submodule candidate is NOT a false positive,
    # while a genuinely hallucinated module still is.
    #
    # Signature: (tree_root_node, src_bytes) -> set[str] of canonical keys.
    extract_keys: Optional[Callable[["object", bytes], Set[str]]] = None

    def __post_init__(self):
        has_simple = self.oracle_query is not None and self.canonical_import is not None
        has_rich = self.extract_keys is not None
        if has_simple == has_rich:
            raise ValueError(
                f"LanguageSpec {self.name!r} must set EXACTLY one detection path: "
                "either (oracle_query + canonical_import) or extract_keys."
            )

    def load_grammar(self):
        """Instantiate the tree-sitter Language (raises if the wheel is absent)."""
        return self.grammar_loader()
