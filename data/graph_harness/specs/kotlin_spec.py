"""
Kotlin LanguageSpec for the conformance harness.

Kotlin is a JVM-family, fully-qualified-name (FQN) language like Java, and is
therefore MULTI-REPO capable (import strings are globally unique). But Kotlin
differs from Java in two ways that shape the node model (documented at length in
data/kotlin_graph_extractor/build_kotlin_graph.py):

  * the filename does NOT determine the symbol name, and
  * ONE ``.kt`` file can declare MANY top-level symbols (classes, objects,
    interfaces, top-level funcs/vals/vars, typealiases), each with its own FQN
    ``<package>.<SymbolName>``.

So ``import com.ex.util.Helper`` names a DECLARATION, not a file. Resolution needs
a SYMBOL -> FILE index; the node unit is one node per declared FQN (see the
extractor / kotlin_nodes.py). Detection, graded here, is independent of that.

Detection path: RICH (``extract_keys``). Unlike Java (whose static imports license
two keys) Kotlin licenses exactly ONE key per non-wildcard import (the dotted FQN,
alias stripped) and NOTHING for a wildcard/on-demand ``import com.ex.*``. The RICH
path is used (rather than the SIMPLE query path) precisely because the wildcard
case must license nothing, and tree-sitter strips the ``.*`` off the captured
``qualified_identifier`` (it becomes just ``com.ex``) — indistinguishable from a
real ``import com.ex.Const`` at the node level. So the walker inspects the whole
``import`` node text/children to detect the trailing star and drop it.

Canonical key space: DOTTED FQN (e.g. ``com.ex.util.Helper``), identical to Java.
Both the oracle and the detector's emitted ``target_str`` live here. Resolution
(dotted FQN -> the file node that DECLARES the symbol) is the RESOLUTION axis
(fixtures + build_kotlin_file_nodes).

Extensions = {"kt"} only. Kotlin scripts (``.kts``) rarely declare importable
intra-repo symbols and are excluded from nodes (documented in the extractor).
"""
from __future__ import annotations

from typing import Optional, Set

from ..spec import LanguageSpec


def _load_grammar():
    import tree_sitter_kotlin
    from tree_sitter import Language
    return Language(tree_sitter_kotlin.language())


def _import_is_wildcard(node, raw: str) -> bool:
    """True if an ``import`` node is an on-demand/wildcard import (``a.b.*``).

    tree-sitter strips the ``.*`` off the ``qualified_identifier`` child, so the
    star surfaces either as a sibling ``*`` token under the ``import`` node or,
    defensively, as a trailing ``*`` in the raw node text.
    """
    if any(c.type == "*" for c in node.children):
        return True
    return raw.rstrip().endswith("*")


def _extract_keys(root, src_bytes: bytes) -> Set[str]:
    """Independent statement finder (tree-sitter) + candidate licensor.

    Walks ``import`` statement nodes. For each non-wildcard import, licenses the
    dotted FQN (the ``qualified_identifier`` child, which already excludes a
    trailing ``as Alias`` — tree-sitter parses the alias as separate ``as`` +
    ``identifier`` siblings). Wildcard imports license NOTHING (no single target
    file), matching the detector which drops them — so a dropped wildcard is not a
    recall miss.

    NOTE: the ``import`` keyword token is ALSO typed ``import`` in this grammar,
    but it is a leaf with no named children, so the ``named_children`` scan for a
    qualified_identifier/identifier naturally skips it.
    """
    keys: Set[str] = set()

    def text(node) -> str:
        return src_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")

    def walk(node):
        if node.type == "import":
            fqn_node = None
            for c in node.named_children:
                if c.type in ("qualified_identifier", "identifier"):
                    fqn_node = c
                    break
            if fqn_node is not None:
                raw = text(node)
                if not _import_is_wildcard(node, raw):
                    # Strip backtick keyword-escapes (`fun`, `object`, ...): the
                    # backticks are a lexical device, not part of the symbol
                    # identity, and the node model (declaration names) drops them,
                    # so the canonical key space is backtick-free on both sides.
                    fqn = text(fqn_node).strip().replace("`", "")
                    if fqn:
                        keys.add(fqn)
        for c in node.children:
            walk(c)

    walk(root)
    return keys


def _canonical_target(target: str) -> Optional[str]:
    """Detector-emitted target -> canonical dotted FQN key.

    The Kotlin detector emits the dotted FQN (alias already stripped, wildcard
    already dropped). Normalizes defensively: strips a trailing ``;`` (Kotlin
    imports have no semicolon, but be forgiving), converts a stray ``.kt`` file
    path back to dotted form, strips a trailing ``.*`` (wildcard), and strips a
    trailing `` as Alias`` so both sides share the oracle's key space.
    """
    t = target.strip().rstrip(";").strip().replace("`", "")
    if " as " in t:
        t = t.split(" as ", 1)[0].strip()
    if t.endswith(".kt"):
        t = t[: -len(".kt")].replace("/", ".")
    if t.endswith(".*"):
        t = t[:-2]
    return t or None


KOTLIN_SPEC = LanguageSpec(
    name="kotlin",
    extensions=frozenset({"kt"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    extract_keys=_extract_keys,
)
