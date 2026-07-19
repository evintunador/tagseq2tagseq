"""
Java LanguageSpec for the conformance harness.

Java maps cleanly onto the file-node model (like Python): a fully-qualified name
in an import (`com.google.gson.internal.Excluder`) maps deterministically to a
file path (`.../com/google/gson/internal/Excluder.java`), and `package` ==
directory. So a Java NODE is a file, keyed by its source-root-relative path.

Detection path: RICH (`extract_keys`). Like Python's from-imports, one Java
STATIC import licenses more than one legitimate detection key: `import static
a.b.C.method;` names the member `a.b.C.method` AND its enclosing type `a.b.C`
(which is the real file the member lives in, so the detector rightly emits it).
If the oracle only licensed the member, that correct enclosing-type candidate
would be a false positive. So the oracle walks import declarations and licenses
the SAME key set the detector produces (member + enclosing type for static;
the type for a normal import; nothing for on-demand `.*`, which is a package).
tree-sitter (independent) still supplies the STATEMENT set, so detection recall
is graded independently of the detector's regex.

Canonical key space: DOTTED type FQN (e.g. `com.google.gson.Gson`). Both the
oracle and the detector's emitted target_str live here. Resolution (dotted FQN ->
file node under a source root) is the RESOLUTION axis (fixtures + build_java_file_nodes).
"""
from __future__ import annotations

from typing import Optional, Set

from ..spec import LanguageSpec


def _load_grammar():
    import tree_sitter_java
    from tree_sitter import Language
    return Language(tree_sitter_java.language())


def _extract_keys(root, src_bytes: bytes) -> Set[str]:
    """Independent statement finder (tree-sitter) + candidate licensor.

    Walks import_declaration nodes. For each, licenses the dotted FQN; if it's a
    `static` import, also licenses the enclosing type (drop the trailing member).
    On-demand (`.*`) imports license the bare package (rarely a file, but kept for
    symmetry with the detector, which emits nothing for them — so it won't hurt
    precision and can only help recall if a package-info node exists). Skips
    nothing by heuristic — the detector's own choices define the shared space.
    """
    keys: Set[str] = set()

    def text(node) -> str:
        return src_bytes[node.start_byte:node.end_byte].decode("utf-8", "replace")

    def walk(node):
        if node.type == "import_declaration":
            is_static = any(c.type == "static" for c in node.children)
            is_star = any(c.type == "asterisk" for c in node.children)
            name = None
            for c in node.named_children:
                if c.type in ("scoped_identifier", "identifier"):
                    name = text(c)
                    break
            if name:
                dotted = name.strip()
                if is_star:
                    keys.add(dotted)  # package
                else:
                    keys.add(dotted)  # full FQN (member for static, type otherwise)
                    if is_static and "." in dotted:
                        keys.add(dotted.rsplit(".", 1)[0])  # enclosing type
        for c in node.children:
            walk(c)

    walk(root)
    return keys


def _canonical_target(target: str) -> Optional[str]:
    """Detector-emitted target -> canonical dotted key.

    The Java detector emits the dotted FQN (optionally a file path). Accept either:
    a path like `com/google/gson/Gson.java` is converted back to dotted form so it
    shares the oracle's key space.
    """
    t = target.strip().rstrip(";").strip()
    if t.endswith(".java"):
        t = t[: -len(".java")].replace("/", ".")
    if t.endswith(".*"):
        t = t[:-2]
    return t or None


JAVA_SPEC = LanguageSpec(
    name="java",
    extensions=frozenset({"java"}),
    grammar_loader=_load_grammar,
    canonical_target=_canonical_target,
    extract_keys=_extract_keys,
)
