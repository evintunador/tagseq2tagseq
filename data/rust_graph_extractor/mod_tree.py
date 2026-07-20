"""
Rust module-tree walker — the RESOLUTION engine shared by the graph extractor and
the harness fixtures node builder.

The core problem (design §4, "Rust"): a Rust file's identity is its MODULE PATH
(``crate::net::tcp``), which is NOT a path convention but the chain of ``mod``
declarations from the crate root down to the file. So resolving ``use crate::a::b``
requires first WALKING the mod-declaration tree to assign every ``.rs`` file its
module path, THEN matching ``use`` targets against those paths.

Crate root: ``src/lib.rs`` or ``src/main.rs`` (or ``src/bin/*.rs``). Its module
path is ``crate``. A file-backed ``mod foo;`` (semicolon, NO body) in a module whose
"module directory" is ``D`` resolves to a SIBLING file ``D/foo.rs`` OR
``D/foo/mod.rs`` and gets module path ``<parent>::foo``. The module directory of:
  * a crate root ``src/lib.rs``      -> ``src``      (mod foo -> src/foo.rs)
  * a ``mod.rs`` file ``src/a/mod.rs`` -> ``src/a``   (mod foo -> src/a/foo.rs)
  * a regular file ``src/a.rs``      -> ``src/a``    (2018 edition: mod foo -> src/a/foo.rs)

Inline ``mod foo { ... }`` (WITH a body) creates NO new file node -- it extends the
current file's namespace. A file-backed ``mod bar;`` nested inside an inline
``mod foo { ... }`` resolves to ``<module_dir>/foo/bar.rs`` with path
``<parent>::foo::bar``. We track that inline nesting.

The Stack has no ``Cargo.toml``, so we cannot read the crate's published name; per
design we use ``crate`` as the fixed root for every crate (valid in the 2018+
edition). Multiple crate roots in one repo share the ``crate::`` namespace (rare in
The Stack, where most repos are one crate); collisions are first-wins.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def _load_parser():
    import tree_sitter_rust
    from tree_sitter import Language, Parser
    return Parser(Language(tree_sitter_rust.language()))


class RustModParser:
    """tree-sitter Rust parser exposing file-backed mod decls + use-tree leaves."""

    def __init__(self):
        self._parser = _load_parser()

    # -- mod-tree side -----------------------------------------------------
    def file_backed_mods(self, source: str) -> List[Tuple[List[str], str]]:
        """Return (inline_prefix, mod_name) for each file-backed ``mod X;``.

        ``inline_prefix`` is the list of enclosing INLINE mod names (usually empty).
        Inline ``mod X { ... }`` (has a declaration_list) is NOT returned as a
        file-backed decl; we only descend into it to find nested file-backed mods.
        """
        src = source.encode("utf-8", "replace")
        tree = self._parser.parse(src)
        out: List[Tuple[List[str], str]] = []

        def name_of(node) -> Optional[str]:
            for c in node.named_children:
                if c.type == "identifier":
                    return src[c.start_byte:c.end_byte].decode("utf-8", "replace")
            return None

        def walk(node, prefix: List[str]):
            for c in node.children:
                if c.type == "mod_item":
                    body = next((g for g in c.children if g.type == "declaration_list"), None)
                    nm = name_of(c)
                    if body is None:
                        if nm:
                            out.append((list(prefix), nm))
                    else:
                        # inline module: descend with extended prefix
                        walk(body, prefix + ([nm] if nm else []))
                else:
                    walk(c, prefix)

        walk(tree.root_node, [])
        return out


def _module_dir(relpath: str) -> str:
    """The directory that a file's file-backed submodules live in."""
    fname = relpath.rsplit("/", 1)[-1]
    parent = relpath.rsplit("/", 1)[0] if "/" in relpath else ""
    stem = fname[:-3] if fname.endswith(".rs") else fname
    # mod.rs / lib.rs / main.rs: submodules live in the SAME directory.
    if stem in ("mod", "lib", "main"):
        return parent
    # a regular file src/foo.rs (2018 edition): submodules live under src/foo/.
    return f"{parent}/{stem}" if parent else stem


def _is_crate_root(relpath: str) -> bool:
    """True for ``src/lib.rs``, ``src/main.rs`` (any depth), or ``src/bin/*.rs``."""
    parts = relpath.split("/")
    fname = parts[-1]
    if len(parts) >= 2 and parts[-2] == "src" and fname in ("lib.rs", "main.rs"):
        return True
    if len(parts) >= 3 and parts[-3] == "src" and parts[-2] == "bin" and fname.endswith(".rs"):
        return True
    # top-level lib.rs / main.rs (no src/) — some Stack fragments
    if len(parts) == 1 and fname in ("lib.rs", "main.rs"):
        return True
    return False


def build_module_paths(
    files: List[Tuple[str, str]],
    parser: Optional[RustModParser] = None,
) -> Dict[str, str]:
    """Assign each reachable ``.rs`` file its crate-relative module path.

    Args:
        files: list of (relpath, content) for one repo/crate.
        parser: optional shared RustModParser.

    Returns:
        dict relpath -> module_path (``crate``, ``crate::a``, ``crate::a::b``...).
        Files not reachable from a crate root are omitted (no module path).
    """
    parser = parser or RustModParser()
    by_path = {rp: c for rp, c in files if rp.endswith(".rs")}
    assigned: Dict[str, str] = {}

    # seed crate roots
    worklist: List[str] = []
    for rp in by_path:
        if _is_crate_root(rp):
            assigned[rp] = "crate"
            worklist.append(rp)

    def resolve_child(module_dir: str, inline_prefix: List[str], name: str) -> Optional[str]:
        base = module_dir
        for p in inline_prefix:
            base = f"{base}/{p}" if base else p
        cand_flat = f"{base}/{name}.rs" if base else f"{name}.rs"
        cand_mod = f"{base}/{name}/mod.rs" if base else f"{name}/mod.rs"
        if cand_flat in by_path:
            return cand_flat
        if cand_mod in by_path:
            return cand_mod
        return None

    while worklist:
        rp = worklist.pop()
        mp = assigned[rp]
        module_dir = _module_dir(rp)
        for inline_prefix, name in parser.file_backed_mods(by_path[rp]):
            child = resolve_child(module_dir, inline_prefix, name)
            if child is None or child in assigned:
                continue
            child_mp = "::".join([mp] + inline_prefix + [name])
            assigned[child] = child_mp
            worklist.append(child)

    return assigned
