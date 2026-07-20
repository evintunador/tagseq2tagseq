"""
Rust module-path node model — shared by the fixtures runner and mirrors the
extractor data/rust_graph_extractor/build_rust_graph.py.

A Rust NODE is a source FILE keyed by its crate-relative MODULE PATH
(``crate::net::tcp``), assigned by WALKING the mod-declaration tree from the crate
root (``src/lib.rs`` / ``src/main.rs`` / ``src/bin/*.rs``). This is the resolution
core (design §4): a file's identity is not a path convention but the chain of
``mod`` decls to it.

`key` and the scoring space are the bare module path (matches edges.json).
`raw_identifier` is repo-qualified (``fixture@crate::net::tcp``), mirroring the
Stack ``owner/repo@module_path`` shape, so RustImportDetector.index_doc_span (which
strips the ``@`` repo prefix) yields the bare module path an emitted target matches.
"""
from __future__ import annotations

from typing import List, Set

from data.rust_graph_extractor.mod_tree import RustModParser, build_module_paths


def build_rust_module_nodes(files, extensions: Set[str]):
    """One node per reachable ``.rs`` file, keyed by its walked module path."""
    from .fixtures import _FixtureNode  # lazy to avoid import cycle

    exts = {e.lstrip(".") for e in extensions}
    src_files = [(f.relpath, f.content) for f in files
                 if (f.relpath.rsplit(".", 1)[-1] if "." in f.relpath else "") in exts]
    module_paths = build_module_paths(src_files, RustModParser())
    content_by_path = {rp: c for rp, c in src_files}

    nodes = []
    seen = set()
    for relpath, mp in sorted(module_paths.items()):
        if mp in seen:
            continue
        seen.add(mp)
        raw = f"fixture@{mp}"
        nodes.append(_FixtureNode(
            key=mp, raw_identifier=raw, normed_identifier=raw,
            content=content_by_path[relpath], relpath=relpath,
        ))
    return nodes
