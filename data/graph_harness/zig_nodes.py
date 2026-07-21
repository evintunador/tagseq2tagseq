"""
Zig file-node model — shared by the fixtures runner (and mirrors the extractor
data/zig_graph_extractor/build_zig_graph.py).

A Zig NODE is a source FILE keyed by its repo-relative path WITH the ``.zig``
extension kept (``src/util/helper.zig``). Zig imports are explicit relative file
paths that resolve to a full ``.zig`` path, so — unlike TypeScript, where nodes are
keyed extension-less and the extension is inferred — the Zig node key keeps
``.zig`` and matches the resolved import path exactly (no candidate expansion).

Both ``key`` and ``normed_identifier`` are the repo-relative ``.zig`` path;
``raw_identifier`` is ``"fixture:<path>"`` mirroring the Stack ``owner/repo:path``
shape so ``ZigImportDetector.index_doc_span`` (which takes the post-``:`` path)
yields the matching key.
"""
from __future__ import annotations

from typing import Set


def build_zig_file_nodes(files, extensions: Set[str]):
    """One node per .zig file, keyed by its repo-relative path (extension kept)."""
    from .fixtures import _FixtureNode  # lazy to avoid import cycle

    exts = {e.lstrip(".") for e in extensions}
    nodes = []
    for f in files:
        rel = f.relpath
        ext = rel.rsplit(".", 1)[-1] if "." in rel else ""
        if ext not in exts:
            continue
        key = rel  # keep the .zig extension
        raw = f"fixture:{rel}"
        nodes.append(_FixtureNode(
            key=key, raw_identifier=raw, normed_identifier=key,
            content=f.content, relpath=rel,
        ))
    return nodes
