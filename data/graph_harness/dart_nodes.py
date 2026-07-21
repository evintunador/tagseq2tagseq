"""
Dart file-node model — shared by the fixtures runner (and mirrors the extractor
data/dart_graph_extractor/build_dart_graph.py).

A Dart NODE is a source FILE keyed by its repo-relative path INCLUDING the
``.dart`` extension (``lib/models/user.dart``). The extension is kept (unlike
TypeScript's extension-less keys) because Dart import URIs always carry ``.dart``
explicitly, so target keys and node keys align only if both keep it. Both ``key``
and ``normed_identifier`` are the repo-relative path with extension;
``raw_identifier`` is ``"fixture:<path>"`` mirroring the Stack ``owner/repo:path``
shape so ``DartImportDetector.index_doc_span`` (which takes the post-``:`` path)
yields the matching key.
"""
from __future__ import annotations

from typing import Set


def build_dart_file_nodes(files, extensions: Set[str]):
    """One node per .dart file, keyed by repo-relative path WITH extension."""
    from .fixtures import _FixtureNode  # lazy to avoid import cycle

    exts = {e.lstrip(".") for e in extensions}
    nodes = []
    for f in files:
        rel = f.relpath
        ext = rel.rsplit(".", 1)[-1] if "." in rel else ""
        if ext not in exts:
            continue
        key = rel
        raw = f"fixture:{rel}"
        nodes.append(_FixtureNode(
            key=key, raw_identifier=raw, normed_identifier=key,
            content=f.content, relpath=rel,
        ))
    return nodes
