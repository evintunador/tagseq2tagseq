"""
JavaScript file-node model — shared by the fixtures runner (and mirrors the
extractor data/javascript_graph_extractor/build_javascript_graph.py).

A JavaScript NODE is a source FILE keyed by its repo-relative path WITHOUT
extension (``src/util/helper``). Both ``key`` and ``normed_identifier`` are the
extension-less repo-relative path; ``raw_identifier`` is ``"fixture:<path-with-ext>"``
mirroring the Stack ``owner/repo:path`` shape so
``JavaScriptImportDetector.index_doc_span`` (which takes the post-``:`` path and
strips the extension) yields the matching key.
"""
from __future__ import annotations

from typing import List, Set

# Match the extension set the detector/spec recognize (longest-first).
_EXTS = (".jsx", ".mjs", ".cjs", ".js")


def _strip_ext(path: str) -> str:
    for ext in _EXTS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def build_javascript_file_nodes(files, extensions: Set[str]):
    """One node per .js/.jsx/.mjs/.cjs file, keyed by extension-less path."""
    from .fixtures import _FixtureNode  # lazy to avoid import cycle

    exts = {e.lstrip(".") for e in extensions}
    nodes = []
    for f in files:
        rel = f.relpath
        ext = rel.rsplit(".", 1)[-1] if "." in rel else ""
        if ext not in exts:
            continue
        key = _strip_ext(rel)
        raw = f"fixture:{rel}"
        nodes.append(_FixtureNode(
            key=key, raw_identifier=raw, normed_identifier=key,
            content=f.content, relpath=rel,
        ))
    return nodes
