"""
TypeScript file-node model — shared by the fixtures runner (and mirrors the
extractor data/typescript_graph_extractor/build_typescript_graph.py).

A TypeScript NODE is a source FILE keyed by its repo-relative path WITHOUT
extension (``src/util/helper``). ``.d.ts`` declaration files are EXCLUDED (they
declare types, are not real import targets, and would create phantom nodes). Both
``key`` and ``normed_identifier`` are the extension-less repo-relative path;
``raw_identifier`` is ``"fixture:<path-with-ext>"`` mirroring the Stack
``owner/repo:path`` shape so ``TypeScriptImportDetector.index_doc_span`` (which
takes the post-``:`` path and strips the extension) yields the matching key.
"""
from __future__ import annotations

from typing import List, Set

# Match the extension set the detector/spec recognize (longest-first).
_EXTS = (".d.ts", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs")


def _strip_ext(path: str) -> str:
    for ext in _EXTS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def build_typescript_file_nodes(files, extensions: Set[str]):
    """One node per .ts/.tsx file (excluding .d.ts), keyed by extension-less path."""
    from .fixtures import _FixtureNode  # lazy to avoid import cycle

    exts = {e.lstrip(".") for e in extensions}
    nodes = []
    for f in files:
        rel = f.relpath
        if rel.endswith(".d.ts"):
            continue  # declaration file, not an import target
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
