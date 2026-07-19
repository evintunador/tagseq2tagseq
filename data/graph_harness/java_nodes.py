"""
Java file-node model — shared by the fixtures runner (and mirrors the extractor
data/java_graph_extractor/build_java_graph.py).

A Java NODE is a source FILE keyed by its fully-qualified type name
(FQN = <package>.<ClassFromFilename>), read from the `package` declaration and the
filename. Both `key` and `raw_identifier` are the FQN, so JavaImportDetector's
index_doc_span (which returns raw_identifier's post-':' path as a dotted name, or
the raw string when there's no ':') matches an emitted import FQN by exact string.
"""
from __future__ import annotations

import re
from typing import List, Set

_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)


def build_java_file_nodes(files, extensions: Set[str]):
    """One node per .java file (excluding module-info/package-info), keyed by FQN."""
    from .fixtures import _FixtureNode  # lazy to avoid import cycle

    nodes = []
    for f in files:
        if not f.relpath.endswith(".java"):
            continue
        stem = f.relpath.rsplit("/", 1)[-1][: -len(".java")]
        if stem in ("module-info", "package-info"):
            continue
        m = _PACKAGE_RE.search(f.content)
        package = m.group(1) if m else ""
        fqn = f"{package}.{stem}" if package else stem
        nodes.append(_FixtureNode(
            key=fqn, raw_identifier=fqn, normed_identifier=fqn,
            content=f.content, relpath=f.relpath,
        ))
    return nodes
