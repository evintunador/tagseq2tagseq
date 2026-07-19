"""
Go package-node model — shared by the fixtures runner and (eventually) the Go
graph extractor.

A Go corpus NODE is a PACKAGE: all non-test ``.go`` files in one directory,
concatenated, keyed by the package's full import path ``<module>/<pkgdir>``. This
is the pilot's resolved node-unit decision (design doc §Go pilot): Go imports
reference directories, never files, and same-directory files share a package
without importing each other.

`build_go_package_nodes` reads the module path from ``go.mod`` and groups source
files by directory into package nodes. Both `key` and `raw_identifier` are the
package's import path, so `GoImportDetector.index_doc_span` (which returns
raw_identifier unchanged) matches an emitted import path exactly.
"""
from __future__ import annotations

import re
from typing import List, Set

# import here would be circular at module import if fixtures imported this at top;
# _FixtureNode/_FixtureFile are simple dataclasses, imported lazily in the builder.

_MODULE_RE = re.compile(r"^module\s+(\S+)", re.MULTILINE)


def _module_path(files) -> str:
    """Read the module path from a go.mod among the files (fallback: 'module')."""
    for f in files:
        if f.relpath == "go.mod" or f.relpath.endswith("/go.mod"):
            m = _MODULE_RE.search(f.content)
            if m:
                return m.group(1)
    return "module"


def build_go_package_nodes(files, extensions: Set[str]):
    """Group .go files into one package-node per directory.

    Args:
        files: list of _FixtureFile(relpath, content).
        extensions: source extensions (expects {"go"}).

    Returns:
        list of _FixtureNode, one per package directory, keyed by import path.
    """
    from .fixtures import _FixtureNode  # lazy to avoid import cycle

    module = _module_path(files)
    exts = {e.lstrip(".") for e in extensions}

    # dir -> list of (relpath, content) for non-test source files
    by_dir: dict[str, list] = {}
    for f in files:
        ext = f.relpath.rsplit(".", 1)[-1] if "." in f.relpath else ""
        if ext not in exts:
            continue
        if f.relpath.endswith("_test.go"):
            continue  # test files are not part of the importable package graph
        d = f.relpath.rsplit("/", 1)[0] if "/" in f.relpath else ""
        by_dir.setdefault(d, []).append(f)

    nodes = []
    for d in sorted(by_dir):
        pkg_files = sorted(by_dir[d], key=lambda x: x.relpath)
        # import path: module + "/" + dir (dir "" -> the module root package)
        import_path = module if d == "" else f"{module}/{d}"
        content = "\n\n".join(f.content for f in pkg_files)
        nodes.append(_FixtureNode(
            key=import_path,
            raw_identifier=import_path,
            normed_identifier=import_path,
            content=content,
            relpath=pkg_files[0].relpath,
        ))
    return nodes
