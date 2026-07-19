"""
Java dependency-graph builder for The Stack (Java subset).

A Java NODE is a source FILE, keyed by its fully-qualified type name (FQN), e.g.
``com.google.gson.Gson``. This is the file-node model (like Python), and the FQN
is what an ``import`` names — so resolution is EXACT string match on the FQN, the
same key ``JavaImportDetector`` emits and ``index_doc_span`` returns.

Deriving the FQN (the Java analogue of Go's module inference):
    A file with ``package com.example;`` whose class is ``Foo`` has FQN
    ``com.example.Foo``, regardless of the source-root prefix (``src/main/java/``,
    etc.). So FQN = ``<package>.<ClassNameFromFilename>`` — read directly from the
    ``package`` declaration + filename. No inference/guessing needed (contrast Go).
    Files with no package declaration (default package) are keyed by bare class
    name; they rarely have unique names across repos, so they are only linked
    WITHIN a repo (same as all edges here).

Edges: an import ``com.example.Bar`` resolves to node ``com.example.Bar`` if that
node exists in the SAME repo. Static imports (``import static a.b.C.m``) resolve
to the enclosing type ``a.b.C``. On-demand (``import a.b.*``) is a package, not a
type — skipped (no file node).

Detection uses tree-sitter (build-time engine); the runtime JavaImportDetector
re-derives the same imports from tokens and is graded against the same oracle.

Outputs graph.jsonl + content.jsonl (one file-node each), consumed by
data.pretokenize_go-style flow via a Java content source.

Usage:
    python -m data.java_graph_extractor.build_java_graph \\
        raw/java/sample_java.jsonl -o graphs/java --min-links 1
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)


class _JavaParser:
    """tree-sitter Java parser + import-extraction (build-time engine)."""

    _QUERY = r"""
    (import_declaration (scoped_identifier) @mod)
    (import_declaration (identifier) @mod)
    """

    def __init__(self):
        import tree_sitter_java
        from tree_sitter import Language, Parser, Query, QueryCursor
        self._lang = Language(tree_sitter_java.language())
        self._parser = Parser(self._lang)
        self._query = Query(self._lang, self._QUERY)
        self._Cursor = QueryCursor

    def imports(self, source: str) -> List[Tuple[str, bool]]:
        """Return (fqn, is_static) for each import. Static detected by text scan
        of the import line (the grammar nests static under the same node)."""
        src = source.encode("utf-8", errors="replace")
        tree = self._parser.parse(src)
        caps = self._Cursor(self._query).captures(tree.root_node)
        out: List[Tuple[str, bool]] = []
        for nodes in caps.values():
            for n in nodes:
                name = src[n.start_byte:n.end_byte].decode("utf-8", "replace").strip()
                # find the enclosing import_declaration to check for 'static'
                decl = n
                while decl is not None and decl.type != "import_declaration":
                    decl = decl.parent
                is_static = False
                is_star = False
                if decl is not None:
                    for c in decl.children:
                        if c.type == "static":
                            is_static = True
                        if c.type == "asterisk":
                            is_star = True
                if name and not is_star:
                    out.append((name, is_static))
        return out


def _fqn_from(package: str, file_path: str) -> Optional[str]:
    """FQN of the type defined by file_path with the given package declaration.

    Uses the filename stem as the public type name (Java convention: one public
    top-level class per file, named after the file). Returns None for non-class
    files (module-info.java, package-info.java).
    """
    fname = file_path.rsplit("/", 1)[-1]
    if not fname.endswith(".java"):
        return None
    stem = fname[: -len(".java")]
    if stem in ("module-info", "package-info"):
        return None
    return f"{package}.{stem}" if package else stem


def build_repo_nodes(
    repo_files: List[Tuple[str, str]],
    parser: _JavaParser,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Build FQN-keyed file nodes + contents for ONE repo, with intra-repo edges."""
    nodes: Dict[str, dict] = {}
    contents: Dict[str, str] = {}
    imports_by_fqn: Dict[str, List[Tuple[str, bool]]] = {}

    for path, content in repo_files:
        if not path.endswith(".java") or path.endswith("Test.java"):
            continue
        m = _PACKAGE_RE.search(content)
        package = m.group(1) if m else ""
        fqn = _fqn_from(package, path)
        if fqn is None or fqn in nodes:
            continue
        contents[fqn] = content
        imps = parser.imports(content)
        imports_by_fqn[fqn] = imps
        nodes[fqn] = {
            "normed_identifier": fqn,
            "raw_identifier": fqn,
            "src_path": path,
            "char_count": len(content),
            "import_count": len(imps),
            "outgoing": [],
            "incoming": [],
            "links_in_repo": 0,
        }

    node_ids = set(nodes)
    for fqn, imps in imports_by_fqn.items():
        outgoing: Set[str] = set()
        for name, is_static in imps:
            # candidate target types in this repo's key space
            cands = [name]
            if is_static and "." in name:
                cands.append(name.rsplit(".", 1)[0])  # enclosing type
            for c in cands:
                if c in node_ids and c != fqn:
                    outgoing.add(c)
        nodes[fqn]["outgoing"] = sorted(outgoing)
    for src_fqn, data in nodes.items():
        for tgt in data["outgoing"]:
            nodes[tgt]["incoming"].append(src_fqn)
    for data in nodes.values():
        data["incoming"] = sorted(set(data["incoming"]))
        data["links_in_repo"] = len(data["outgoing"]) + len(data["incoming"])

    return nodes, contents


def build(jsonl_path: Path, out_dir: Path, min_links: int = 1) -> dict:
    parser = _JavaParser()
    repos: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_records = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            repo = rec.get("max_stars_repo_name")
            path = rec.get("max_stars_repo_path")
            content = rec.get("content")
            if not (repo and path and content and path.endswith(".java")):
                continue
            repos[repo].append((path, content))
            n_records += 1
    logger.info("Read %d Java records across %d repos", n_records, len(repos))

    all_nodes: Dict[str, dict] = {}
    all_contents: Dict[str, str] = {}
    repos_kept = 0
    nodes_before = 0
    for repo, files in repos.items():
        if len(files) < 2:
            continue
        nodes, contents = build_repo_nodes(files, parser)
        if not nodes:
            continue
        nodes_before += len(nodes)
        kept = {k: n for k, n in nodes.items() if n["links_in_repo"] >= min_links}
        if not kept:
            continue
        kept_ids = set(kept)
        for n in kept.values():
            n["outgoing"] = [t for t in n["outgoing"] if t in kept_ids]
            n["incoming"] = [s for s in n["incoming"] if s in kept_ids]
            n["links_in_repo"] = len(n["outgoing"]) + len(n["incoming"])
        repos_kept += 1
        for k, n in kept.items():
            # FQNs are globally unique enough, but guard against cross-repo key
            # collisions (same FQN in two repos): first repo wins, later dropped.
            if k not in all_nodes:
                all_nodes[k] = n
                all_contents[k] = contents[k]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "graph.jsonl", "w", encoding="utf-8") as gf:
        for n in all_nodes.values():
            gf.write(json.dumps(n) + "\n")
    with open(out_dir / "content.jsonl", "w", encoding="utf-8") as cf:
        for k, content in all_contents.items():
            cf.write(json.dumps({"normed_identifier": k, "content": content}) + "\n")

    n_edges = sum(len(n["outgoing"]) for n in all_nodes.values())
    stats = {
        "records_read": n_records,
        "repos_total": len(repos),
        "repos_kept": repos_kept,
        "nodes_before_filter": nodes_before,
        "graph_nodes_final": len(all_nodes),
        "graph_edges_internal_final": n_edges,
        "avg_out_degree": (n_edges / len(all_nodes)) if all_nodes else 0.0,
        "min_links": min_links,
    }
    with open(out_dir / "graph_stats.json", "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    logger.info("Java graph: %d nodes, %d edges (%.2f avg out-degree) from %d repos",
                stats["graph_nodes_final"], n_edges, stats["avg_out_degree"], repos_kept)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl_file", type=Path)
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--min-links", type=int, default=1)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s: %(message)s")
    stats = build(args.jsonl_file, args.out_dir, args.min_links)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
