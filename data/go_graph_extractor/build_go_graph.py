"""
Go dependency-graph builder for The Stack (Go subset).

Reads a downloaded Stack Go JSONL (records with max_stars_repo_name /
max_stars_repo_path / content, ext == "go") and produces a PACKAGE-level
dependency graph plus the concatenated package contents:

    graph.jsonl    — one node per package (a directory of .go files), with
                     outgoing/incoming import edges to other in-repo packages.
    content.jsonl  — one {"normed_identifier", "content"} per package (all its
                     non-test .go files concatenated), consumed by pretokenize_go.

Why package-nodes (not file-nodes): Go imports reference a *package* = a
directory; files in one directory share a `package` and never import each other.
So the natural graph node is the package. See docs/multilang_code_datasets_DESIGN.md
(Go pilot) and data/graph_harness/go_nodes.py (the same model, shared).

Node identity / resolution:
    normed_identifier = raw_identifier = the package's FULL import path
        "<module>/<pkgdir>"   (module read from the repo's go.mod)
    An import "github.com/x/y/pkg" resolves by EXACT match to that node's id —
    the same key GoImportDetector.index_doc_span returns. No prefix stripping,
    no candidate expansion (contrast the Python builder).

Detection uses tree-sitter (the build-time extractor engine, per the design
decision). The runtime GoImportDetector re-derives the same imports from tokens
and is graded against the same tree-sitter oracle; this builder and that detector
are independent implementations that must agree (harness enforces it).

Single-repo vs multi-repo: edges are only created BETWEEN packages of the SAME
repo (module). The identifier is globally unique either way, so relaxing to
multi-repo later needs no id change — only removing the same-repo edge guard and
the precompute dispatcher fix (design §7).

Usage:
    python -m data.go_graph_extractor.build_go_graph \\
        data/go_graph_extractor/sample_go.jsonl \\
        -o data/go_graph_extractor \\
        --min-package-links 1
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

_MODULE_RE = re.compile(r"^module\s+(\S+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Tree-sitter import extraction (build-time engine)
# ---------------------------------------------------------------------------

_GO_IMPORT_QUERY = r"""
(import_spec path: (interpreted_string_literal) @p)
(import_spec path: (raw_string_literal) @p)
"""


class _GoParser:
    """Lazily-constructed tree-sitter Go parser + import query."""

    def __init__(self):
        import tree_sitter_go
        from tree_sitter import Language, Parser, Query, QueryCursor
        self._lang = Language(tree_sitter_go.language())
        self._parser = Parser(self._lang)
        self._query = Query(self._lang, _GO_IMPORT_QUERY)
        self._Cursor = QueryCursor

    def imports(self, source: str) -> List[str]:
        src = source.encode("utf-8", errors="replace")
        tree = self._parser.parse(src)
        caps = self._Cursor(self._query).captures(tree.root_node)
        out: List[str] = []
        for nodes in caps.values():
            for n in nodes:
                raw = src[n.start_byte:n.end_byte].decode("utf-8", "replace").strip()
                if len(raw) >= 2 and raw[0] in "\"`" and raw[-1] in "\"`":
                    raw = raw[1:-1]
                if raw:
                    out.append(raw)
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _package_dir(file_path: str) -> str:
    """Directory portion of a repo-relative file path ('' for repo root)."""
    return file_path.rsplit("/", 1)[0] if "/" in file_path else ""


def _import_path(module: str, pkg_dir: str) -> str:
    return module if pkg_dir == "" else f"{module}/{pkg_dir}"


def _is_vendored(path: str) -> bool:
    """True for third-party code copied into a vendor/ tree (not the repo's own)."""
    return path == "vendor" or path.startswith("vendor/") or "/vendor/" in path


def infer_module_path(
    pkg_dirs: Set[str],
    all_imports: Set[str],
    go_mod_module: Optional[str] = None,
) -> Optional[str]:
    """Infer the repo's Go module path.

    The Stack (dedup) contains NO go.mod files (filtered to ext=="go"), so the
    module path — needed to form globally-unique package import paths — must be
    inferred. Strategy (robust to custom hosts like k8s.io, gopkg.in; needs no
    go.mod): find the prefix P such that ``P/<pkgdir>`` appears as an import for
    the most of the repo's OWN package directories. Because P is only established
    from imports that actually reference this repo's directories, it is
    self-validating: a repo whose imports never reference its own subpackages
    yields None (no module, no intra-repo edges — correct for single-package repos).

    If ``go_mod_module`` is provided (a future dataset that keeps go.mod), it wins.
    """
    if go_mod_module:
        return go_mod_module
    from collections import Counter
    votes: Counter = Counter()
    subdirs = [d for d in pkg_dirs if d]
    for imp in all_imports:
        for d in subdirs:
            if imp == d or imp.endswith("/" + d):
                prefix = imp[: len(imp) - len(d)].rstrip("/")
                # a real module prefix is a host-qualified path: its first segment
                # is a domain (contains a dot, no '..', not a relative marker).
                # This rejects stdlib-like short imports and dodges './..' paths
                # that survive from relative-import artifacts.
                host = prefix.split("/", 1)[0]
                if (prefix and "." in host and ".." not in prefix
                        and not prefix.startswith(".")):
                    votes[prefix] += 1
    if not votes:
        return None
    return votes.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Core build (per-repo, package-level)
# ---------------------------------------------------------------------------

def build_repo_packages(
    repo_files: List[Tuple[str, str]],
    parser: _GoParser,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Build package nodes + contents for ONE repo.

    Args:
        repo_files: list of (file_path, content) for every .go file in the repo.
        parser: shared _GoParser.

    Returns:
        (nodes, contents) where
          nodes:    import_path -> node dict (schema below), edges filled.
          contents: import_path -> concatenated non-test .go source.

    A node with no in-repo edges is still returned; the caller applies the
    min-package-links filter so stats can see the pre-filter distribution.
    """
    # Optional go.mod (absent in The Stack dedup; kept for future datasets).
    go_mod_module = None
    for path, content in repo_files:
        if path == "go.mod" or path.endswith("/go.mod"):
            m = _MODULE_RE.search(content)
            if m:
                go_mod_module = m.group(1)
                break

    # group non-test, non-vendored .go files by package dir; detect imports.
    pkg_files: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for path, content in repo_files:
        if not path.endswith(".go") or path.endswith("_test.go"):
            continue
        if _is_vendored(path):
            continue  # vendored third-party copies are not this repo's packages
        pkg_files[_package_dir(path)].append((path, content))

    if len(pkg_files) < 1:
        return {}, {}

    # first pass: concat each package + detect its imports (needed for inference)
    pkg_concat: Dict[str, str] = {}
    imports_by_dir: Dict[str, Set[str]] = {}
    all_imports: Set[str] = set()
    for pkg_dir, files in pkg_files.items():
        files_sorted = sorted(files, key=lambda x: x[0])
        concat = "\n\n".join(c for _p, c in files_sorted)
        pkg_concat[pkg_dir] = concat
        imps = set(parser.imports(concat))
        imports_by_dir[pkg_dir] = imps
        all_imports |= imps

    # infer the module path from the repo's own imports vs. its directory layout.
    module = infer_module_path(set(pkg_files.keys()), all_imports, go_mod_module)
    if module is None:
        # Can't form globally-unique import paths (e.g. a single-package repo with
        # no self-references). Skip — it would contribute only edgeless nodes.
        return {}, {}

    nodes: Dict[str, dict] = {}
    contents: Dict[str, str] = {}
    imports_by_pkg: Dict[str, Set[str]] = {}
    for pkg_dir, files in pkg_files.items():
        ip = _import_path(module, pkg_dir)
        concat = pkg_concat[pkg_dir]
        contents[ip] = concat
        imports_by_pkg[ip] = imports_by_dir[pkg_dir]
        nodes[ip] = {
            "normed_identifier": ip,
            "raw_identifier": ip,
            "n_files": len(files),
            "char_count": len(concat),
            "import_count": len(imports_by_dir[pkg_dir]),
            "outgoing": [],
            "incoming": [],
            "links_in_repo": 0,
        }

    node_ids = set(nodes.keys())
    # outgoing: an import that exactly matches another in-repo package id.
    for ip, imps in imports_by_pkg.items():
        outgoing = {t for t in imps if t in node_ids and t != ip}
        nodes[ip]["outgoing"] = sorted(outgoing)
    # incoming (O(E))
    for src_ip, data in nodes.items():
        for tgt_ip in data["outgoing"]:
            nodes[tgt_ip]["incoming"].append(src_ip)
    for data in nodes.values():
        data["incoming"] = sorted(set(data["incoming"]))
        data["links_in_repo"] = len(data["outgoing"]) + len(data["incoming"])

    return nodes, contents


# ---------------------------------------------------------------------------
# Streaming driver
# ---------------------------------------------------------------------------

def build(
    jsonl_path: Path,
    out_dir: Path,
    min_package_links: int = 1,
) -> dict:
    """Stream a Stack Go JSONL, build the package graph + contents, write outputs.

    Records are grouped by repo in memory by streaming once and bucketing on
    repo name. (Stack Go is far smaller than Python; a single pass with a dict of
    repos is fine. If memory becomes a concern, switch to the 256-bucket
    hash-partition approach the Python builder uses.)

    Returns a stats dict (also written to graph_stats.json).
    """
    parser = _GoParser()
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
            if not (repo and path and content):
                continue
            # keep go source + go.mod (needed for module path)
            if not (path.endswith(".go") or path == "go.mod" or path.endswith("/go.mod")):
                continue
            repos[repo].append((path, content))
            n_records += 1

    logger.info("Read %d Go records across %d repos", n_records, len(repos))

    all_nodes: Dict[str, dict] = {}
    all_contents: Dict[str, str] = {}
    repos_kept = 0
    pkgs_before_filter = 0

    for repo, files in repos.items():
        if len(files) < 2:
            continue
        nodes, contents = build_repo_packages(files, parser)
        if not nodes:
            continue
        pkgs_before_filter += len(nodes)
        # keep only packages meeting the link threshold (edge-bearing packages),
        # matching the Python builder's links_in_repo >= 2 spirit but tunable.
        kept = {ip: n for ip, n in nodes.items() if n["links_in_repo"] >= min_package_links}
        if not kept:
            continue
        # prune edges to dropped packages so no dangling edges are written
        kept_ids = set(kept.keys())
        for n in kept.values():
            n["outgoing"] = [t for t in n["outgoing"] if t in kept_ids]
            n["incoming"] = [s for s in n["incoming"] if s in kept_ids]
            n["links_in_repo"] = len(n["outgoing"]) + len(n["incoming"])
        repos_kept += 1
        for ip, n in kept.items():
            all_nodes[ip] = n
            all_contents[ip] = contents[ip]

    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "graph.jsonl"
    content_path = out_dir / "content.jsonl"
    with open(graph_path, "w", encoding="utf-8") as gf:
        for n in all_nodes.values():
            gf.write(json.dumps(n) + "\n")
    with open(content_path, "w", encoding="utf-8") as cf:
        for ip, content in all_contents.items():
            cf.write(json.dumps({"normed_identifier": ip, "content": content}) + "\n")

    n_edges = sum(len(n["outgoing"]) for n in all_nodes.values())
    stats = {
        "records_read": n_records,
        "repos_total": len(repos),
        "repos_kept": repos_kept,
        "packages_before_filter": pkgs_before_filter,
        "graph_nodes_final": len(all_nodes),
        "graph_edges_internal_final": n_edges,
        "avg_out_degree": (n_edges / len(all_nodes)) if all_nodes else 0.0,
        "min_package_links": min_package_links,
    }
    with open(out_dir / "graph_stats.json", "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    logger.info("Go graph: %d nodes, %d edges (%.2f avg out-degree) from %d repos",
                stats["graph_nodes_final"], n_edges, stats["avg_out_degree"], repos_kept)
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl_file", type=Path, help="Stack Go JSONL dump")
    ap.add_argument("-o", "--out-dir", type=Path, required=True)
    ap.add_argument("--min-package-links", type=int, default=1,
                    help="keep packages with at least this many in-repo edges "
                         "(default 1: drop edgeless packages)")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(levelname)s: %(message)s")
    stats = build(args.jsonl_file, args.out_dir, args.min_package_links)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
