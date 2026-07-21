"""
Dart dependency-graph builder for The Stack (Dart subset).

A Dart NODE is a source FILE, keyed by its repo-relative path INCLUDING the
``.dart`` extension (``lib/models/user.dart``). This is the file-node model (like
Python/Java/TypeScript). Dart intra-repo imports are RELATIVE URIs
(``import '../models/user.dart'``, ``import 'src/foo.dart'``); a relative URI
resolves deterministically to another file in the SAME repo (Dart requires the
``.dart`` extension explicitly — no extension inference, no ``index`` convention):

    'foo.dart'            -> <dir>/foo.dart
    'src/api.dart'        -> <dir>/src/api.dart
    '../models/user.dart' -> resolve '..' up a dir
    './widget.dart'       -> <dir>/widget.dart

URIs with a scheme are external and produce NO edge:
  * ``dart:core`` / ``dart:async``   — Dart SDK.
  * ``package:flutter/material.dart`` / ``package:myapp/foo.dart`` — pub / own-package.
    ALL ``package:`` URIs are treated external: The Stack has no ``pubspec.yaml`` so
    the repo's own package name can't be inferred. This undercounts intra-repo edges
    written as ``package:<own>/...`` but keeps precision honest; relative imports
    carry the graph.

Edges: a relative import/export URI resolves to the repo-relative path (WITH
``.dart``) of another file node in the SAME repo (edges are intra-repo by
construction, since ``build_repo_nodes`` runs per-repo — no cross-repo
contamination). ``export '...';`` re-exports create edges too. ``part '...';``
directives are NOT edges (a part-file is spliced into the same library).

Detection uses tree-sitter (build-time engine); the runtime DartImportDetector
re-derives the same imports from tokens and is graded against the same tree-sitter
oracle.

Outputs graph.jsonl + content.jsonl + graph_stats.json, consumed by
data.pretokenize_dart via ContentJsonlSource.

Usage:
    python -m data.dart_graph_extractor.build_dart_graph \\
        raw/dart/sample_dart.jsonl -o graphs/dart --min-links 1
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from data.jsonl_shards import iter_jsonl_records

logger = logging.getLogger(__name__)

_NODE_EXT = ".dart"


def _leading_scheme(uri: str) -> Optional[str]:
    """Return the URI scheme or None for a relative URI (mirrors the detector)."""
    i, n = 0, len(uri)
    if i >= n or not uri[i].isalpha():
        return None
    while i < n and (uri[i].isalnum() or uri[i] in "+.-"):
        i += 1
    if i < n and uri[i] == ":":
        return uri[:i]
    return None


class _DartParser:
    """tree-sitter Dart parser + relative-URI extraction."""

    def __init__(self):
        import tree_sitter_dart
        from tree_sitter import Language, Parser
        self._lang = Language(tree_sitter_dart.language())
        self._parser = Parser(self._lang)

    def _uri_text(self, uri_node, src: bytes) -> Optional[str]:
        """Extract the URI string (without quotes) from a ``uri`` node."""
        if uri_node is None:
            return None
        for c in uri_node.children:
            if c.type == "string_literal":
                raw = src[c.start_byte:c.end_byte].decode("utf-8", "replace").strip()
                if len(raw) >= 2 and raw[0] in "\"'" and raw[-1] == raw[0]:
                    return raw[1:-1]
                return raw
        return None

    def imports(self, source: str) -> List[str]:
        """Return the RELATIVE URIs imported/exported by *source* (externals dropped).

        Only the PRIMARY ``configurable_uri`` of each ``library_import`` /
        ``library_export`` is read (a conditional ``if (...) 'other.dart'`` fallback
        lives in a nested ``configuration_uri`` and is ignored — matching the oracle).
        """
        src = source.encode("utf-8", errors="replace")
        tree = self._parser.parse(src)
        specs: List[str] = []

        def add(uri: Optional[str]):
            if uri and _leading_scheme(uri) is None:
                specs.append(uri)

        # Iterative (explicit-stack) walk — a recursive walk can overflow Python's
        # ~1000-frame stack on pathologically deep real files.
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            t = node.type
            if t == "library_import":
                spec_node = None
                for c in node.children:
                    if c.type == "import_specification":
                        spec_node = c
                        break
                target = spec_node if spec_node is not None else node
                for c in target.children:
                    if c.type == "configurable_uri":
                        # the primary URI is the direct ``uri`` child
                        for cc in c.children:
                            if cc.type == "uri":
                                add(self._uri_text(cc, src))
                                break
                        break
            elif t == "library_export":
                for c in node.children:
                    if c.type == "configurable_uri":
                        for cc in c.children:
                            if cc.type == "uri":
                                add(self._uri_text(cc, src))
                                break
                        break
            stack.extend(node.children)

        return specs


def _resolve_relative_uri(uri: str, source_path: str, node_keys: Set[str]) -> Optional[str]:
    """Resolve a relative URI to an existing node key (or None).

    ``source_path`` is the repo-relative path (WITH ``.dart``) of the importing file;
    ``node_keys`` is the set of repo-relative node keys in the repo.
    """
    if _leading_scheme(uri) is not None:
        return None
    base_dir = source_path.split("/")[:-1]
    cur = list(base_dir)
    for seg in uri.strip().split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if cur:
                cur.pop()
            else:
                return None  # escapes repo root
        else:
            cur.append(seg)
    resolved = "/".join(cur)
    if not resolved:
        return None
    return resolved if resolved in node_keys else None


def build_repo_nodes(
    repo_files: List[Tuple[str, str]],
    parser: _DartParser,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Build path-keyed file nodes + contents for ONE repo, with intra-repo edges."""
    nodes: Dict[str, dict] = {}
    contents: Dict[str, str] = {}
    imports_by_key: Dict[str, List[str]] = {}

    for path, content in repo_files:
        if not path.endswith(_NODE_EXT):
            continue
        key = path
        if key in nodes:
            continue
        contents[key] = content
        imps = parser.imports(content)
        imports_by_key[key] = imps
        nodes[key] = {
            "normed_identifier": key,
            "raw_identifier": key,
            "src_path": path,
            "char_count": len(content),
            "import_count": len(imps),
            "outgoing": [],
            "incoming": [],
            "links_in_repo": 0,
        }

    node_keys = set(nodes)
    for key, imps in imports_by_key.items():
        outgoing: Set[str] = set()
        for uri in imps:
            tgt = _resolve_relative_uri(uri, key, node_keys)
            if tgt is not None and tgt != key:
                outgoing.add(tgt)
        nodes[key]["outgoing"] = sorted(outgoing)
    for src_key, data in nodes.items():
        for tgt in data["outgoing"]:
            nodes[tgt]["incoming"].append(src_key)
    for data in nodes.values():
        data["incoming"] = sorted(set(data["incoming"]))
        data["links_in_repo"] = len(data["outgoing"]) + len(data["incoming"])

    return nodes, contents


def build(jsonl_path: Path, out_dir: Path, min_links: int = 1) -> dict:
    parser = _DartParser()
    repos: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_records = 0
    for rec in iter_jsonl_records(jsonl_path):
        repo = rec.get("max_stars_repo_name")
        path = rec.get("max_stars_repo_path")
        content = rec.get("content")
        if not (repo and path and content):
            continue
        if not path.endswith(".dart"):
            continue
        repos[repo].append((path, content))
        n_records += 1
    logger.info("Read %d Dart records across %d repos", n_records, len(repos))

    # Keys are repo-relative (not globally unique across repos), so unlike Go/Java
    # we MUST namespace stored node ids by repo to avoid cross-repo key collisions.
    # A Dart corpus is single-repo-per-pack (relative imports); the ``repo:path``
    # identifier keeps nodes distinct while index_doc_span strips the repo prefix.
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
            global_id = f"{repo}:{k}"
            n["normed_identifier"] = global_id
            n["raw_identifier"] = global_id
            n["outgoing"] = [f"{repo}:{t}" for t in n["outgoing"]]
            n["incoming"] = [f"{repo}:{s}" for s in n["incoming"]]
            all_nodes[global_id] = n
            all_contents[global_id] = contents[k]

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
    logger.info("Dart graph: %d nodes, %d edges (%.2f avg out-degree) from %d repos",
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
