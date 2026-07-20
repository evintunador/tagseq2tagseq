"""
Rust dependency-graph builder for The Stack (Rust subset).

A Rust NODE is a source FILE, keyed by its crate-relative MODULE PATH
(``crate::net::tcp``), repo-qualified for global uniqueness
(``owner/repo@crate::net::tcp``) because module paths collide across crates (every
crate calls itself ``crate::``). This is single-crate resolution (like Python's
relative paths), NOT globally-unique like Go/Java.

Two passes per repo:
  1. WALK the mod-declaration tree (data.rust_graph_extractor.mod_tree) from each
     crate root (``src/lib.rs`` / ``src/main.rs`` / ``src/bin/*.rs``) to assign
     every reachable ``.rs`` file its module path.
  2. EXTRACT ``use`` edges: for each file, detect its use-targets (via the runtime
     RustImportDetector re-used as the build engine, resolving ``self::``/``super::``
     against the file's module path), then resolve each candidate against the SAME
     repo's module-path set (exact match). ``std::`` / external crate roots and
     ``crate::`` targets with no matching node simply don't resolve (dropped).

Detection is graded independently by the tree-sitter oracle (harness); this builder
and the runtime detector share the ``use``-parsing helper (one implementation), but
the oracle is a THIRD, independent tree-sitter walk — so agreement is still checked.

The runtime RustImportDetector's ``detect_links_for_doc`` re-derives the same edges
from tokens at train time; here we call the shared string-space parser directly on
raw source (no tokenization round-trip) for speed and exactness.

Outputs graph.jsonl + content.jsonl + graph_stats.json.

Usage:
    python -m data.rust_graph_extractor.build_rust_graph \\
        raw/rust/sample_rust.jsonl -o graphs/rust --min-links 1
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from data.jsonl_shards import iter_jsonl_records
from typing import Dict, List, Set, Tuple

from data.rust_graph_extractor.mod_tree import RustModParser, build_module_paths
from model.graph_traversal.rust_import_detector import _parse_targets

logger = logging.getLogger(__name__)


def _file_targets(content: str, module_path: str) -> List[str]:
    """Crate-relative use/mod targets for a file, with self/super rewritten."""
    return [t for t, _end in _parse_targets(content, module_path=module_path)]


def build_repo_nodes(
    repo: str,
    repo_files: List[Tuple[str, str]],
    parser: RustModParser,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Build module-path-keyed file nodes + contents for ONE repo (intra-crate edges)."""
    module_paths = build_module_paths(repo_files, parser)  # relpath -> module_path
    if not module_paths:
        return {}, {}

    content_by_path = {rp: c for rp, c in repo_files if rp.endswith(".rs")}

    # module_path -> node id ("repo@module_path"); first file wins on path collision
    mp_to_id: Dict[str, str] = {}
    id_to_relpath: Dict[str, str] = {}
    for relpath, mp in sorted(module_paths.items()):
        if mp in mp_to_id:
            continue
        node_id = f"{repo}@{mp}"
        mp_to_id[mp] = node_id
        id_to_relpath[node_id] = relpath

    nodes: Dict[str, dict] = {}
    contents: Dict[str, str] = {}
    targets_by_id: Dict[str, List[str]] = {}
    for mp, node_id in mp_to_id.items():
        relpath = id_to_relpath[node_id]
        content = content_by_path[relpath]
        contents[node_id] = content
        targets = _file_targets(content, mp)
        targets_by_id[node_id] = targets
        nodes[node_id] = {
            "normed_identifier": node_id,
            "raw_identifier": node_id,
            "src_path": relpath,
            "module_path": mp,
            "char_count": len(content),
            "import_count": len(targets),
            "outgoing": [],
            "incoming": [],
            "links_in_repo": 0,
        }

    # resolve targets against this repo's module-path set (exact match)
    for node_id, targets in targets_by_id.items():
        outgoing: Set[str] = set()
        for cand in targets:
            tgt_id = mp_to_id.get(cand)
            if tgt_id is not None and tgt_id != node_id:
                outgoing.add(tgt_id)
        nodes[node_id]["outgoing"] = sorted(outgoing)
    for src_id, data in nodes.items():
        for tgt in data["outgoing"]:
            nodes[tgt]["incoming"].append(src_id)
    for data in nodes.values():
        data["incoming"] = sorted(set(data["incoming"]))
        data["links_in_repo"] = len(data["outgoing"]) + len(data["incoming"])

    return nodes, contents


def build(jsonl_path: Path, out_dir: Path, min_links: int = 1) -> dict:
    parser = RustModParser()
    repos: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_records = 0
    for rec in iter_jsonl_records(jsonl_path):
        repo = rec.get("max_stars_repo_name")
        path = rec.get("max_stars_repo_path")
        content = rec.get("content")
        if not (repo and path and content and path.endswith(".rs")):
            continue
        repos[repo].append((path, content))
        n_records += 1
    logger.info("Read %d Rust records across %d repos", n_records, len(repos))

    all_nodes: Dict[str, dict] = {}
    all_contents: Dict[str, str] = {}
    repos_kept = 0
    nodes_before = 0
    repos_with_root = 0
    for repo, files in repos.items():
        if len(files) < 2:
            continue
        nodes, contents = build_repo_nodes(repo, files, parser)
        if not nodes:
            continue
        repos_with_root += 1
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
        "repos_with_crate_root": repos_with_root,
        "repos_kept": repos_kept,
        "nodes_before_filter": nodes_before,
        "graph_nodes_final": len(all_nodes),
        "graph_edges_internal_final": n_edges,
        "avg_out_degree": (n_edges / len(all_nodes)) if all_nodes else 0.0,
        "min_links": min_links,
    }
    with open(out_dir / "graph_stats.json", "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)
    logger.info("Rust graph: %d nodes, %d edges (%.2f avg out-degree) from %d repos",
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
