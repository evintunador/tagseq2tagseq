"""
Zig dependency-graph builder for The Stack (Zig subset).

A Zig NODE is a source FILE, keyed by its repo-relative path WITH the ``.zig``
extension (``src/util/helper.zig``). This is the file-node model (like
Python/Java/TypeScript). Zig intra-repo imports are EXPLICIT relative FILE PATHS
— the cleanest resolution here: an ``@import`` literally names a sibling file,
resolved against the importing file's DIRECTORY with no extension inference and no
directory/index-file candidates (unlike TS):

    ``@import("foo.zig")``      from ``src/main.zig``  -> ``src/foo.zig``
    ``@import("lib/bar.zig")``  from ``src/main.zig``  -> ``src/lib/bar.zig``
    ``@import("../up/x.zig")``  from ``src/a/b.zig``   -> ``src/up/x.zig``

Bare specifiers (``std``, ``builtin``, and package names wired via ``build.zig``,
which The Stack filters out) are external stdlib/package deps that legitimately
don't resolve and produce NO edge.

Edges: a relative import resolves to another ``.zig`` file node in the SAME repo
(edges are intra-repo by construction, since ``build_repo_nodes`` runs per-repo —
no cross-repo contamination).

Detection uses tree-sitter (build-time engine); the runtime ZigImportDetector
re-derives the same imports from tokens and is graded against the same tree-sitter
oracle.

Outputs graph.jsonl + content.jsonl + graph_stats.json, consumed by
data.pretokenize_zig via ContentJsonlSource.

Usage:
    python -m data.zig_graph_extractor.build_zig_graph \\
        raw/zig/sample_zig.jsonl -o graphs/zig --min-links 1
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


class _ZigParser:
    """tree-sitter Zig parser + relative-@import extraction (build-time engine)."""

    def __init__(self):
        import tree_sitter_zig
        from tree_sitter import Language, Parser
        self._lang = Language(tree_sitter_zig.language())
        self._parser = Parser(self._lang)

    def _string_arg(self, arguments_node, src: bytes) -> Optional[str]:
        """First string-literal argument's content, or None if not a literal."""
        if arguments_node is None:
            return None
        for c in arguments_node.named_children:
            if c.type == "string":
                for cc in c.named_children:
                    if cc.type == "string_content":
                        return src[cc.start_byte:cc.end_byte].decode("utf-8", "replace")
                return ""  # empty string literal
            return None  # first arg is not a literal string
        return None

    def imports(self, source: str) -> List[str]:
        """Return the RELATIVE (``.zig``) specifiers imported by *source*.

        Bare stdlib/package imports (``std``, ``builtin``, ...) are dropped.
        Iterative (explicit-stack) walk to avoid Python recursion overflow on
        pathologically deep real files (the TS RecursionError lesson).
        """
        src = source.encode("utf-8", errors="replace")
        tree = self._parser.parse(src)
        specs: List[str] = []

        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "builtin_function":
                bid = None
                args = None
                for c in node.named_children:
                    if c.type == "builtin_identifier" and bid is None:
                        bid = src[c.start_byte:c.end_byte].decode("utf-8", "replace")
                    elif c.type == "arguments" and args is None:
                        args = c
                if bid == "@import":
                    spec = self._string_arg(args, src)
                    if spec and spec.endswith(".zig"):
                        specs.append(spec)
            stack.extend(node.children)

        return specs


def _resolve_relative_spec(spec: str, source_path: str, node_keys: Set[str]) -> Optional[str]:
    """Resolve a relative ``.zig`` specifier to an existing node key (or None).

    ``source_path`` is the repo-relative path (WITH ``.zig``) of the importing
    file; ``node_keys`` is the set of repo-relative ``.zig`` node keys in the repo.
    """
    if not spec.endswith(".zig"):
        return None
    stripped = spec[: -len(".zig")]
    base_dir = source_path.split("/")[:-1]
    cur = list(base_dir)
    for seg in stripped.split("/"):
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
    key = f"{resolved}.zig"
    return key if key in node_keys else None


def build_repo_nodes(
    repo_files: List[Tuple[str, str]],
    parser: _ZigParser,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Build path-keyed file nodes + contents for ONE repo, with intra-repo edges."""
    nodes: Dict[str, dict] = {}
    contents: Dict[str, str] = {}
    imports_by_key: Dict[str, List[str]] = {}

    for path, content in repo_files:
        if not path.endswith(".zig"):
            continue
        key = path  # keep the .zig extension
        if key in nodes:
            continue  # first file wins on a key collision
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
        for spec in imps:
            tgt = _resolve_relative_spec(spec, key, node_keys)
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
    parser = _ZigParser()
    repos: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_records = 0
    for rec in iter_jsonl_records(jsonl_path):
        repo = rec.get("max_stars_repo_name")
        path = rec.get("max_stars_repo_path")
        content = rec.get("content")
        if not (repo and path and content):
            continue
        if not path.endswith(".zig"):
            continue
        repos[repo].append((path, content))
        n_records += 1
    logger.info("Read %d Zig records across %d repos", n_records, len(repos))

    # Keys are repo-relative (not globally unique across repos), so — like TS —
    # stored node ids are namespaced by repo (``repo:path``) to avoid cross-repo
    # collisions. A Zig corpus is single-repo-per-pack (relative imports); the
    # ``repo:path`` id keeps nodes distinct while index_doc_span strips the prefix.
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
    logger.info("Zig graph: %d nodes, %d edges (%.2f avg out-degree) from %d repos",
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
