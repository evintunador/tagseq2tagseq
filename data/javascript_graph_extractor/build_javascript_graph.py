"""
JavaScript dependency-graph builder for The Stack (JavaScript subset).

A JavaScript NODE is a source FILE, keyed by its repo-relative path WITHOUT
extension (``src/util/helper``). This is the file-node model (like
Python/Java/TypeScript). JS intra-repo imports are RELATIVE (``./foo``, ``../x/y``);
a relative specifier resolves deterministically to another file in the SAME repo by
Node/ESM module resolution rules (implemented here without touching a filesystem):

    ``./foo``  -> ``<dir>/foo.js`` | ``foo.jsx`` | ``foo.mjs`` | ``foo.cjs`` |
                  ``foo/index.js`` | ...
    ``../x/y`` -> resolve ``..`` up a directory

Extension is usually omitted and inferred; a directory import resolves to its
``index`` file. Bare specifiers (``react``, ``lodash``) are external node_modules
deps that legitimately don't resolve and produce NO edge.

Import forms that create edges: ES ``import ... from "./x"``, side-effect
``import "./x"``, re-exports (``export ... from "./q"``), CommonJS ``require("./z")``
(VERY common in JS), and dynamic ``import("./x")`` with a literal string. JavaScript
has no type-only imports.

``.min.js`` (and ``.min.jsx`` etc.) minified bundles are EXCLUDED from nodes: they
are machine-generated single-line concatenations of many modules, not authored
source, and add huge tokens with no meaningful import graph. This is the JS analogue
of TypeScript excluding ``.d.ts`` declaration files.

Edges: a relative import resolves to the extension-less repo-relative path of
another file node in the SAME repo (edges are intra-repo by construction, since
``build_repo_nodes`` runs per-repo — no cross-repo contamination).

Detection uses tree-sitter (build-time engine); the runtime
JavaScriptImportDetector re-derives the same imports from tokens and is graded
against the same tree-sitter oracle.

Outputs graph.jsonl + content.jsonl + graph_stats.json, consumed by
data.pretokenize_javascript via ContentJsonlSource.

Usage:
    python -m data.javascript_graph_extractor.build_javascript_graph \\
        raw/javascript/sample_javascript.jsonl -o graphs/javascript --min-links 1
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from data.jsonl_shards import iter_jsonl_records
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Recognized module extensions, longest-first.
_EXTS = (".jsx", ".mjs", ".cjs", ".js")
# Extensions that become NODES (source files).
_NODE_EXTS = ("js", "jsx", "mjs", "cjs")


def _strip_ext(path: str) -> str:
    for ext in _EXTS:
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def _is_node_path(path: str) -> bool:
    """A .js/.jsx/.mjs/.cjs file that is NOT a minified bundle."""
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    if ext not in _NODE_EXTS:
        return False
    # exclude minified bundles: foo.min.js / foo.min.jsx / .min.mjs / .min.cjs
    stem = _strip_ext(path)
    if stem.endswith(".min"):
        return False
    return True


class _JSParser:
    """tree-sitter JavaScript parser + relative-specifier extraction."""

    def __init__(self):
        import tree_sitter_javascript
        from tree_sitter import Language, Parser
        self._lang = Language(tree_sitter_javascript.language())
        self._parser = Parser(self._lang)

    def _spec_text(self, string_node, src: bytes) -> Optional[str]:
        if string_node is None or string_node.type != "string":
            return None
        for c in string_node.named_children:
            if c.type == "string_fragment":
                return src[c.start_byte:c.end_byte].decode("utf-8", "replace")
        return ""

    def _first_string_arg(self, arguments_node):
        if arguments_node is None:
            return None
        for c in arguments_node.named_children:
            if c.type == "string":
                return c
            return None
        return None

    def imports(self, source: str) -> List[str]:
        """Return the RELATIVE specifiers imported by *source* (bare deps dropped)."""
        src = source.encode("utf-8", errors="replace")
        tree = self._parser.parse(src)
        specs: List[str] = []

        def add(spec: Optional[str]):
            if spec and (spec.startswith("./") or spec.startswith("../")):
                specs.append(spec)

        # Iterative (explicit-stack) walk — a recursive walk overflows Python's
        # ~1000-frame stack on pathologically deep real JS files (deeply nested
        # ternaries/objects), which crashes the whole build.
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            t = node.type
            if t == "import_statement":
                add(self._spec_text(node.child_by_field_name("source"), src))
            elif t == "export_statement":
                add(self._spec_text(node.child_by_field_name("source"), src))
            elif t == "call_expression":
                fn = node.child_by_field_name("function")
                args = node.child_by_field_name("arguments")
                if fn is not None:
                    fn_txt = src[fn.start_byte:fn.end_byte].decode("utf-8", "replace")
                    if (fn.type == "identifier" and fn_txt == "require") or fn.type == "import":
                        add(self._spec_text(self._first_string_arg(args), src))
            stack.extend(node.children)

        return specs


def _resolve_relative_spec(spec: str, source_path: str, node_keys: Set[str]) -> Optional[str]:
    """Resolve a relative specifier to an existing node key (or None).

    ``source_path`` is the extension-less repo-relative path of the importing file;
    ``node_keys`` is the set of extension-less repo-relative node keys in the repo.
    Tries the resolved path, then ``<resolved>/index``.
    """
    stripped = _strip_ext(spec)
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
    if resolved in node_keys:
        return resolved
    idx = f"{resolved}/index"
    if idx in node_keys:
        return idx
    return None


def build_repo_nodes(
    repo_files: List[Tuple[str, str]],
    parser: _JSParser,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Build path-keyed file nodes + contents for ONE repo, with intra-repo edges."""
    nodes: Dict[str, dict] = {}
    contents: Dict[str, str] = {}
    imports_by_key: Dict[str, List[str]] = {}

    for path, content in repo_files:
        if not _is_node_path(path):
            continue  # skip non-node ext + minified bundles
        key = _strip_ext(path)
        if key in nodes:
            continue  # first file wins on a key collision (e.g. foo.js + foo.jsx)
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
    parser = _JSParser()
    repos: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    n_records = 0
    for rec in iter_jsonl_records(jsonl_path):
        repo = rec.get("max_stars_repo_name")
        path = rec.get("max_stars_repo_path")
        content = rec.get("content")
        if not (repo and path and content):
            continue
        if not _is_node_path(path):
            continue
        repos[repo].append((path, content))
        n_records += 1
    logger.info("Read %d JavaScript records across %d repos", n_records, len(repos))

    # Keys are repo-relative (not globally unique across repos), so unlike Go/Java
    # we MUST namespace stored node ids by repo to avoid cross-repo key collisions.
    # A JS corpus is single-repo-per-pack (relative imports); the ``repo:path``
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
            # rewrite node identity + edge endpoints into the repo-namespaced space
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
    logger.info("JavaScript graph: %d nodes, %d edges (%.2f avg out-degree) from %d repos",
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
