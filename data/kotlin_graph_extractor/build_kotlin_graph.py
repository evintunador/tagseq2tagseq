"""
Kotlin dependency-graph builder for The Stack (Kotlin subset).

Node model — CRITICAL difference from Java (decided empirically; documented here)
--------------------------------------------------------------------------------
Kotlin is FQN/JVM-family like Java (imports are globally unique dotted names, so
the corpus is MULTI-REPO capable), BUT unlike Java:

  * the FILENAME does NOT determine the class/symbol name, and
  * ONE ``.kt`` file can declare MANY top-level symbols (classes, objects,
    interfaces, top-level funcs/vals/vars, typealiases), each with its own FQN
    ``<package>.<SymbolName>``.

``import com.ex.util.Helper`` names a DECLARATION, not a file — so we CANNOT map
filename -> FQN like Java. Instead we build a SYMBOL -> FILE index: for each file,
parse its ``package`` header + ALL top-level declaration names and register FQN
``<package>.<Name>`` -> that file. An import of any of those FQNs resolves to the
declaring file.

The frozen resolver (model/document_corpus._build_indexes and
cross_doc_mask._match_links_to_docs) keys each corpus document by exactly ONE
string (``index_doc_span``). A Kotlin file exposing several FQNs therefore cannot
be a single node keyed by all of them. So the NODE UNIT is ONE NODE PER DECLARED
FQN: a file that declares N top-level symbols yields N nodes, each keyed by one
FQN and each carrying the SAME file content (the file is the smallest unit a model
would fetch when following any of its symbols). ``raw_identifier`` = the FQN, so
KotlinImportDetector.index_doc_span (returns raw_identifier) matches an emitted
import FQN by exact string.

Edges: an import FQN ``com.ex.Bar`` resolves to the node keyed ``com.ex.Bar`` iff
that FQN is declared by some file IN THE SAME REPO. Wildcard imports (``a.b.*``)
name no single symbol and are skipped. ``kotlin.*`` / ``java.*`` / other external
imports simply don't match an in-repo node and produce no edge.

Multi-node-per-file consequence: two nodes that come from the SAME file (sibling
top-level symbols) are NOT linked to each other unless one file's symbol imports
the other's FQN — Kotlin same-file symbols reference each other WITHOUT an import,
so there is legitimately no edge (mirrors Go's same-package no-import property).

Extensions: ``.kt`` only. Kotlin scripts (``.kts``) rarely declare importable
intra-repo symbols and are EXCLUDED from nodes.

Detection uses tree-sitter (build-time engine); the runtime KotlinImportDetector
re-derives the same imports from tokens and is graded against the same oracle.

Outputs graph.jsonl + content.jsonl (one record per FQN node) + graph_stats.json.

Usage:
    python -m data.kotlin_graph_extractor.build_kotlin_graph \\
        raw/kotlin/sample_kotlin.jsonl -o graphs/kotlin --min-links 1
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Top-level declaration node types whose FQN <package>.<Name> is importable.
_DECL_TYPES = frozenset({
    "class_declaration",       # class / interface / enum / annotation / data / sealed
    "object_declaration",
    "function_declaration",
    "property_declaration",
    "type_alias",
})


class _KotlinParser:
    """tree-sitter Kotlin parser: package header, top-level decl names, imports."""

    def __init__(self):
        import tree_sitter_kotlin
        from tree_sitter import Language, Parser
        self._lang = Language(tree_sitter_kotlin.language())
        self._parser = Parser(self._lang)

    def _text(self, src: bytes, n) -> str:
        return src[n.start_byte:n.end_byte].decode("utf-8", "replace")

    def _decl_name(self, src: bytes, node) -> Optional[str]:
        """Name of a top-level declaration node (its declared symbol name)."""
        for c in node.named_children:
            if c.type == "identifier":
                return self._text(src, c).strip().strip("`")
            if c.type == "variable_declaration":  # property: val/var
                for v in c.named_children:
                    if v.type == "identifier":
                        return self._text(src, v).strip().strip("`")
        return None

    def parse(self, source: str) -> Tuple[str, List[str], List[str]]:
        """Return (package, [declared_names], [imported_fqns_non_wildcard])."""
        src = source.encode("utf-8", errors="replace")
        tree = self._parser.parse(src)
        root = tree.root_node

        package = ""
        names: List[str] = []
        imports: List[str] = []

        for c in root.children:
            if c.type == "package_header":
                for cc in c.named_children:
                    if cc.type in ("qualified_identifier", "identifier"):
                        package = self._text(src, cc).strip()
                        break
            elif c.type == "import":
                # skip the leaf 'import' keyword (no named children)
                fqn_node = None
                for cc in c.named_children:
                    if cc.type in ("qualified_identifier", "identifier"):
                        fqn_node = cc
                        break
                if fqn_node is None:
                    continue
                raw = self._text(src, c)
                is_star = any(x.type == "*" for x in c.children) or raw.rstrip().endswith("*")
                if is_star:
                    continue
                fqn = self._text(src, fqn_node).strip().strip("`")
                if fqn:
                    imports.append(fqn)
            elif c.type in _DECL_TYPES:
                name = self._decl_name(src, c)
                if name:
                    names.append(name)

        return package, names, imports


def build_repo_nodes(
    repo_files: List[Tuple[str, str]],
    parser: _KotlinParser,
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Build FQN-keyed nodes + contents for ONE repo, with intra-repo edges.

    Each top-level declaration FQN becomes a node whose content is its declaring
    file. Edges: an import FQN resolves to a node with that exact FQN key in this
    repo (built via the symbol->file index, which here IS the node key set).
    """
    nodes: Dict[str, dict] = {}
    contents: Dict[str, str] = {}
    imports_by_fqn: Dict[str, List[str]] = {}

    for path, content in repo_files:
        if not path.endswith(".kt") or path.endswith(".kts"):
            continue
        package, names, imports = parser.parse(content)
        for name in names:
            fqn = f"{package}.{name}" if package else name
            if fqn in nodes:
                # FQN collision within a repo (e.g. same symbol in two files):
                # first file wins; later declarations of the same FQN dropped.
                continue
            contents[fqn] = content
            imports_by_fqn[fqn] = imports
            nodes[fqn] = {
                "normed_identifier": fqn,
                "raw_identifier": fqn,
                "src_path": path,
                "char_count": len(content),
                "import_count": len(imports),
                "decls_in_file": len(names),
                "outgoing": [],
                "incoming": [],
                "links_in_repo": 0,
            }

    node_ids = set(nodes)
    for fqn, imps in imports_by_fqn.items():
        outgoing: Set[str] = set()
        for name in imps:
            if name in node_ids and name != fqn:
                outgoing.add(name)
        nodes[fqn]["outgoing"] = sorted(outgoing)
    for src_fqn, data in nodes.items():
        for tgt in data["outgoing"]:
            nodes[tgt]["incoming"].append(src_fqn)
    for data in nodes.values():
        data["incoming"] = sorted(set(data["incoming"]))
        data["links_in_repo"] = len(data["outgoing"]) + len(data["incoming"])

    return nodes, contents


def build(jsonl_path: Path, out_dir: Path, min_links: int = 1) -> dict:
    parser = _KotlinParser()
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
            if not (repo and path and content and path.endswith(".kt")):
                continue
            if path.endswith(".kts"):
                continue
            repos[repo].append((path, content))
            n_records += 1
    logger.info("Read %d Kotlin records across %d repos", n_records, len(repos))

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
    logger.info("Kotlin graph: %d nodes, %d edges (%.2f avg out-degree) from %d repos",
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
