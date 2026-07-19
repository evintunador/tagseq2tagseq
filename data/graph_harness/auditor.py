"""
Dataset auditor — checkpoint-free graph-quality report for any built dataset.

Loads a pretokenized dataset directory (via GraphIndex) and reports the structural
health of its graph, PLUS an optional resolvability sample when a link detector is
given. Fills the gap the design doc §3b identifies: extractor stats today are
build-time-only and per-dataset bespoke; this runs on any finished dataset dir and
doubles as the Python-corpus sanity check.

Metrics (pure graph, no model, no tokens needed):
  * node / edge counts, repo count (for repo-partitioned corpora)
  * out/in-degree: mean, median, % zero (leaf / source rate)
  * dangling-edge rate: outgoing targets that are NOT nodes in this corpus
  * self-link rate: edges whose source == target
  * isolated-node fraction: nodes with no in AND no out edges
  * reciprocal-edge rate: edges A->B where B->A also exists

Resolvability sample (needs a LinkDetector + token backend):
  * for a sample of nodes, decode tokens, run detect_links, and report what
    fraction of emitted target_str resolve via PretokCorpus-style resolution to a
    node in THIS corpus. This is the resolution axis on real built data — reported
    ALONGSIDE structural metrics, never as the sole number (design doc §2).

Everything is returned as a dataclass AND printable, so it feeds both automated
gates and the human review bundle.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from data.dataset import GraphIndex


@dataclass
class GraphAudit:
    dataset_dir: str
    n_nodes: int
    n_edges: int
    n_repos: Optional[int]
    out_degree_mean: float
    out_degree_median: float
    in_degree_mean: float
    pct_out_zero: float
    pct_in_zero: float
    dangling_edge_rate: float
    self_link_rate: float
    isolated_node_frac: float
    reciprocal_edge_rate: float
    # populated only when a detector+backend resolvability sample was run
    resolvability: Optional["ResolvabilitySample"] = None
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.dataset_dir}",
            f"  nodes={self.n_nodes:,}  edges={self.n_edges:,}"
            + (f"  repos={self.n_repos:,}" if self.n_repos is not None else ""),
            f"  out-degree: mean={self.out_degree_mean:.2f} "
            f"median={self.out_degree_median:.1f}  in-degree mean={self.in_degree_mean:.2f}",
            f"  %out=0 (leaves)={self.pct_out_zero:.1%}  "
            f"%in=0 (sources)={self.pct_in_zero:.1%}  "
            f"isolated={self.isolated_node_frac:.1%}",
            f"  dangling-edge rate={self.dangling_edge_rate:.2%}  "
            f"self-link rate={self.self_link_rate:.2%}  "
            f"reciprocal-edge rate={self.reciprocal_edge_rate:.1%}",
        ]
        if self.resolvability is not None:
            lines.append("  " + self.resolvability.summary())
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


@dataclass
class ResolvabilitySample:
    n_nodes_sampled: int
    n_targets_emitted: int
    n_targets_resolved: int
    n_resolved_in_corpus: int
    examples_unresolved: List[str] = field(default_factory=list)

    @property
    def resolve_rate(self) -> float:
        if self.n_targets_emitted == 0:
            return 0.0
        return self.n_targets_resolved / self.n_targets_emitted

    def summary(self) -> str:
        return (
            f"resolvability: {self.n_targets_resolved}/{self.n_targets_emitted} "
            f"targets resolve ({self.resolve_rate:.1%}) over "
            f"{self.n_nodes_sampled} sampled nodes"
        )


def audit_graph(dataset_dir: "str | Path") -> GraphAudit:
    """Compute the pure-graph structural audit (no model, no tokens)."""
    graph = GraphIndex(Path(dataset_dir))
    nodes = graph.nodes
    n_nodes = len(nodes)

    out_degs: List[int] = []
    in_degs: List[int] = []
    n_edges = 0
    n_self = 0
    n_dangling = 0
    n_isolated = 0
    n_reciprocal = 0

    # repo count only meaningful for repo-partitioned corpora ("owner/repo:path")
    repos = set()
    repo_partitioned = False

    node_keys = set(nodes.keys())
    for normed, node in nodes.items():
        raw = node.get("raw_identifier", "")
        if ":" in raw:
            repo_partitioned = True
            repos.add(raw.split(":", 1)[0])
        outgoing = node.get("outgoing", []) or []
        incoming = node.get("incoming", []) or []
        out_degs.append(len(outgoing))
        in_degs.append(len(incoming))
        n_edges += len(outgoing)
        if not outgoing and not incoming:
            n_isolated += 1
        for tgt in outgoing:
            if tgt == normed:
                n_self += 1
            if tgt not in node_keys:
                n_dangling += 1
            elif tgt != normed:
                # reciprocal if target lists this node as its own outgoing.
                # Self-links are excluded — a node pointing at itself is not a
                # mutual A<->B relationship.
                tgt_out = nodes[tgt].get("outgoing", []) or []
                if normed in tgt_out:
                    n_reciprocal += 1

    def pct_zero(degs: List[int]) -> float:
        return (sum(1 for d in degs if d == 0) / len(degs)) if degs else 0.0

    warnings: List[str] = []
    if n_edges == 0:
        warnings.append("graph has ZERO edges — link masks will degrade to doc_causal")
    dangling_rate = (n_dangling / n_edges) if n_edges else 0.0
    if dangling_rate > 0.01:
        warnings.append(
            f"{dangling_rate:.1%} of edges are dangling (target not a node) — "
            "check edge filtering after splits/carving"
        )
    if n_self > 0:
        warnings.append(f"{n_self} self-links present (should normally be 0)")

    return GraphAudit(
        dataset_dir=str(dataset_dir),
        n_nodes=n_nodes,
        n_edges=n_edges,
        n_repos=len(repos) if repo_partitioned else None,
        out_degree_mean=(statistics.fmean(out_degs) if out_degs else 0.0),
        out_degree_median=(statistics.median(out_degs) if out_degs else 0.0),
        in_degree_mean=(statistics.fmean(in_degs) if in_degs else 0.0),
        pct_out_zero=pct_zero(out_degs),
        pct_in_zero=pct_zero(in_degs),
        dangling_edge_rate=dangling_rate,
        self_link_rate=(n_self / n_edges) if n_edges else 0.0,
        isolated_node_frac=(n_isolated / n_nodes) if n_nodes else 0.0,
        reciprocal_edge_rate=(n_reciprocal / n_edges) if n_edges else 0.0,
        warnings=warnings,
    )
