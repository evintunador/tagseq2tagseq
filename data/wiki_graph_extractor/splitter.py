#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAGWikiGraphSplitter: Splits the Wikipedia link graph into train,
community-validation, and random-validation subsets.

Community-validation is built from whole Louvain communities selected to
approximate the global degree and char-count distribution.  Every node in
the community-validation split is guaranteed to have induced degree >=
min_degree within the split's own subgraph.

Random-validation is a uniform random sample of article IDs.
"""
import argparse
import hashlib
import json
import logging
import os
import random
from collections import defaultdict
from timeit import default_timer

import networkx as nx

logger = logging.getLogger(__name__)

# ── Bucket helpers ──────────────────────────────────────────────────────

DEGREE_BUCKET_EDGES = [2, 4, 8, 16]
CHAR_BUCKET_EDGES = [1_000, 5_000, 20_000]


def _bucket_index(value, edges):
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def _bucket_counts(values, edges):
    counts = [0] * (len(edges) + 1)
    for v in values:
        counts[_bucket_index(v, edges)] += 1
    return counts


def _l1_distance(vec_a, vec_b):
    """L1 distance between two equal-length numeric vectors."""
    return sum(abs(a - b) for a, b in zip(vec_a, vec_b))


def _normalize_vec(counts):
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [c / total for c in counts]


# ── Graph helpers ───────────────────────────────────────────────────────

def _load_graph_jsonl(path):
    """Load graph.jsonl into a dict keyed by normed_identifier."""
    graph_data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            node = json.loads(line)
            graph_data[node["normed_identifier"]] = node
    return graph_data


def _undirected_degree(node_id, graph_data, node_set):
    """Undirected degree of *node_id* restricted to *node_set*."""
    node = graph_data[node_id]
    neighbors = set()
    for n in node.get("outgoing", []):
        if n in node_set:
            neighbors.add(n)
    for n in node.get("incoming", []):
        if n in node_set:
            neighbors.add(n)
    neighbors.discard(node_id)
    return len(neighbors)


def _build_undirected_nx(node_ids, graph_data):
    """Build a networkx undirected Graph over *node_ids*."""
    node_set = set(node_ids)
    G = nx.Graph()
    G.add_nodes_from(node_ids)
    for nid in node_ids:
        node = graph_data[nid]
        for target in node.get("outgoing", []):
            if target in node_set and target != nid:
                G.add_edge(nid, target)
        for source in node.get("incoming", []):
            if source in node_set and source != nid:
                G.add_edge(nid, source)
    return G


# ── Core split logic ───────────────────────────────────────────────────

def create_splits(
    graph_data: dict,
    seed: int = 42,
    random_val_frac: float = 0.05,
    comm_val_frac: float = 0.15,
    min_degree: int = 2,
) -> dict:
    """
    Partition eligible graph nodes into train / community-val / random-val.

    Parameters
    ----------
    graph_data : dict
        normed_identifier -> {outgoing, incoming, char_count, ...}
    seed : int
        Master random seed for reproducibility.
    random_val_frac : float
        Fraction of eligible nodes to put in random-val.
    comm_val_frac : float
        Target fraction of eligible nodes for community-val.
    min_degree : int
        Minimum undirected degree for eligibility AND the hard density
        constraint inside community-val (every node must meet this).

    Returns
    -------
    dict  – full split metadata (see ``splits.json`` schema in the plan).
    """
    total_graph_nodes = len(graph_data)
    all_ids = set(graph_data.keys())

    # ── A) Filter to eligible nodes ─────────────────────────────────
    eligible_ids = set()
    for nid in all_ids:
        if _undirected_degree(nid, graph_data, all_ids) >= min_degree:
            eligible_ids.add(nid)

    ineligible_count = total_graph_nodes - len(eligible_ids)
    logger.info(
        f"Eligible nodes: {len(eligible_ids):,} / {total_graph_nodes:,} "
        f"(filtered {ineligible_count:,} with undirected degree < {min_degree})"
    )

    total_eligible = len(eligible_ids)
    if total_eligible == 0:
        raise ValueError("No eligible nodes after degree filtering.")

    # ── B) Random-val (5 %) ─────────────────────────────────────────
    rng = random.Random(seed)
    eligible_list = sorted(eligible_ids)
    rng.shuffle(eligible_list)

    n_random_val = round(random_val_frac * total_eligible)
    val_rand_ids = set(eligible_list[:n_random_val])
    remaining_ids = eligible_ids - val_rand_ids

    logger.info(f"Random-val: {len(val_rand_ids):,} nodes ({len(val_rand_ids)/total_eligible:.2%})")

    # ── C) Community-val (15 %) ─────────────────────────────────────
    target_comm_size = round(comm_val_frac * total_eligible)

    logger.info("Building undirected graph for community detection...")
    G_remaining = _build_undirected_nx(sorted(remaining_ids), graph_data)
    logger.info(
        f"Undirected graph: {G_remaining.number_of_nodes():,} nodes, "
        f"{G_remaining.number_of_edges():,} edges"
    )

    logger.info("Running Louvain community detection...")
    try:
        communities = nx.community.louvain_communities(G_remaining, seed=seed)
    except AttributeError:
        logger.warning("Louvain not available, falling back to greedy modularity.")
        communities = list(nx.community.greedy_modularity_communities(G_remaining))

    communities = [frozenset(c) for c in communities]
    logger.info(f"Detected {len(communities):,} communities")

    # Pre-compute per-node undirected degree and char_count in remaining pool
    remaining_set = set(remaining_ids)
    node_degree = {}
    node_char = {}
    for nid in remaining_ids:
        node_degree[nid] = G_remaining.degree(nid)
        node_char[nid] = graph_data[nid].get("char_count", 0)

    # Global bucket proportions over the remaining pool
    global_deg_counts = _bucket_counts(
        [node_degree[n] for n in remaining_ids], DEGREE_BUCKET_EDGES
    )
    global_char_counts = _bucket_counts(
        [node_char[n] for n in remaining_ids], CHAR_BUCKET_EDGES
    )
    global_proportions = _normalize_vec(global_deg_counts + global_char_counts)
    n_buckets = len(global_proportions)

    # Per-community bucket counts
    comm_profiles = []
    for idx, comm in enumerate(communities):
        deg_counts = _bucket_counts([node_degree[n] for n in comm], DEGREE_BUCKET_EDGES)
        char_counts = _bucket_counts([node_char[n] for n in comm], CHAR_BUCKET_EDGES)
        comm_profiles.append({
            "idx": idx,
            "size": len(comm),
            "deg_counts": deg_counts,
            "char_counts": char_counts,
        })

    # Greedy community selection with iterative pruning
    val_comm_ids = set()
    selected_indices = set()
    selected_deg = [0] * len(DEGREE_BUCKET_EDGES + [None])   # len = buckets
    selected_char = [0] * len(CHAR_BUCKET_EDGES + [None])
    # Correct bucket lengths
    selected_deg = [0] * (len(DEGREE_BUCKET_EDGES) + 1)
    selected_char = [0] * (len(CHAR_BUCKET_EDGES) + 1)

    tolerance = 0.01
    max_iterations = 5  # pruning-then-refill iterations

    for iteration in range(max_iterations):
        candidates = [
            p for p in comm_profiles
            if p["idx"] not in selected_indices
        ]
        if not candidates:
            break

        # Greedy: keep adding communities until we hit the target
        while len(val_comm_ids) < target_comm_size and candidates:
            best_score = float("inf")
            best_idx_in_list = -1

            for ci, cp in enumerate(candidates):
                trial_deg = [s + c for s, c in zip(selected_deg, cp["deg_counts"])]
                trial_char = [s + c for s, c in zip(selected_char, cp["char_counts"])]
                trial_proportions = _normalize_vec(trial_deg + trial_char)
                score = _l1_distance(trial_proportions, global_proportions)
                if score < best_score:
                    best_score = score
                    best_idx_in_list = ci

            cp = candidates.pop(best_idx_in_list)

            # If adding this community overshoots by more than 1%, skip it
            # and try the next candidate, unless we haven't reached threshold yet
            overshoot = (len(val_comm_ids) + cp["size"]) / total_eligible
            if overshoot > (comm_val_frac + tolerance) and len(val_comm_ids) >= round((comm_val_frac - tolerance) * total_eligible):
                continue

            selected_indices.add(cp["idx"])
            val_comm_ids |= communities[cp["idx"]]
            for b in range(len(selected_deg)):
                selected_deg[b] += cp["deg_counts"][b]
            for b in range(len(selected_char)):
                selected_char[b] += cp["char_counts"][b]

        # ── Hard density pruning ────────────────────────────────────
        # Every community-val node must have induced degree >= min_degree
        # within the val_comm subgraph.
        G_val = _build_undirected_nx(sorted(val_comm_ids), graph_data)
        pruned_count = 0
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for nid in list(G_val.nodes()):
                if G_val.degree(nid) < min_degree:
                    to_remove.add(nid)
            if to_remove:
                G_val.remove_nodes_from(to_remove)
                val_comm_ids -= to_remove
                pruned_count += len(to_remove)
                changed = True

        if pruned_count > 0:
            logger.info(
                f"Iteration {iteration + 1}: pruned {pruned_count:,} nodes "
                f"with induced degree < {min_degree}. "
                f"Community-val now {len(val_comm_ids):,} nodes "
                f"({len(val_comm_ids)/total_eligible:.2%})"
            )

        # If we're within tolerance, we're done
        current_frac = len(val_comm_ids) / total_eligible
        if current_frac >= (comm_val_frac - tolerance):
            break

        # Otherwise, reset bucket accumulators to match the surviving set
        # and try adding more communities
        selected_deg = _bucket_counts(
            [node_degree[n] for n in val_comm_ids], DEGREE_BUCKET_EDGES
        )
        selected_char = _bucket_counts(
            [node_char[n] for n in val_comm_ids], CHAR_BUCKET_EDGES
        )
        logger.info(
            f"Iteration {iteration + 1}: community-val at "
            f"{len(val_comm_ids)/total_eligible:.2%}, below target "
            f"{comm_val_frac:.0%}. Selecting more communities..."
        )

    logger.info(
        f"Community-val: {len(val_comm_ids):,} nodes "
        f"({len(val_comm_ids)/total_eligible:.2%})"
    )

    # ── D) Train = remaining ────────────────────────────────────────
    train_ids = eligible_ids - val_rand_ids - val_comm_ids

    logger.info(f"Train: {len(train_ids):,} nodes ({len(train_ids)/total_eligible:.2%})")

    # ── Stats for community-val ─────────────────────────────────────
    if val_comm_ids:
        G_val_final = _build_undirected_nx(sorted(val_comm_ids), graph_data)
        induced_degrees = [G_val_final.degree(n) for n in G_val_final.nodes()]
        avg_induced_degree = sum(induced_degrees) / len(induced_degrees)
        min_induced_degree_actual = min(induced_degrees)
        internal_edges = G_val_final.number_of_edges()

        # Count cross edges between community-val and train
        train_set = set(train_ids)
        cross_edges = 0
        for nid in val_comm_ids:
            node = graph_data[nid]
            for t in node.get("outgoing", []):
                if t in train_set:
                    cross_edges += 1
            for s in node.get("incoming", []):
                if s in train_set:
                    cross_edges += 1
    else:
        avg_induced_degree = 0.0
        min_induced_degree_actual = 0
        internal_edges = 0
        cross_edges = 0

    # ── Assemble result ─────────────────────────────────────────────
    node_list_hash = hashlib.sha256(
        ",".join(sorted(eligible_ids)).encode()
    ).hexdigest()[:16]

    result = {
        "seed": seed,
        "min_degree": min_degree,
        "total_graph_nodes": total_graph_nodes,
        "eligible_nodes": total_eligible,
        "ineligible_node_count": ineligible_count,
        "node_list_hash": node_list_hash,
        "splits": {
            "train": {
                "count": len(train_ids),
                "fraction": round(len(train_ids) / total_eligible, 6),
                "ids": sorted(train_ids),
            },
            "val_community": {
                "count": len(val_comm_ids),
                "fraction": round(len(val_comm_ids) / total_eligible, 6),
                "ids": sorted(val_comm_ids),
            },
            "val_random": {
                "count": len(val_rand_ids),
                "fraction": round(len(val_rand_ids) / total_eligible, 6),
                "ids": sorted(val_rand_ids),
            },
        },
        "community_val_stats": {
            "num_communities_selected": len(selected_indices),
            "avg_induced_degree": round(avg_induced_degree, 4),
            "min_induced_degree": min_induced_degree_actual,
            "nodes_pruned_by_density_constraint": sum(
                len(communities[i]) for i in selected_indices
            ) - len(val_comm_ids),
            "internal_edges": internal_edges,
            "cross_edges_to_train": cross_edges,
        },
    }
    return result


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="DAGWikiGraphSplitter",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "graph_file",
        help="Path to the graph.jsonl file produced by build_graph.py.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for splits.json (default: splits.json next to graph file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--min-degree",
        type=int,
        default=2,
        help="Minimum undirected degree for eligibility and community-val density (default: 2).",
    )
    parser.add_argument(
        "--random-val-frac",
        type=float,
        default=0.05,
        help="Fraction of eligible nodes for random-val (default: 0.05).",
    )
    parser.add_argument(
        "--comm-val-frac",
        type=float,
        default=0.15,
        help="Target fraction of eligible nodes for community-val (default: 0.15).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress reporting.",
    )
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    if args.output is None:
        base = os.path.splitext(args.graph_file)[0]
        args.output = base + "_splits.json"

    logger.info(f"Loading graph from {args.graph_file}...")
    start = default_timer()
    graph_data = _load_graph_jsonl(args.graph_file)
    logger.info(f"Loaded {len(graph_data):,} nodes in {default_timer() - start:.2f}s")

    start = default_timer()
    result = create_splits(
        graph_data,
        seed=args.seed,
        random_val_frac=args.random_val_frac,
        comm_val_frac=args.comm_val_frac,
        min_degree=args.min_degree,
    )
    logger.info(f"Splitting completed in {default_timer() - start:.2f}s")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Wrote splits to {args.output}")

    # Summary
    for name in ("train", "val_community", "val_random"):
        s = result["splits"][name]
        logger.info(f"  {name}: {s['count']:,} ({s['fraction']:.2%})")
    cs = result["community_val_stats"]
    logger.info(
        f"  community-val induced: avg_deg={cs['avg_induced_degree']:.2f}, "
        f"min_deg={cs['min_induced_degree']}, "
        f"internal_edges={cs['internal_edges']:,}, "
        f"cross_edges={cs['cross_edges_to_train']:,}, "
        f"pruned={cs['nodes_pruned_by_density_constraint']:,}"
    )


if __name__ == "__main__":
    main()
