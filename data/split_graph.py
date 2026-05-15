"""
data/split_graph.py — split a pretokenized graph into separate subdirectories.

Writes dataset_dir/splits/{train,val_community,val_random,test_community,
test_random}/ where each subdirectory is a self-contained GraphIndex-compatible
dataset: its own tokenized_graph.jsonl (edges filtered to same-split nodes
only) plus a metadata.json that inherits all fields from the parent but with
shard_filenames as absolute paths (the binary shards are shared, not copied).

Split design (default 2.5% each):
  val_community   — BFS-identified subgraphs; community link structure intact.
  val_random      — uniform random sample from non-community nodes.
  test_community  — same structure as val_community; hold back for paper.
  test_random     — same structure as val_random; hold back for paper.
  train           — all remaining nodes.

Each split's tokenized_graph.jsonl contains only nodes belonging to that
split. Outgoing/incoming edge lists are filtered to reference only nodes
within the same split. This gives each split a clean, self-contained graph
with no cross-split edge leakage.

Usage:
    python data/split_graph.py \\
        --dataset-dir data/pretokenized_datasets/simplewiki \\
        [--val-frac 0.025] [--test-frac 0.025] \\
        [--community-size-min 50] [--community-size-max 5000] \\
        [--seed 42] [--dry-run]

The output goes to dataset_dir/splits/ and can be pointed to directly:
    GraphIndex("data/pretokenized_datasets/simplewiki/splits/train")
    GraphIndex("data/pretokenized_datasets/simplewiki/splits/val_community")
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

ALL_SPLITS = ("train", "val_community", "val_random", "test_community", "test_random")


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def _load_graph(dataset_dir: Path):
    """Load tokenized_graph.jsonl.

    Returns:
        nodes: list of raw node dicts in load order
        normed_ids: list of normed_identifier strings (index = position in file)
        nid_to_idx: normed_identifier -> position index
        out_adj: position -> list of outgoing neighbour positions
        in_adj:  position -> list of incoming neighbour positions
    """
    graph_path = dataset_dir / "tokenized_graph.jsonl"
    if not graph_path.exists():
        raise FileNotFoundError(f"tokenized_graph.jsonl not found in {dataset_dir}")

    nodes: List[dict] = []
    normed_ids: List[str] = []
    nid_to_idx: Dict[str, int] = {}

    with open(graph_path, encoding="utf-8") as f:
        for line in f:
            node = json.loads(line)
            nid = node["normed_identifier"]
            nid_to_idx[nid] = len(normed_ids)
            normed_ids.append(nid)
            nodes.append(node)

    n = len(nodes)
    out_adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    in_adj:  Dict[int, List[int]] = {i: [] for i in range(n)}

    for src_idx, node in enumerate(nodes):
        for tgt_nid in node.get("outgoing", []):
            tgt_idx = nid_to_idx.get(tgt_nid)
            if tgt_idx is not None:
                out_adj[src_idx].append(tgt_idx)
                in_adj[tgt_idx].append(src_idx)

    logger.info("Loaded %d nodes from %s", n, dataset_dir)
    return nodes, normed_ids, nid_to_idx, out_adj, in_adj


# ---------------------------------------------------------------------------
# Community extraction via BFS
# ---------------------------------------------------------------------------

def _extract_communities(
    n_nodes: int,
    out_adj: Dict[int, List[int]],
    in_adj: Dict[int, List[int]],
    target_node_count: int,
    community_size_min: int,
    community_size_max: int,
    rng,
    excluded: Set[int],
) -> List[List[int]]:
    """Extract BFS communities until target_node_count nodes are collected.

    Seeds are tried in descending degree order. Communities smaller than
    community_size_min are discarded (isolated/stub nodes go to train).
    BFS is bidirectional so communities capture both linkers and targets.
    """
    import random

    degree = [len(out_adj[i]) + len(in_adj[i]) for i in range(n_nodes)]
    candidates = sorted(range(n_nodes), key=lambda i: degree[i], reverse=True)
    rng.shuffle(candidates)

    communities: List[List[int]] = []
    total_collected = 0

    for seed in candidates:
        if total_collected >= target_node_count:
            break
        if seed in excluded:
            continue

        visited: List[int] = []
        queue = collections.deque([seed])
        seen: Set[int] = {seed}

        while queue and len(visited) < community_size_max:
            node = queue.popleft()
            visited.append(node)
            for nbr in out_adj[node] + in_adj[node]:
                if nbr not in seen and nbr not in excluded:
                    seen.add(nbr)
                    queue.append(nbr)

        if len(visited) < community_size_min:
            continue

        communities.append(visited)
        excluded.update(visited)
        total_collected += len(visited)

    return communities


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_splits(
    nodes: List[dict],
    normed_ids: List[str],
    out_adj: Dict[int, List[int]],
    in_adj: Dict[int, List[int]],
    val_frac: float,
    test_frac: float,
    community_size_min: int,
    community_size_max: int,
    seed: int,
) -> Dict[str, List[int]]:
    """Return a dict mapping split name -> list of node indices."""
    import random

    n = len(nodes)
    rng = random.Random(seed)
    assigned: Set[int] = set()

    community_target = int(round(n * (val_frac + test_frac)))

    logger.info(
        "Extracting communities: target=%d nodes (%.1f%% of %d), size [%d, %d]",
        community_target, (val_frac + test_frac) * 100, n,
        community_size_min, community_size_max,
    )
    communities = _extract_communities(
        n_nodes=n,
        out_adj=out_adj,
        in_adj=in_adj,
        target_node_count=community_target,
        community_size_min=community_size_min,
        community_size_max=community_size_max,
        rng=rng,
        excluded=assigned,
    )

    # Shuffle so val/test communities are interspersed across the degree spectrum.
    rng.shuffle(communities)
    all_community_ids = [idx for comm in communities for idx in comm]
    midpoint = len(all_community_ids) // 2
    val_community  = all_community_ids[:midpoint]
    test_community = all_community_ids[midpoint:]

    # Random splits from remaining nodes.
    remaining = [i for i in range(n) if i not in assigned]
    rng.shuffle(remaining)
    val_n   = int(round(n * val_frac))
    test_n  = int(round(n * test_frac))
    val_random   = remaining[:val_n]
    test_random  = remaining[val_n : val_n + test_n]
    train        = remaining[val_n + test_n:]

    split_map = {
        "train":          train,
        "val_community":  val_community,
        "val_random":     val_random,
        "test_community": test_community,
        "test_random":    test_random,
    }

    counts = {k: len(v) for k, v in split_map.items()}
    assert sum(counts.values()) == n, f"Sum {sum(counts.values())} != {n}"
    logger.info("Split counts: %s", counts)
    return split_map


# ---------------------------------------------------------------------------
# Writing split subdirectories
# ---------------------------------------------------------------------------

def write_splits(
    dataset_dir: Path,
    nodes: List[dict],
    normed_ids: List[str],
    split_map: Dict[str, List[int]],
    parent_metadata: dict,
) -> None:
    """Write one subdirectory per split under dataset_dir/splits/.

    Each subdir contains:
      - tokenized_graph.jsonl: only nodes in this split; outgoing/incoming
        edges filtered to reference only same-split nodes.
      - metadata.json: parent metadata with shard_filenames as absolute paths
        (shards are shared, not copied).
    """
    splits_dir = dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # Build per-split node-index sets for fast edge filtering.
    split_sets: Dict[str, Set[int]] = {k: set(v) for k, v in split_map.items()}

    # Resolve shard filenames to absolute paths once.
    abs_shards = [
        str((dataset_dir / fname).resolve())
        for fname in parent_metadata.get("shard_filenames", [])
    ]
    split_metadata = {**parent_metadata, "shard_filenames": abs_shards}

    for split_name, indices in split_map.items():
        split_dir = splits_dir / split_name
        split_dir.mkdir(exist_ok=True)

        idx_set = split_sets[split_name]
        nid_set = {normed_ids[j] for j in idx_set}

        # Write tokenized_graph.jsonl with edges filtered to same-split nodes.
        jsonl_path = split_dir / "tokenized_graph.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for idx in indices:
                node = dict(nodes[idx])
                node["outgoing"] = [nid for nid in node.get("outgoing", []) if nid in nid_set]
                node["incoming"] = [nid for nid in node.get("incoming", []) if nid in nid_set]
                f.write(json.dumps(node) + "\n")

        # Write metadata.json.
        meta_path = split_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(split_metadata, f, ensure_ascii=False)

        logger.info(
            "Wrote %s: %d nodes → %s",
            split_name, len(indices), split_dir,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Split a pretokenized graph into train/val/test subdirectories. "
            "Output: dataset_dir/splits/{train,val_community,val_random,"
            "test_community,test_random}/ — each a self-contained GraphIndex "
            "directory with filtered edges and shared binary shards."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to pretokenized dataset directory.",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.025,
        help="Fraction of nodes for each val split (community + random).",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.025,
        help="Fraction of nodes for each test split (community + random).",
    )
    parser.add_argument(
        "--community-size-min", type=int, default=50,
        help="Discard BFS communities smaller than this.",
    )
    parser.add_argument(
        "--community-size-max", type=int, default=5000,
        help="Cap BFS expansion per community at this many nodes.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print split counts without writing any files.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    nodes, normed_ids, nid_to_idx, out_adj, in_adj = _load_graph(dataset_dir)

    split_map = assign_splits(
        nodes=nodes,
        normed_ids=normed_ids,
        out_adj=out_adj,
        in_adj=in_adj,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        community_size_min=args.community_size_min,
        community_size_max=args.community_size_max,
        seed=args.seed,
    )

    n = len(nodes)
    print(f"\nSplit summary ({n} total nodes, seed={args.seed}):")
    for split_name in ALL_SPLITS:
        count = len(split_map[split_name])
        print(f"  {split_name:<18} {count:>7}  ({count / n * 100:.2f}%)")

    if args.dry_run:
        print("\n[dry-run] Not writing splits.")
        return

    meta_path = dataset_dir / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        parent_metadata = json.load(f)

    write_splits(
        dataset_dir=dataset_dir,
        nodes=nodes,
        normed_ids=normed_ids,
        split_map=split_map,
        parent_metadata=parent_metadata,
    )
    print(f"\nSplits written to {dataset_dir / 'splits'}/")


if __name__ == "__main__":
    main()
