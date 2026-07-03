"""Tests for data/split_graph.py."""
import json
from pathlib import Path

import pytest

from data.split_graph import assign_splits, write_splits, _load_graph, ALL_SPLITS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_dataset(tmp_path: Path, nodes: list) -> Path:
    """Write a minimal pretokenized dataset to tmp_path."""
    (tmp_path / "tokenized_graph.jsonl").write_text(
        "\n".join(json.dumps(n) for n in nodes)
    )
    (tmp_path / "metadata.json").write_text(json.dumps({
        "dtype_str": "uint16",
        "shard_filenames": ["shard_000000.bin"],
        "tokenizer": "gpt2",
    }))
    # Create a dummy shard so absolute path resolution works.
    (tmp_path / "shard_000000.bin").write_bytes(b"")
    return tmp_path


def _linear_graph(n: int) -> list:
    """n nodes in a chain: 0→1→2→...→n-1."""
    return [
        {
            "normed_identifier": str(i),
            "raw_identifier": str(i),
            "outgoing": [str(i + 1)] if i < n - 1 else [],
            "incoming": [str(i - 1)] if i > 0 else [],
            "tok_shard_idx": 0,
            "tok_offset_bytes": i * 10,
            "tok_len": 5,
        }
        for i in range(n)
    ]


def _isolated_graph(n: int) -> list:
    """n fully isolated nodes (no edges)."""
    return [
        {
            "normed_identifier": str(i),
            "raw_identifier": str(i),
            "outgoing": [],
            "incoming": [],
            "tok_shard_idx": 0,
            "tok_offset_bytes": i * 10,
            "tok_len": 5,
        }
        for i in range(n)
    ]


def _cluster_graph(cluster_size: int, n_clusters: int) -> list:
    """n_clusters fully-connected cliques of cluster_size nodes each."""
    nodes = []
    idx = 0
    for _ in range(n_clusters):
        cluster_ids = list(range(idx, idx + cluster_size))
        for i in cluster_ids:
            nodes.append({
                "normed_identifier": str(i),
                "raw_identifier": str(i),
                "outgoing": [str(j) for j in cluster_ids if j != i],
                "incoming": [str(j) for j in cluster_ids if j != i],
                "tok_shard_idx": 0,
                "tok_offset_bytes": i * 10,
                "tok_len": 5,
            })
        idx += cluster_size
    return nodes


def _run_assign(tmp_path, nodes, **kwargs):
    dataset_dir = _write_dataset(tmp_path, nodes)
    loaded = _load_graph(dataset_dir)
    nodes_l, normed_ids, nid_to_idx, out_adj, in_adj = loaded
    params = dict(val_frac=0.025, test_frac=0.025, community_size_min=50,
                  community_size_max=5000, seed=42)
    params.update(kwargs)
    return assign_splits(nodes_l, normed_ids, out_adj, in_adj, **params), dataset_dir, loaded


# ---------------------------------------------------------------------------
# assign_splits tests
# ---------------------------------------------------------------------------

def test_all_nodes_assigned(tmp_path):
    nodes = _linear_graph(200)
    split_map, _, loaded = _run_assign(tmp_path, nodes)
    total = sum(len(v) for v in split_map.values())
    assert total == 200


def test_splits_are_disjoint(tmp_path):
    nodes = _linear_graph(400)
    split_map, _, _ = _run_assign(tmp_path, nodes)
    sets = [set(v) for v in split_map.values()]
    for i, s1 in enumerate(sets):
        for j, s2 in enumerate(sets):
            if i != j:
                assert s1.isdisjoint(s2)


def test_val_random_fraction_approximately_correct(tmp_path):
    n = 1000
    nodes = _isolated_graph(n)
    split_map, _, _ = _run_assign(tmp_path, nodes, community_size_min=50)
    expected = int(round(n * 0.025))
    assert abs(len(split_map["val_random"]) - expected) <= 1


def test_isolated_nodes_go_to_train(tmp_path):
    nodes = _isolated_graph(200)
    split_map, _, _ = _run_assign(tmp_path, nodes, community_size_min=50)
    assert len(split_map["val_community"]) == 0
    assert len(split_map["test_community"]) == 0


def test_community_splits_contain_connected_nodes(tmp_path):
    nodes = _cluster_graph(cluster_size=100, n_clusters=4)
    split_map, dataset_dir, loaded = _run_assign(
        tmp_path, nodes, val_frac=0.05, test_frac=0.05,
        community_size_min=10, community_size_max=200,
    )
    _, normed_ids, _, _, _ = loaded
    val_nids = set(normed_ids[i] for i in split_map["val_community"])
    if not val_nids:
        pytest.skip("No val_community nodes extracted.")
    nid_to_out = {n["normed_identifier"]: set(n["outgoing"]) for n in nodes}
    for nid in val_nids:
        assert nid_to_out.get(nid, set()) & val_nids, (
            f"Node {nid} in val_community has no neighbour also in val_community"
        )


def test_seed_determinism(tmp_path):
    nodes = _linear_graph(300)
    s1, _, loaded = _run_assign(tmp_path, nodes, seed=7)
    s2, _, _ = _run_assign(tmp_path, nodes, seed=7)
    assert s1 == s2


def test_different_seeds_produce_different_splits(tmp_path):
    nodes = _isolated_graph(300)
    s1, _, loaded1 = _run_assign(tmp_path, nodes, seed=1, community_size_min=50)
    s2, _, loaded2 = _run_assign(tmp_path, nodes, seed=2, community_size_min=50)
    _, nids1, _, _, _ = loaded1
    _, nids2, _, _, _ = loaded2
    ids1 = [nids1[i] for i in s1["val_random"]]
    ids2 = [nids2[i] for i in s2["val_random"]]
    assert ids1 != ids2


# ---------------------------------------------------------------------------
# write_splits tests
# ---------------------------------------------------------------------------

def test_write_splits_creates_subdirs(tmp_path):
    nodes = _isolated_graph(200)
    split_map, dataset_dir, loaded = _run_assign(tmp_path, nodes, community_size_min=50)
    nodes_l, normed_ids, _, _, _ = loaded
    parent_meta = json.loads((dataset_dir / "metadata.json").read_text())
    write_splits(dataset_dir, nodes_l, normed_ids, split_map, parent_meta)

    for split_name in ALL_SPLITS:
        split_dir = dataset_dir / "splits" / split_name
        assert split_dir.is_dir(), f"Missing split dir: {split_dir}"
        assert (split_dir / "tokenized_graph.jsonl").exists()
        assert (split_dir / "metadata.json").exists()


def test_write_splits_node_counts_correct(tmp_path):
    nodes = _isolated_graph(200)
    split_map, dataset_dir, loaded = _run_assign(tmp_path, nodes, community_size_min=50)
    nodes_l, normed_ids, _, _, _ = loaded
    parent_meta = json.loads((dataset_dir / "metadata.json").read_text())
    write_splits(dataset_dir, nodes_l, normed_ids, split_map, parent_meta)

    for split_name, indices in split_map.items():
        split_dir = dataset_dir / "splits" / split_name
        written = (split_dir / "tokenized_graph.jsonl").read_text().strip().splitlines()
        assert len(written) == len(indices), (
            f"{split_name}: expected {len(indices)} nodes, got {len(written)}"
        )


def test_write_splits_edges_filtered(tmp_path):
    """Cross-split edges must not appear in any split's JSONL."""
    nodes = _cluster_graph(cluster_size=100, n_clusters=4)
    split_map, dataset_dir, loaded = _run_assign(
        tmp_path, nodes, val_frac=0.05, test_frac=0.05,
        community_size_min=10, community_size_max=200,
    )
    nodes_l, normed_ids, _, _, _ = loaded
    parent_meta = json.loads((dataset_dir / "metadata.json").read_text())
    write_splits(dataset_dir, nodes_l, normed_ids, split_map, parent_meta)

    for split_name, indices in split_map.items():
        if not indices:
            continue
        split_dir = dataset_dir / "splits" / split_name
        nid_set = {normed_ids[i] for i in indices}
        for line in (split_dir / "tokenized_graph.jsonl").read_text().splitlines():
            node = json.loads(line)
            for tgt in node.get("outgoing", []):
                assert tgt in nid_set, (
                    f"Cross-split edge in {split_name}: {node['normed_identifier']} → {tgt}"
                )


def test_write_splits_metadata_has_absolute_shard_paths(tmp_path):
    nodes = _isolated_graph(100)
    split_map, dataset_dir, loaded = _run_assign(tmp_path, nodes, community_size_min=50)
    nodes_l, normed_ids, _, _, _ = loaded
    parent_meta = json.loads((dataset_dir / "metadata.json").read_text())
    write_splits(dataset_dir, nodes_l, normed_ids, split_map, parent_meta)

    train_meta = json.loads((dataset_dir / "splits" / "train" / "metadata.json").read_text())
    for shard_path in train_meta["shard_filenames"]:
        assert Path(shard_path).is_absolute(), f"Shard path not absolute: {shard_path}"


def test_graphindex_can_load_split_dir(tmp_path):
    """GraphIndex should load a split subdir without errors."""
    from data.dataset import GraphIndex
    nodes = _isolated_graph(200)
    split_map, dataset_dir, loaded = _run_assign(tmp_path, nodes, community_size_min=50)
    nodes_l, normed_ids, _, _, _ = loaded
    parent_meta = json.loads((dataset_dir / "metadata.json").read_text())
    write_splits(dataset_dir, nodes_l, normed_ids, split_map, parent_meta)

    train_dir = dataset_dir / "splits" / "train"
    graph = GraphIndex(train_dir)
    assert len(graph) == len(split_map["train"])


# ---------------------------------------------------------------------------
# Source-stratified split tests (merge_datasets provenance)
# ---------------------------------------------------------------------------

def _two_source_clusters(cluster_size: int, n_per_source: int) -> list:
    """n_per_source cliques tagged source 'a' then the same for source 'b'.

    Sources are disjoint (no cross-source edges), so a fair stratified split
    must draw community nodes from BOTH, not just the larger/denser one.
    """
    nodes = []
    idx = 0
    for source, count in (("a", n_per_source), ("b", n_per_source)):
        for _ in range(count):
            cluster_ids = list(range(idx, idx + cluster_size))
            for i in cluster_ids:
                nodes.append({
                    "normed_identifier": str(i),
                    "raw_identifier": str(i),
                    "source": source,
                    "outgoing": [str(j) for j in cluster_ids if j != i],
                    "incoming": [str(j) for j in cluster_ids if j != i],
                    "tok_shard_idx": 0,
                    "tok_offset_bytes": i * 10,
                    "tok_len": 5,
                })
            idx += cluster_size
    return nodes


def test_stratified_all_assigned_and_disjoint(tmp_path):
    nodes = _two_source_clusters(cluster_size=100, n_per_source=4)
    split_map, _, _ = _run_assign(
        tmp_path, nodes, stratify_by_source=True,
        val_frac=0.05, test_frac=0.05, community_size_min=50,
    )
    total = sum(len(v) for v in split_map.values())
    assert total == len(nodes)
    sets = [set(v) for v in split_map.values()]
    for i, s1 in enumerate(sets):
        for j, s2 in enumerate(sets):
            if i != j:
                assert s1.isdisjoint(s2)


def test_stratified_community_draws_from_both_sources(tmp_path):
    nodes = _two_source_clusters(cluster_size=100, n_per_source=4)
    split_map, dataset_dir, loaded = _run_assign(
        tmp_path, nodes, stratify_by_source=True,
        val_frac=0.05, test_frac=0.05, community_size_min=50,
    )
    _, normed_ids, _, _, _ = loaded
    # Map each community node back to its source.
    src_of = {n["normed_identifier"]: n["source"] for n in nodes}
    comm_sources = {
        src_of[normed_ids[idx]]
        for split in ("val_community", "test_community")
        for idx in split_map[split]
    }
    assert comm_sources == {"a", "b"}, (
        "stratified community split must include both sources"
    )
