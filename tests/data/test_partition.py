"""
Tests for document partitioning in epoch_precompute.

Covers both partition keys:
  - _partition_repos          : TheStack "owner/repo:path" identifiers
  - _partition_graph_communities : flat-identifier BFS Voronoi (Wikipedia/ArXiv)
  - _partition_documents      : dispatch between the two

The graph-community tests use a synthetic in-memory graph (no real dataset) and
verify: full coverage with no duplicates, the size cap, that connected
components stay together, and that the intra-shard edge fraction beats naive
random chunking (the whole point of the partitioner).
"""

import collections
import random

import pytest

from data.epoch_precompute import (
    _is_repo_partitioned,
    _partition_documents,
    _partition_graph_communities,
    _partition_repos,
)


# ---------------------------------------------------------------------------
# Synthetic graph stand-in
# ---------------------------------------------------------------------------

class _FakeGraph:
    """Minimal GraphIndex stand-in: integer doc ids 0..n-1 with directed edges.

    ``edges`` maps doc_id -> list of outgoing neighbor doc_ids.  ``incoming`` is
    derived.  ``identifiers`` controls get_normed_identifier for dispatch tests.
    """

    def __init__(self, n, edges=None, identifiers=None):
        self._n = n
        self._out = collections.defaultdict(list)
        self._in = collections.defaultdict(list)
        for src, dsts in (edges or {}).items():
            for dst in dsts:
                self._out[src].append(dst)
                self._in[dst].append(src)
        self._identifiers = identifiers

    def __len__(self):
        return self._n

    def get_normed_identifier(self, doc_id):
        if self._identifiers is not None:
            return self._identifiers[doc_id]
        return f"doc_{doc_id}"

    def neighbors_out(self, doc_id):
        return list(self._out[doc_id])

    def neighbors_in(self, doc_id):
        return list(self._in[doc_id])


def _disjoint_cliques(n_cliques, clique_size):
    """Build n_cliques disjoint fully-connected components.  Returns (graph, comp_of)."""
    edges = {}
    comp_of = {}
    doc = 0
    members = []
    for c in range(n_cliques):
        ids = list(range(doc, doc + clique_size))
        members.append(ids)
        for i in ids:
            comp_of[i] = c
            edges[i] = [j for j in ids if j != i]
        doc += clique_size
    return _FakeGraph(doc, edges), comp_of, members


def _assert_valid_partition(shards, n):
    """Every doc in 0..n-1 appears in exactly one shard."""
    flat = [d for s in shards for d in s]
    assert sorted(flat) == list(range(n)), "partition is not a permutation of doc ids"
    assert len(flat) == len(set(flat)), "duplicate doc ids across shards"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_repo_identifiers_detected(self):
        g = _FakeGraph(2, identifiers=["owner/repo_x:a.py", "owner/repo_x:b.py"])
        assert _is_repo_partitioned(g) is True

    def test_flat_identifiers_detected(self):
        g = _FakeGraph(2, identifiers=["United_States_abc", "Canada_def"])
        assert _is_repo_partitioned(g) is False

    def test_empty_graph_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _is_repo_partitioned(_FakeGraph(0))

    def test_dispatch_routes_flat_to_communities(self):
        # 3 disjoint cliques, flat ids → community partition keeps cliques whole.
        g, comp_of, _ = _disjoint_cliques(n_cliques=3, clique_size=6)
        shards = _partition_documents(g, n_workers=3, seed=0)
        _assert_valid_partition(shards, len(g))


# ---------------------------------------------------------------------------
# Graph-community partitioner
# ---------------------------------------------------------------------------

class TestGraphCommunities:
    def test_full_coverage_no_duplicates(self):
        g, _, _ = _disjoint_cliques(n_cliques=8, clique_size=10)
        shards = _partition_graph_communities(g, n_workers=4, seed=7)
        _assert_valid_partition(shards, len(g))

    def test_single_worker_returns_everything(self):
        g, _, _ = _disjoint_cliques(n_cliques=4, clique_size=5)
        shards = _partition_graph_communities(g, n_workers=1, seed=0)
        assert len(shards) == 1
        assert sorted(shards[0]) == list(range(len(g)))

    def test_isolated_nodes_distributed(self):
        # No edges at all → every node is a leftover, assigned round-robin.
        g = _FakeGraph(20, edges={})
        shards = _partition_graph_communities(g, n_workers=4, seed=1)
        _assert_valid_partition(shards, 20)
        # Round-robin of 20 isolated nodes over 4 workers → 5 each.
        assert all(len(s) == 5 for s in shards)

    def test_size_cap_respected(self):
        # One giant hub connected to everything; cap must stop it swallowing all.
        n = 100
        edges = {0: list(range(1, n))}  # node 0 links to all others
        g = _FakeGraph(n, edges=edges)
        n_workers = 5
        cap_factor = 1.5
        shards = _partition_graph_communities(
            g, n_workers=n_workers, seed=3, cap_factor=cap_factor,
        )
        _assert_valid_partition(shards, n)
        cap = int(cap_factor * n / n_workers)
        # The hub's cell may overshoot the cap by at most the neighbors enqueued
        # for a single dequeued node before the mid-expansion break; with a star
        # graph that break fires immediately, so no shard wildly exceeds cap.
        for s in shards:
            assert len(s) <= cap + 1, f"shard size {len(s)} exceeds cap {cap}"

    def test_connected_components_mostly_stay_together(self):
        # BFS claims a whole component from a single seed, so a component only
        # fragments when >1 seed lands inside it (inherent to multi-source
        # Voronoi).  With many more components than workers, that's rare — the
        # vast majority of components should stay intact on one shard.
        g, comp_of, members = _disjoint_cliques(n_cliques=40, clique_size=8)
        shards = _partition_graph_communities(g, n_workers=4, seed=11, cap_factor=3.0)
        _assert_valid_partition(shards, len(g))
        shard_of = {}
        for w, s in enumerate(shards):
            for d in s:
                shard_of[d] = w
        intact = sum(1 for ids in members if len({shard_of[d] for d in ids}) == 1)
        assert intact >= 0.8 * len(members), (
            f"only {intact}/{len(members)} components stayed intact on one shard"
        )

    def test_intra_shard_edge_fraction_beats_random(self):
        """The partitioner should keep far more edges intra-shard than random chunking.

        Build a graph with strong community structure (dense within blocks, sparse
        across), then compare the fraction of edges whose endpoints land on the
        same worker under the community partition vs. a random round-robin chunk.
        """
        rng = random.Random(123)
        n_blocks, block = 10, 40
        n = n_blocks * block
        edges = collections.defaultdict(list)
        for b in range(n_blocks):
            base = b * block
            ids = list(range(base, base + block))
            # dense intra-block edges
            for i in ids:
                for j in rng.sample(ids, 8):
                    if i != j:
                        edges[i].append(j)
            # a few cross-block edges
            for i in rng.sample(ids, 4):
                edges[i].append(rng.randrange(n))
        g = _FakeGraph(n, edges=edges)

        n_workers = 10
        comm = _partition_graph_communities(g, n_workers=n_workers, seed=5, cap_factor=1.5)
        _assert_valid_partition(comm, n)

        # naive random chunk baseline
        order = list(range(n))
        random.Random(5).shuffle(order)
        rand_shards = [order[i::n_workers] for i in range(n_workers)]

        def intra_fraction(shards):
            shard_of = {}
            for w, s in enumerate(shards):
                for d in s:
                    shard_of[d] = w
            tot = same = 0
            for src, dsts in edges.items():
                for dst in dsts:
                    tot += 1
                    if shard_of[src] == shard_of[dst]:
                        same += 1
            return same / tot

        comm_frac = intra_fraction(comm)
        rand_frac = intra_fraction(rand_shards)
        # Random chunking keeps ~1/n_workers of edges intra-shard (~0.1 here).
        # The community partition should be dramatically higher.
        assert comm_frac > 0.6, f"community intra-shard fraction too low: {comm_frac:.2f}"
        assert comm_frac > 3 * rand_frac, (
            f"community ({comm_frac:.2f}) not clearly better than random ({rand_frac:.2f})"
        )


# ---------------------------------------------------------------------------
# Repo partitioner (TheStack) — unchanged behavior
# ---------------------------------------------------------------------------

class TestRepoPartition:
    def test_files_in_repo_stay_together(self):
        ids = [
            "a/r1:f1.py", "a/r1:f2.py", "a/r1:f3.py",
            "b/r2:g1.py", "b/r2:g2.py",
            "c/r3:h1.py",
        ]
        g = _FakeGraph(len(ids), identifiers=ids)
        shards = _partition_repos(g, n_workers=2, seed=0)
        _assert_valid_partition(shards, len(ids))
        shard_of = {}
        for w, s in enumerate(shards):
            for d in s:
                shard_of[d] = w
        # All files sharing a repo prefix must be on the same shard.
        repo_to_shards = collections.defaultdict(set)
        for d, ident in enumerate(ids):
            repo_to_shards[ident.split(":")[0]].add(shard_of[d])
        for repo, ws in repo_to_shards.items():
            assert len(ws) == 1, f"repo {repo} split across shards {ws}"
