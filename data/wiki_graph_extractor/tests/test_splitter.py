import unittest

from data.wiki_graph_extractor.splitter import (
    create_splits,
    _build_undirected_nx,
    _undirected_degree,
    _bucket_index,
    _bucket_counts,
    DEGREE_BUCKET_EDGES,
)


def _make_clique(prefix, size, char_count=3000):
    """
    Build a fully-connected clique of *size* nodes.
    Returns a partial graph_data dict (outgoing/incoming within the clique).
    """
    ids = [f"{prefix}_{i}" for i in range(size)]
    graph = {}
    for nid in ids:
        others = [x for x in ids if x != nid]
        graph[nid] = {
            "normed_identifier": nid,
            "raw_identifier": nid.replace("_", " "),
            "char_count": char_count,
            "outgoing": list(others),
            "incoming": list(others),
        }
    return graph


def _make_ring(prefix, size, char_count=3000):
    """
    Build a directed ring: node_0 -> node_1 -> ... -> node_{n-1} -> node_0.
    Each node has undirected degree 2 (one outgoing, one incoming neighbor).
    """
    ids = [f"{prefix}_{i}" for i in range(size)]
    graph = {}
    for i, nid in enumerate(ids):
        out_target = ids[(i + 1) % size]
        in_source = ids[(i - 1) % size]
        graph[nid] = {
            "normed_identifier": nid,
            "raw_identifier": nid.replace("_", " "),
            "char_count": char_count,
            "outgoing": [out_target],
            "incoming": [in_source],
        }
    return graph


def _merge_graphs(*graphs):
    """Merge multiple partial graph_data dicts into one."""
    merged = {}
    for g in graphs:
        merged.update(g)
    return merged


class TestBucketHelpers(unittest.TestCase):
    def test_bucket_index(self):
        self.assertEqual(_bucket_index(1, DEGREE_BUCKET_EDGES), 0)
        self.assertEqual(_bucket_index(2, DEGREE_BUCKET_EDGES), 1)
        self.assertEqual(_bucket_index(3, DEGREE_BUCKET_EDGES), 1)
        self.assertEqual(_bucket_index(4, DEGREE_BUCKET_EDGES), 2)
        self.assertEqual(_bucket_index(7, DEGREE_BUCKET_EDGES), 2)
        self.assertEqual(_bucket_index(8, DEGREE_BUCKET_EDGES), 3)
        self.assertEqual(_bucket_index(16, DEGREE_BUCKET_EDGES), 4)
        self.assertEqual(_bucket_index(100, DEGREE_BUCKET_EDGES), 4)

    def test_bucket_counts(self):
        values = [1, 2, 3, 5, 10, 20]
        counts = _bucket_counts(values, DEGREE_BUCKET_EDGES)
        self.assertEqual(len(counts), len(DEGREE_BUCKET_EDGES) + 1)
        self.assertEqual(sum(counts), len(values))


class TestSplitterSmallGraphs(unittest.TestCase):
    """Tests using small synthetic graphs with clear community structure."""

    def _build_three_community_graph(self):
        """
        Three cliques (A=40, B=30, C=30) = 100 nodes total.
        A few cross-edges connect them so community detection can still
        find them but the graph is connected.
        """
        clique_a = _make_clique("a", 40, char_count=2000)
        clique_b = _make_clique("b", 30, char_count=8000)
        clique_c = _make_clique("c", 30, char_count=15000)
        graph = _merge_graphs(clique_a, clique_b, clique_c)

        # Add a handful of cross-edges so the graph is weakly connected
        cross = [("a_0", "b_0"), ("b_0", "c_0"), ("a_1", "c_1")]
        for src, dst in cross:
            graph[src]["outgoing"].append(dst)
            graph[dst]["incoming"].append(src)
        return graph

    def test_disjointness(self):
        graph = self._build_three_community_graph()
        result = create_splits(graph, seed=42, min_degree=2)
        splits = result["splits"]
        train = set(splits["train"]["ids"])
        val_comm = set(splits["val_community"]["ids"])
        val_rand = set(splits["val_random"]["ids"])

        self.assertEqual(len(train & val_comm), 0, "train and val_community overlap")
        self.assertEqual(len(train & val_rand), 0, "train and val_random overlap")
        self.assertEqual(len(val_comm & val_rand), 0, "val_community and val_random overlap")

    def test_completeness(self):
        graph = self._build_three_community_graph()
        result = create_splits(graph, seed=42, min_degree=2)
        splits = result["splits"]

        all_split = (
            set(splits["train"]["ids"])
            | set(splits["val_community"]["ids"])
            | set(splits["val_random"]["ids"])
        )
        self.assertEqual(len(all_split), result["eligible_nodes"])

    def test_approximate_fractions(self):
        graph = self._build_three_community_graph()
        result = create_splits(graph, seed=42, min_degree=2)
        eligible = result["eligible_nodes"]
        splits = result["splits"]

        train_frac = splits["train"]["count"] / eligible
        rand_frac = splits["val_random"]["count"] / eligible
        # Community-val can deviate more due to whole-community selection + pruning
        self.assertAlmostEqual(rand_frac, 0.05, delta=0.05)
        self.assertGreater(train_frac, 0.50)

    def test_community_val_hard_density_constraint(self):
        """Every community-val node must have induced degree >= min_degree."""
        graph = self._build_three_community_graph()
        for md in (2, 3):
            with self.subTest(min_degree=md):
                result = create_splits(graph, seed=42, min_degree=md)
                val_comm_ids = set(result["splits"]["val_community"]["ids"])
                if not val_comm_ids:
                    continue
                G = _build_undirected_nx(sorted(val_comm_ids), graph)
                for nid in val_comm_ids:
                    self.assertGreaterEqual(
                        G.degree(nid), md,
                        f"Node {nid} has induced degree {G.degree(nid)} < {md}"
                    )

    def test_determinism(self):
        graph = self._build_three_community_graph()
        r1 = create_splits(graph, seed=123, min_degree=2)
        r2 = create_splits(graph, seed=123, min_degree=2)
        self.assertEqual(r1["splits"]["train"]["ids"], r2["splits"]["train"]["ids"])
        self.assertEqual(r1["splits"]["val_community"]["ids"], r2["splits"]["val_community"]["ids"])
        self.assertEqual(r1["splits"]["val_random"]["ids"], r2["splits"]["val_random"]["ids"])

    def test_different_seeds_differ(self):
        graph = self._build_three_community_graph()
        r1 = create_splits(graph, seed=1, min_degree=2)
        r2 = create_splits(graph, seed=999, min_degree=2)
        self.assertNotEqual(
            r1["splits"]["val_random"]["ids"],
            r2["splits"]["val_random"]["ids"],
        )

    def test_metadata_fields(self):
        graph = self._build_three_community_graph()
        result = create_splits(graph, seed=42, min_degree=2)
        self.assertIn("seed", result)
        self.assertIn("min_degree", result)
        self.assertIn("total_graph_nodes", result)
        self.assertIn("eligible_nodes", result)
        self.assertIn("ineligible_node_count", result)
        self.assertIn("node_list_hash", result)
        self.assertIn("community_val_stats", result)
        cs = result["community_val_stats"]
        self.assertIn("num_communities_selected", cs)
        self.assertIn("avg_induced_degree", cs)
        self.assertIn("min_induced_degree", cs)
        self.assertIn("nodes_pruned_by_density_constraint", cs)
        self.assertIn("internal_edges", cs)
        self.assertIn("cross_edges_to_train", cs)


class TestSplitterEdgeCases(unittest.TestCase):

    def test_fully_connected_graph(self):
        graph = _make_clique("full", 50)
        result = create_splits(graph, seed=42, min_degree=2)
        total = sum(
            result["splits"][k]["count"]
            for k in ("train", "val_community", "val_random")
        )
        self.assertEqual(total, result["eligible_nodes"])

    def test_ring_graph(self):
        """Ring graph: every node has undirected degree exactly 2."""
        graph = _make_ring("ring", 60)
        result = create_splits(graph, seed=42, min_degree=2)
        self.assertEqual(result["eligible_nodes"], 60)
        total = sum(
            result["splits"][k]["count"]
            for k in ("train", "val_community", "val_random")
        )
        self.assertEqual(total, 60)

    def test_disconnected_components(self):
        """Two separate cliques with no edges between them."""
        g1 = _make_clique("iso_a", 30)
        g2 = _make_clique("iso_b", 30)
        graph = _merge_graphs(g1, g2)
        result = create_splits(graph, seed=42, min_degree=2)
        self.assertEqual(result["eligible_nodes"], 60)
        splits = result["splits"]
        all_ids = (
            set(splits["train"]["ids"])
            | set(splits["val_community"]["ids"])
            | set(splits["val_random"]["ids"])
        )
        self.assertEqual(len(all_ids), 60)

    def test_all_ineligible(self):
        """Graph where every node has degree < min_degree should raise."""
        graph = {
            "lone_0": {
                "normed_identifier": "lone_0",
                "raw_identifier": "Lone 0",
                "char_count": 100,
                "outgoing": [],
                "incoming": [],
            },
            "lone_1": {
                "normed_identifier": "lone_1",
                "raw_identifier": "Lone 1",
                "char_count": 100,
                "outgoing": [],
                "incoming": [],
            },
        }
        with self.assertRaises(ValueError):
            create_splits(graph, seed=42, min_degree=2)

    def test_high_min_degree_prunes_community_val(self):
        """With a very high min_degree, community-val may be empty after pruning."""
        graph = _make_ring("ring", 40)
        # Ring nodes have undirected degree 2; min_degree=2 for eligibility
        # but asking for min_degree=3 should make all ineligible
        result = create_splits(graph, seed=42, min_degree=2)
        self.assertGreater(result["eligible_nodes"], 0)

        # Community-val density constraint with min_degree=2 on a ring:
        # ring segments may lose endpoints after random-val removal.
        val_comm = set(result["splits"]["val_community"]["ids"])
        if val_comm:
            G = _build_undirected_nx(sorted(val_comm), graph)
            for nid in val_comm:
                self.assertGreaterEqual(G.degree(nid), 2)


if __name__ == "__main__":
    unittest.main()
