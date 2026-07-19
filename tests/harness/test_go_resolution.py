"""
Go resolution-axis self-test (mirrors the Python one).

The GoImportDetector + package-node model + PretokCorpus resolution must recover
exactly the hand-labeled intra-module edges on a multi-package fixture, scoring in
IMPORT-PATH (package) key space — the pilot's resolved node unit. Confirms:
  * package-node grouping (store's two files = one node);
  * exact-string resolution of full import paths;
  * stdlib imports (fmt) correctly do NOT resolve (excluded from edges.json).
"""
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

import tiktoken
import torch

from data.graph_harness.fixtures import score_resolution
from data.graph_harness.go_nodes import build_go_package_nodes
from model.graph_traversal.go_import_detector import GoImportDetector

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "graph_harness" / "fixtures_data"


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


def test_go_resolution_on_fixture(enc):
    detector = GoImportDetector(decode_fn=enc.decode)

    def edge_producer(nodes):
        # Each node is a package; run detect_links over the package's concatenated
        # source, emit (from_import_path, raw_target_str) for every detected import.
        edges = []
        for n in nodes:
            ids = torch.tensor(enc.encode(n.content), dtype=torch.long)
            for li in detector.detect_links(ids):
                edges.append((n.key, li.target_str))
        return edges

    score = score_resolution(
        FIXTURES / "go" / "simple_module",
        extensions={"go"},
        edge_producer=edge_producer,
        link_detector=detector,
        node_builder=build_go_package_nodes,
    )
    assert score.recall == 1.0, f"missed edges: {score.missing_edges}; {score.summary()}"
    assert score.precision == 1.0, f"spurious edges: {score.wrong_edges}; {score.summary()}"


def test_go_package_node_grouping(enc):
    """store/ has two .go files but must collapse to ONE package node."""
    from data.graph_harness.fixtures import _FixtureFile
    files = [
        _FixtureFile("go.mod", "module example.com/proj\n"),
        _FixtureFile("internal/store/store.go", "package store\n"),
        _FixtureFile("internal/store/store_helpers.go", "package store\n"),
        _FixtureFile("util/util.go", "package util\n"),
        _FixtureFile("util/util_test.go", "package util\n"),  # test excluded
    ]
    nodes = build_go_package_nodes(files, {"go"})
    keys = sorted(n.key for n in nodes)
    assert keys == ["example.com/proj/internal/store", "example.com/proj/util"]
    # store node concatenates both source files
    store = next(n for n in nodes if n.key.endswith("/store"))
    assert "store.go content" not in store.content  # sanity: not literal
    assert store.content.count("package store") == 2
