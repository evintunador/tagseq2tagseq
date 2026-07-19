"""
Java resolution-axis self-test (mirrors Go/Python).

JavaImportDetector + FQN file-node model + PretokCorpus resolution must recover the
hand-labeled edges on a multi-file fixture, in FQN key space. Confirms: type
imports resolve; a static import resolves to the ENCLOSING type; a JDK import
(java.util.List) correctly does NOT resolve (excluded from edges.json).
"""
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

import tiktoken
import torch

from data.graph_harness.fixtures import score_resolution
from data.graph_harness.java_nodes import build_java_file_nodes
from model.graph_traversal.java_import_detector import JavaImportDetector

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "graph_harness" / "fixtures_data"


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


def test_java_resolution_on_fixture(enc):
    detector = JavaImportDetector(decode_fn=enc.decode)

    def edge_producer(nodes):
        edges = []
        for n in nodes:
            ids = torch.tensor(enc.encode(n.content), dtype=torch.long)
            for li in detector.detect_links(ids):
                edges.append((n.key, li.target_str))
        return edges

    score = score_resolution(
        FIXTURES / "java" / "simple_pkg",
        extensions={"java"},
        edge_producer=edge_producer,
        link_detector=detector,
        node_builder=build_java_file_nodes,
    )
    assert score.recall == 1.0, f"missed: {score.missing_edges}; {score.summary()}"
    assert score.precision == 1.0, f"spurious: {score.wrong_edges}; {score.summary()}"
