"""
Zig resolution-axis self-test (mirrors Go/Java/Python/TypeScript).

ZigImportDetector + path-keyed file-node model + PretokCorpus resolution must
recover the hand-labeled edges on a multi-file fixture, in repo-relative ``.zig``
path key space. Confirms: a sibling import resolves; a subdir import resolves; an
up-dir (``../``) import resolves; bare stdlib imports (``std``, ``builtin``)
correctly do NOT resolve (excluded from edges.json); an ``@import`` inside a ``//``
comment and one inside a string literal correctly produce NO edge.

Because Zig relative imports need the importing file's path to resolve, the fixture
uses ``detect_links_for_doc`` (the per-doc, path-aware method), like Python/TS.
"""
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

import tiktoken
import torch

from data.graph_harness.fixtures import score_resolution
from data.graph_harness.zig_nodes import build_zig_file_nodes
from model.graph_traversal.zig_import_detector import ZigImportDetector

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "graph_harness" / "fixtures_data"


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


def test_zig_resolution_on_fixture(enc):
    detector = ZigImportDetector(decode_fn=enc.decode)

    def edge_producer(nodes):
        edges = []
        for n in nodes:
            ids = torch.tensor(enc.encode(n.content), dtype=torch.long)
            for li in detector.detect_links_for_doc(ids, n.raw_identifier):
                edges.append((n.key, li.target_str))
        return edges

    score = score_resolution(
        FIXTURES / "zig" / "simple_pkg",
        extensions={"zig"},
        edge_producer=edge_producer,
        link_detector=detector,
        node_builder=build_zig_file_nodes,
    )
    assert score.recall == 1.0, f"missed: {score.missing_edges}; {score.summary()}"
    assert score.precision == 1.0, f"spurious: {score.wrong_edges}; {score.summary()}"
