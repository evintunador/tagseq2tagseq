"""
Dart resolution-axis self-test (mirrors Go/Java/Python/TypeScript).

DartImportDetector + path-keyed file-node model (keys keep the ``.dart`` extension)
+ PretokCorpus resolution must recover the hand-labeled edges on a multi-file
fixture, in repo-relative path key space. Confirms: relative imports resolve
(same-dir, subdir, ``../`` up-dir, explicit ``./``); an ``export`` re-export creates
a file edge; a combinator clause (``show User``) is stripped and the URI resolves;
a Dart SDK import (``dart:math``) and a pub-dep import (``package:flutter/...``)
correctly do NOT resolve (excluded from edges.json).

Because Dart relative imports need the importing file's path to resolve, the
fixture uses ``detect_links_for_doc`` (the per-doc, path-aware method), like Python
and TypeScript.
"""
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

import tiktoken
import torch

from data.graph_harness.fixtures import score_resolution
from data.graph_harness.dart_nodes import build_dart_file_nodes
from model.graph_traversal.dart_import_detector import DartImportDetector

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "graph_harness" / "fixtures_data"


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


def test_dart_resolution_on_fixture(enc):
    detector = DartImportDetector(decode_fn=enc.decode)

    def edge_producer(nodes):
        edges = []
        for n in nodes:
            ids = torch.tensor(enc.encode(n.content), dtype=torch.long)
            for li in detector.detect_links_for_doc(ids, n.raw_identifier):
                edges.append((n.key, li.target_str))
        return edges

    score = score_resolution(
        FIXTURES / "dart" / "simple_pkg",
        extensions={"dart"},
        edge_producer=edge_producer,
        link_detector=detector,
        node_builder=build_dart_file_nodes,
    )
    assert score.recall == 1.0, f"missed: {score.missing_edges}; {score.summary()}"
    assert score.precision == 1.0, f"spurious: {score.wrong_edges}; {score.summary()}"
