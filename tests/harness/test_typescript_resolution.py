"""
TypeScript resolution-axis self-test (mirrors Go/Java/Python).

TypeScriptImportDetector + path-keyed file-node model + PretokCorpus resolution
must recover the hand-labeled edges on a multi-file fixture, in repo-relative
(extension-less) path key space. Confirms: relative imports resolve; a directory
import resolves to its ``index`` file; a ``../`` up-dir import resolves; a re-export
(``export ... from``) and a type-only import both create file edges; a bare
external import (``react``) and a bare ``require("lodash")`` correctly do NOT
resolve (excluded from edges.json).

Because TS relative imports need the importing file's path to resolve, the fixture
uses ``detect_links_for_doc`` (the per-doc, path-aware method), like Python.
"""
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

import tiktoken
import torch

from data.graph_harness.fixtures import score_resolution
from data.graph_harness.typescript_nodes import build_typescript_file_nodes
from model.graph_traversal.typescript_import_detector import TypeScriptImportDetector

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "graph_harness" / "fixtures_data"


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


def test_typescript_resolution_on_fixture(enc):
    detector = TypeScriptImportDetector(decode_fn=enc.decode)

    def edge_producer(nodes):
        edges = []
        for n in nodes:
            ids = torch.tensor(enc.encode(n.content), dtype=torch.long)
            for li in detector.detect_links_for_doc(ids, n.raw_identifier):
                edges.append((n.key, li.target_str))
        return edges

    score = score_resolution(
        FIXTURES / "typescript" / "simple_pkg",
        extensions={"ts", "tsx"},
        edge_producer=edge_producer,
        link_detector=detector,
        node_builder=build_typescript_file_nodes,
    )
    assert score.recall == 1.0, f"missed: {score.missing_edges}; {score.summary()}"
    assert score.precision == 1.0, f"spurious: {score.wrong_edges}; {score.summary()}"
