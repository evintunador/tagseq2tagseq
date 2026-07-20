"""
Kotlin resolution-axis self-test (mirrors Go/Java/Python).

KotlinImportDetector + SYMBOL->FILE (per-FQN) node model + PretokCorpus resolution
must recover the hand-labeled edges on a multi-file fixture, in dotted-FQN key
space. Confirms:
  * a single import (com.ex.util.Helper) resolves to the file declaring Helper;
  * a top-level FUNCTION import (com.ex.util.helperFn) resolves to the SAME file
    (multi-symbol-per-file — the case the symbol->file model exists for);
  * an object import (com.ex.Consts) resolves;
  * an alias import (com.ex.foo.bar as Baz) resolves to the FQN with the alias
    STRIPPED (and since nothing declares com.ex.foo.bar, produces no edge — not a
    spurious one);
  * a wildcard import (com.ex.*) is dropped (no edge);
  * a stdlib import (kotlin.collections.List) does NOT resolve (excluded from edges).
"""
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_kotlin")

import tiktoken
import torch

from data.graph_harness.fixtures import score_resolution
from data.graph_harness.kotlin_nodes import build_kotlin_file_nodes
from model.graph_traversal.kotlin_import_detector import KotlinImportDetector

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "graph_harness" / "fixtures_data"


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


def test_kotlin_resolution_on_fixture(enc):
    detector = KotlinImportDetector(decode_fn=enc.decode)

    def edge_producer(nodes):
        # Multiple nodes may share one file's content (sibling top-level symbols);
        # detect_links per node key is correct — the (from_key, target) pairs are
        # deduped in set space by the runner.
        edges = []
        for n in nodes:
            ids = torch.tensor(enc.encode(n.content), dtype=torch.long)
            for li in detector.detect_links(ids):
                edges.append((n.key, li.target_str))
        return edges

    score = score_resolution(
        FIXTURES / "kotlin" / "simple_pkg",
        extensions={"kt"},
        edge_producer=edge_producer,
        link_detector=detector,
        node_builder=build_kotlin_file_nodes,
    )
    assert score.recall == 1.0, f"missed: {score.missing_edges}; {score.summary()}"
    assert score.precision == 1.0, f"spurious: {score.wrong_edges}; {score.summary()}"
