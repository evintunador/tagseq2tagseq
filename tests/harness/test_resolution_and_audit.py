"""
Resolution-axis + auditor self-tests.

Load-bearing (mirrors the detection test): the trusted PythonImportDetector +
PretokCorpus resolution logic must score high resolution precision/recall on a
hand-labeled fixture. And the auditor must compute correct structural metrics on a
synthetic graph with known dangling/self/isolated edges.
"""
from pathlib import Path

import pytest

tree_sitter = pytest.importorskip("tree_sitter")

import tiktoken
import torch

from data.graph_harness.fixtures import score_resolution
from model.graph_traversal.python_import_detector import PythonImportDetector

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "graph_harness" / "fixtures_data"


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


def test_python_resolution_on_fixture(enc):
    detector = PythonImportDetector(decode_fn=enc.decode)

    def edge_producer(nodes):
        # For each source node, run the per-doc detector (handles relative imports
        # too) and emit (from_relpath, raw_target_str) for every detected link.
        edges = []
        by_rel = {n.relpath: n for n in nodes}
        for n in nodes:
            if not n.relpath.endswith(".py"):
                continue
            ids = torch.tensor(enc.encode(n.content), dtype=torch.long)
            links = detector.detect_links_for_doc(ids, n.raw_identifier)
            for li in links:
                edges.append((n.relpath, li.target_str))
        return edges

    score = score_resolution(
        FIXTURES / "python" / "simple_pkg",
        extensions={"py"},
        edge_producer=edge_producer,
        link_detector=detector,
    )
    assert score.recall == 1.0, (
        f"missed real edges: {score.missing_edges}; {score.summary()}"
    )
    # precision may be <1 only if the detector resolves spurious edges; for this
    # fixture every real import resolves to exactly one node, so expect perfect.
    assert score.precision == 1.0, (
        f"resolved spurious edges: {score.wrong_edges}; {score.summary()}"
    )


def test_auditor_on_synthetic_graph(tmp_path):
    """Auditor computes dangling/self/isolated/reciprocal correctly."""
    import json
    # Build a minimal valid pretokenized dataset dir.
    # nodes: a<->b reciprocal, a->b, b->a; c self-link + dangling; d isolated.
    nodes = [
        {"normed_identifier": "r:a", "raw_identifier": "r:a",
         "outgoing": ["r:b"], "incoming": ["r:b"],
         "tok_shard_idx": 0, "tok_offset_bytes": 1024, "tok_len": 1},
        {"normed_identifier": "r:b", "raw_identifier": "r:b",
         "outgoing": ["r:a"], "incoming": ["r:a"],
         "tok_shard_idx": 0, "tok_offset_bytes": 1026, "tok_len": 1},
        {"normed_identifier": "r:c", "raw_identifier": "r:c",
         "outgoing": ["r:c", "r:missing"], "incoming": [],
         "tok_shard_idx": 0, "tok_offset_bytes": 1028, "tok_len": 1},
        {"normed_identifier": "r:d", "raw_identifier": "r:d",
         "outgoing": [], "incoming": [],
         "tok_shard_idx": 0, "tok_offset_bytes": 1030, "tok_len": 1},
    ]
    (tmp_path / "tokenized_graph.jsonl").write_text(
        "\n".join(json.dumps(n) for n in nodes)
    )
    (tmp_path / "metadata.json").write_text(json.dumps({
        "tokenizer": "gpt2", "dtype_str": "uint16", "shard_filenames": ["s0.bin"],
    }))

    from data.graph_harness.auditor import audit_graph
    a = audit_graph(tmp_path)
    assert a.n_nodes == 4
    assert a.n_edges == 4  # a->b, b->a, c->c, c->missing
    assert a.n_repos == 1  # all "r:*"
    # c->missing is dangling: 1/4
    assert a.dangling_edge_rate == pytest.approx(0.25)
    # c->c is a self-link: 1/4
    assert a.self_link_rate == pytest.approx(0.25)
    # d is isolated: 1/4
    assert a.isolated_node_frac == pytest.approx(0.25)
    # a<->b reciprocal: both a->b and b->a counted = 2/4
    assert a.reciprocal_edge_rate == pytest.approx(0.5)
    assert any("dangling" in w for w in a.warnings)
    assert any("self-link" in w for w in a.warnings)


def test_auditor_flags_edgeless_graph(tmp_path):
    import json
    nodes = [
        {"normed_identifier": "x", "raw_identifier": "x",
         "outgoing": [], "incoming": [],
         "tok_shard_idx": 0, "tok_offset_bytes": 1024, "tok_len": 1},
    ]
    (tmp_path / "tokenized_graph.jsonl").write_text(json.dumps(nodes[0]))
    (tmp_path / "metadata.json").write_text(json.dumps({
        "tokenizer": "gpt2", "dtype_str": "uint16", "shard_filenames": ["s0.bin"],
    }))
    from data.graph_harness.auditor import audit_graph
    a = audit_graph(tmp_path)
    assert a.n_edges == 0
    assert a.n_repos is None  # no ":" in identifiers -> not repo-partitioned
    assert any("ZERO edges" in w for w in a.warnings)
