"""
Rust resolution-axis self-test (mirrors Go/Java/Python).

RustImportDetector + module-path node model (mod-tree walk) + PretokCorpus
resolution must recover the hand-labeled intra-crate edges on a multi-file crate,
in MODULE-PATH key space. Confirms:
  * the mod-tree walk assigns correct module paths (crate, crate::a, crate::a::c,
    crate::b) from src/lib.rs + `mod` declarations across SIBLING files;
  * `self::` / `super::` rewrite against the file's module path;
  * grouped/nested `use crate::a::{c::Helper, self}` expands to several edges;
  * a glob `use crate::b::*` resolves to the target MODULE `crate::b`;
  * a re-export `pub use crate::b::ReExported` is a real edge;
  * `use std::...` (external) correctly does NOT resolve (excluded from edges.json);
  * an inline `mod inner_inline { .. }` creates NO file node / edge;
  * a self-referential `use self::X` in crate::b does NOT create a self-link edge.
"""
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_rust")

import tiktoken
import torch

from data.graph_harness.fixtures import score_resolution
from data.graph_harness.rust_nodes import build_rust_module_nodes
from model.graph_traversal.rust_import_detector import RustImportDetector

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "graph_harness" / "fixtures_data"


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


def test_rust_resolution_on_fixture(enc):
    detector = RustImportDetector(decode_fn=enc.decode)

    def edge_producer(nodes):
        # module_path (n.key) of each node, to drop self-referential candidates —
        # the build (build_rust_graph.py) drops self-links (tgt_id != node_id), and
        # a `use crate::a::{self}` / `use self::X` legitimately produces a
        # parent-module candidate that points back at the emitting file. A self-link
        # is definitionally not a cross-doc edge, so we filter it here, mirroring the
        # builder's invariant. (edges.json lists only cross-file edges.)
        edges = []
        for n in nodes:
            ids = torch.tensor(enc.encode(n.content), dtype=torch.long)
            # per-doc detection resolves self::/super:: against the module path
            for li in detector.detect_links_for_doc(ids, n.raw_identifier):
                if li.target_str == n.key:
                    continue  # self-link: dropped by the builder
                edges.append((n.key, li.target_str))
        return edges

    score = score_resolution(
        FIXTURES / "rust" / "simple_crate",
        extensions={"rs"},
        edge_producer=edge_producer,
        link_detector=detector,
        node_builder=build_rust_module_nodes,
    )
    assert score.recall == 1.0, f"missed: {score.missing_edges}; {score.summary()}"
    assert score.precision == 1.0, f"spurious: {score.wrong_edges}; {score.summary()}"


def test_rust_module_paths(enc):
    """The mod-tree walk assigns the expected module path to each file."""
    from data.graph_harness.fixtures import _FixtureFile
    from data.rust_graph_extractor.mod_tree import build_module_paths

    files_dir = FIXTURES / "rust" / "simple_crate" / "files"
    files = [(p.relative_to(files_dir).as_posix(), p.read_text())
             for p in files_dir.rglob("*.rs")]
    mp = build_module_paths(files)
    assert mp["src/lib.rs"] == "crate"
    assert mp["src/a/mod.rs"] == "crate::a"
    assert mp["src/a/c.rs"] == "crate::a::c"
    assert mp["src/b.rs"] == "crate::b"
