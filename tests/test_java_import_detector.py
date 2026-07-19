"""
Tests for java_import_detector.py

Coverage: _parse_imports (single, static, on-demand, nested), detect_links
end-to-end with GPT-2, index_doc_span FQN derivation + shared-space contract.
"""
import pytest
import tiktoken
import torch

from model.graph_traversal.java_import_detector import (
    JavaImportDetector,
    _parse_imports,
    _fqn_candidates,
)


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return JavaImportDetector(decode_fn=enc.decode)


def _cands(text):
    return [c for c, _e in _parse_imports(text)]


def test_single_import():
    assert _cands("import com.google.gson.Gson;\n") == ["com.google.gson.Gson"]


def test_static_import_emits_member_and_type():
    got = _cands("import static com.google.gson.GsonBuilder.newImmutableList;\n")
    assert "com.google.gson.GsonBuilder.newImmutableList" in got
    assert "com.google.gson.GsonBuilder" in got  # enclosing type


def test_on_demand_import_emits_nothing():
    # a.b.* is a package, not a type/file
    assert _cands("import com.google.gson.*;\n") == []


def test_multiple_imports():
    src = (
        "package p;\n"
        "import a.b.C;\n"
        "import a.b.d.E;\n"
        "import java.util.List;\n"
    )
    assert _cands(src) == ["a.b.C", "a.b.d.E", "java.util.List"]


def test_fqn_candidates_helper():
    assert _fqn_candidates("a.b.C", is_static=False, is_star=False) == ["a.b.C"]
    assert _fqn_candidates("a.b.C.m", is_static=True, is_star=False) == ["a.b.C.m", "a.b.C"]
    assert _fqn_candidates("a.b", is_static=False, is_star=True) == []


def test_detect_links_targets(enc, detector):
    src = "package p;\nimport com.google.gson.Gson;\nimport a.b.C;\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    targets = {li.target_str for li in detector.detect_links(ids)}
    assert targets == {"com.google.gson.Gson", "a.b.C"}


def test_link_end_pos_after_import(enc, detector):
    src = "import com.google.gson.Gson;\nclass X {}\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    li = detector.detect_links(ids)[0]
    prefix = enc.decode(ids[: li.link_end_pos].tolist())
    assert "com.google.gson.Gson" in prefix


class _Span:
    def __init__(self, raw):
        self.raw_identifier = raw


def test_index_doc_span_derives_fqn(detector):
    span = _Span("owner/repo:src/main/java/com/google/gson/Gson.java")
    # NOTE: index_doc_span returns the post-':' path as a dotted name; the source
    # root is NOT stripped here (see build_java_file_nodes for FQN-relative keys).
    assert detector.index_doc_span(span) == "src.main.java.com.google.gson.Gson"


def test_detect_and_index_share_space_with_fqn_relative_node(enc, detector):
    """When a node is keyed FQN-relative, target_str matches index_doc_span."""
    src = "import com.google.gson.Gson;\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    target = detector.detect_links(ids)[0].target_str
    # a node whose raw_identifier is "repo:com/google/gson/Gson.java" (source root
    # already stripped at build time) yields the matching dotted key.
    node = _Span("repo:com/google/gson/Gson.java")
    assert target == detector.index_doc_span(node)
