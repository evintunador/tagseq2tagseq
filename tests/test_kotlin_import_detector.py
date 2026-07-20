"""
Tests for kotlin_import_detector.py

Coverage: _parse_imports (single, function/property import, alias strip, wildcard
drop, stdlib, comment-blanking), detect_links end-to-end with GPT-2,
index_doc_span FQN identity + shared-space contract.
"""
import pytest
import tiktoken
import torch

from model.graph_traversal.kotlin_import_detector import (
    KotlinImportDetector,
    _parse_imports,
    _blank_comments,
)


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return KotlinImportDetector(decode_fn=enc.decode)


def _fqns(text):
    return [f for f, _e in _parse_imports(text)]


def test_single_import():
    assert _fqns("import com.ex.util.Helper\n") == ["com.ex.util.Helper"]


def test_function_import():
    # Kotlin can import a top-level function/property (not just a type)
    assert _fqns("import com.ex.util.helperFn\n") == ["com.ex.util.helperFn"]


def test_alias_import_strips_alias():
    # `import a.b.c as X` -> emit the FQN, NOT "a.b.c as X"
    assert _fqns("import com.ex.foo.bar as Baz\n") == ["com.ex.foo.bar"]


def test_wildcard_import_dropped():
    # on-demand/wildcard import has no single target -> emit nothing
    assert _fqns("import com.ex.*\n") == []


def test_stdlib_import_detected_as_fqn():
    # stdlib is emitted like any FQN; it just won't resolve to a corpus node
    assert _fqns("import kotlin.math.max\n") == ["kotlin.math.max"]


def test_multiple_imports():
    src = (
        "package p\n"
        "import a.b.C\n"
        "import a.b.d.e\n"
        "import kotlin.collections.List\n"
    )
    assert _fqns(src) == ["a.b.C", "a.b.d.e", "kotlin.collections.List"]


def test_import_in_line_comment_ignored():
    src = "package p\n// import com.fake.Bad\nimport a.b.C\n"
    assert _fqns(src) == ["a.b.C"]


def test_import_in_block_comment_ignored():
    src = "package p\n/* import com.fake.Bad */\nimport a.b.C\n"
    assert _fqns(src) == ["a.b.C"]


def test_blank_comments_preserves_length():
    src = "import a.b.C // hi\n/* x */import d.e.F\n"
    blanked = _blank_comments(src)
    assert len(blanked) == len(src)
    assert "a.b.C" in blanked and "d.e.F" in blanked
    assert "hi" not in blanked


def test_detect_links_targets(enc, detector):
    src = "package p\nimport com.ex.util.Helper\nimport a.b.C\nimport com.ex.*\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    targets = {li.target_str for li in detector.detect_links(ids)}
    assert targets == {"com.ex.util.Helper", "a.b.C"}


def test_link_end_pos_after_import(enc, detector):
    src = "import com.ex.util.Helper\nclass X\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    li = detector.detect_links(ids)[0]
    prefix = enc.decode(ids[: li.link_end_pos].tolist())
    assert "com.ex.util.Helper" in prefix


class _Span:
    def __init__(self, raw):
        self.raw_identifier = raw


def test_index_doc_span_is_fqn_identity(detector):
    span = _Span("com.ex.util.Helper")
    assert detector.index_doc_span(span) == "com.ex.util.Helper"


def test_index_doc_span_strips_repo_prefix(detector):
    span = _Span("owner/repo:com.ex.util.Helper")
    assert detector.index_doc_span(span) == "com.ex.util.Helper"


def test_detect_and_index_share_space(enc, detector):
    src = "import com.ex.util.Helper\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    target = detector.detect_links(ids)[0].target_str
    node = _Span("com.ex.util.Helper")
    assert target == detector.index_doc_span(node)
