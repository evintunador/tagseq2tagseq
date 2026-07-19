"""
Tests for go_import_detector.py

Coverage:
  - _parse_imports: single, grouped, aliased, blank, dot, backtick, stdlib
  - GoImportDetector.detect_links: end-to-end with tiktoken GPT-2, link_end_pos
  - GoImportDetector.index_doc_span: returns full import path unchanged
"""
import pytest
import tiktoken
import torch

from model.graph_traversal.go_import_detector import (
    GoImportDetector,
    _parse_imports,
)


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return GoImportDetector(decode_fn=enc.decode)


def _paths(text):
    return [p for p, _end in _parse_imports(text)]


# ---------------------------------------------------------------------------
# _parse_imports
# ---------------------------------------------------------------------------

def test_single_import():
    assert _paths('import "fmt"\n') == ["fmt"]


def test_single_import_module_path():
    assert _paths('import "github.com/x/y/pkg"\n') == ["github.com/x/y/pkg"]


def test_single_import_aliased():
    assert _paths('import m "github.com/x/y/mod"\n') == ["github.com/x/y/mod"]


def test_grouped_imports():
    src = (
        "import (\n"
        '    "fmt"\n'
        '    "github.com/x/y/a"\n'
        '    "os"\n'
        ")\n"
    )
    assert _paths(src) == ["fmt", "github.com/x/y/a", "os"]


def test_grouped_with_alias_blank_dot():
    src = (
        "import (\n"
        '    alias "github.com/x/y/a"\n'
        '    _ "github.com/x/y/driver"\n'
        '    . "github.com/x/y/dsl"\n'
        ")\n"
    )
    assert _paths(src) == [
        "github.com/x/y/a",
        "github.com/x/y/driver",
        "github.com/x/y/dsl",
    ]


def test_backtick_path():
    assert _paths('import (\n\t`github.com/x/y/z`\n)\n') == ["github.com/x/y/z"]


def test_no_imports():
    assert _paths("package main\n\nfunc main() {}\n") == []


def test_import_path_with_dots_and_hyphens():
    src = 'import "github.com/go-chi/chi/v5"\n'
    assert _paths(src) == ["github.com/go-chi/chi/v5"]


def test_import_inside_block_comment_ignored():
    """Import declarations in a /* */ doc comment are NOT real code (harness FP)."""
    src = (
        "/*\n"
        "Example usage:\n"
        "\timport (\n"
        '\t\tlog "github.com/sirupsen/logrus"\n'
        "\t)\n"
        "*/\n"
        "package logrus\n"
    )
    assert _paths(src) == []


def test_import_inside_line_comment_ignored():
    src = '// import "github.com/x/y/z"\npackage main\n'
    assert _paths(src) == []


def test_real_import_after_comment_still_found():
    src = (
        "// leading comment mentioning import \"github.com/fake/one\"\n"
        'import "github.com/real/two"\n'
    )
    assert _paths(src) == ["github.com/real/two"]


def test_url_in_comment_not_treated_as_import():
    # a // inside a real import path's context should not corrupt scanning
    src = (
        "/* see https://github.com/x/y for docs */\n"
        'import "github.com/x/y"\n'
    )
    assert _paths(src) == ["github.com/x/y"]


# ---------------------------------------------------------------------------
# detect_links
# ---------------------------------------------------------------------------

def test_detect_links_emits_one_per_path(enc, detector):
    src = (
        "package main\n"
        "import (\n"
        '    "fmt"\n'
        '    "github.com/x/y/pkg"\n'
        ")\n"
    )
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    links = detector.detect_links(ids)
    targets = {li.target_str for li in links}
    assert targets == {"fmt", "github.com/x/y/pkg"}
    # NO candidate expansion: exactly one LinkInfo per import path
    assert len(links) == 2


def test_link_end_pos_after_import(enc, detector):
    src = 'import "github.com/x/y/pkg"\nvar z = 1\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    links = detector.detect_links(ids)
    assert len(links) == 1
    li = links[0]
    # Decoding up to link_end_pos should include the full import path.
    decoded_prefix = enc.decode(ids[: li.link_end_pos].tolist())
    assert "github.com/x/y/pkg" in decoded_prefix
    # and the token right after should not have consumed 'var z' yet
    assert li.link_end_pos <= len(ids)


def test_detect_links_empty_on_no_imports(enc, detector):
    ids = torch.tensor(enc.encode("package main\nfunc f() {}\n"), dtype=torch.long)
    assert detector.detect_links(ids) == []


# ---------------------------------------------------------------------------
# index_doc_span
# ---------------------------------------------------------------------------

class _Span:
    def __init__(self, raw):
        self.raw_identifier = raw


def test_index_doc_span_returns_full_path(detector):
    span = _Span("github.com/owner/repo/pkg/sub")
    assert detector.index_doc_span(span) == "github.com/owner/repo/pkg/sub"


def test_detect_and_index_share_string_space(enc, detector):
    """The contract: target_str from detect_links matches index_doc_span(node)."""
    src = 'import "github.com/owner/repo/pkg"\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    target = detector.detect_links(ids)[0].target_str
    node = _Span("github.com/owner/repo/pkg")
    assert target == detector.index_doc_span(node)
