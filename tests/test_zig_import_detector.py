"""
Tests for zig_import_detector.py

Coverage: _parse_import_specs (@import extraction, comment/string blanking,
char-literal skip), _spec_to_detection_key (.zig stripped, bare stdlib dropped,
../ retained), _resolve_relative_spec (sibling, subdir, ../ , escape, bare),
detect_links (specifier-space key), detect_links_for_doc (path-resolved),
index_doc_span.
"""
import pytest
import tiktoken
import torch

from model.graph_traversal.zig_import_detector import (
    ZigImportDetector,
    _parse_import_specs,
    _spec_to_detection_key,
    _resolve_relative_spec,
)


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return ZigImportDetector(decode_fn=enc.decode)


def _specs(text):
    return [s for s, _e in _parse_import_specs(text)]


# --- specifier extraction ---------------------------------------------------

def test_sibling_import():
    assert _specs('const foo = @import("foo.zig");\n') == ["foo.zig"]


def test_subdir_and_updir_import():
    src = 'const a = @import("lib/bar.zig");\nconst b = @import("../up/x.zig");\n'
    assert _specs(src) == ["lib/bar.zig", "../up/x.zig"]


def test_stdlib_imports_extracted_raw():
    # _parse_import_specs returns the raw specifier; key normalization drops bare.
    src = 'const std = @import("std");\nconst b = @import("builtin");\n'
    assert _specs(src) == ["std", "builtin"]


def test_import_in_comment_blanked():
    src = ('// const nope = @import("nope.zig");\n'
           '/// doc: @import("doc.zig")\n'
           'const real = @import("real.zig");\n')
    assert _specs(src) == ["real.zig"]


def test_import_in_string_literal_not_matched():
    # a string literal that looks like an import is not a real @import call
    src = 'const s = "const x = @import(\\"strlit.zig\\");";\n'
    assert _specs(src) == []


def test_import_in_multiline_string_not_matched():
    # Zig multiline-string lines (\\...) commonly embed codegen `@import(...)`
    # text that is NOT a real import — tree-sitter treats it as string content.
    src = ('const gen =\n'
           '    \\\\const client = @import("wayland.zig");\n'
           '    \\\\const common = @import("common.zig");\n'
           ';\n'
           'const real = @import("real.zig");\n')
    assert _specs(src) == ["real.zig"]


def test_whitespace_tolerant():
    assert _specs('const foo = @import ( "foo.zig" ) ;\n') == ["foo.zig"]


def test_char_literal_does_not_break_scan():
    # a char literal containing a quote must not desync the string scanner
    src = "const q = '\\'';\nconst foo = @import(\"foo.zig\");\n"
    assert _specs(src) == ["foo.zig"]


# --- detection key normalization --------------------------------------------

def test_detection_key_strips_ext():
    assert _spec_to_detection_key("foo.zig") == "foo"
    assert _spec_to_detection_key("lib/bar.zig") == "lib/bar"


def test_detection_key_dotdot_retained():
    assert _spec_to_detection_key("../up/x.zig") == "../up/x"


def test_detection_key_currentdir_collapsed():
    assert _spec_to_detection_key("./foo.zig") == "foo"


def test_detection_key_bare_is_none():
    assert _spec_to_detection_key("std") is None
    assert _spec_to_detection_key("builtin") is None
    assert _spec_to_detection_key("mypkg") is None


# --- relative resolution ----------------------------------------------------

def test_resolve_sibling():
    assert _resolve_relative_spec("foo.zig", "src/main.zig") == "src/foo.zig"


def test_resolve_subdir():
    assert _resolve_relative_spec("lib/bar.zig", "src/main.zig") == "src/lib/bar.zig"


def test_resolve_updir():
    assert _resolve_relative_spec("../consts.zig", "src/util/helper.zig") == "src/consts.zig"


def test_resolve_escape_returns_none():
    assert _resolve_relative_spec("../../../x.zig", "a/b.zig") is None


def test_resolve_bare_returns_none():
    assert _resolve_relative_spec("std", "src/main.zig") is None


# --- detect_links (specifier space) -----------------------------------------

def test_detect_links_specifier_space(enc, detector):
    src = 'const h = @import("util/helper.zig");\nconst std = @import("std");\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    targets = {li.target_str for li in detector.detect_links(ids)}
    assert targets == {"util/helper"}  # std licenses nothing


def test_link_end_pos_after_import(enc, detector):
    src = 'const h = @import("util/helper.zig");\nconst x = 1;\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    li = detector.detect_links(ids)[0]
    prefix = enc.decode(ids[: li.link_end_pos].tolist())
    assert "util/helper.zig" in prefix


# --- detect_links_for_doc (path resolved) -----------------------------------

def test_detect_links_for_doc_resolves(enc, detector):
    src = 'const c = @import("../consts.zig");\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    links = detector.detect_links_for_doc(ids, "repo:src/util/helper.zig")
    targets = {li.target_str for li in links}
    assert targets == {"src/consts.zig"}


def test_detect_links_for_doc_drops_stdlib(enc, detector):
    src = 'const std = @import("std");\nconst c = @import("consts.zig");\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    links = detector.detect_links_for_doc(ids, "repo:src/main.zig")
    assert {li.target_str for li in links} == {"src/consts.zig"}


# --- index_doc_span ---------------------------------------------------------

class _Span:
    def __init__(self, raw):
        self.raw_identifier = raw


def test_index_doc_span_strips_prefix_keeps_ext(detector):
    assert detector.index_doc_span(_Span("owner/repo:src/util/helper.zig")) == "src/util/helper.zig"
