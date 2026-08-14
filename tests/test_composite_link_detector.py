"""
Tests for composite_link_detector.py

Coverage:
  - _sniff_by_identifier: extension / '::' / go-host / ambiguous → detector name
  - _sniff_by_content: distinctive per-language syntax → detector name; prose → None
  - CompositeLinkDetector.detect_links: content-only dispatch (generation path)
  - CompositeLinkDetector.detect_links_for_doc: identifier + content dispatch,
    routes to sub-detector's per-doc path, resolves relative imports
  - CompositeLinkDetector.index_doc_span: dispatches key transform per source
  - make_link_detector('composite') + registration + inference layout mapping
  - cross-firing guard: the wrong syntax in the wrong file does not win
"""
import pytest
import tiktoken
import torch

from model.graph_traversal.composite_link_detector import (
    CompositeLinkDetector,
    COMPOSITE_MEMBERS,
    _sniff_by_content,
    _sniff_by_identifier,
)
from model.graph_traversal.link_detector import (
    LINK_DETECTOR_NAMES,
    make_link_detector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return CompositeLinkDetector(decode_fn=enc.decode)


def _t(enc, code: str) -> torch.Tensor:
    return torch.tensor(enc.encode(code), dtype=torch.long)


class _Span:
    """Minimal DocSpan shim: index_doc_span reads only raw_identifier."""

    def __init__(self, raw_identifier: str):
        self.raw_identifier = raw_identifier


# ===========================================================================
# _sniff_by_identifier
# ===========================================================================


class TestSniffByIdentifier:
    @pytest.mark.parametrize(
        "ident,expected",
        [
            ("owner/repo:pkg/sub/mod.py", "python"),
            ("repo:src/main/java/com/foo/Bar.java", "java"),
            ("repo:app/Helper.kt", "kotlin"),
            ("repo:lib/models/user.dart", "dart"),
            ("repo:src/util/helper.zig", "zig"),
            ("repo:src/util/helper.ts", "typescript"),
            ("repo:src/index.tsx", "typescript"),
            ("repo:src/util/helper.js", "javascript"),
            ("repo:src/x.mjs", "javascript"),
            ("owner/repo:pkg/thing.go", "go"),
            ("owner/repo@crate::net::tcp", "rust"),
            ("crate::net::tcp", "rust"),
            ("github.com/owner/repo/pkg", "go"),  # go host-rooted import path
            ("gopkg.in/yaml.v2", "go"),
        ],
    )
    def test_recognised(self, ident, expected):
        assert _sniff_by_identifier(ident) == expected

    @pytest.mark.parametrize(
        "ident",
        [
            "",                       # empty
            "Fluid dynamics",         # wiki title (has a space, no ext)
            "Some Paper Title 2019",  # arxiv title
        ],
    )
    def test_ambiguous_returns_none(self, ident):
        assert _sniff_by_identifier(ident) is None

    def test_all_members_have_a_name(self):
        # every returned name must be a real sub-detector key
        for ident in ("a:b.py", "a:b.java", "crate::x", "github.com/a/b"):
            name = _sniff_by_identifier(ident)
            assert name in COMPOSITE_MEMBERS


# ===========================================================================
# _sniff_by_content
# ===========================================================================


class TestSniffByContent:
    def test_python(self):
        assert _sniff_by_content("from foo.bar import baz\n\ndef f():\n    pass\n") == "python"

    def test_go(self):
        assert _sniff_by_content('package main\n\nimport (\n\t"fmt"\n)\n\nfunc main() {}\n') == "go"

    def test_java(self):
        assert _sniff_by_content("import java.util.List;\n\npublic class Foo {}\n") == "java"

    def test_rust(self):
        assert _sniff_by_content("use crate::net::tcp;\n\npub fn go() {}\n") == "rust"

    def test_arxiv(self):
        assert _sniff_by_content("As shown in \\cite{Smith2019} the result holds.") == "arxiv"

    def test_zig(self):
        assert _sniff_by_content('const std = @import("std");\n') == "zig"

    def test_dart(self):
        assert _sniff_by_content("import 'package:foo/bar.dart';\n\nclass X {}\n") == "dart"

    def test_kotlin(self):
        assert _sniff_by_content("import com.example.Foo\n\nfun main() {}\n") == "kotlin"

    def test_typescript(self):
        code = 'import { Foo } from "./foo";\ninterface Bar { x: number }\n'
        assert _sniff_by_content(code) == "typescript"

    def test_javascript(self):
        code = 'const x = require("./foo");\nmodule.exports = x;\n'
        assert _sniff_by_content(code) == "javascript"

    def test_markdown(self):
        assert _sniff_by_content("See [Fluid dynamics](Fluid dynamics) for more.\n") == "markdown"

    def test_plain_prose_returns_none(self):
        assert _sniff_by_content("The quick brown fox jumps over the lazy dog.\n") is None

    def test_empty_returns_none(self):
        assert _sniff_by_content("") is None


# ===========================================================================
# detect_links — content-only dispatch (generation-loop path)
# ===========================================================================


class TestDetectLinks:
    def test_markdown_link_detected(self, detector, enc):
        # A resolvable [text](Target) — markdown detector should fire.
        links = detector.detect_links(_t(enc, "Intro. See [Lever](Lever) here.\n"))
        assert any(l.target_str == "Lever" for l in links)

    def test_python_import_detected(self, detector, enc):
        code = "from chess.board import Board\n\ndef play():\n    pass\n"
        links = detector.detect_links(_t(enc, code))
        assert any("chess/board" in l.target_str for l in links)

    def test_arxiv_cite_detected(self, detector, enc):
        links = detector.detect_links(_t(enc, "We build on \\cite{Vaswani2017} here.\n"))
        assert any("Vaswani2017" in l.target_str for l in links)

    def test_prose_yields_no_links(self, detector, enc):
        assert detector.detect_links(_t(enc, "Just some ordinary sentence.\n")) == []

    def test_cross_fire_guard_python_not_markdown(self, detector, enc):
        # Python file whose STRING contains ](-like text must classify as python,
        # not markdown, and must not emit a bogus markdown link.
        code = "from a.b import c\n\nx = 'see [foo](bar)'\n"
        links = detector.detect_links(_t(enc, code))
        assert all(l.target_str != "bar" for l in links)


# ===========================================================================
# detect_links_for_doc — identifier + content dispatch (mask per-doc path)
# ===========================================================================


class TestDetectLinksForDoc:
    def test_identifier_dispatch_python_relative(self, detector, enc):
        # Relative import resolves ONLY on the per-doc path with the identifier.
        code = "from .board import Board\n"
        links = detector.detect_links_for_doc(_t(enc, code), "owner/repo:chess/mcts.py")
        assert any(l.target_str == "chess/board.py" for l in links)

    def test_identifier_beats_misleading_content(self, detector, enc):
        # Identifier says python; content also python — sanity that .py routes py.
        code = "import os\n"
        links = detector.detect_links_for_doc(_t(enc, code), "r:pkg/m.py")
        assert any(l.target_str == "os.py" for l in links)

    def test_content_fallback_when_identifier_ambiguous(self, detector, enc):
        # Bare title identifier (wiki) → content sniff picks markdown.
        code = "See [Lever](Lever).\n"
        links = detector.detect_links_for_doc(_t(enc, code), "Archimedes screw")
        assert any(l.target_str == "Lever" for l in links)

    def test_go_span_slice_path(self, detector, enc):
        # Go has no detect_links_for_doc → composite falls back to detect_links.
        code = 'package main\nimport "github.com/x/y/pkg"\n'
        links = detector.detect_links_for_doc(_t(enc, code), "github.com/a/b/main.go")
        assert any("github.com/x/y/pkg" in l.target_str for l in links)


# ===========================================================================
# index_doc_span — per-source key transform
# ===========================================================================


class TestIndexDocSpan:
    def test_python_strips_repo_prefix(self, detector):
        assert detector.index_doc_span(_Span("owner/repo:chess/board.py")) == "chess/board.py"

    def test_typescript_strips_ext(self, detector):
        assert detector.index_doc_span(_Span("repo:src/util/helper.ts")) == "src/util/helper"

    def test_dart_keeps_ext(self, detector):
        assert detector.index_doc_span(_Span("repo:lib/models/user.dart")) == "lib/models/user.dart"

    def test_rust_module_path(self, detector):
        assert detector.index_doc_span(_Span("owner/repo@crate::net::tcp")) == "crate::net::tcp"

    def test_go_identity(self, detector):
        assert detector.index_doc_span(_Span("github.com/owner/repo/pkg")) == "github.com/owner/repo/pkg"

    def test_ambiguous_title_identity(self, detector):
        # wiki/arxiv bare title → identity key
        assert detector.index_doc_span(_Span("Fluid dynamics")) == "Fluid dynamics"

    def test_matches_standalone_python_detector(self, detector, enc):
        # composite's key transform must equal the standalone detector's.
        py = make_link_detector("python", enc.decode)
        span = _Span("owner/repo:a/b/c.py")
        assert detector.index_doc_span(span) == py.index_doc_span(span)


# ===========================================================================
# Registration + factory
# ===========================================================================


class TestRegistration:
    def test_in_names(self):
        assert "composite" in LINK_DETECTOR_NAMES

    def test_factory_builds_it(self, enc):
        det = make_link_detector("composite", enc.decode)
        assert isinstance(det, CompositeLinkDetector)

    def test_has_per_doc_method(self, detector):
        # The mask creator selects the per-doc path via hasattr — must be present.
        assert hasattr(detector, "detect_links_for_doc")

    def test_inference_layout_mapping(self):
        from data.layout import inference_layout_for_detector
        # must not raise, and must return a real layout name
        assert inference_layout_for_detector("composite") == "identifier_prefix_eos"

    def test_builds_all_members(self, detector):
        assert set(detector._subs) == set(COMPOSITE_MEMBERS)
        assert len(COMPOSITE_MEMBERS) == 11
