"""
Tests for python_import_detector.py

Coverage:
  - module_path_to_file_paths: all import forms and edge cases
  - _parse_imports: every supported syntax variant + relative/edge cases
  - PythonImportDetector.detect_links: end-to-end with tiktoken GPT-2
  - PythonImportDetector.index_doc_span: repo-prefix stripping
  - PythonImportDetector._build_char_to_token_index / _char_pos_to_token_pos
"""

import pytest
import tiktoken
import torch

from model.graph_traversal.python_import_detector import (
    PythonImportDetector,
    _blank_comments_and_strings,
    _parse_imports,
    _parse_relative_imports,
    _resolve_relative_import,
    _strip_alias,
    module_path_to_file_paths,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return PythonImportDetector(decode_fn=enc.decode)


def _encode(enc, code: str) -> torch.Tensor:
    return torch.tensor(enc.encode(code), dtype=torch.long)


# ===========================================================================
# module_path_to_file_paths
# ===========================================================================


class TestModulePathToFilePaths:
    def test_single_component_no_from(self):
        assert module_path_to_file_paths("os") == ["os.py", "os/__init__.py"]

    def test_two_components_no_from(self):
        assert module_path_to_file_paths("os.path") == [
            "os/path.py",
            "os/path/__init__.py",
        ]

    def test_three_components_no_from(self):
        assert module_path_to_file_paths("foo.bar.baz") == [
            "foo/bar/baz.py",
            "foo/bar/baz/__init__.py",
        ]

    def test_with_from_name(self):
        paths = module_path_to_file_paths("foo.bar", "baz")
        # baz-as-submodule candidates come first
        assert paths[0] == "foo/bar/baz.py"
        assert paths[1] == "foo/bar/baz/__init__.py"
        # then the parent module itself
        assert "foo/bar.py" in paths
        assert "foo/bar/__init__.py" in paths

    def test_star_import_ignores_from_name(self):
        assert module_path_to_file_paths("foo.bar", "*") == [
            "foo/bar.py",
            "foo/bar/__init__.py",
        ]

    def test_empty_from_name_same_as_no_from(self):
        assert module_path_to_file_paths("foo.bar", "") == module_path_to_file_paths(
            "foo.bar"
        )

    def test_single_component_with_from_name(self):
        paths = module_path_to_file_paths("foo", "bar")
        assert "foo/bar.py" in paths
        assert "foo/bar/__init__.py" in paths
        assert "foo.py" in paths
        assert "foo/__init__.py" in paths

    def test_no_duplicate_paths(self):
        paths = module_path_to_file_paths("foo.bar", "baz")
        assert len(paths) == len(set(paths))

    def test_order_most_specific_first(self):
        # Submodule candidates should appear before parent-module candidates
        paths = module_path_to_file_paths("foo.bar", "baz")
        sub_idx = paths.index("foo/bar/baz.py")
        par_idx = paths.index("foo/bar.py")
        assert sub_idx < par_idx


# ===========================================================================
# _parse_imports
# ===========================================================================


class TestParseImports:

    # --- plain import ---

    def test_simple_import(self):
        result = _parse_imports("import os\n")
        assert len(result) == 1
        assert result[0][0] == "os"
        assert result[0][1] == ""

    def test_dotted_import(self):
        result = _parse_imports("import foo.bar.baz\n")
        assert result[0][0] == "foo.bar.baz"
        assert result[0][1] == ""

    def test_aliased_import(self):
        result = _parse_imports("import numpy as np\n")
        assert result[0][0] == "numpy"
        assert result[0][1] == ""

    def test_comma_separated_import(self):
        result = _parse_imports("import os, sys\n")
        modules = [r[0] for r in result]
        assert "os" in modules
        assert "sys" in modules

    def test_comma_separated_aliased(self):
        result = _parse_imports("import os as operating_system, sys as system\n")
        modules = [r[0] for r in result]
        assert "os" in modules
        assert "sys" in modules

    def test_dotted_comma_separated(self):
        result = _parse_imports("import foo.bar, baz.qux\n")
        modules = [r[0] for r in result]
        assert "foo.bar" in modules
        assert "baz.qux" in modules

    # --- from import ---

    def test_from_import_single(self):
        result = _parse_imports("from os.path import join\n")
        assert result[0][0] == "os.path"
        assert result[0][1] == "join"

    def test_from_import_multiple(self):
        result = _parse_imports("from os import path, getcwd\n")
        entries = {r[1]: r[0] for r in result}
        assert "path" in entries
        assert "getcwd" in entries
        assert entries["path"] == "os"

    def test_from_import_star(self):
        result = _parse_imports("from foo.bar import *\n")
        assert result[0][0] == "foo.bar"
        assert result[0][1] == "*"

    def test_from_import_parenthesised_single_line(self):
        result = _parse_imports("from foo.bar import (baz)\n")
        assert any(r[0] == "foo.bar" and r[1] == "baz" for r in result)

    def test_from_import_parenthesised_multiline(self):
        code = "from foo.bar import (\n    baz,\n    qux,\n)\n"
        result = _parse_imports(code)
        entries = {r[1] for r in result}
        assert "baz" in entries
        assert "qux" in entries

    def test_from_import_parenthesised_trailing_comma(self):
        code = "from foo import (\n    bar,\n    baz,\n)\n"
        result = _parse_imports(code)
        entries = {r[1] for r in result}
        assert "bar" in entries
        assert "baz" in entries
        # trailing comma should not produce an empty entry
        assert "" not in entries

    def test_from_import_parenthesised_with_comments(self):
        code = "from foo import (\n    bar,  # the bar module\n    baz,\n)\n"
        result = _parse_imports(code)
        entries = {r[1] for r in result}
        assert "bar" in entries
        assert "baz" in entries
        # comment text should not appear as a name
        assert any("the bar module" in r[1] for r in result) is False

    # --- relative imports (should all be skipped) ---

    def test_relative_dot_import(self):
        assert _parse_imports("from . import foo\n") == []

    def test_relative_dot_module_import(self):
        assert _parse_imports("from .foo import bar\n") == []

    def test_relative_dotdot_import(self):
        assert _parse_imports("from .. import baz\n") == []

    def test_relative_dotdot_module_import(self):
        assert _parse_imports("from ..foo.bar import baz\n") == []

    # --- character positions ---

    def test_char_positions_match_text(self):
        code = "x = 1\nimport os\ny = 2\n"
        result = _parse_imports(code)
        assert len(result) == 1
        module_path, from_name, start, end = result[0]
        # The sliced text should be the import statement itself
        assert code[start:end].startswith("import os")

    def test_char_positions_from_import(self):
        code = "x = 1\nfrom os import path\ny = 2\n"
        result = _parse_imports(code)
        assert len(result) == 1
        _, _, start, end = result[0]
        assert code[start:end].startswith("from os import path")

    def test_multiple_imports_positions_are_different(self):
        code = "import os\nimport sys\n"
        result = _parse_imports(code)
        positions = [(r[2], r[3]) for r in result]
        # Each import statement occupies a different span
        assert positions[0] != positions[1]

    # --- indented imports (valid Python inside functions) ---

    def test_indented_import(self):
        code = "def foo():\n    import bar\n"
        result = _parse_imports(code)
        assert any(r[0] == "bar" for r in result)

    # --- no-import code ---

    def test_no_imports(self):
        assert _parse_imports("x = 1\ny = x + 2\n") == []

    def test_empty_string(self):
        assert _parse_imports("") == []

    # --- import in string/comment (false positive is acceptable) ---

    def test_import_in_comment_not_at_line_start(self):
        # A comment in the middle of a line should NOT be matched
        code = "x = 1  # import os\n"
        result = _parse_imports(code)
        assert result == []

    # --- deduplication / ordering ---

    def test_from_import_multiple_same_module(self):
        code = "from foo.bar import baz\nfrom foo.bar import qux\n"
        result = _parse_imports(code)
        entries = [(r[0], r[1]) for r in result]
        assert ("foo.bar", "baz") in entries
        assert ("foo.bar", "qux") in entries


# ===========================================================================
# PythonImportDetector.detect_links
# ===========================================================================


class TestDetectLinks:

    def test_simple_import_produces_links(self, detector, enc):
        ids = _encode(enc, "import os\n")
        links = detector.detect_links(ids)
        target_strs = {l.target_str for l in links}
        assert "os.py" in target_strs
        assert "os/__init__.py" in target_strs

    def test_dotted_import(self, detector, enc):
        ids = _encode(enc, "import tensorflow.python.distributions\n")
        links = detector.detect_links(ids)
        target_strs = {l.target_str for l in links}
        assert "tensorflow/python/distributions.py" in target_strs
        assert "tensorflow/python/distributions/__init__.py" in target_strs

    def test_from_import(self, detector, enc):
        ids = _encode(enc, "from tensorflow.python.distributions import gamma\n")
        links = detector.detect_links(ids)
        target_strs = {l.target_str for l in links}
        # gamma as submodule
        assert "tensorflow/python/distributions/gamma.py" in target_strs
        # parent module
        assert "tensorflow/python/distributions.py" in target_strs

    def test_no_imports_returns_empty(self, detector, enc):
        ids = _encode(enc, "x = 1\ny = x + 2\n")
        assert detector.detect_links(ids) == []

    def test_relative_imports_skipped(self, detector, enc):
        ids = _encode(enc, "from . import foo\nfrom ..bar import baz\n")
        assert detector.detect_links(ids) == []

    def test_link_end_pos_is_after_import(self, detector, enc):
        # Verify link_end_pos points somewhere AFTER the import statement,
        # not before it or at token 0.
        code = "x = 1\nimport os\nresult = 42\n"
        ids = _encode(enc, code)
        links = detector.detect_links(ids)
        assert links, "expected at least one link"
        import_token_approx = len(enc.encode("x = 1\n"))
        for l in links:
            assert l.link_end_pos > import_token_approx, (
                f"link_end_pos={l.link_end_pos} should be after the import "
                f"(approx token {import_token_approx})"
            )

    def test_multiple_imports_produce_multiple_links(self, detector, enc):
        code = "import os\nimport sys\n"
        ids = _encode(enc, code)
        links = detector.detect_links(ids)
        target_strs = {l.target_str for l in links}
        assert "os.py" in target_strs
        assert "sys.py" in target_strs

    def test_from_import_multiline_parens(self, detector, enc):
        code = "from os import (\n    path,\n    getcwd,\n)\n"
        ids = _encode(enc, code)
        links = detector.detect_links(ids)
        target_strs = {l.target_str for l in links}
        # path and getcwd as submodules of os
        assert "os/path.py" in target_strs
        assert "os/getcwd.py" in target_strs

    def test_aliased_import(self, detector, enc):
        ids = _encode(enc, "import numpy as np\n")
        links = detector.detect_links(ids)
        target_strs = {l.target_str for l in links}
        assert "numpy.py" in target_strs

    def test_empty_sequence(self, detector, enc):
        ids = _encode(enc, "")
        assert detector.detect_links(ids) == []

    def test_realistic_python_file(self, detector, enc):
        # A realistic snippet; smoke-test that it doesn't crash and finds links.
        code = (
            "import os\n"
            "import sys\n"
            "from typing import List, Dict\n"
            "from . import local_module\n"  # relative — should be skipped
            "\n"
            "def main():\n"
            "    import json  # lazy import\n"
            "    return json.dumps({})\n"
        )
        ids = _encode(enc, code)
        links = detector.detect_links(ids)
        target_strs = {l.target_str for l in links}
        assert "os.py" in target_strs
        assert "sys.py" in target_strs
        assert "typing.py" in target_strs
        assert "json.py" in target_strs
        # relative import must not appear
        assert all("local_module" not in s for s in target_strs)

    def test_link_end_pos_within_sequence(self, detector, enc):
        code = "import os\nx = 1\n"
        ids = _encode(enc, code)
        links = detector.detect_links(ids)
        for l in links:
            assert 0 <= l.link_end_pos <= ids.shape[0], (
                f"link_end_pos={l.link_end_pos} out of range [0, {ids.shape[0]}]"
            )


# ===========================================================================
# PythonImportDetector.index_doc_span
# ===========================================================================


class TestIndexDocSpan:

    class _Span:
        def __init__(self, raw_identifier):
            self.raw_identifier = raw_identifier

    def test_strips_repo_prefix(self, detector):
        span = self._Span("myrepo_abc123:src/foo/bar.py")
        assert detector.index_doc_span(span) == "src/foo/bar.py"

    def test_preserves_path_with_nested_colon(self, detector):
        # Only the first colon is used as separator
        span = self._Span("myrepo_abc123:src/foo:bar.py")
        assert detector.index_doc_span(span) == "src/foo:bar.py"

    def test_no_colon_returns_full_title(self, detector):
        span = self._Span("no_colon_title")
        assert detector.index_doc_span(span) == "no_colon_title"

    def test_empty_path_after_colon(self, detector):
        span = self._Span("repo_abc123:")
        assert detector.index_doc_span(span) == ""

    def test_realistic_stack_identifier(self, detector):
        span = self._Span(
            "leroidauphin/probability:tensorflow_probability/"
            "python/distributions/gamma.py"
        )
        assert detector.index_doc_span(span) == (
            "tensorflow_probability/python/distributions/gamma.py"
        )


# ===========================================================================
# Internal helpers: _build_char_to_token_index / _char_pos_to_token_pos
# ===========================================================================


class TestCharToTokenIndex:

    def test_cumulative_starts_at_zero(self, detector, enc):
        tokens = enc.encode("hello world")
        cum = detector._build_char_to_token_index(tokens)
        assert cum[0] == 0

    def test_cumulative_length_matches_token_count(self, detector, enc):
        tokens = enc.encode("hello world")
        cum = detector._build_char_to_token_index(tokens)
        assert len(cum) == len(tokens) + 1

    def test_cumulative_is_non_decreasing(self, detector, enc):
        tokens = enc.encode("import os\nimport sys\n")
        cum = detector._build_char_to_token_index(tokens)
        assert all(cum[i] <= cum[i + 1] for i in range(len(cum) - 1))

    def test_char_pos_at_zero_returns_zero(self, detector, enc):
        tokens = enc.encode("import os\n")
        cum = detector._build_char_to_token_index(tokens)
        assert detector._char_pos_to_token_pos(cum, 0) == 0

    def test_char_pos_beyond_end_clamped(self, detector, enc):
        tokens = enc.encode("abc")
        cum = detector._build_char_to_token_index(tokens)
        result = detector._char_pos_to_token_pos(cum, 10_000)
        assert result <= len(tokens)

    def test_char_pos_within_ascii_sequence(self, detector, enc):
        # For pure ASCII "import os\n", the cumulative sum should be
        # monotonically increasing by at least 1 per token.
        code = "import os\n"
        tokens = enc.encode(code)
        cum = detector._build_char_to_token_index(tokens)
        # Character at position len(code)-1 should map to the last token or close
        last_pos = detector._char_pos_to_token_pos(cum, len(code) - 1)
        assert 0 < last_pos <= len(tokens)


# ---------------------------------------------------------------------------
# _parse_relative_imports
# ---------------------------------------------------------------------------


class TestParseRelativeImports:
    def test_single_dot_inline(self):
        results = _parse_relative_imports("from . import foo\n")
        assert len(results) == 1
        mp, fn, cs, ce = results[0]
        assert mp == "."
        assert fn == "foo"
        assert ce > cs

    def test_dotted_module_inline(self):
        results = _parse_relative_imports("from .utils import helper\n")
        assert len(results) == 1
        mp, fn, _, _ = results[0]
        assert mp == ".utils"
        assert fn == "helper"

    def test_parent_package(self):
        results = _parse_relative_imports("from .. import models\n")
        assert len(results) == 1
        mp, fn, _, _ = results[0]
        assert mp == ".."
        assert fn == "models"

    def test_parent_dotted(self):
        results = _parse_relative_imports("from ..pkg import thing\n")
        assert len(results) == 1
        mp, fn, _, _ = results[0]
        assert mp == "..pkg"
        assert fn == "thing"

    def test_multiple_names_inline(self):
        results = _parse_relative_imports("from .schema import User, Role\n")
        assert len(results) == 2
        names = {fn for _, fn, _, _ in results}
        assert names == {"User", "Role"}
        # Both share same char range
        assert results[0][2] == results[1][2]

    def test_parenthesized(self):
        code = "from .schema import (\n    User,\n    Role,\n)\n"
        results = _parse_relative_imports(code)
        assert len(results) == 2
        names = {fn for _, fn, _, _ in results}
        assert names == {"User", "Role"}

    def test_absolute_imports_excluded(self):
        results = _parse_relative_imports("import os\nfrom foo import bar\n")
        assert results == []

    def test_mixed_absolute_and_relative(self):
        code = "from foo import bar\nfrom . import utils\n"
        results = _parse_relative_imports(code)
        assert len(results) == 1
        assert results[0][0] == "."

    def test_char_positions_nonzero(self):
        code = "x = 1\nfrom . import foo\n"
        results = _parse_relative_imports(code)
        assert len(results) == 1
        _, _, cs, ce = results[0]
        assert cs > 0   # statement starts after "x = 1\n"
        assert ce > cs


# ---------------------------------------------------------------------------
# _resolve_relative_import
# ---------------------------------------------------------------------------


class TestResolveRelativeImport:
    def test_single_dot_import(self):
        paths = _resolve_relative_import(".", "utils", "pkg/sub/mod.py")
        assert "pkg/sub/utils.py" in paths
        assert "pkg/sub/utils/__init__.py" in paths

    def test_double_dot_import(self):
        paths = _resolve_relative_import("..", "models", "pkg/sub/mod.py")
        assert "pkg/models.py" in paths
        assert "pkg/models/__init__.py" in paths

    def test_dotted_submodule(self):
        paths = _resolve_relative_import(".schema", "User", "pkg/sub/mod.py")
        # Most-specific first: schema/User.py before schema.py
        assert paths.index("pkg/sub/schema/User.py") < paths.index("pkg/sub/schema.py")

    def test_root_level_file(self):
        # Source file at repo root: "from . import utils" in top.py
        paths = _resolve_relative_import(".", "utils", "top.py")
        assert paths == ["utils.py", "utils/__init__.py"]

    def test_star_import_returns_package(self):
        # "from . import *" — targets the current package's __init__.py / module file.
        paths = _resolve_relative_import(".", "*", "pkg/sub/mod.py")
        assert "pkg/sub.py" in paths or "pkg/sub/__init__.py" in paths

    def test_over_deep_returns_empty(self):
        # Source is "a/mod.py" but import uses "../../.." — walks above root.
        paths = _resolve_relative_import("...", "foo", "a/mod.py")
        assert paths == []

    def test_backslash_separator_normalised(self):
        paths = _resolve_relative_import(".", "utils", "pkg\\sub\\mod.py")
        assert "pkg/sub/utils.py" in paths

    def test_from_name_empty_string_returns_package(self):
        # Empty from_name behaves like no submodule specified — returns the package.
        paths = _resolve_relative_import(".", "", "pkg/sub/mod.py")
        assert "pkg/sub.py" in paths or "pkg/sub/__init__.py" in paths


# ---------------------------------------------------------------------------
# PythonImportDetector.detect_links_for_doc
# ---------------------------------------------------------------------------


class TestDetectLinksForDoc:
    def test_relative_import_resolved(self, enc):
        detector = PythonImportDetector(decode_fn=enc.decode)
        code = "from . import utils\n"
        tokens = torch.tensor(enc.encode(code), dtype=torch.long)
        links = detector.detect_links_for_doc(tokens, "myrepo/myrepo:pkg/sub/mod.py")
        targets = {lk.target_str for lk in links}
        assert "pkg/sub/utils.py" in targets

    def test_absolute_import_also_present(self, enc):
        # detect_links_for_doc is a superset of detect_links for absolute imports.
        detector = PythonImportDetector(decode_fn=enc.decode)
        code = "import os.path\nfrom . import utils\n"
        tokens = torch.tensor(enc.encode(code), dtype=torch.long)
        links = detector.detect_links_for_doc(tokens, "myrepo/myrepo:pkg/sub/mod.py")
        targets = {lk.target_str for lk in links}
        assert "os/path.py" in targets           # absolute
        assert "pkg/sub/utils.py" in targets     # relative

    def test_positions_are_local(self, enc):
        # Returned link_end_pos must be < len(span tokens), not a global offset.
        detector = PythonImportDetector(decode_fn=enc.decode)
        code = "from . import utils\n"
        tokens = torch.tensor(enc.encode(code), dtype=torch.long)
        links = detector.detect_links_for_doc(tokens, "myrepo/myrepo:pkg/sub/mod.py")
        assert links, "expected at least one link"
        for lk in links:
            assert lk.link_end_pos <= len(tokens), (
                f"link_end_pos {lk.link_end_pos} exceeds span length {len(tokens)}"
            )

    def test_no_imports_returns_empty(self, enc):
        detector = PythonImportDetector(decode_fn=enc.decode)
        code = "x = 1\ny = x + 2\n"
        tokens = torch.tensor(enc.encode(code), dtype=torch.long)
        links = detector.detect_links_for_doc(tokens, "myrepo/myrepo:pkg/mod.py")
        assert links == []

    def test_raw_identifier_without_colon(self, enc):
        # Fallback: no repo prefix — treat whole string as file path.
        detector = PythonImportDetector(decode_fn=enc.decode)
        code = "from . import foo\n"
        tokens = torch.tensor(enc.encode(code), dtype=torch.long)
        links = detector.detect_links_for_doc(tokens, "pkg/sub/mod.py")
        targets = {lk.target_str for lk in links}
        assert "pkg/sub/foo.py" in targets


# ===========================================================================
# _strip_alias
# ===========================================================================


class TestStripAlias:
    def test_no_alias(self):
        assert _strip_alias("foo") == "foo"

    def test_simple_alias(self):
        assert _strip_alias("foo as bar") == "foo"

    def test_extra_whitespace(self):
        assert _strip_alias("  foo   as   bar  ") == "foo"

    def test_empty(self):
        assert _strip_alias("   ") == ""


# ===========================================================================
# The `from x import y as z` alias bug (design §10a) — REGRESSION TESTS
# ===========================================================================


class TestAliasedFromImportBug:
    """The pre-migration regex missed inline-aliased from-imports entirely and,
    where it matched, emitted a target_str of literally ``x/y as z`` (design
    §10a). These lock in the fix."""

    def test_inline_aliased_from_import_detected(self):
        # Previously MISSED completely (inline regex forbade the `as` clause).
        result = _parse_imports("from gettext import gettext as _\n")
        assert len(result) == 1
        module_path, from_name, _cs, _ce = result[0]
        assert module_path == "gettext"
        assert from_name == "gettext"  # NOT "gettext as _"

    def test_alias_stripped_from_dotted_module(self):
        result = _parse_imports("from keyword import iskeyword as is_kw\n")
        assert result == [("keyword", "iskeyword", result[0][2], result[0][3])]

    def test_target_str_has_no_alias(self, detector, enc):
        ids = _encode(enc, "from a.b import c as d\n")
        targets = {l.target_str for l in detector.detect_links(ids)}
        # No emitted target may contain " as " (the old bug produced "a/b/c as d").
        assert all(" as " not in t for t in targets)
        assert "a/b/c.py" in targets

    def test_multiple_names_with_aliases(self):
        result = _parse_imports("from mod import a as x, b, c as z\n")
        names = {fn for _, fn, _, _ in result}
        assert names == {"a", "b", "c"}

    def test_parenthesised_alias_stripped(self):
        code = "from mod import (\n    a as x,\n    b,\n)\n"
        names = {fn for _, fn, _, _ in _parse_imports(code)}
        assert names == {"a", "b"}

    def test_relative_inline_alias_stripped(self):
        result = _parse_relative_imports("from .schema import User as U\n")
        assert len(result) == 1
        assert result[0][1] == "User"


# ===========================================================================
# Comment / string blanking (docstring false-positive fix)
# ===========================================================================


class TestBlankCommentsAndStrings:
    def test_offsets_preserved(self):
        code = "import os\n# comment\nimport sys\n"
        blanked = _blank_comments_and_strings(code)
        assert len(blanked) == len(code)
        # newlines preserved so MULTILINE ^ anchors still work
        assert blanked.count("\n") == code.count("\n")

    def test_import_in_line_comment_not_detected(self):
        # A line comment that itself contains an import statement.
        code = "x = 1\n#import evil\nimport real\n"
        modules = {mp for mp, _, _, _ in _parse_imports(code)}
        assert "real" in modules
        assert "evil" not in modules

    def test_import_in_docstring_not_detected(self):
        code = (
            '"""\n'
            "Example usage::\n\n"
            "    from jinja2 import BaseLoader\n"
            "    from os.path import join\n"
            '"""\n'
            "import real_module\n"
        )
        result = _parse_imports(code)
        modules = {mp for mp, _, _, _ in result}
        assert "real_module" in modules
        assert "jinja2" not in modules
        assert "os.path" not in modules

    def test_import_in_single_quoted_string_not_detected(self):
        code = "s = 'from secret import key'\nimport real\n"
        modules = {mp for mp, _, _, _ in _parse_imports(code)}
        assert modules == {"real"}

    def test_hash_inside_string_not_treated_as_comment(self):
        # '#' inside a string must not start a comment that eats the next import.
        code = "s = 'a # b'\nimport real\n"
        modules = {mp for mp, _, _, _ in _parse_imports(code)}
        assert "real" in modules

    def test_real_imports_still_detected_after_docstring(self, detector, enc):
        code = (
            '"""module docstring: import fake"""\n'
            "import os\n"
            "from typing import List\n"
        )
        ids = _encode(enc, code)
        targets = {l.target_str for l in detector.detect_links(ids)}
        assert "os.py" in targets
        assert "typing.py" in targets
        assert all("fake" not in t for t in targets)


# ===========================================================================
# Batch-decode char->token index (perf path) equals per-token path
# ===========================================================================


class TestBatchDecodeIndex:
    def test_batch_path_active_for_tiktoken(self, detector):
        # tiktoken exposes decode_tokens_bytes -> fast path must be selected.
        assert detector._decode_tokens_bytes is not None

    def test_batch_index_matches_per_token(self, detector, enc):
        code = "import os\nfrom a.b import c\nx = 'hello world'\n"
        tokens = enc.encode(code)
        fast = detector._build_char_to_token_index(tokens)
        # force the fallback and recompute
        saved = detector._decode_tokens_bytes
        detector._decode_tokens_bytes = None
        try:
            slow = detector._build_char_to_token_index(tokens)
        finally:
            detector._decode_tokens_bytes = saved
        assert fast == slow

    def test_no_batch_capability_falls_back(self):
        # A plain decode callable (no __self__ encoder) uses the per-token path.
        enc2 = tiktoken.get_encoding("gpt2")
        det = PythonImportDetector(decode_fn=lambda ids: enc2.decode(ids))
        assert det._decode_tokens_bytes is None
        idx = det._build_char_to_token_index(enc2.encode("import os\n"))
        assert idx[0] == 0 and idx[-1] > 0
