"""
Tests for python_import_detector.py

Coverage:
  - module_path_to_file_paths: all import forms and edge cases
  - _parse_imports: every supported syntax variant + relative/edge cases
  - PythonImportDetector.detect_links: end-to-end with tiktoken GPT-2
  - PythonImportDetector.index_doc_span: repo-prefix stripping
  - PythonImportDetector._build_char_to_token_index / _char_pos_to_token_pos
"""

import sys
from pathlib import Path

import pytest
import tiktoken
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python_import_detector import (
    PythonImportDetector,
    _parse_imports,
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
