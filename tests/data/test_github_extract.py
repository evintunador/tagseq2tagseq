"""Tests for the tree-sitter build-time import extractor.

`data/github_graph_extractor/extract.py` builds the shipped TheStack graph. It
was migrated from hand-regexes to a tree-sitter engine (mirroring Go/Java/Rust/TS
builders). These lock in the OUTPUT contract (a set of dotted module-name
strings) and the specific correctness fixes that motivated the migration:
  - `import a, b` captures BOTH modules (old regex emitted `{'a,'}`, dropping b);
  - `as` aliases are stripped;
  - imports inside docstrings/strings/comments are ignored;
  - relative imports keep their leading dots.
"""
import pytest

from data.github_graph_extractor.extract import (
    extract_file_imports,
    _PyImportParser,
)


def _mods(content):
    return extract_file_imports(content, "pkg/mod.py", "owner/repo")


class TestTreeSitterAvailable:
    def test_parser_constructs(self):
        # tree_sitter_python must be importable in this env (peer of go/java/ts).
        parser = _PyImportParser.get()
        assert parser is not None


class TestModuleNameContract:
    def test_plain_import(self):
        assert _mods("import mymod\n") == {"mymod"}

    def test_dotted_import(self):
        assert _mods("import a.b.c\n") == {"a.b.c"}

    def test_comma_separated_both_captured(self):
        # The old regex bug: emitted {'local_a,'} and dropped local_b.
        assert _mods("import local_a, local_b\n") == {"local_a", "local_b"}

    def test_aliased_import_alias_stripped(self):
        assert _mods("import a.b.c as x\n") == {"a.b.c"}

    def test_from_import_module_only(self):
        # From-imports emit the MODULE path, not the imported names.
        assert _mods("from a.b import c, d\n") == {"a.b"}

    def test_from_import_aliased_module_only(self):
        assert _mods("from a.b import c as d\n") == {"a.b"}

    def test_relative_dot(self):
        assert _mods("from . import foo\n") == {"."}

    def test_relative_dotted(self):
        assert _mods("from ..pkg import bar\n") == {"..pkg"}

    def test_future_import(self):
        # `from __future__ import ...` parses to a distinct tree-sitter node type;
        # captured as module '__future__' (matches the harness oracle; never
        # resolves to a repo file, so graph impact is nil).
        assert _mods("from __future__ import annotations\n") == {"__future__"}


class TestIgnoresStringsAndComments:
    def test_import_in_docstring_ignored(self):
        content = '"""\nimport in_docstring\n"""\nimport real\n'
        assert _mods(content) == {"real"}

    def test_import_in_comment_ignored(self):
        assert _mods("# import commented\nimport real\n") == {"real"}

    def test_import_in_string_ignored(self):
        assert _mods("s = 'import in_string'\nimport real\n") == {"real"}


class TestDenylist:
    def test_stdlib_skipped(self):
        # os/sys are on the denylist -> never emitted as candidate repo files.
        assert _mods("import os\nimport sys\nimport mylocal\n") == {"mylocal"}


class TestRegexFallbackParity:
    """The regex fallback (used only if tree_sitter is unavailable) should agree
    with tree-sitter on the common cases."""

    def test_fallback_comma_and_alias(self):
        from data.github_graph_extractor.extract import _extract_module_names_regex
        got = _extract_module_names_regex("import a, b as c\nfrom d.e import f\n")
        assert got == {"a", "b", "d.e"}

    def test_fallback_relative(self):
        from data.github_graph_extractor.extract import _extract_module_names_regex
        assert "..pkg" in _extract_module_names_regex("from ..pkg import x\n")
