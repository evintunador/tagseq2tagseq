"""
Tests for dart_import_detector.py

Coverage: _parse_import_uris (import/export/combinators/comment+string blanking),
_normalize_relative_uri (relative kept, dart:/package: excluded), _leading_scheme,
_resolve_relative_uri (./ , ../ , escape, external), detect_links (specifier-space),
detect_links_for_doc (path-resolved), index_doc_span.
"""
import pytest
import tiktoken
import torch

from model.graph_traversal.dart_import_detector import (
    DartImportDetector,
    _parse_import_uris,
    _normalize_relative_uri,
    _leading_scheme,
    _resolve_relative_uri,
)


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return DartImportDetector(decode_fn=enc.decode)


def _uris(text):
    return [u for u, _e in _parse_import_uris(text)]


# --- URI extraction ---------------------------------------------------------

def test_basic_import():
    assert _uris("import 'foo.dart';\n") == ["foo.dart"]


def test_double_quotes():
    assert _uris('import "foo.dart";\n') == ["foo.dart"]


def test_import_with_combinators():
    assert _uris("import '../models/user.dart' show User hide Bar;\n") == ["../models/user.dart"]


def test_import_with_alias_and_deferred():
    assert _uris("import 'bar.dart' deferred as b;\n") == ["bar.dart"]


def test_export_creates_uri():
    assert _uris("export 'src/api.dart' show A;\n") == ["src/api.dart"]


def test_dart_and_package_uris_extracted_but_normalize_to_none():
    got = _uris("import 'dart:async';\nimport 'package:flutter/material.dart';\n")
    assert got == ["dart:async", "package:flutter/material.dart"]
    assert _normalize_relative_uri("dart:async") is None
    assert _normalize_relative_uri("package:flutter/material.dart") is None


def test_part_directive_not_matched():
    # `part` splices a part-file into the same library — not a dependency edge.
    assert _uris("part 'gen.dart';\npart of 'lib.dart';\n") == []


def test_import_in_line_comment_blanked():
    assert _uris("// import 'nope.dart';\nimport 'real.dart';\n") == ["real.dart"]


def test_import_in_block_comment_blanked():
    assert _uris("/* import 'nope.dart'; */\nimport 'real.dart';\n") == ["real.dart"]


def test_import_in_doc_comment_blanked():
    assert _uris("/// import 'nope.dart';\nimport 'real.dart';\n") == ["real.dart"]


def test_import_in_triple_quoted_string_blanked():
    src = "const t = '''\nimport 'nope.dart';\n''';\nimport 'real.dart';\n"
    assert _uris(src) == ["real.dart"]


def test_conditional_import_primary_uri_only():
    # The `if (...) 'other.dart'` fallback is ignored (matches the tree-sitter
    # oracle, which captures only the primary configurable_uri).
    assert _uris("import 'a.dart' if (dart.library.io) 'a_io.dart';\n") == ["a.dart"]


# --- scheme + normalization -------------------------------------------------

def test_leading_scheme():
    assert _leading_scheme("dart:core") == "dart"
    assert _leading_scheme("package:x/y.dart") == "package"
    assert _leading_scheme("../a.dart") is None
    assert _leading_scheme("src/a.dart") is None
    assert _leading_scheme("./a.dart") is None


def test_normalize_relative_keeps_ext_and_dotdot():
    assert _normalize_relative_uri("foo.dart") == "foo.dart"
    assert _normalize_relative_uri("./foo.dart") == "foo.dart"
    assert _normalize_relative_uri("src/a.dart") == "src/a.dart"
    assert _normalize_relative_uri("../models/user.dart") == "../models/user.dart"


# --- relative resolution ----------------------------------------------------

def test_resolve_same_dir():
    assert _resolve_relative_uri("helper.dart", "lib/main.dart") == "lib/helper.dart"


def test_resolve_subdir():
    assert _resolve_relative_uri("util/helper.dart", "lib/main.dart") == "lib/util/helper.dart"


def test_resolve_updir():
    assert _resolve_relative_uri("../consts.dart", "lib/util/helper.dart") == "lib/consts.dart"


def test_resolve_explicit_current_dir():
    assert _resolve_relative_uri("./consts.dart", "lib/main.dart") == "lib/consts.dart"


def test_resolve_escape_returns_none():
    assert _resolve_relative_uri("../../../x.dart", "a/b.dart") is None


def test_resolve_external_returns_none():
    assert _resolve_relative_uri("dart:math", "lib/main.dart") is None
    assert _resolve_relative_uri("package:flutter/material.dart", "lib/main.dart") is None


# --- detect_links (specifier space) -----------------------------------------

def test_detect_links_specifier_space(enc, detector):
    src = "import 'util/helper.dart';\nimport 'dart:async';\nimport 'package:x/y.dart';\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    targets = {li.target_str for li in detector.detect_links(ids)}
    assert targets == {"util/helper.dart"}  # dart:/package: license nothing


def test_link_end_pos_after_import(enc, detector):
    src = "import 'util/helper.dart';\nvoid main() {}\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    li = detector.detect_links(ids)[0]
    prefix = enc.decode(ids[: li.link_end_pos].tolist())
    assert "util/helper.dart" in prefix


# --- detect_links_for_doc (path resolved) -----------------------------------

def test_detect_links_for_doc_resolves(enc, detector):
    src = "import '../consts.dart';\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    links = detector.detect_links_for_doc(ids, "repo:lib/util/helper.dart")
    targets = {li.target_str for li in links}
    assert targets == {"lib/consts.dart"}


# --- index_doc_span ---------------------------------------------------------

class _Span:
    def __init__(self, raw):
        self.raw_identifier = raw


def test_index_doc_span_strips_repo_prefix(detector):
    assert detector.index_doc_span(_Span("owner/repo:lib/models/user.dart")) == "lib/models/user.dart"
