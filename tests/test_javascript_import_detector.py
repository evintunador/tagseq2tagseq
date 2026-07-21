"""
Tests for javascript_import_detector.py

Coverage: _parse_import_specs (import/from/side-effect/re-export/require/dynamic,
comment- and template-blanking), _spec_to_detection_keys (index candidate, ext
strip, bare excluded), _resolve_relative_spec (./ , ../ , index, escape),
detect_links (specifier-space), detect_links_for_doc (path-resolved), index_doc_span.
"""
import pytest
import tiktoken
import torch

from model.graph_traversal.javascript_import_detector import (
    JavaScriptImportDetector,
    _parse_import_specs,
    _spec_to_detection_keys,
    _resolve_relative_spec,
)


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return JavaScriptImportDetector(decode_fn=enc.decode)


def _specs(text):
    return [s for s, _e in _parse_import_specs(text)]


# --- specifier extraction ---------------------------------------------------

def test_named_import():
    assert _specs('import { a, b } from "./mod";\n') == ["./mod"]


def test_default_and_namespace_import():
    assert _specs('import D from "./d";\nimport * as ns from "../x/y";\n') == ["./d", "../x/y"]


def test_side_effect_import():
    assert _specs('import "./side";\n') == ["./side"]


def test_reexport():
    assert _specs('export { q } from "./q";\nexport * from "./r";\n') == ["./q", "./r"]


def test_require_and_dynamic_import():
    got = _specs('const z = require("./z");\nconst d = import("./dyn");\n')
    assert "./z" in got and "./dyn" in got


def test_local_export_has_no_specifier():
    assert _specs('export const x = 1;\nexport { onlyLocal };\n') == []


def test_member_require_not_matched():
    # foo.require("./x") is not a CommonJS require
    assert _specs('foo.require("./x");\n') == []


def test_nonliteral_dynamic_import_dropped():
    assert _specs('const d = import(varName);\n') == []


def test_nonliteral_require_dropped():
    assert _specs('const d = require(varName);\n') == []


def test_import_in_comment_blanked():
    src = '// import { x } from "./nope";\n/* require("./also-nope"); */\nimport { y } from "./real";\n'
    assert _specs(src) == ["./real"]


def test_require_in_template_blanked():
    # a require(...) STRING inside a backtick template is codegen text, not an import
    src = 'const t = `const x = require("./ghost");`;\nconst z = require("./real");\n'
    assert _specs(src) == ["./real"]


def test_bare_specifier_still_extracted_but_licenses_nothing():
    # _parse_import_specs returns the raw specifier; key normalization drops bare.
    assert _specs('import React from "react";\n') == ["react"]
    assert _spec_to_detection_keys("react") == set()


def test_bare_require_licenses_nothing():
    assert _specs('const _ = require("lodash");\n') == ["lodash"]
    assert _spec_to_detection_keys("lodash") == set()


# --- detection key normalization --------------------------------------------

def test_detection_keys_index_candidate():
    assert _spec_to_detection_keys("./foo") == {"foo", "foo/index"}


def test_detection_keys_strip_ext():
    assert _spec_to_detection_keys("./sub/b.js") == {"sub/b", "sub/b/index"}
    assert _spec_to_detection_keys("./sub/b.jsx") == {"sub/b", "sub/b/index"}
    assert _spec_to_detection_keys("./sub/b.mjs") == {"sub/b", "sub/b/index"}
    assert _spec_to_detection_keys("./sub/b.cjs") == {"sub/b", "sub/b/index"}


def test_detection_keys_dotdot_retained():
    assert _spec_to_detection_keys("../x/y") == {"../x/y", "../x/y/index"}


def test_detection_keys_bare_empty():
    assert _spec_to_detection_keys("lodash") == set()
    assert _spec_to_detection_keys("@angular/core") == set()


# --- relative resolution ----------------------------------------------------

def test_resolve_same_dir():
    assert _resolve_relative_spec("./helper", "src/main") == ["src/helper", "src/helper/index"]


def test_resolve_updir():
    assert _resolve_relative_spec("../consts", "src/util/helper") == [
        "src/consts", "src/consts/index"]


def test_resolve_index_candidate_emitted():
    assert _resolve_relative_spec("./models", "src/main") == ["src/models", "src/models/index"]


def test_resolve_escape_returns_empty():
    assert _resolve_relative_spec("../../../x", "a/b") == []


def test_resolve_bare_returns_empty():
    assert _resolve_relative_spec("react", "src/main") == []


# --- detect_links (specifier space) -----------------------------------------

def test_detect_links_specifier_space(enc, detector):
    src = 'import { helper } from "./util/helper";\nimport React from "react";\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    targets = {li.target_str for li in detector.detect_links(ids)}
    assert targets == {"util/helper", "util/helper/index"}  # react licenses nothing


def test_detect_links_require(enc, detector):
    src = 'const z = require("./util/helper");\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    targets = {li.target_str for li in detector.detect_links(ids)}
    assert targets == {"util/helper", "util/helper/index"}


def test_link_end_pos_after_import(enc, detector):
    src = 'import { helper } from "./util/helper";\nconst x = 1;\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    li = detector.detect_links(ids)[0]
    prefix = enc.decode(ids[: li.link_end_pos].tolist())
    assert "./util/helper" in prefix


# --- detect_links_for_doc (path resolved) -----------------------------------

def test_detect_links_for_doc_resolves(enc, detector):
    src = 'const { VALUE } = require("../consts");\n'
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    links = detector.detect_links_for_doc(ids, "repo:src/util/helper.js")
    targets = {li.target_str for li in links}
    assert "src/consts" in targets
    assert "src/consts/index" in targets


# --- index_doc_span ---------------------------------------------------------

class _Span:
    def __init__(self, raw):
        self.raw_identifier = raw


def test_index_doc_span_strips_prefix_and_ext(detector):
    assert detector.index_doc_span(_Span("owner/repo:src/util/helper.js")) == "src/util/helper"
    assert detector.index_doc_span(_Span("owner/repo:src/models/index")) == "src/models/index"
