"""
Tests for rust_import_detector.py

Coverage: use-tree expansion (simple, grouped, nested, self/super, glob, alias,
re-export), candidate projection, comment/string blanking, self/super rewriting,
detect_links end-to-end with GPT-2, index_doc_span repo-prefix stripping.
"""
import pytest
import tiktoken
import torch

from model.graph_traversal.rust_import_detector import (
    RustImportDetector,
    _blank_comments_and_strings,
    _parse_targets,
    expand_use_tree_string,
    leaf_to_candidates,
    leaf_paths_to_candidates,
    rewrite_relative,
)


@pytest.fixture(scope="module")
def enc():
    return tiktoken.get_encoding("gpt2")


@pytest.fixture(scope="module")
def detector(enc):
    return RustImportDetector(decode_fn=enc.decode)


def _targets(text, module_path=None):
    return [t for t, _e in _parse_targets(text, module_path=module_path)]


# --- use-tree string expansion ---

def test_expand_simple():
    assert expand_use_tree_string("crate::net::tcp::Conn") == ["crate::net::tcp::Conn"]


def test_expand_grouped():
    assert expand_use_tree_string("crate::a::{b::C, d::E}") == [
        "crate::a::b::C", "crate::a::d::E"]


def test_expand_group_with_self():
    assert expand_use_tree_string("crate::a::{self, b}") == ["crate::a", "crate::a::b"]


def test_expand_nested():
    assert expand_use_tree_string("a::{b::{c, d}, e}") == ["a::b::c", "a::b::d", "a::e"]


def test_expand_glob():
    assert expand_use_tree_string("crate::foo::*") == ["crate::foo::*"]


def test_expand_alias_dropped():
    assert expand_use_tree_string("crate::a as foo") == ["crate::a"]


# --- candidate projection ---

def test_leaf_candidates_parent_and_full():
    assert leaf_to_candidates("crate::a::b::C") == ["crate::a::b", "crate::a::b::C"]


def test_leaf_candidates_glob_is_module():
    assert leaf_to_candidates("crate::foo::*") == ["crate::foo"]


def test_leaf_candidates_single_segment():
    assert leaf_to_candidates("crate") == ["crate"]


def test_leaf_candidates_self_alone_empty():
    assert leaf_to_candidates("self") == []


# --- self/super rewriting ---

def test_rewrite_self():
    assert rewrite_relative("self::x", "crate::net::tcp") == "crate::net::tcp::x"


def test_rewrite_super():
    assert rewrite_relative("super::y::Z", "crate::net::tcp") == "crate::net::y::Z"


def test_rewrite_super_super():
    assert rewrite_relative("super::super::z", "crate::net::tcp") == "crate::z"


def test_rewrite_absolute_unchanged():
    assert rewrite_relative("crate::a::B", "crate::net") == "crate::a::B"
    assert rewrite_relative("std::io::Read", "crate::net") == "std::io::Read"


# --- comment / string blanking ---

def test_blank_line_comment():
    blanked = _blank_comments_and_strings("// use crate::x::Y;\nuse crate::a::B;\n")
    assert "crate::x" not in blanked
    assert "crate::a" in blanked


def test_blank_block_comment_nested():
    src = "/* outer /* inner use crate::x::Y; */ still */ use crate::a::B;\n"
    blanked = _blank_comments_and_strings(src)
    assert "crate::x" not in blanked
    assert "crate::a" in blanked


def test_blank_string_literal():
    blanked = _blank_comments_and_strings('let s = "use crate::x::Y;";\nuse crate::a::B;\n')
    assert "crate::x" not in blanked
    assert "crate::a" in blanked


def test_lifetime_not_treated_as_string():
    # a lifetime 'a must not swallow the following `use`
    src = "fn f<'a>(x: &'a str) {}\nuse crate::a::B;\n"
    assert "crate::a::B" in _blank_comments_and_strings(src)


# --- parse_targets integration ---

def test_parse_use_and_mod():
    src = "mod net;\nuse crate::a::B;\n"
    got = _targets(src)
    assert "net" in got
    assert "crate::a" in got and "crate::a::B" in got


def test_inline_mod_not_a_target():
    # inline mod (with body) does NOT emit a mod target
    src = "mod inline { fn f() {} }\n"
    assert _targets(src) == []


def test_self_super_relative_when_no_module():
    src = "use self::x;\nuse super::y::Z;\n"
    got = _targets(src)
    assert "self::x" in got
    assert "super::y" in got and "super::y::Z" in got


def test_self_super_rewritten_with_module():
    src = "use self::x;\nuse super::y::Z;\nmod child;\n"
    got = _targets(src, module_path="crate::net::tcp")
    assert "crate::net::tcp::x" in got
    assert "crate::net::y" in got and "crate::net::y::Z" in got
    assert "crate::net::tcp::child" in got  # mod resolved against module path


# --- detect_links end-to-end ---

def test_detect_links_targets(enc, detector):
    src = "use crate::a::B;\nuse std::io::Read;\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    targets = {li.target_str for li in detector.detect_links(ids)}
    assert "crate::a" in targets and "crate::a::B" in targets
    assert "std::io" in targets and "std::io::Read" in targets


def test_link_end_pos_after_use(enc, detector):
    src = "use crate::a::B;\nfn main() {}\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    li = detector.detect_links(ids)[0]
    prefix = enc.decode(ids[: li.link_end_pos].tolist())
    assert "crate::a::B" in prefix


class _Span:
    def __init__(self, raw):
        self.raw_identifier = raw


def test_index_doc_span_strips_repo_prefix(detector):
    assert detector.index_doc_span(_Span("owner/repo@crate::net::tcp")) == "crate::net::tcp"
    # no prefix -> returned unchanged
    assert detector.index_doc_span(_Span("crate::net::tcp")) == "crate::net::tcp"


def test_detect_and_index_share_space(enc, detector):
    src = "use crate::net::tcp::Conn;\n"
    ids = torch.tensor(enc.encode(src), dtype=torch.long)
    targets = {li.target_str for li in detector.detect_links(ids)}
    # parent-module candidate matches a node keyed by that module path
    node = _Span("owner/repo@crate::net::tcp")
    assert detector.index_doc_span(node) in targets
