from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter")

from eval.benchmark_harness.ports.internal_community import (
    INTERNAL_PORTS,
    _DATASET_DIR,
    _RELATIVE_LANGS,
    _aux_path,
    _basename_identifier,
    _build_examples,
    _split_dir,
)
from eval.benchmark_harness.ports import PORTS, get_port
from eval.benchmark_harness.schema import CrossDocExample, encode_example
from eval.benchmark_harness.scopes import _DECL_NODE_TYPES, scope_example


ALL_LANGS = sorted(_DATASET_DIR)
# Languages whose test_community split is materialized on this box; skip the
# data-backed tests for any that are absent so the suite stays green off-cluster.
_AVAILABLE = [l for l in ALL_LANGS if _split_dir(l).exists()]


def _some_available():
    if not _AVAILABLE:
        pytest.skip("no test_community splits materialized on this host")
    return _AVAILABLE


# ─── registration + pure-function invariants (no data needed) ──────────────────

def test_all_nine_ports_registered():
    for lang in ALL_LANGS:
        name = f"internal_{lang}"
        assert name in INTERNAL_PORTS
        assert name in PORTS
        assert get_port(name).language == lang


def test_scopes_cover_every_internal_language():
    # Every internal port language must have a declaration-type set, else the
    # use_line/use_block/rest_of_doc scopes silently drop every example.
    for lang in ALL_LANGS:
        assert _DECL_NODE_TYPES.get(lang), f"{lang} missing from _DECL_NODE_TYPES"


def test_identity_identifier_for_absolute_langs():
    for lang in ALL_LANGS:
        if lang in _RELATIVE_LANGS:
            continue
        fn = get_port(f"internal_{lang}").identifier_fn
        assert fn("repo", "some/key.path", "content") == "some/key.path"


def test_aux_path_strips_repo_prefix():
    # python/ts/js/dart/zig glue repo before ':'; rust before '@'; go/java/kotlin none.
    assert _aux_path("python", "owner_repo_ab12:virtool/foo.py") == "virtool/foo.py"
    assert _aux_path("rust", "owner/repo@crate::a::b") == "crate::a::b"
    assert _aux_path("typescript", "owner/repo:src/components/Foo") == "src/components/Foo"
    assert _aux_path("go", "github.com/o/r/pkg/sub") == "github.com/o/r/pkg/sub"
    assert _aux_path("java", "com.foo.Bar") == "com.foo.Bar"


def test_aux_path_is_tier0_clean():
    # No leading '/' and no '..' component (Tier 0 rejects those).
    for lang in ALL_LANGS:
        p = _aux_path(lang, "owner/repo:a/b/c.ext" if lang != "rust"
                      else "owner/repo@crate::a::b")
        assert not p.startswith("/")
        assert ".." not in p.split("/")


def test_basename_identifier_directory_index_keeps_parent():
    # foo/index must not collapse to bare 'index' (collides across a repo).
    assert _basename_identifier("typescript", "r", "src/foo/index.ts") == "r:foo/index"
    assert _basename_identifier("typescript", "r", "src/util/helper.ts") == "r:helper"


def test_basename_identifier_dart_keeps_extension():
    # dart's detector keeps '.dart' in its emitted key; ts/js/zig strip it.
    assert _basename_identifier("dart", "r", "lib/models/user.dart") == "r:user.dart"
    assert _basename_identifier("zig", "r", "src/system/ole.zig") == "r:ole"


# ─── data-backed generation (skips if splits are not on this host) ─────────────

@pytest.mark.parametrize("lang", ALL_LANGS)
def test_examples_build_and_validate(lang):
    if lang not in _AVAILABLE:
        pytest.skip(f"{lang} test_community split not materialized")
    exs = _build_examples(lang, max_examples=25)
    assert exs, f"{lang}: no examples built from test_community"
    for e in exs:
        assert isinstance(e, CrossDocExample)
        assert e.context.strip()
        assert e.target.strip()
        assert e.aux, "cross-doc example must have ≥1 aux"
        assert e.full_file, "full_file needed for use-site scopes"
        # Tier-0 aux-path invariants.
        for a in e.aux:
            assert not a.path.startswith("/")
            assert ".." not in a.path.split("/")
            assert a.content.strip()
        assert e.meta["split"] == "test_community"


def test_generation_is_deterministic():
    lang = _some_available()[0]
    a = _build_examples(lang, max_examples=20)
    b = _build_examples(lang, max_examples=20)
    assert len(a) == len(b)
    assert [e.meta["source_id"] for e in a] == [e.meta["source_id"] for e in b]
    assert [(e.context, e.target) for e in a] == [(e.context, e.target) for e in b]


def test_encode_example_roundtrips_completion_tokens():
    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    enc = lambda t: tok.encode(t, disallowed_special=())
    lang = _some_available()[0]
    port = get_port(f"internal_{lang}")
    ex = _build_examples(lang, max_examples=1)[0]
    packed = encode_example(ex, enc, port.identifier_fn)
    assert packed["completion_tokens"] == enc(ex.target)
    assert packed["context_tokens"] == enc(ex.context)
    assert len(packed["aux_token_lists"]) == len(packed["aux_raw_identifiers"])


def test_use_line_scope_available_for_some_examples():
    # The headline scope must resolve on at least one example per available lang
    # (proves _DECL_NODE_TYPES + full_file wiring actually yields use sites).
    lang = _some_available()[0]
    exs = _build_examples(lang, max_examples=40)
    assert any(scope_example(e, lang, "use_line") is not None for e in exs)
