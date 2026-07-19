"""
Tests for data/go_graph_extractor/build_go_graph.py

Focus on the parts that are NOT already covered by the harness detection/resolution
tests: module-path inference (The Stack has no go.mod), vendor exclusion, and
package-node edge building with no dangling/self edges.
"""
import pytest

pytest.importorskip("tree_sitter_go")

from data.go_graph_extractor.build_go_graph import (
    _GoParser,
    _is_vendored,
    build_repo_packages,
    infer_module_path,
)


def test_infer_module_from_self_imports():
    # repo has packages a, a/b, a/b/c; imports reference them under a host prefix
    pkg_dirs = {"", "b", "b/c"}
    imports = {
        "github.com/me/proj/b",
        "github.com/me/proj/b/c",
        "fmt",  # stdlib noise
    }
    assert infer_module_path(pkg_dirs, imports) == "github.com/me/proj"


def test_infer_module_prefers_go_mod_when_present():
    assert infer_module_path({"b"}, {"x/y/b"}, go_mod_module="real/module") == "real/module"


def test_infer_module_none_when_no_self_refs():
    # single-package repo whose imports never reference its own dirs -> no module
    assert infer_module_path({""}, {"fmt", "net/http"}) is None


def test_infer_module_rejects_relative_and_dotless():
    # a '..' artifact or a dotless (stdlib-like) prefix must not win
    assert infer_module_path({"b"}, {"../b", "internal/b"}) is None


def test_is_vendored():
    assert _is_vendored("vendor/github.com/x/y/z.go")
    assert _is_vendored("pkg/vendor/a/b.go")
    assert not _is_vendored("pkg/util/util.go")


def test_build_repo_packages_edges():
    parser = _GoParser()
    files = [
        ("cmd/app/main.go",
         'package main\nimport (\n\t"fmt"\n\t"github.com/me/proj/store"\n)\n'
         'func main(){ _ = store.X; fmt.Println("hi") }\n'),
        ("store/store.go",
         'package store\nimport "github.com/me/proj/util"\nvar X = util.V\n'),
        ("store/extra.go", "package store\n"),  # same package, 2nd file
        ("util/util.go", "package util\nvar V = 1\n"),
        # a vendored file that must be ignored
        ("vendor/github.com/other/lib/lib.go", "package lib\n"),
        # a test file that must be ignored
        ("util/util_test.go", 'package util\nimport "testing"\n'),
    ]
    nodes, contents = build_repo_packages(files, parser)
    ids = set(nodes)
    assert ids == {
        "github.com/me/proj/cmd/app",
        "github.com/me/proj/store",
        "github.com/me/proj/util",
    }
    # store is ONE node from two files
    assert nodes["github.com/me/proj/store"]["n_files"] == 2
    # edges: cmd/app -> store, store -> util
    assert nodes["github.com/me/proj/cmd/app"]["outgoing"] == ["github.com/me/proj/store"]
    assert nodes["github.com/me/proj/store"]["outgoing"] == ["github.com/me/proj/util"]
    # incoming mirrors
    assert nodes["github.com/me/proj/util"]["incoming"] == ["github.com/me/proj/store"]
    # no dangling / self
    node_ids = set(nodes)
    for n in nodes.values():
        for o in n["outgoing"]:
            assert o in node_ids and o != n["normed_identifier"]
    # vendored package not present
    assert not any("other/lib" in i for i in ids)


def test_build_repo_skips_when_no_module_inferable():
    parser = _GoParser()
    files = [("main.go", 'package main\nimport "fmt"\nfunc main(){}\n')]
    nodes, contents = build_repo_packages(files, parser)
    assert nodes == {}
