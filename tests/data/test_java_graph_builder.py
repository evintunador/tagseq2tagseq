"""
Tests for data/java_graph_extractor/build_java_graph.py — FQN derivation from
package+filename, static-import enclosing-type edges, no dangling/self edges.
"""
import pytest

pytest.importorskip("tree_sitter_java")

from data.java_graph_extractor.build_java_graph import (
    _JavaParser,
    _fqn_from,
    build_repo_nodes,
)


def test_fqn_from_package_and_path():
    assert _fqn_from("com.example", "src/main/java/com/example/Foo.java") == "com.example.Foo"
    assert _fqn_from("", "Foo.java") == "Foo"  # default package
    assert _fqn_from("a.b", "x/a/b/module-info.java") is None
    assert _fqn_from("a.b", "x/a/b/package-info.java") is None


def test_build_repo_nodes_edges():
    parser = _JavaParser()
    files = [
        ("src/main/java/com/ex/Main.java",
         "package com.ex;\nimport com.ex.util.Helper;\n"
         "import static com.ex.Consts.MAX;\npublic class Main {}\n"),
        ("src/main/java/com/ex/util/Helper.java",
         "package com.ex.util;\npublic class Helper {}\n"),
        ("src/main/java/com/ex/Consts.java",
         "package com.ex;\npublic class Consts { public static int MAX = 1; }\n"),
        # external import that isn't a node -> no edge
        ("src/main/java/com/ex/Uses.java",
         "package com.ex;\nimport java.util.List;\npublic class Uses {}\n"),
    ]
    nodes, contents = build_repo_nodes(files, parser)
    # build_repo_nodes does NOT apply the min-links filter (that's in build()),
    # so all four class files are present here; Uses just has no in-repo edges.
    assert set(nodes) == {
        "com.ex.Main", "com.ex.util.Helper", "com.ex.Consts", "com.ex.Uses",
    }
    # Main imports Helper (type) and Consts.MAX (static -> enclosing type Consts)
    assert nodes["com.ex.Main"]["outgoing"] == ["com.ex.Consts", "com.ex.util.Helper"]
    # incoming mirror
    assert "com.ex.Main" in nodes["com.ex.util.Helper"]["incoming"]
    assert "com.ex.Main" in nodes["com.ex.Consts"]["incoming"]
    # no dangling / self
    ids = set(nodes)
    for n in nodes.values():
        for o in n["outgoing"]:
            assert o in ids and o != n["normed_identifier"]
    # java.util.List import produced no edge (not a node)
    assert "java.util.List" not in ids
