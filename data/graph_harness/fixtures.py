"""
Fixtures runner — the RESOLUTION-axis oracle.

Tree-sitter validates DETECTION ("what does this file import?"). It cannot validate
RESOLUTION ("which node does that import point to?") — for that we need ground
truth about a corpus. A fixture is a tiny, self-contained, hand-labeled repo where
we know every correct edge, so we can score resolution precision/recall directly,
with no model and no trained checkpoint.

Fixture format (a directory `<lang>/<fixture_name>/`):
    files/                       # the source tree (real <lang> files)
        go.mod                   # (language-specific project files as needed)
        main.go
        pkg/util/util.go
    edges.json                   # ground-truth: list of {"from": path, "to": path}
                                 #   paths are repo-relative; each is a directed
                                 #   file->file (or file->package) dependency edge
                                 #   the resolver SHOULD produce. Imports of
                                 #   external deps are simply omitted.

The runner:
  1. builds the in-memory node set from files/ (raw_identifier = "fixture:relpath",
     mirroring the Stack "owner/repo:path" shape so index_doc_span works);
  2. runs the language's build-time extractor OR link detector to produce edges;
  3. resolves each emitted target via the SAME PretokCorpus resolution logic
     training/generation use (`_build_indexes` + `_resolve_target`);
  4. scores resolved edges vs. edges.json (precision/recall), reporting the
     concrete wrong / missing edges.

This is deliberately model-free and corpus-free: fixtures are the broad-coverage
resolution oracle. The narrow gold oracle (language toolchain: `go list`, etc.) is
a separate, optional cross-check layered on top for buildable code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from model.document_corpus import _build_indexes, _resolve_target


@dataclass
class ResolutionScore:
    fixture: str
    tp: int
    fp: int
    fn: int
    wrong_edges: List[Tuple[str, str]] = field(default_factory=list)   # (from, bad_to)
    missing_edges: List[Tuple[str, str]] = field(default_factory=list)  # (from, to)

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return 1.0 if d == 0 else self.tp / d

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return 1.0 if d == 0 else self.tp / d

    def passes(self, min_precision: float, min_recall: float) -> bool:
        return self.precision >= min_precision and self.recall >= min_recall

    def summary(self) -> str:
        return (
            f"[{self.fixture}] resolution P={self.precision:.3f} R={self.recall:.3f} "
            f"(tp={self.tp} fp={self.fp} fn={self.fn})"
        )


@dataclass
class _FixtureNode:
    # `key`: the node's identity in edges.json + scoring space (relpath for
    # file-node languages like Python; import path for package-node languages like
    # Go). `raw_identifier`: fed to the detector's index_doc_span to build the
    # resolution index (for Python "fixture:relpath"; for Go the import path). The
    # two differ because index_doc_span may strip a prefix (Python) or not (Go).
    key: str
    raw_identifier: str
    normed_identifier: str
    content: str
    relpath: str = ""  # a representative source path (first file, for a package)


@dataclass
class _FixtureFile:
    relpath: str
    content: str


def default_file_node_builder(files: List[_FixtureFile], extensions: Set[str]) -> List[_FixtureNode]:
    """File-per-node model (Python and any language where import ≈ one file).

    key = relpath; raw_identifier = "fixture:relpath" (so a detector's
    index_doc_span that strips a "<repo>:" prefix yields the bare relpath key).
    """
    nodes: List[_FixtureNode] = []
    for f in files:
        raw = f"fixture:{f.relpath}"
        nodes.append(_FixtureNode(
            key=f.relpath, raw_identifier=raw, normed_identifier=raw,
            content=f.content, relpath=f.relpath,
        ))
    return nodes


def _load_fixture(
    fixture_dir: Path,
    extensions: Set[str],
    node_builder,
) -> Tuple[List[_FixtureNode], Set[Tuple[str, str]]]:
    files_dir = fixture_dir / "files"
    edges_path = fixture_dir / "edges.json"
    if not files_dir.is_dir():
        raise FileNotFoundError(f"fixture {fixture_dir} has no files/ dir")

    files: List[_FixtureFile] = []
    for p in sorted(files_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(files_dir).as_posix()
        files.append(_FixtureFile(rel, p.read_text(encoding="utf-8", errors="replace")))

    nodes = node_builder(files, extensions)

    gold: Set[Tuple[str, str]] = set()
    if edges_path.exists():
        data = json.loads(edges_path.read_text())
        for e in data:
            gold.add((e["from"], e["to"]))
    return nodes, gold


def score_resolution(
    fixture_dir: "str | Path",
    extensions: Set[str],
    edge_producer: Callable[[List[_FixtureNode]], List[Tuple[str, str]]],
    link_detector=None,
    node_builder=default_file_node_builder,
) -> ResolutionScore:
    """Score resolved edges from a fixture against its hand-labeled edges.json.

    Args:
        fixture_dir: directory with files/ and edges.json.
        extensions: source extensions for this language.
        edge_producer: callable taking the fixture nodes and returning the list of
            (from_node_key, raw_target_str) pairs the implementation emits — i.e.
            detected imports with their emitted target strings, BEFORE resolution.
            The runner resolves each raw_target_str itself via PretokCorpus logic,
            so this callable only owns detection, keeping resolution consistent
            with training/generation. ``from_node_key`` must be in the same space
            as the ``key`` of the fixture nodes and edges.json (relpath for
            file-node languages, import path for package-node languages).
        link_detector: the LinkDetector, used to build the detector-key index
            exactly as PretokCorpus does. Required (resolution is detector-keyed).
        node_builder: maps the fixture's files to nodes. Default is one node per
            file (Python); Go supplies a package-grouping builder so a node is a
            directory of .go files (see design doc §Go pilot).
    """
    fixture_dir = Path(fixture_dir)
    nodes, gold = _load_fixture(fixture_dir, extensions, node_builder)

    node_dicts: Dict[str, Dict] = {
        n.normed_identifier: {
            "raw_identifier": n.raw_identifier,
            "normed_identifier": n.normed_identifier,
        }
        for n in nodes
    }
    raw_to_normed, key_to_normed = _build_indexes(node_dicts, link_detector)

    # Map a resolved normed_identifier back to its node key (scoring space).
    normed_to_key: Dict[str, str] = {n.normed_identifier: n.key for n in nodes}

    produced_edges: Set[Tuple[str, str]] = set()
    for from_key, raw_target in edge_producer(nodes):
        normed = _resolve_target(
            raw_target, raw_to_normed, key_to_normed,
            has_detector=link_detector is not None,
        )
        to_key = normed_to_key.get(normed) if normed is not None else None
        if to_key is not None:
            produced_edges.add((from_key, to_key))

    tp = produced_edges & gold
    fp = produced_edges - gold
    fn = gold - produced_edges
    return ResolutionScore(
        fixture=fixture_dir.name,
        tp=len(tp), fp=len(fp), fn=len(fn),
        wrong_edges=sorted(fp)[:20],
        missing_edges=sorted(fn)[:20],
    )
