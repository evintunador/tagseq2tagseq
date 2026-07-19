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
    relpath: str
    raw_identifier: str
    normed_identifier: str
    content: str


def _load_fixture(fixture_dir: Path, extensions: Set[str]) -> Tuple[List[_FixtureNode], Set[Tuple[str, str]]]:
    files_dir = fixture_dir / "files"
    edges_path = fixture_dir / "edges.json"
    if not files_dir.is_dir():
        raise FileNotFoundError(f"fixture {fixture_dir} has no files/ dir")

    nodes: List[_FixtureNode] = []
    for p in sorted(files_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(files_dir).as_posix()
        ext = rel.rsplit(".", 1)[-1] if "." in rel else ""
        # include language source files as nodes; project files (go.mod, etc.) are
        # readable by the extractor but are not themselves link targets unless the
        # extractor decides so — we still register them as nodes so a resolver may
        # legitimately reference them.
        content = p.read_text(encoding="utf-8", errors="replace")
        raw = f"fixture:{rel}"
        nodes.append(_FixtureNode(rel, raw, raw, content))

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
) -> ResolutionScore:
    """Score resolved edges from a fixture against its hand-labeled edges.json.

    Args:
        fixture_dir: directory with files/ and edges.json.
        extensions: source extensions for this language.
        edge_producer: callable taking the fixture nodes and returning the list of
            (from_relpath, raw_target_str) pairs the implementation emits — i.e.
            detected imports with their emitted target strings, BEFORE resolution.
            The runner resolves each raw_target_str itself via PretokCorpus logic,
            so this callable only owns detection, keeping resolution consistent
            with training/generation.
        link_detector: the LinkDetector, used to build the detector-key index
            exactly as PretokCorpus does. Required (resolution is detector-keyed).
    """
    fixture_dir = Path(fixture_dir)
    nodes, gold = _load_fixture(fixture_dir, extensions)

    node_dicts: Dict[str, Dict] = {
        n.normed_identifier: {
            "raw_identifier": n.raw_identifier,
            "normed_identifier": n.normed_identifier,
        }
        for n in nodes
    }
    raw_to_normed, key_to_normed = _build_indexes(node_dicts, link_detector)

    # normed_identifier "fixture:relpath" -> relpath, for scoring in relpath space
    def relpath_of(normed: Optional[str]) -> Optional[str]:
        if normed is None:
            return None
        return normed.split(":", 1)[1] if ":" in normed else normed

    produced_edges: Set[Tuple[str, str]] = set()
    for from_rel, raw_target in edge_producer(nodes):
        normed = _resolve_target(
            raw_target, raw_to_normed, key_to_normed,
            has_detector=link_detector is not None,
        )
        to_rel = relpath_of(normed)
        if to_rel is not None:
            produced_edges.add((from_rel, to_rel))

    tp = produced_edges & gold
    fp = produced_edges - gold
    fn = gold - produced_edges
    return ResolutionScore(
        fixture=fixture_dir.name,
        tp=len(tp), fp=len(fp), fn=len(fn),
        wrong_edges=sorted(fp)[:20],
        missing_edges=sorted(fn)[:20],
    )
