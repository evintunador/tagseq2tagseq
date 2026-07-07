"""
PretokCorpus — dataset-agnostic corpus access for TS2TS generation and eval.

Thin wrapper around GraphIndex + PretokShardedBackend that satisfies the corpus
protocol expected by the generation loop:

    has_document(target_str) -> bool
    get_document(target_str) -> Iterator[int]

Link resolution matches the string a detector emits (``target_str``) against the
key a corpus document was filed under. Two indexes are maintained:

  1. Exact: raw_identifier -> normed_identifier (works for Wikipedia / arXiv,
     where target_str equals the corpus raw_identifier).
  2. Detector-key: link_detector.index_doc_span(node) -> normed_identifier. This
     is the SAME match key training uses (see cross_doc_mask._match_links_to_docs).
     For the Python import detector it strips the repo prefix
     ("owner/repo:path/to/file.py" -> "path/to/file.py"), so a bare relative
     import path resolves; for markdown / arxiv / null detectors it returns the
     raw_identifier unchanged, so the detector-key index simply mirrors index 1.

Lookups try the exact index first (backward compatible) then the detector-key
index. The corpus is intentionally "dumb": it indexes whatever directory it is
pointed at and knows nothing about repos or scoping. Single-repo code corpora
(needed because bare-path keys collide across repos) are produced upstream as a
data-prep step — see data/make_repo_corpus.py — and this class is simply pointed
at the resulting directory.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from data.dataset import GraphIndex, PretokShardedBackend


class _NodeSpan:
    """Minimal span shim exposing only ``raw_identifier``.

    LinkDetector.index_doc_span reads nothing but ``raw_identifier``, so we avoid
    constructing a full data.collate.DocSpan (which needs 7 fields) and keep this
    module decoupled from the packing layer.
    """
    __slots__ = ("raw_identifier",)

    def __init__(self, raw_identifier: str):
        self.raw_identifier = raw_identifier


def _build_indexes(
    nodes: Dict[str, Dict[str, Any]],
    link_detector=None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build the (exact, detector_key) lookup indexes from graph nodes.

    Pure function of ``nodes`` (a mapping of normed_identifier -> node dict) so it
    can be unit-tested without a GraphIndex / on-disk shards.

    Returns:
        exact:      raw_identifier -> normed_identifier for every node.
        detector_key: index_doc_span(node) -> normed_identifier, or {} when no
                      link_detector is given. First-wins on key collisions.
    """
    exact: Dict[str, str] = {}
    detector_key: Dict[str, str] = {}
    for node in nodes.values():
        raw = node.get("raw_identifier")
        normed = node.get("normed_identifier")
        if raw is None or normed is None:
            continue
        exact.setdefault(raw, normed)
        if link_detector is not None:
            key = link_detector.index_doc_span(_NodeSpan(raw))
            detector_key.setdefault(key, normed)
    return exact, detector_key


class PretokCorpus:
    """
    Wrapper around GraphIndex and PretokShardedBackend for corpus access.

    Args:
        dataset_dir: Path to a pretokenized dataset directory (metadata.json,
            tokenized_graph.jsonl, and shard .bin files, or a split/repo dir
            whose metadata references the parent's shards by absolute path).
        link_detector: Optional LinkDetector. When provided, a detector-key index
            is built so detector-emitted targets that differ from raw_identifier
            (e.g. bare Python import paths) resolve. None = exact match only.
    """

    def __init__(self, dataset_dir: "str | Path", link_detector=None):
        dataset_dir = Path(dataset_dir)
        self._graph = GraphIndex(dataset_dir)
        self._backend = PretokShardedBackend(self._graph)
        self._link_detector = link_detector
        # A miss in both indexes means the target is not in the corpus (e.g. a
        # hallucinated title) -> has_document returns False.
        self._raw_to_normed, self._key_to_normed = _build_indexes(
            self._graph.nodes, link_detector
        )

    def _resolve(self, target: str) -> Optional[str]:
        """Resolve a detector target string to a corpus normed_identifier.

        Exact raw_identifier match first (backward compatible), then the
        detector-key index. Returns None if the target is not in the corpus.
        """
        normed = self._raw_to_normed.get(target)
        if normed is not None:
            return normed
        if self._link_detector is not None:
            return self._key_to_normed.get(target)
        return None

    def has_document(self, target: str) -> bool:
        """True if a document matching ``target`` exists in the corpus."""
        return self._resolve(target) is not None

    def get_document(self, target: str):
        """Yield the token IDs of the document matching ``target``.

        Returns an empty iterator if the target is not in the corpus (or the
        backend has no tokens for it).
        """
        normed = self._resolve(target)
        if normed is None:
            return iter([])
        tokens = self._backend.get_tokens(normed)
        if tokens is None:
            return iter([])
        return iter(tokens.tolist())

    def close(self):
        """Close backend resources (memory-mapped files)."""
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
