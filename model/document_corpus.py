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

Lookups try the exact index first (backward compatible), then the detector-key
index, and finally — only when ``fuzzy_match=True`` — a cascading fuzzy match
(exact → norm → word_overlap → edit_distance) via the SAME ``HashNormTitleIndex``
the eval annotators use (eval/title_index.py). The fuzzy tier fires only after
both exact indexes miss, so enabling it can only ADD resolutions (near-miss
titles the model emitted) — it never overrides or changes an existing exact hit.
This makes generation-time link resolution match the eval annotator cascade
(previously eval could recover a near-miss title that generation would drop).

The corpus is intentionally "dumb": it indexes whatever directory it is
pointed at and knows nothing about repos or scoping. Single-repo code corpora
(needed because bare-path keys collide across repos) are produced upstream as a
data-prep step — see data/make_repo_corpus.py — and this class is simply pointed
at the resulting directory.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from data.dataset import GraphIndex, PretokShardedBackend
from eval.title_index import HashNormTitleIndex


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


def _resolve_target(
    target: str,
    raw_to_normed: Dict[str, str],
    key_to_normed: Dict[str, str],
    has_detector: bool,
    title_index=None,
) -> Optional[str]:
    """Resolve a detector target string to a corpus normed_identifier.

    Pure function of its index arguments (no GraphIndex / on-disk shards), so the
    exact single source of truth for resolution ordering is unit-testable. Tiers,
    in order (first hit wins):

      1. Exact raw_identifier match (backward compatible).
      2. Detector-key match (only if a detector is present).
      3. Fuzzy HashNormTitleIndex cascade (only if title_index given). lookup
         returns a corpus raw_identifier, mapped back to its normed id.

    Returns None if the target resolves by no tier.
    """
    normed = raw_to_normed.get(target)
    if normed is not None:
        return normed
    if has_detector:
        normed = key_to_normed.get(target)
        if normed is not None:
            return normed
    if title_index is not None:
        raw = title_index.lookup(target)
        if raw is not None:
            return raw_to_normed.get(raw)
    return None


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
        fuzzy_match: When True, build a HashNormTitleIndex over the corpus
            raw_identifiers and consult it as a final tier after both exact
            indexes miss. Gives generation the same near-miss recovery
            (casing/punctuation/word-order/typo variants) the eval annotators
            get. Default False — preserves exact-only behavior and avoids the
            index-build cost for callers that don't need it.
        fuzzy_strategies: Ordered HashNormTitleIndex strategy cascade. Only used
            when fuzzy_match=True. Default matches the eval annotator default.
        edit_distance_threshold: Max normalized edit distance for the
            "edit_distance" strategy (only used if present in fuzzy_strategies).
        edit_distance_min_chars: Minimum query length before "edit_distance"
            fires (suppresses false positives on short strings).
    """

    _DEFAULT_FUZZY_STRATEGIES: Tuple[str, ...] = (
        "exact", "norm", "word_overlap_ordered", "edit_distance",
    )

    def __init__(
        self,
        dataset_dir: "str | Path",
        link_detector=None,
        fuzzy_match: bool = False,
        fuzzy_strategies: Sequence[str] = _DEFAULT_FUZZY_STRATEGIES,
        edit_distance_threshold: float = 0.2,
        edit_distance_min_chars: int = 5,
    ):
        dataset_dir = Path(dataset_dir)
        self._graph = GraphIndex(dataset_dir)
        self._backend = PretokShardedBackend(self._graph)
        self._link_detector = link_detector
        # A miss in both indexes means the target is not in the corpus (e.g. a
        # hallucinated title) -> has_document returns False (unless the fuzzy
        # tier below recovers it).
        self._raw_to_normed, self._key_to_normed = _build_indexes(
            self._graph.nodes, link_detector
        )

        # Optional fuzzy tier: reuse the eval annotator's HashNormTitleIndex so
        # generation resolves near-miss titles identically to eval.
        self._title_index = None
        if fuzzy_match:
            self._title_index = HashNormTitleIndex(
                self._raw_to_normed.keys(),
                strategies=tuple(fuzzy_strategies),
                edit_distance_threshold=edit_distance_threshold,
                edit_distance_min_chars=edit_distance_min_chars,
            )

    def _resolve(self, target: str) -> Optional[str]:
        """Resolve a detector target string to a corpus normed_identifier.

        Delegates to the pure ``_resolve_target`` helper (exact → detector-key →
        fuzzy). Returns None if the target is not in the corpus by any tier.
        """
        return _resolve_target(
            target,
            self._raw_to_normed,
            self._key_to_normed,
            has_detector=self._link_detector is not None,
            title_index=self._title_index,
        )

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
