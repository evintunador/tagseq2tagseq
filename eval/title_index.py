"""
eval/title_index.py — corpus title lookup for link annotation.

Two index classes, different use cases:

  HashNormTitleIndex — post-hoc lookup. After free title generation produces a
      string, tries cascading strategies (exact, norm, word_overlap, edit_distance)
      to find the closest corpus entry. Good when generation mode is
      corpus_then_generate or generate, or when the corpus is too large to build
      a trie (TrieTitleIndex tokenizes every title at construction time).
      Use-case framing: "accept whatever the model generates and find the nearest
      corpus match, if any."

  TrieTitleIndex — constrained generation (in eval/link_annotator.py). Builds a
      BPE-level prefix trie over all corpus titles at construction time; during
      title generation restricts each next-token step to tokens that continue at
      least one valid corpus path. Uses beam search to explore the top-beam_width
      paths simultaneously and returns the completed title with the highest
      length-normalized joint log-prob. Guaranteed to return a valid corpus title
      on success (no post-hoc lookup needed). Falls back to None (→ free generation
      + HashNormTitleIndex cascade) only when min_joint_logprob threshold is hit.
      Use-case framing: "force the model to generate a title that exists in the
      corpus; prefer corpus coverage over generation freedom."

HashNormTitleIndex strategies (tried in caller-specified order, first hit wins):

  "exact"                  — exact (case-insensitive): raw.lower() == generated.lower()
  "norm"                   — normalized: strip_hash(normalize_wiki_title(s)) — reuses
                             the training normalization pipeline so casing/hyphenation/
                             punctuation variants resolve.
  "word_overlap_ordered"   — all words of the generated title must appear as a contiguous
                             subsequence in the candidate's word list (split on
                             non-alphanumeric chars); tiebreaks by shortest candidate.
                             Designed for the truncation-recovery case: the model generates
                             the first N words of a title and stops (e.g. "Russian Civil"
                             → "Russian Civil War"). Word order is enforced so "Civil
                             Russian" would not match. Not in the default strategy list.
  "word_overlap_unordered" — same as ordered but without the contiguous-subsequence
                             constraint: all query words must appear somewhere in the
                             candidate's word set, in any order. Strictly more permissive
                             than ordered; use when the model may paraphrase or reorder
                             words. Misfires more readily on short ambiguous queries
                             (e.g. "Apple" can hit "Apple (fruit)" or "Apple Inc.").

Default strategies: ("exact", "norm", "word_overlap_ordered").
Opt-in to unordered overlap by passing e.g.:
    strategies=("exact", "norm", "word_overlap_ordered", "word_overlap_unordered")

Implemented strategies (opt-in):

  "edit_distance" — fuzzy match via Levenshtein normalized similarity (rapidfuzz).
                   Lossy — may return a wrong doc. Recommend placing last so lossless
                   strategies take priority. Tunable via edit_distance_threshold and
                   edit_distance_min_chars.

Future HashNormTitleIndex strategies (not yet implemented):

  "prefix_commit" — after _generate_title produces a target_str, find all corpus
                   titles that share the longest common word-level prefix. Covers
                   early-halt ("Russian Civil" → "Russian Civil War") and overshoot
                   cases. Orthogonal to edit_distance.

TODO (MarkdownPromptAnnotator._fetch_aux):
  Display-text fallback — when all strategies miss, retry lookup using the anchor
  text between '[' and ']('. No new strategy needed; just a second lookup() call.
"""

import re as _re
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from data.normalization import normalize_wiki_title, strip_hash


@runtime_checkable
class TitleIndex(Protocol):
    """Minimal interface for corpus title lookup."""

    def lookup(self, generated_str: str) -> Optional[str]:
        """Return the corpus raw_identifier for generated_str, or None on miss."""
        ...


_VALID_STRATEGIES = frozenset({
    "exact", "norm", "word_overlap_ordered", "word_overlap_unordered", "edit_distance",
})
_DEFAULT_STRATEGIES: tuple = ("exact", "norm", "word_overlap_ordered")
_WORD_OVERLAP_STRATEGIES = frozenset({"word_overlap_ordered", "word_overlap_unordered"})


class HashNormTitleIndex:
    """
    Lookup index with configurable cascading strategies (first hit wins).

    Strategies are tried in the order given by the ``strategies`` parameter:

    * ``"exact"``                  — case-insensitive verbatim match.
    * ``"norm"``                   — normalization-based match; handles punctuation,
      casing, and hyphen variants that survive ``normalize_wiki_title``.
    * ``"word_overlap_ordered"``   — query words must appear as a contiguous subsequence
      in the candidate's word list; tiebreak by shortest candidate. Good for truncated
      titles ("Russian Civil" → "Russian Civil War").
    * ``"word_overlap_unordered"`` — query words must all appear in the candidate's word
      set (any order); tiebreak by shortest candidate. More permissive than ordered.
    * ``"edit_distance"``          — fuzzy match using Levenshtein normalized
      similarity. Lossy — may return a wrong doc. Recommended last in the
      sequence so lossless strategies take priority, but not structurally
      enforced. Requires ``rapidfuzz`` (``pip install rapidfuzz``). Tunable
      via ``edit_distance_threshold`` and ``edit_distance_min_chars``.

    For strategies that can collide (norm, word_overlap_*), first-inserted entry wins.

    Args:
        raw_identifiers: Iterable of raw identifier strings from the corpus.
        strategies: Ordered sequence of strategy names to try in order.
            Default: ``("exact", "norm", "word_overlap_ordered")``.
        edit_distance_threshold: Maximum normalized edit distance allowed for a
            match (1 - normalized_similarity). Range [0, 1]; default 0.2 means
            ≥80% of characters must match. Only used with ``"edit_distance"``.
        edit_distance_min_chars: Minimum length (chars) of the normalized query
            before ``edit_distance`` will fire. Queries shorter than this are
            skipped to suppress false positives on short ambiguous strings.
            Default 5. Only used with ``"edit_distance"``.
    """

    def __init__(
        self,
        raw_identifiers: Iterable[str],
        strategies: Sequence[str] = _DEFAULT_STRATEGIES,
        edit_distance_threshold: float = 0.2,
        edit_distance_min_chars: int = 5,
    ) -> None:
        unknown = set(strategies) - _VALID_STRATEGIES
        if unknown:
            raise ValueError(
                f"Unknown strategies: {sorted(unknown)}. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self._strategies: tuple = tuple(strategies)
        self._ed_threshold: float = edit_distance_threshold
        self._ed_min_chars: int = edit_distance_min_chars

        self._exact: Dict[str, str] = {}          # raw.lower() -> raw
        self._index: Dict[str, str] = {}          # strip_hash(norm(raw)) -> raw
        # word -> [raw, ...] in insertion order (for word_overlap_* strategies)
        self._word_index: Dict[str, List[str]] = {}
        # normed word list per raw title (for ordered subsequence check)
        self._word_lists: Dict[str, List[str]] = {}  # raw -> [word, ...]
        # parallel lists for edit_distance: normed key + raw (positional index)
        self._ed_keys: List[str] = []
        self._ed_raws: List[str] = []

        _need_word = bool(set(strategies) & _WORD_OVERLAP_STRATEGIES)
        _need_ed = "edit_distance" in strategies

        for raw in raw_identifiers:
            lower = raw.lower()
            if lower not in self._exact:
                self._exact[lower] = raw

            key = strip_hash(normalize_wiki_title(raw))
            if key and key not in self._index:
                self._index[key] = raw

            if _need_word:
                words = [w for w in _re.split(r'[^a-z0-9]+', lower) if w]
                self._word_lists[raw] = words
                for word in words:
                    if word not in self._word_index:
                        self._word_index[word] = []
                    self._word_index[word].append(raw)

            if _need_ed:
                ed_key = strip_hash(normalize_wiki_title(raw))
                if ed_key:
                    self._ed_keys.append(ed_key)
                    self._ed_raws.append(raw)

    @property
    def strategies(self) -> tuple:
        return self._strategies

    def lookup(self, generated_str: str) -> Optional[str]:
        """Return matching raw_identifier, or None if not found."""
        if not generated_str:
            return None
        for strategy in self._strategies:
            hit = self._lookup_one(strategy, generated_str)
            if hit is not None:
                return hit
        return None

    def _lookup_one(self, strategy: str, generated_str: str) -> Optional[str]:
        if strategy == "exact":
            return self._exact.get(generated_str.lower())
        if strategy == "norm":
            key = strip_hash(normalize_wiki_title(generated_str))
            return self._index.get(key) if key else None
        if strategy == "word_overlap_ordered":
            return self._word_overlap(generated_str, ordered=True)
        if strategy == "word_overlap_unordered":
            return self._word_overlap(generated_str, ordered=False)
        if strategy == "edit_distance":
            return self._edit_distance_lookup(generated_str)
        return None

    def _word_overlap(self, generated_str: str, ordered: bool) -> Optional[str]:
        """Return shortest corpus entry containing all query words.

        ordered=True:  query words must appear as a contiguous subsequence in
                       the candidate's word list.
        ordered=False: query words must all appear somewhere in the candidate's
                       word set (any order).
        """
        query_words = [w for w in _re.split(r'[^a-z0-9]+', generated_str.lower()) if w]
        if not query_words:
            return None

        # Candidate set: corpus entries that contain all query words (unordered gate).
        candidates: Optional[set] = None
        for word in query_words:
            word_hits = set(self._word_index.get(word, []))
            candidates = word_hits if candidates is None else candidates & word_hits
            if not candidates:
                return None
        if not candidates:
            return None

        if ordered:
            # Further filter to those where query_words appear as a contiguous
            # subsequence in the candidate's stored word list.
            filtered = []
            for raw in candidates:
                cand_words = self._word_lists.get(raw, [])
                if _is_contiguous_subsequence(query_words, cand_words):
                    filtered.append(raw)
            candidates = filtered
            if not candidates:
                return None

        return min(candidates, key=len)

    def _edit_distance_lookup(self, generated_str: str) -> Optional[str]:
        """Return closest corpus entry by Levenshtein normalized similarity, or None.

        Compares normalized forms (same pipeline as the ``norm`` strategy) to avoid
        spurious distance inflation from punctuation differences. Returns None when
        the query is too short (< edit_distance_min_chars) or no entry exceeds the
        similarity cutoff (1 - edit_distance_threshold).
        """
        try:
            from rapidfuzz.distance import Levenshtein as _Lev
            from rapidfuzz import process as _rfp
        except ImportError:
            raise ImportError(
                "edit_distance strategy requires rapidfuzz: pip install rapidfuzz"
            )
        query_norm = strip_hash(normalize_wiki_title(generated_str))
        if not query_norm or len(query_norm) < self._ed_min_chars:
            return None
        if not self._ed_keys:
            return None
        cutoff = 1.0 - self._ed_threshold
        result = _rfp.extractOne(
            query_norm,
            self._ed_keys,
            scorer=_Lev.normalized_similarity,
            score_cutoff=cutoff,
        )
        if result is None:
            return None
        _match_key, _score, idx = result
        return self._ed_raws[idx]

    def __len__(self) -> int:
        return len(self._exact)


def _is_contiguous_subsequence(needle: List[str], haystack: List[str]) -> bool:
    """Return True if needle appears as a contiguous subsequence in haystack."""
    n, h = len(needle), len(haystack)
    if n > h:
        return False
    for i in range(h - n + 1):
        if haystack[i:i + n] == needle:
            return True
    return False
