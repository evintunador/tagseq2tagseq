"""
eval/title_index.py — corpus title lookup for link annotation.

TitleIndex protocol: lookup(generated_str) -> Optional[raw_identifier].
HashNormTitleIndex: configurable cascading match — strategies tried in caller-specified
  order, first hit wins:

  "exact"                  — exact (case-insensitive): raw.lower() == generated.lower()
  "norm"                   — normalized: strip_hash(create_normed_identifier(s)) — reuses
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

Future strategies (not yet implemented):

  "trie_constrained" — build a BPE-level prefix trie over all corpus titles at
                   construction time (tokenizer required). During autoregressive title
                   generation, at each step restrict the next-token distribution to
                   tokens that continue at least one valid corpus path in the trie.
                   Track the running joint log-prob of the current path; if it exceeds
                   a caller-supplied threshold, commit to the highest-probability corpus
                   leaf and return it. Falls back gracefully when the model is not
                   confident enough (prob < threshold) — no worse than the existing
                   strategies. This is the most powerful strategy: it guarantees a
                   corpus match when the model is on the right track from the first
                   token, and the threshold prevents forcing spurious matches when it
                   isn't. Requires replacing the free autoregressive loop in
                   MarkdownPromptAnnotator._generate_title with a trie-constrained
                   variant, and is best implemented as a new TitleIndex subclass
                   (TrieTitleIndex) that also owns the constrained generation loop.

  "beam_trie"    — top-k token branching during generation: at each step keep top-k
                   next-token candidates, probe "exact"/"norm" on each partial
                   decoded string, return first hit. k=2 doubles forward passes but
                   recovers single wrong-character errors cheaply.
"""

import re as _re
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

from model.identifier_utils import create_normed_identifier, strip_hash


@runtime_checkable
class TitleIndex(Protocol):
    """Minimal interface for corpus title lookup."""

    def lookup(self, generated_str: str) -> Optional[str]:
        """Return the corpus raw_identifier for generated_str, or None on miss."""
        ...


_VALID_STRATEGIES = frozenset({
    "exact", "norm", "word_overlap_ordered", "word_overlap_unordered",
})
_DEFAULT_STRATEGIES: tuple = ("exact", "norm", "word_overlap_ordered")
_WORD_OVERLAP_STRATEGIES = frozenset({"word_overlap_ordered", "word_overlap_unordered"})


class HashNormTitleIndex:
    """
    Lookup index with configurable cascading strategies (first hit wins).

    Strategies are tried in the order given by the ``strategies`` parameter:

    * ``"exact"``                  — case-insensitive verbatim match.
    * ``"norm"``                   — normalization-based match; handles punctuation,
      casing, and hyphen variants that survive ``create_normed_identifier``.
    * ``"word_overlap_ordered"``   — query words must appear as a contiguous subsequence
      in the candidate's word list; tiebreak by shortest candidate. Good for truncated
      titles ("Russian Civil" → "Russian Civil War").
    * ``"word_overlap_unordered"`` — query words must all appear in the candidate's word
      set (any order); tiebreak by shortest candidate. More permissive than ordered.

    For strategies that can collide (norm, word_overlap_*), first-inserted entry wins.

    Args:
        raw_identifiers: Iterable of raw identifier strings from the corpus.
        strategies: Ordered sequence of strategy names to try in order.
            Default: ``("exact", "norm", "word_overlap_ordered")``.
    """

    def __init__(
        self,
        raw_identifiers: Iterable[str],
        strategies: Sequence[str] = _DEFAULT_STRATEGIES,
    ) -> None:
        unknown = set(strategies) - _VALID_STRATEGIES
        if unknown:
            raise ValueError(
                f"Unknown strategies: {sorted(unknown)}. "
                f"Valid: {sorted(_VALID_STRATEGIES)}"
            )
        self._strategies: tuple = tuple(strategies)

        self._exact: Dict[str, str] = {}          # raw.lower() -> raw
        self._index: Dict[str, str] = {}          # strip_hash(norm(raw)) -> raw
        # word -> [raw, ...] in insertion order (for word_overlap_* strategies)
        self._word_index: Dict[str, List[str]] = {}
        # normed word list per raw title (for ordered subsequence check)
        self._word_lists: Dict[str, List[str]] = {}  # raw -> [word, ...]

        _need_word = bool(set(strategies) & _WORD_OVERLAP_STRATEGIES)

        for raw in raw_identifiers:
            lower = raw.lower()
            if lower not in self._exact:
                self._exact[lower] = raw

            key = strip_hash(create_normed_identifier(raw))
            if key and key not in self._index:
                self._index[key] = raw

            if _need_word:
                words = [w for w in _re.split(r'[^a-z0-9]+', lower) if w]
                self._word_lists[raw] = words
                for word in words:
                    if word not in self._word_index:
                        self._word_index[word] = []
                    self._word_index[word].append(raw)

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
            key = strip_hash(create_normed_identifier(generated_str))
            return self._index.get(key) if key else None
        if strategy == "word_overlap_ordered":
            return self._word_overlap(generated_str, ordered=True)
        if strategy == "word_overlap_unordered":
            return self._word_overlap(generated_str, ordered=False)
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
