"""
eval/title_index.py — corpus title lookup for link annotation.

TitleIndex protocol: lookup(generated_str) -> Optional[raw_identifier].
HashNormTitleIndex: cascading match — strategies tried cheapest-first:
  1. Exact (case-insensitive): raw.lower() == generated.lower()
  2. Normalized: strip_hash(create_normed_identifier(s)) — reuses the training
     normalization pipeline so casing/hyphenation/punctuation variants resolve.

Future strategies (not yet implemented, require model + device):
  3. Beam/trie title generation: top-k tokens per step (~k=2), custom flex mask
     so all candidate suffixes attend to shared context but not each other;
     returns multiple candidate strings to probe strategies 1 & 2 above.
  4. NLL ranking: pre-filter corpus titles (e.g. prefix or BM25) to a tractable
     candidate set, then score all candidates in one forward pass via a batched
     flex mask; return the highest-probability match.
"""

from typing import Dict, Iterable, Optional, Protocol, runtime_checkable

from model.identifier_utils import create_normed_identifier, strip_hash


@runtime_checkable
class TitleIndex(Protocol):
    """Minimal interface for corpus title lookup."""

    def lookup(self, generated_str: str) -> Optional[str]:
        """Return the corpus raw_identifier for generated_str, or None on miss."""
        ...


class HashNormTitleIndex:
    """
    Lookup index with two cascading strategies (cheapest first):

    1. Exact (case-insensitive): ``raw.lower() == generated.lower()``.
       Tried first so that normalization collisions (e.g. "C" and "C++" both
       normalize to "c") resolve to the correct entry rather than first-wins.
    2. Normalized: ``strip_hash(create_normed_identifier(raw))`` — covers
       casing/hyphenation/punctuation variants that survive normalization
       (e.g. "SpiderMan" → "Spider-Man", "PYTHON (LANGUAGE)" → "Python (language)").

    For both levels, first entry wins when two raws share the same key.

    Args:
        raw_identifiers: Iterable of raw identifier strings from the corpus.
    """

    def __init__(self, raw_identifiers: Iterable[str]) -> None:
        self._exact: Dict[str, str] = {}   # raw.lower() -> raw
        self._index: Dict[str, str] = {}   # strip_hash(norm(raw)) -> raw
        for raw in raw_identifiers:
            lower = raw.lower()
            if lower not in self._exact:
                self._exact[lower] = raw
            key = strip_hash(create_normed_identifier(raw))
            if key and key not in self._index:
                self._index[key] = raw

    def lookup(self, generated_str: str) -> Optional[str]:
        """Return matching raw_identifier, or None if not found."""
        if not generated_str:
            return None
        # Strategy 1: exact (case-insensitive)
        hit = self._exact.get(generated_str.lower())
        if hit is not None:
            return hit
        # Strategy 2: normalized
        key = strip_hash(create_normed_identifier(generated_str))
        return self._index.get(key) if key else None

    def __len__(self) -> int:
        return len(self._exact)
