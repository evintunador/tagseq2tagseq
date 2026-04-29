"""
eval/title_index.py — corpus title lookup for link annotation.

TitleIndex protocol: lookup(generated_str) -> Optional[raw_identifier].
HashNormTitleIndex: matches by strip_hash(create_normed_identifier(s)) — reuses
the training normalization pipeline but relaxes the hash uniqueness constraint,
so casing/hyphenation/punctuation variants of the same title resolve correctly.
"""

from typing import Dict, Iterable, List, Optional, Protocol, runtime_checkable

from model.identifier_utils import create_normed_identifier, strip_hash


@runtime_checkable
class TitleIndex(Protocol):
    """Minimal interface for corpus title lookup."""

    def lookup(self, generated_str: str) -> Optional[str]:
        """Return the corpus raw_identifier for generated_str, or None on miss."""
        ...


class HashNormTitleIndex:
    """
    Lookup index keyed by strip_hash(create_normed_identifier(raw_identifier)).

    Built once from an iterable of raw_identifiers. On collision (two raw
    identifiers with identical stripped-norm keys), the first entry wins.

    Args:
        raw_identifiers: Iterable of raw identifier strings from the corpus.
    """

    def __init__(self, raw_identifiers: Iterable[str]) -> None:
        self._index: Dict[str, str] = {}
        for raw in raw_identifiers:
            key = strip_hash(create_normed_identifier(raw))
            if key not in self._index:
                self._index[key] = raw

    def lookup(self, generated_str: str) -> Optional[str]:
        """Return matching raw_identifier, or None if not found."""
        key = strip_hash(create_normed_identifier(generated_str))
        return self._index.get(key)

    def __len__(self) -> int:
        return len(self._index)
