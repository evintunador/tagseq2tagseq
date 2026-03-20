"""
Document context management for TS2TS generation.

Manages the growing packed token sequence during generation, tracking DocSpan
metadata for every document and supporting efficient extension.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from data.collate import DocSpan
from data.layout import DocLayoutInfo
from model.generation_result import GeneratedDocument
from model.identifier_utils import create_normed_identifier


@dataclass
class _DocEntry:
    """Internal per-document state. Not exported."""
    normed_identifier: str        # normalized form (for DocSpan.normed_identifier)
    raw_identifier: str           # human-readable form (for DocSpan.raw_identifier)
    prefix_tokens: List[int]      # layout prefix (fixed at construction time)
    tokens: List[int]             # body token IDs only (accumulates during generation)
    suffix_tokens: List[int]      # layout suffix (set at mark_done / add_corpus_doc)
    done: bool
    truncated: bool
    doc_id: int                   # sequential counter, not tied to corpus
    source: Literal["generated", "corpus"]
    is_root: bool
    parent_raw_identifier: Optional[str]
    depth: int                    # recursion depth at which this doc was created


class DocumentContext:
    """
    Manages the growing packed sequence during generation.

    Tracks DocSpan metadata for every document in the context, maintaining
    correct start/end indices as tokens are appended or new documents are
    inserted.

    Args:
        max_context_length: Maximum total tokens in the packed sequence.
        max_auxiliary_documents: Maximum number of non-root documents.
        eviction_policy: 'drop_oldest' or 'stop_new'.
        device: Torch device string for tensors returned by build_sequence().
    """

    def __init__(
        self,
        max_context_length: int,
        max_auxiliary_documents: int,
        eviction_policy: str,
        device: str,
    ):
        self.max_context_length = max_context_length
        self.max_auxiliary_documents = max_auxiliary_documents
        self.eviction_policy = eviction_policy
        self.device = device

        self._docs: List[_DocEntry] = []
        self._root: Optional[_DocEntry] = None
        self._evicted: List[_DocEntry] = []
        self._next_doc_id: int = 0

    @property
    def total_tokens(self) -> int:
        return sum(len(e.prefix_tokens) + len(e.tokens) + len(e.suffix_tokens) for e in self._docs)

    @property
    def num_aux_docs(self) -> int:
        return len(self._docs) - 1

    def add_root(
        self,
        raw_identifier: str,
        prompt_tokens: List[int],
        layout_policy=None,
    ) -> _DocEntry:
        """
        Add the root document, seeded with prompt_tokens.

        raw_identifier should be "" — the root has no natural document identifier.
        layout_policy may be None (equivalent to NullLayoutPolicy).
        """
        assert self._root is None, "Root document already added"
        doc_id = self._next_doc_id
        self._next_doc_id += 1

        normed = create_normed_identifier(raw_identifier) if raw_identifier else ""
        if layout_policy is not None:
            info = DocLayoutInfo(
                raw_identifier=raw_identifier,
                normed_identifier=normed,
                body_tokens=list(prompt_tokens),
            )
            prefix = list(layout_policy.prefix_tokens(info))
        else:
            prefix = []

        entry = _DocEntry(
            normed_identifier=normed,
            raw_identifier=raw_identifier,
            prefix_tokens=prefix,
            tokens=list(prompt_tokens),
            suffix_tokens=[],
            done=False,
            truncated=False,
            doc_id=doc_id,
            source="generated",
            is_root=True,
            parent_raw_identifier=None,
            depth=0,
        )
        self._docs.append(entry)
        self._root = entry
        return entry

    def append_token(self, entry: _DocEntry, token_id: int) -> None:
        """Append a single generated token to the given document."""
        entry.tokens.append(token_id)

    def mark_done(self, entry: _DocEntry, layout_policy=None) -> None:
        """Mark the document as complete, storing any layout suffix tokens separately."""
        if layout_policy is not None:
            info = DocLayoutInfo(
                raw_identifier=entry.raw_identifier,
                normed_identifier=entry.normed_identifier,
                body_tokens=list(entry.tokens),
            )
            entry.suffix_tokens = list(layout_policy.suffix_tokens(info))
        entry.done = True

    def build_sequence(self) -> Tuple[Tensor, List[DocSpan]]:
        """
        Build the current packed token tensor and DocSpan list.

        Recomputes all DocSpan offsets from scratch (O(total_tokens)).
        Each document contributes prefix_tokens + tokens (body) + suffix_tokens.

        Returns:
            tokens: LongTensor of shape [1, T] on self.device
            doc_spans: List[DocSpan] with consistent start/end indices
        """
        all_tokens: List[int] = []
        doc_spans: List[DocSpan] = []
        offset = 0
        for entry in self._docs:
            start = offset
            end = offset + len(entry.prefix_tokens) + len(entry.tokens) + len(entry.suffix_tokens)
            doc_spans.append(DocSpan(
                doc_id=entry.doc_id,
                normed_identifier=entry.normed_identifier,
                start=start,
                end=end,
                truncated=entry.truncated,
                outgoing_identifiers=[],
                raw_identifier=entry.raw_identifier,
            ))
            all_tokens.extend(entry.prefix_tokens)
            all_tokens.extend(entry.tokens)
            all_tokens.extend(entry.suffix_tokens)
            offset = end

        tokens_tensor = torch.tensor(
            all_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)  # [1, T]
        return tokens_tensor, doc_spans

    def can_add_document(self, num_new_tokens: int) -> bool:
        """True if adding a document of num_new_tokens fits within both limits."""
        return (
            self.total_tokens + num_new_tokens <= self.max_context_length
            and self.num_aux_docs < self.max_auxiliary_documents
        )

    def evict_oldest_aux(self) -> _DocEntry:
        """Remove and return the leftmost non-root entry; append to evicted list."""
        for i, entry in enumerate(self._docs):
            if not entry.is_root:
                self._docs.pop(i)
                self._evicted.append(entry)
                return entry
        raise RuntimeError("No auxiliary documents to evict")

    def make_room(self, num_tokens_needed: int) -> bool:
        """
        Evict oldest aux docs until there is room for num_tokens_needed tokens.

        Returns True if room was successfully made, False if impossible (only
        root remains or root alone already exceeds the budget).

        Only call when eviction_policy == 'drop_oldest'; caller is responsible
        for checking the policy.

        TODO: consider more efficient eviction strategies (e.g. evict largest
        doc first) rather than always evicting the oldest.
        """
        while not self.can_add_document(num_tokens_needed):
            if self.num_aux_docs == 0:
                return False
            self.evict_oldest_aux()
        return True

    def has_identifier(self, raw_identifier: str) -> bool:
        """True if raw_identifier is found in the active window (not evicted docs)."""
        return any(e.raw_identifier == raw_identifier for e in self._docs)

    def find_evicted(self, raw_identifier: str) -> Optional[_DocEntry]:
        """Return the evicted entry matching raw_identifier, or None."""
        for entry in self._evicted:
            if entry.raw_identifier == raw_identifier:
                return entry
        return None

    def restore_evicted(self, entry: _DocEntry, before_entry: _DocEntry) -> None:
        """Re-insert a previously evicted entry before before_entry in _docs."""
        self._evicted.remove(entry)
        idx = self._docs.index(before_entry)
        self._docs.insert(idx, entry)

    def add_corpus_doc(
        self,
        raw_identifier: str,
        corpus_tokens: List[int],
        layout_policy,
        parent_raw_identifier: Optional[str],
        depth: int,
        before_entry: _DocEntry,
    ) -> _DocEntry:
        """
        Insert a completed corpus document before before_entry in the context.

        Applies both layout prefix and suffix so the token sequence matches the
        training distribution (prefix + body + suffix).

        The caller is responsible for ensuring space exists (via can_add_document
        or make_room) before calling this method.
        """
        doc_id = self._next_doc_id
        self._next_doc_id += 1
        normed = create_normed_identifier(raw_identifier)

        if layout_policy is not None:
            info = DocLayoutInfo(
                raw_identifier=raw_identifier,
                normed_identifier=normed,
                body_tokens=list(corpus_tokens),
            )
            prefix = list(layout_policy.prefix_tokens(info))
            suffix = list(layout_policy.suffix_tokens(info))
        else:
            prefix = []
            suffix = []

        entry = _DocEntry(
            normed_identifier=normed,
            raw_identifier=raw_identifier,
            prefix_tokens=prefix,
            tokens=list(corpus_tokens),
            suffix_tokens=suffix,
            done=True,
            truncated=False,
            doc_id=doc_id,
            source="corpus",
            is_root=False,
            parent_raw_identifier=parent_raw_identifier,
            depth=depth,
        )
        idx = self._docs.index(before_entry)
        self._docs.insert(idx, entry)
        return entry

    def add_generated_doc(
        self,
        raw_identifier: str,
        layout_policy,
        parent_raw_identifier: Optional[str],
        depth: int,
        before_entry: _DocEntry,
    ) -> _DocEntry:
        """
        Insert an empty generated document before before_entry in the context.

        Seeded with the layout prefix; body tokens are accumulated later via
        append_token.

        The caller is responsible for ensuring space exists (via can_add_document
        or make_room) before calling this method.
        """
        doc_id = self._next_doc_id
        self._next_doc_id += 1
        normed = create_normed_identifier(raw_identifier)

        if layout_policy is not None:
            info = DocLayoutInfo(
                raw_identifier=raw_identifier,
                normed_identifier=normed,
                body_tokens=[],
            )
            prefix = list(layout_policy.prefix_tokens(info))
        else:
            prefix = []

        entry = _DocEntry(
            normed_identifier=normed,
            raw_identifier=raw_identifier,
            prefix_tokens=prefix,
            tokens=[],
            suffix_tokens=[],
            done=False,
            truncated=False,
            doc_id=doc_id,
            source="generated",
            is_root=False,
            parent_raw_identifier=parent_raw_identifier,
            depth=depth,
        )
        idx = self._docs.index(before_entry)
        self._docs.insert(idx, entry)
        return entry

    def get_all_documents(self) -> List[GeneratedDocument]:
        """
        Convert all tracked entries to GeneratedDocument objects.

        Returns root first, then active aux docs in topological order
        (i.e. their order in _docs, which places dependencies before dependents),
        then evicted docs in eviction order.
        """
        result = [_entry_to_doc(self._root)]
        for entry in self._docs:
            if not entry.is_root:
                result.append(_entry_to_doc(entry))
        for entry in self._evicted:
            result.append(_entry_to_doc(entry))
        return result


def _entry_to_doc(entry: _DocEntry) -> GeneratedDocument:
    return GeneratedDocument(
        raw_identifier=entry.raw_identifier,
        normed_identifier=entry.normed_identifier,
        tokens=np.array(
            entry.prefix_tokens + entry.tokens + entry.suffix_tokens, dtype=np.int32
        ),
        text=None,
        source=entry.source,
        is_root=entry.is_root,
        parent_raw_identifier=entry.parent_raw_identifier,
        depth=entry.depth,
        truncated=entry.truncated,
    )
