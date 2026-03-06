"""
Document context management for TS2TS generation.

Manages the growing packed token sequence during generation, tracking DocSpan
metadata for every document and supporting efficient extension.

Stage 1: single-document subset only (add_root, append_token, mark_done,
build_sequence). Multi-document support (add_corpus_doc, add_generated_doc,
eviction, re-eviction) is deferred to Stage 2.
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
    tokens: List[int]             # accumulated token IDs: prefix + body only
    suffix_tokens: List[int]      # layout suffix, stored separately from body
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
        return sum(len(e.tokens) + len(e.suffix_tokens) for e in self._docs)

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
            tokens=prefix + list(prompt_tokens),
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

        Returns:
            tokens: LongTensor of shape [1, T] on self.device
            doc_spans: List[DocSpan] with consistent start/end indices
        """
        all_tokens: List[int] = []
        doc_spans: List[DocSpan] = []
        offset = 0
        for entry in self._docs:
            start = offset
            end = offset + len(entry.tokens) + len(entry.suffix_tokens)
            doc_spans.append(DocSpan(
                doc_id=entry.doc_id,
                normed_identifier=entry.normed_identifier,
                start=start,
                end=end,
                truncated=entry.truncated,
                outgoing_identifiers=[],
                raw_identifier=entry.raw_identifier,
            ))
            all_tokens.extend(entry.tokens)
            all_tokens.extend(entry.suffix_tokens)
            offset = end

        tokens_tensor = torch.tensor(
            all_tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)  # [1, T]
        return tokens_tensor, doc_spans

    def get_all_documents(self) -> List[GeneratedDocument]:
        """
        Convert all tracked entries to GeneratedDocument objects.

        Returns root first, then active aux docs in topological order,
        then evicted docs in eviction order.
        """
        result = []
        for entry in self._docs:
            result.append(_entry_to_doc(entry))
        for entry in self._evicted:
            result.append(_entry_to_doc(entry))
        return result


def _entry_to_doc(entry: _DocEntry) -> GeneratedDocument:
    return GeneratedDocument(
        raw_identifier=entry.raw_identifier,
        normed_identifier=entry.normed_identifier,
        tokens=np.array(entry.tokens + entry.suffix_tokens, dtype=np.int32),
        text=None,
        source=entry.source,
        is_root=entry.is_root,
        parent_raw_identifier=entry.parent_raw_identifier,
        depth=entry.depth,
        truncated=entry.truncated,
    )
