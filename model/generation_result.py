"""
Data structures for TS2TS generation results.

Defines GeneratedDocument and GenerationResult classes for storing and
accessing the outputs of a generation run.
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import numpy as np

from .identifier_utils import create_normed_identifier


@dataclass
class GeneratedDocument:
    """Represents a single document in the generation result."""

    raw_identifier: str       # Human-readable identifier as decoded from link (or "root")
    normed_identifier: str    # Normalized+hashed identifier for corpus lookup (e.g., "python_a7f8c3")
    tokens: Optional[np.ndarray]  # Token IDs (may be None for corpus docs if only text loaded)
    text: Optional[str]       # Decoded text (may be None if only tokens stored)
    source: Literal["generated", "corpus"]  # Where doc came from
    is_root: bool             # Whether this is the root document
    parent_raw_identifier: Optional[str]  # raw_identifier of the doc that linked here (None for root)
    depth: int = 0            # Recursion depth at which this doc was created (0 for root)
    truncated: bool = False   # True when max_tokens_per_document was hit mid-generation

    def __post_init__(self):
        """Validate that at least one of tokens or text is provided."""
        if self.tokens is None and self.text is None:
            raise ValueError(
                f"GeneratedDocument '{self.raw_identifier}' must have at least one of "
                f"tokens or text non-None"
            )


@dataclass
class GenerationResult:
    """Complete result of a generation run."""

    root_document: GeneratedDocument
    auxiliary_documents: List[GeneratedDocument]
    # Ordering: Topologically sorted where possible (dependencies before dependents),
    # with ties broken by creation/access order.

    # Metadata
    generation_config: dict  # Parameters used for generation

    def get_all_documents(self) -> List[GeneratedDocument]:
        """
        Return all documents (root + auxiliary) in order.

        Returns:
            List with root document first, followed by auxiliary documents
        """
        return [self.root_document] + self.auxiliary_documents

    def get_document_by_identifier(self, identifier: str) -> Optional[GeneratedDocument]:
        """
        Retrieve a document by its identifier.

        Matches against both raw_identifier and normed_identifier.

        Args:
            identifier: Document identifier (raw or normalized+hashed)

        Returns:
            GeneratedDocument if found, None otherwise

        Examples:
            >>> result.get_document_by_identifier("Python")
            GeneratedDocument(raw_identifier="Python", ...)
            >>> result.get_document_by_identifier("python_a7f8c3")
            GeneratedDocument(raw_identifier="Python", ...)  # Same document
        """
        for doc in self.get_all_documents():
            if doc.raw_identifier == identifier or doc.normed_identifier == identifier:
                return doc
        return None

    def get_generated_documents(self) -> List[GeneratedDocument]:
        """
        Return only documents that were generated (not from corpus).

        Returns:
            List of GeneratedDocuments where source == "generated"
        """
        return [doc for doc in self.get_all_documents() if doc.source == "generated"]

    def get_corpus_documents(self) -> List[GeneratedDocument]:
        """
        Return only documents that came from the corpus.

        Returns:
            List of GeneratedDocuments where source == "corpus"
        """
        return [doc for doc in self.get_all_documents() if doc.source == "corpus"]

    def get_document_count(self) -> int:
        """
        Get total number of documents in result.

        Returns:
            Total count (root + auxiliary)
        """
        return 1 + len(self.auxiliary_documents)

    def __repr__(self) -> str:
        """Readable representation of the generation result."""
        total_docs = self.get_document_count()
        generated_count = len(self.get_generated_documents())
        corpus_count = len(self.get_corpus_documents())

        return (
            f"GenerationResult("
            f"total_docs={total_docs}, "
            f"generated={generated_count}, "
            f"corpus={corpus_count}, "
            f"root='{self.root_document.raw_identifier}'"
            f")"
        )
