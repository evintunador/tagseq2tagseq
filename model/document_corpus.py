"""
DocumentCorpus wrapper for clean corpus access during generation.

Provides a simplified interface around GraphIndex and PretokShardedBackend
for retrieving documents during generation.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from data.dataset import GraphIndex, PretokShardedBackend
from .identifier_utils import create_normed_identifier


class DocumentCorpus:
    """
    Wrapper around GraphIndex and PretokShardedBackend for corpus access.

    Provides a clean interface for retrieving documents during generation.
    Handles identifier normalization and hash matching automatically.
    """

    def __init__(self, dataset_path: Path):
        """
        Initialize the corpus from a pretokenized dataset directory.

        Args:
            dataset_path: Path to directory containing metadata.json,
                         tokenized_graph.jsonl, and shard files

        Raises:
            FileNotFoundError: If dataset_path doesn't exist or is missing required files
        """
        self.dataset_path = Path(dataset_path)

        # Initialize GraphIndex and PretokShardedBackend
        self.index = GraphIndex(self.dataset_path)
        self.backend = PretokShardedBackend(self.index)

    def get_document(self, identifier: str) -> Optional[np.ndarray]:
        """
        Retrieve a document's tokens by its identifier.

        The identifier can be either:
        - The raw identifier (e.g., "Python")
        - The normed_identifier (e.g., "python_a7f8c3")

        Args:
            identifier: Document identifier (raw or normalized+hashed)

        Returns:
            Numpy array of token IDs, or None if not found

        Examples:
            >>> corpus = DocumentCorpus(Path("data/wiki_pretok"))
            >>> tokens = corpus.get_document("Python")
            >>> tokens = corpus.get_document("python_a7f8c3")  # Also works
        """
        # Try direct lookup first (in case it's already a normed_identifier)
        if identifier in self.index:
            return self.backend.get_tokens(identifier)

        # Try constructing the normed_identifier from the raw identifier
        normed = create_normed_identifier(identifier)
        if normed in self.index:
            return self.backend.get_tokens(normed)

        return None

    def has_document(self, identifier: str) -> bool:
        """
        Check if a document exists in the corpus.

        Args:
            identifier: Document identifier (raw or normalized+hashed)

        Returns:
            True if the document exists, False otherwise
        """
        if identifier in self.index:
            return True
        normed = create_normed_identifier(identifier)
        return normed in self.index

    def get_normed_identifier(self, identifier: str) -> Optional[str]:
        """
        Get the normed_identifier as stored in the corpus.

        Given either a raw identifier or a normed_identifier, returns the
        exact key used in the corpus index.

        Args:
            identifier: Document identifier (raw or normalized+hashed)

        Returns:
            normed_identifier as stored in corpus, or None if not found

        Examples:
            >>> corpus.get_normed_identifier("Python")
            'python_a7f8c3'
        """
        if identifier in self.index:
            return identifier
        normed = create_normed_identifier(identifier)
        if normed in self.index:
            return normed
        return None

    def close(self):
        """Close backend resources (memory-mapped files)."""
        self.backend.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes resources."""
        self.close()
        return False
