"""
DocumentCorpus wrapper for clean corpus access during generation.

Provides a simplified interface around GraphIndex and PretokShardedBackend
for retrieving documents during generation.
"""
from pathlib import Path
from typing import Optional

import numpy as np

from data.dataset import GraphIndex, PretokShardedBackend
from .title_utils import create_filename


class DocumentCorpus:
    """
    Wrapper around GraphIndex and PretokShardedBackend for corpus access.
    
    Provides a clean interface for retrieving documents during generation.
    Handles title normalization and hash matching automatically.
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
    
    def get_document(self, title: str) -> Optional[np.ndarray]:
        """
        Retrieve a document's tokens by its title.
        
        The title can be either:
        - The original title (e.g., "Python")
        - The normalized+hashed title (e.g., "python_a7f8c3")
        
        Args:
            title: Document title (original or normalized+hashed)
            
        Returns:
            Numpy array of token IDs, or None if not found
            
        Examples:
            >>> corpus = DocumentCorpus(Path("data/wiki_pretok"))
            >>> tokens = corpus.get_document("Python")
            >>> tokens = corpus.get_document("python_a7f8c3")  # Also works
        """
        # Try direct lookup first (in case it's already normalized+hashed)
        if title in self.index:
            return self.backend.get_tokens(title)
        
        # Try creating the filename from the original title
        filename = create_filename(title)
        if filename in self.index:
            return self.backend.get_tokens(filename)
        
        # Not found
        return None
    
    def has_document(self, title: str) -> bool:
        """
        Check if a document exists in the corpus.
        
        Args:
            title: Document title (original or normalized+hashed)
            
        Returns:
            True if the document exists, False otherwise
        """
        # Try direct lookup
        if title in self.index:
            return True
        
        # Try creating filename from original title
        filename = create_filename(title)
        return filename in self.index
    
    def get_title(self, title: str) -> Optional[str]:
        """
        Get the exact title as stored in the corpus.
        
        Given either an original title or normalized+hashed title,
        returns the exact title string used in the corpus index.
        
        Args:
            title: Document title (original or normalized+hashed)
            
        Returns:
            Exact title as stored in corpus, or None if not found
            
        Examples:
            >>> corpus.get_title("Python")
            'python_a7f8c3'  # Returns the normalized+hashed version
        """
        # Try direct lookup
        if title in self.index:
            return title
        
        # Try creating filename from original title
        filename = create_filename(title)
        if filename in self.index:
            return filename
        
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
