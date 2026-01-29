"""
Core protocols defining the interfaces for graph extraction components.
"""
from typing import Protocol, Iterator, Set, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    """A document from any source."""
    identifier: str          # Filename, repo:path, etc.
    content: str
    metadata: Dict[str, Any]


@dataclass  
class LinkContext:
    """Context for link extraction."""
    document: Document
    source_type: str         # "wiki", "github", etc.


class LinkExtractor(Protocol):
    """Extract raw link targets from document content."""
    
    def extract_links(self, content: str, context: LinkContext) -> Set[str]:
        """Returns set of raw link targets found in content."""
        ...


class LinkNormalizer(Protocol):
    """Normalize raw links into canonical identifiers."""
    
    def normalize(self, link: str, context: LinkContext) -> str:
        """Returns normalized, filesystem-safe identifier."""
        ...


class ContentSource(Protocol):
    """Iterator over documents from a data source."""
    
    def iter_documents(self) -> Iterator[Document]:
        """Yields Document objects."""
        ...
