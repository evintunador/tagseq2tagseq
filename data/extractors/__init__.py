"""
Graph extraction framework with pluggable components.

This package provides shared abstractions for building document graphs
from various sources (Wikipedia, GitHub, LaTeX, etc.).
"""

from .protocols import Document, LinkContext, LinkExtractor, LinkNormalizer, ContentSource
from .normalization import FilesafeNormalizer, PassthroughNormalizer, WikiTitleNormalizer, PythonModuleNormalizer
from .sources import FileSource, MarkdownFileSource, JSONLSource
from .link_extractors import MarkdownLinkExtractor, PythonImportExtractor
from .graph_builder import GraphBuilder, GraphNode

__all__ = [
    # Protocols
    "Document",
    "LinkContext",
    "LinkExtractor",
    "LinkNormalizer",
    "ContentSource",
    # Normalizers
    "FilesafeNormalizer",
    "PassthroughNormalizer",
    "WikiTitleNormalizer",
    "PythonModuleNormalizer",
    # Core Builder
    "GraphBuilder",
    "GraphNode",
    # Sources
    "FileSource",
    "MarkdownFileSource",  # Backward compatibility alias
    "JSONLSource",
    # Link Extractors
    "MarkdownLinkExtractor",
    "PythonImportExtractor",
]
