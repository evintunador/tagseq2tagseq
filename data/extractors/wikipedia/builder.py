"""
Wikipedia graph builder using the shared framework.

This module composes the generic GraphBuilder with Wikipedia-specific components:
- MarkdownFileSource: Reads .md files from directory
- MarkdownLinkExtractor: Extracts [text](target) links
- PassthroughNormalizer: No normalization (files already normalized by dump_extractor.py)
"""
from pathlib import Path
from typing import Dict

from data.extractors.graph_builder import GraphBuilder, GraphNode
from data.extractors.sources import MarkdownFileSource
from data.extractors.link_extractors import MarkdownLinkExtractor
from data.extractors.normalization import PassthroughNormalizer


def build_wiki_graph(
    input_dir: Path,
    output_path: Path,
    show_progress: bool = True,
) -> Dict[str, GraphNode]:
    """
    Build Wikipedia link graph from extracted markdown files.
    
    This is a convenience function that sets up the GraphBuilder
    with Wikipedia-specific components.
    
    Args:
        input_dir: Directory containing .md files (from dump_extractor.py)
        output_path: Path for output graph.jsonl
        show_progress: Show progress bars
    
    Returns:
        Dictionary mapping normalized titles to GraphNode objects
    
    Example:
        >>> from pathlib import Path
        >>> graph = build_wiki_graph(
        ...     input_dir=Path("extracted_wiki/"),
        ...     output_path=Path("wiki_graph.jsonl")
        ... )
        >>> print(f"Built graph with {len(graph)} nodes")
    """
    builder = GraphBuilder(
        source=MarkdownFileSource(input_dir),
        link_extractor=MarkdownLinkExtractor(),
        normalizer=PassthroughNormalizer(),
        source_type="wiki",
        show_progress=show_progress,
    )
    
    return builder.build_graph(output_path)
