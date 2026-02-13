"""
Wikipedia graph builder using the shared framework.

This module composes the generic GraphBuilder with Wikipedia-specific components:
- MarkdownFileSource: Reads .md files from directory
- MarkdownLinkExtractor: Extracts [text](target) links
- Normalizer from DatasetConfig (typically PassthroughNormalizer for pre-normalized files)
"""
from pathlib import Path
from typing import Dict, Optional

from data.extractors.graph_builder import GraphBuilder, GraphNode
from data.extractors.sources import MarkdownFileSource
from data.extractors.link_extractors import MarkdownLinkExtractor
from data.dataset_config import DatasetConfig, WIKIPEDIA_CONFIG


def build_wiki_graph(
    input_dir: Path,
    output_path: Path,
    show_progress: bool = True,
    dataset_config: Optional[DatasetConfig] = None,
) -> Dict[str, GraphNode]:
    """
    Build Wikipedia link graph from extracted markdown files.
    
    This is a convenience function that sets up the GraphBuilder
    with Wikipedia-specific components.
    
    Args:
        input_dir: Directory containing .md files (from dump_extractor.py)
        output_path: Path for output graph.jsonl
        show_progress: Show progress bars
        dataset_config: Optional DatasetConfig to use. If None, uses WIKIPEDIA_CONFIG.
                       The normalizer will be obtained from this config.
    
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
    # Use default Wikipedia config if not provided
    if dataset_config is None:
        dataset_config = WIKIPEDIA_CONFIG
    
    # Get normalizer from config (ensures hash_length is consistent)
    normalizer = dataset_config.get_normalizer()
    
    builder = GraphBuilder(
        source=MarkdownFileSource(input_dir),
        link_extractor=MarkdownLinkExtractor(),
        normalizer=normalizer,
        source_type="wiki",
        show_progress=show_progress,
    )
    
    graph = builder.build_graph(output_path)
    
    # Save dataset config alongside graph
    from data.dataset_config import save_config_to_pretokenized_dir
    config_path = output_path.parent / "dataset_config.json"
    dataset_config.save(config_path)
    
    return graph
