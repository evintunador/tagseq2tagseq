"""
Core graph building logic using pluggable components.

Combines logic from:
- wiki_graph_extractor/build_graph.py (lines 156-288)
- github_graph_extractor/build_graph_streaming.py (lines 248-336)
"""
import json
import logging
from pathlib import Path
from typing import Dict, Set, Optional, Any
from tqdm import tqdm

from .protocols import (
    Document, LinkContext, ContentSource, 
    LinkExtractor, LinkNormalizer
)

logger = logging.getLogger(__name__)


class GraphNode:
    """A node in the document graph."""
    
    def __init__(self, title: str, char_count: int = 0):
        """
        Args:
            title: Normalized title/identifier for this node
            char_count: Number of characters in document content
        """
        self.title = title
        self.char_count = char_count
        self.outgoing: Set[str] = set()
        self.incoming: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
    
    def __repr__(self):
        return (f"GraphNode(title={self.title!r}, char_count={self.char_count}, "
                f"outgoing={len(self.outgoing)}, incoming={len(self.incoming)})")


class GraphBuilder:
    """
    Generic graph builder using pluggable components.
    
    This class implements the core graph construction algorithm:
    1. Extract links from all documents
    2. Normalize identifiers and link targets
    3. Build bidirectional graph (outgoing + incoming links)
    4. Write to JSONL format
    
    The builder is configured with:
    - ContentSource: Where to read documents from
    - LinkExtractor: How to find links in content
    - LinkNormalizer: How to normalize link identifiers
    """
    
    def __init__(
        self,
        source: ContentSource,
        link_extractor: LinkExtractor,
        normalizer: LinkNormalizer,
        source_type: str = "generic",
        show_progress: bool = True,
    ):
        """
        Args:
            source: Content source providing documents
            link_extractor: Extracts raw links from content
            normalizer: Normalizes links to canonical identifiers
            source_type: Type of source for logging/context ("wiki", "github", etc.)
            show_progress: Show progress bars via tqdm
        """
        self.source = source
        self.link_extractor = link_extractor
        self.normalizer = normalizer
        self.source_type = source_type
        self.show_progress = show_progress
    
    def build_graph(self, output_path: Path) -> Dict[str, GraphNode]:
        """
        Build graph from source documents and write to JSONL.
        
        Args:
            output_path: Where to write the graph.jsonl file
        
        Returns:
            Dictionary mapping normalized titles to GraphNode objects
        """
        logger.info(f"Building {self.source_type} graph...")
        
        # Phase 1: Extract links from all documents
        graph: Dict[str, GraphNode] = {}
        
        # Load all documents into memory first so we can show progress
        # For very large datasets, this could be made streaming
        documents = list(self.source.iter_documents())
        logger.info(f"Processing {len(documents)} documents...")
        
        iterator = tqdm(
            documents, 
            disable=not self.show_progress, 
            desc="Extracting links", 
            unit="docs"
        )
        
        for doc in iterator:
            # Normalize document identifier
            context = LinkContext(doc, self.source_type)
            normalized_title = self.normalizer.normalize(doc.identifier, context)
            
            # Create node if it doesn't exist
            if normalized_title not in graph:
                graph[normalized_title] = GraphNode(
                    title=normalized_title,
                    char_count=len(doc.content)
                )
            
            # Extract links from content
            raw_links = self.link_extractor.extract_links(doc.content, context)
            
            # Normalize link targets and add to outgoing set
            for raw_link in raw_links:
                normalized_link = self.normalizer.normalize(raw_link, context)
                graph[normalized_title].outgoing.add(normalized_link)
        
        # Phase 2: Build incoming links
        logger.info("Building incoming link relationships...")
        
        for source_title, node in tqdm(
            graph.items(), 
            disable=not self.show_progress,
            desc="Computing incoming"
        ):
            for target_title in node.outgoing:
                # Only add incoming link if target exists in graph
                if target_title in graph:
                    graph[target_title].incoming.add(source_title)
        
        # Phase 3: Write graph to JSONL
        self._write_graph(graph, output_path)
        
        # Log summary statistics
        total_edges = sum(len(n.outgoing) for n in graph.values())
        logger.info(
            f"Graph complete: {len(graph)} nodes, {total_edges} edges"
        )
        
        return graph
    
    def _write_graph(self, graph: Dict[str, GraphNode], output_path: Path):
        """
        Write graph to JSONL format (common format for both extractors).
        
        Args:
            graph: Graph nodes to write
            output_path: Output file path
        """
        logger.info(f"Writing graph to {output_path}...")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort titles for deterministic output
        sorted_titles = sorted(graph.keys())
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for title in tqdm(
                sorted_titles, 
                disable=not self.show_progress,
                desc="Writing JSONL"
            ):
                node = graph[title]
                node_data = {
                    'title': title,
                    'char_count': node.char_count,
                    'outgoing': sorted(list(node.outgoing)),
                    'incoming': sorted(list(node.incoming)),
                }
                # Add any additional metadata
                node_data.update(node.metadata)
                
                f.write(json.dumps(node_data) + '\n')
        
        logger.info(f"Graph written to {output_path}")
