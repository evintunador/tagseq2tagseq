"""
Integration tests for the graph builder.
"""
import unittest
import tempfile
import json
from pathlib import Path

from data.extractors.graph_builder import GraphBuilder, GraphNode
from data.extractors.sources import MarkdownFileSource
from data.extractors.link_extractors import MarkdownLinkExtractor
from data.extractors.normalization import WikiTitleNormalizer


class TestGraphBuilderIntegration(unittest.TestCase):
    def setUp(self):
        # Create temporary directory with test markdown files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test files with links between them
        # Note: We use the actual normalized format that would be generated
        (self.temp_path / "page1.md").write_text(
            "This is page 1. See [Page 2](page2) and [Page 3](page3)."
        )
        (self.temp_path / "page2.md").write_text(
            "This is page 2. See [Page 1](page1)."
        )
        (self.temp_path / "page3.md").write_text(
            "This is page 3. No links here."
        )
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_build_graph(self):
        """Test basic graph building."""
        output_file = self.temp_path / "graph.jsonl"
        
        builder = GraphBuilder(
            source=MarkdownFileSource(self.temp_path),
            link_extractor=MarkdownLinkExtractor(),
            normalizer=WikiTitleNormalizer(),
            source_type="wiki",
            show_progress=False,
        )
        
        graph = builder.build_graph(output_file)
        
        # Check graph was built
        self.assertGreater(len(graph), 0)
        self.assertEqual(len(graph), 3)  # 3 pages
        
        # Check all nodes are GraphNode instances
        for node in graph.values():
            self.assertIsInstance(node, GraphNode)
    
    def test_outgoing_links(self):
        """Test that outgoing links are captured."""
        output_file = self.temp_path / "graph.jsonl"
        
        builder = GraphBuilder(
            source=MarkdownFileSource(self.temp_path),
            link_extractor=MarkdownLinkExtractor(),
            normalizer=WikiTitleNormalizer(),
            source_type="wiki",
            show_progress=False,
        )
        
        graph = builder.build_graph(output_file)
        
        # Find page1 node (need to find by normalized title)
        page1_nodes = [n for n in graph.values() if 'page1' in n.title]
        self.assertEqual(len(page1_nodes), 1)
        page1 = page1_nodes[0]
        
        # page1 should have 2 outgoing links
        self.assertEqual(len(page1.outgoing), 2)
    
    def test_incoming_links(self):
        """Test that incoming links are computed."""
        output_file = self.temp_path / "graph.jsonl"
        
        builder = GraphBuilder(
            source=MarkdownFileSource(self.temp_path),
            link_extractor=MarkdownLinkExtractor(),
            normalizer=WikiTitleNormalizer(),
            source_type="wiki",
            show_progress=False,
        )
        
        graph = builder.build_graph(output_file)
        
        # Find page2 node
        page2_nodes = [n for n in graph.values() if 'page2' in n.title]
        self.assertEqual(len(page2_nodes), 1)
        page2 = page2_nodes[0]
        
        # page2 should have 1 incoming link (from page1)
        self.assertGreaterEqual(len(page2.incoming), 1)
    
    def test_output_file_created(self):
        """Test that output JSONL file is created."""
        output_file = self.temp_path / "graph.jsonl"
        
        builder = GraphBuilder(
            source=MarkdownFileSource(self.temp_path),
            link_extractor=MarkdownLinkExtractor(),
            normalizer=WikiTitleNormalizer(),
            source_type="wiki",
            show_progress=False,
        )
        
        builder.build_graph(output_file)
        
        # Check output file exists
        self.assertTrue(output_file.exists())
    
    def test_output_jsonl_format(self):
        """Test that output file has correct JSONL format."""
        output_file = self.temp_path / "graph.jsonl"
        
        builder = GraphBuilder(
            source=MarkdownFileSource(self.temp_path),
            link_extractor=MarkdownLinkExtractor(),
            normalizer=WikiTitleNormalizer(),
            source_type="wiki",
            show_progress=False,
        )
        
        builder.build_graph(output_file)
        
        # Check JSONL format
        with open(output_file) as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)  # 3 nodes
            
            for line in lines:
                data = json.loads(line)
                # Check required fields
                self.assertIn('title', data)
                self.assertIn('outgoing', data)
                self.assertIn('incoming', data)
                self.assertIn('char_count', data)
                
                # Check types
                self.assertIsInstance(data['title'], str)
                self.assertIsInstance(data['outgoing'], list)
                self.assertIsInstance(data['incoming'], list)
                self.assertIsInstance(data['char_count'], int)
    
    def test_char_count(self):
        """Test that character counts are recorded."""
        output_file = self.temp_path / "graph.jsonl"
        
        builder = GraphBuilder(
            source=MarkdownFileSource(self.temp_path),
            link_extractor=MarkdownLinkExtractor(),
            normalizer=WikiTitleNormalizer(),
            source_type="wiki",
            show_progress=False,
        )
        
        graph = builder.build_graph(output_file)
        
        # All nodes should have positive char counts
        for node in graph.values():
            self.assertGreater(node.char_count, 0)


if __name__ == '__main__':
    unittest.main()
