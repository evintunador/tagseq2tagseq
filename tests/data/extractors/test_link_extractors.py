"""
Tests for link extraction logic.
"""
import unittest
from data.extractors.link_extractors import (
    MarkdownLinkExtractor, PythonImportExtractor
)
from data.extractors.protocols import Document, LinkContext


class TestMarkdownLinkExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MarkdownLinkExtractor()
    
    def test_basic_links(self):
        """Test extraction of basic markdown links."""
        content = "See [Page](target1) and [Another](target2)."
        doc = Document("test", content, {})
        context = LinkContext(doc, "wiki")
        
        links = self.extractor.extract_links(content, context)
        self.assertEqual(links, {"target1", "target2"})
    
    def test_skips_images(self):
        """Test that image links are skipped."""
        content = "![Image](img.png) and [Link](page)."
        doc = Document("test", content, {})
        context = LinkContext(doc, "wiki")
        
        links = self.extractor.extract_links(content, context)
        self.assertEqual(links, {"page"})
        self.assertNotIn("img.png", links)
    
    def test_empty_content(self):
        """Test extraction from empty content."""
        content = ""
        doc = Document("test", content, {})
        context = LinkContext(doc, "wiki")
        
        links = self.extractor.extract_links(content, context)
        self.assertEqual(links, set())
    
    def test_url_decoding(self):
        """Test that URL-encoded links are decoded."""
        content = "[Test](test%20page)"
        doc = Document("test", content, {})
        context = LinkContext(doc, "wiki")
        
        links = self.extractor.extract_links(content, context)
        self.assertEqual(links, {"test page"})
    
    def test_multiple_links_to_same_target(self):
        """Test that duplicate links are deduplicated."""
        content = "[Link1](target) and [Link2](target)."
        doc = Document("test", content, {})
        context = LinkContext(doc, "wiki")
        
        links = self.extractor.extract_links(content, context)
        self.assertEqual(links, {"target"})
        self.assertEqual(len(links), 1)


class TestPythonImportExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = PythonImportExtractor()
    
    def test_import_statement(self):
        """Test extraction of simple import statements."""
        content = "import mymodule\nimport another"
        doc = Document("test.py", content, {})
        context = LinkContext(doc, "github")
        
        links = self.extractor.extract_links(content, context)
        self.assertIn("mymodule", links)
        self.assertIn("another", links)
    
    def test_from_import(self):
        """Test extraction of from...import statements."""
        content = "from mypackage.submodule import func"
        doc = Document("test.py", content, {})
        context = LinkContext(doc, "github")
        
        links = self.extractor.extract_links(content, context)
        self.assertIn("mypackage.submodule", links)
    
    def test_skips_stdlib(self):
        """Test that standard library imports are filtered out."""
        content = "import os\nimport sys\nimport mymodule"
        doc = Document("test.py", content, {})
        context = LinkContext(doc, "github")
        
        links = self.extractor.extract_links(content, context)
        self.assertNotIn("os", links)
        self.assertNotIn("sys", links)
        self.assertIn("mymodule", links)
    
    def test_relative_imports(self):
        """Test extraction of relative imports."""
        content = "from .module import func\nfrom ..package import other"
        doc = Document("test.py", content, {})
        context = LinkContext(doc, "github")
        
        links = self.extractor.extract_links(content, context)
        self.assertIn(".module", links)
        self.assertIn("..package", links)
    
    def test_skips_common_packages(self):
        """Test that common packages like numpy, pandas are filtered."""
        content = "import numpy\nimport pandas\nimport torch\nimport mymodule"
        doc = Document("test.py", content, {})
        context = LinkContext(doc, "github")
        
        links = self.extractor.extract_links(content, context)
        self.assertNotIn("numpy", links)
        self.assertNotIn("pandas", links)
        self.assertNotIn("torch", links)
        self.assertIn("mymodule", links)
    
    def test_multiline_imports(self):
        """Test extraction from multiline import statements."""
        content = """
import module1
import module2

from package import thing
"""
        doc = Document("test.py", content, {})
        context = LinkContext(doc, "github")
        
        links = self.extractor.extract_links(content, context)
        self.assertIn("module1", links)
        self.assertIn("module2", links)
        self.assertIn("package", links)


if __name__ == '__main__':
    unittest.main()
