"""
Tests for content source implementations.
"""
import unittest
import tempfile
import json
from pathlib import Path

from data.extractors.sources import FileSource, MarkdownFileSource, JSONLSource


class TestFileSource(unittest.TestCase):
    def setUp(self):
        # Create temporary directory with test markdown files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test files
        (self.temp_path / "page1.md").write_text("Content of page 1")
        (self.temp_path / "page2.md").write_text("Content of page 2")
        
        # Create subdirectory
        subdir = self.temp_path / "subdir"
        subdir.mkdir()
        (subdir / "page3.md").write_text("Content of page 3")
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_iterates_markdown_files(self):
        """Test that source iterates over .md files."""
        source = FileSource(self.temp_path, extension='.md')
        documents = list(source.iter_documents())
        
        # Should find all 3 files (recursive by default)
        self.assertEqual(len(documents), 3)
        
        # Check identifiers
        identifiers = {doc.identifier for doc in documents}
        self.assertEqual(identifiers, {"page1", "page2", "page3"})
    
    def test_non_recursive(self):
        """Test non-recursive mode."""
        source = FileSource(self.temp_path, extension='.md', recursive=False)
        documents = list(source.iter_documents())
        
        # Should only find 2 files in root
        self.assertEqual(len(documents), 2)
        identifiers = {doc.identifier for doc in documents}
        self.assertEqual(identifiers, {"page1", "page2"})
    
    def test_content_is_read(self):
        """Test that file content is read correctly."""
        source = FileSource(self.temp_path, extension='.md')
        documents = list(source.iter_documents())
        
        # Find page1
        page1 = next(d for d in documents if d.identifier == "page1")
        self.assertEqual(page1.content, "Content of page 1")
    
    def test_metadata_includes_filepath(self):
        """Test that metadata includes filepath."""
        source = FileSource(self.temp_path, extension='.md')
        documents = list(source.iter_documents())
        
        for doc in documents:
            self.assertIn('filepath', doc.metadata)
            self.assertTrue(doc.metadata['filepath'].endswith('.md'))
    
    def test_invalid_directory_raises_error(self):
        """Test that non-existent directory raises error."""
        with self.assertRaises(ValueError):
            FileSource(Path("/nonexistent/path"))
    
    def test_default_extension_is_md(self):
        """Test that default extension is .md for backward compatibility."""
        source = FileSource(self.temp_path)
        documents = list(source.iter_documents())
        
        # Should find all .md files with default extension
        self.assertEqual(len(documents), 3)
        identifiers = {doc.identifier for doc in documents}
        self.assertEqual(identifiers, {"page1", "page2", "page3"})
    
    def test_custom_extension(self):
        """Test reading files with custom extension."""
        # Create .py files
        (self.temp_path / "script1.py").write_text("print('hello')")
        (self.temp_path / "script2.py").write_text("print('world')")
        
        source = FileSource(self.temp_path, extension='.py')
        documents = list(source.iter_documents())
        
        # Should only find .py files, not .md
        identifiers = {doc.identifier for doc in documents}
        self.assertEqual(identifiers, {"script1", "script2"})
    
    def test_extension_without_dot(self):
        """Test that extension without dot is handled correctly."""
        # Create .txt files
        (self.temp_path / "file1.txt").write_text("Text content 1")
        (self.temp_path / "file2.txt").write_text("Text content 2")
        
        # Should work with or without leading dot
        source = FileSource(self.temp_path, extension='txt')
        documents = list(source.iter_documents())
        
        identifiers = {doc.identifier for doc in documents}
        self.assertEqual(identifiers, {"file1", "file2"})
    
    def test_markdown_file_source_alias(self):
        """Test that MarkdownFileSource still works as backward compatibility alias."""
        source = MarkdownFileSource(self.temp_path)
        documents = list(source.iter_documents())
        
        # Should work exactly like FileSource with .md extension
        self.assertEqual(len(documents), 3)
        identifiers = {doc.identifier for doc in documents}
        self.assertEqual(identifiers, {"page1", "page2", "page3"})


class TestJSONLSource(unittest.TestCase):
    def setUp(self):
        # Create temporary JSONL file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        self.temp_path = Path(self.temp_file.name)
        
        # Write test data
        test_data = [
            {"id": "doc1", "content": "Content 1", "extra": "value1"},
            {"id": "doc2", "content": "Content 2", "extra": "value2"},
            {"id": "doc3", "content": "Content 3"},  # Missing extra field
        ]
        for item in test_data:
            self.temp_file.write(json.dumps(item) + '\n')
        self.temp_file.close()
    
    def tearDown(self):
        self.temp_path.unlink()
    
    def test_iterates_jsonl_lines(self):
        """Test that source iterates over JSONL lines."""
        source = JSONLSource(
            self.temp_path,
            identifier_field="id",
            content_field="content"
        )
        documents = list(source.iter_documents())
        
        self.assertEqual(len(documents), 3)
        identifiers = {doc.identifier for doc in documents}
        self.assertEqual(identifiers, {"doc1", "doc2", "doc3"})
    
    def test_content_is_extracted(self):
        """Test that content field is extracted."""
        source = JSONLSource(
            self.temp_path,
            identifier_field="id",
            content_field="content"
        )
        documents = list(source.iter_documents())
        
        doc1 = next(d for d in documents if d.identifier == "doc1")
        self.assertEqual(doc1.content, "Content 1")
    
    def test_additional_fields_in_metadata(self):
        """Test that additional fields are captured in metadata."""
        source = JSONLSource(
            self.temp_path,
            identifier_field="id",
            content_field="content",
            additional_fields=["extra"]
        )
        documents = list(source.iter_documents())
        
        doc1 = next(d for d in documents if d.identifier == "doc1")
        self.assertEqual(doc1.metadata["extra"], "value1")
        
        # doc3 doesn't have extra field, should be None
        doc3 = next(d for d in documents if d.identifier == "doc3")
        self.assertIsNone(doc3.metadata["extra"])
    
    def test_skips_malformed_json(self):
        """Test that malformed JSON lines are skipped."""
        # Create file with malformed JSON
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        temp_path = Path(temp_file.name)
        
        temp_file.write('{"id": "doc1", "content": "Good"}\n')
        temp_file.write('{"invalid json\n')  # Malformed
        temp_file.write('{"id": "doc2", "content": "Also good"}\n')
        temp_file.close()
        
        try:
            source = JSONLSource(temp_path, identifier_field="id")
            documents = list(source.iter_documents())
            
            # Should skip malformed line
            self.assertEqual(len(documents), 2)
            identifiers = {doc.identifier for doc in documents}
            self.assertEqual(identifiers, {"doc1", "doc2"})
        finally:
            temp_path.unlink()
    
    def test_invalid_file_raises_error(self):
        """Test that non-existent file raises error."""
        with self.assertRaises(ValueError):
            JSONLSource(Path("/nonexistent.jsonl"), identifier_field="id")


if __name__ == '__main__':
    unittest.main()
