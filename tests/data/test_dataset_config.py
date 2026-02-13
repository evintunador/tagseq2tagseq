"""
Tests for DatasetConfig normalization integration.
"""
import unittest
import re
from pathlib import Path
import tempfile
import json

from data.dataset_config import (
    DatasetConfig, 
    WIKIPEDIA_CONFIG,
    THESTACK_CONFIG,
    DOCUMENTATION_CONFIG,
)
from data.extractors.normalization import (
    FilesafeNormalizer,
    PassthroughNormalizer,
    WikiTitleNormalizer,
    PythonModuleNormalizer,
)


class TestDatasetConfigNormalization(unittest.TestCase):
    """Test that DatasetConfig properly creates and configures normalizers."""
    
    def test_get_normalizer_filesafe(self):
        """Test that filesafe normalizer is created with correct hash_length."""
        config = DatasetConfig(
            name="Test",
            title_format="flat",
            normalizer_type="filesafe",
            hash_length=8,
        )
        normalizer = config.get_normalizer()
        
        self.assertIsInstance(normalizer, FilesafeNormalizer)
        self.assertEqual(normalizer.hash_length, 8)
        self.assertEqual(normalizer.max_length, 200 - 1 - 8)  # 191
        
        # Test that normalization produces correct hash length
        result = normalizer.normalize("test")
        self.assertRegex(result, r"^test_[0-9a-f]{8}$")
    
    def test_get_normalizer_passthrough(self):
        """Test that passthrough normalizer is created."""
        config = DatasetConfig(
            name="Test",
            title_format="flat",
            normalizer_type="passthrough",
            hash_length=6,
        )
        normalizer = config.get_normalizer()
        
        self.assertIsInstance(normalizer, PassthroughNormalizer)
        
        # Passthrough should return input unchanged
        result = normalizer.normalize("test_123abc")
        self.assertEqual(result, "test_123abc")
    
    def test_get_normalizer_wiki(self):
        """Test that wiki normalizer is created with correct hash_length."""
        config = DatasetConfig(
            name="Test",
            title_format="flat",
            normalizer_type="wiki",
            hash_length=10,
        )
        normalizer = config.get_normalizer()
        
        self.assertIsInstance(normalizer, WikiTitleNormalizer)
        self.assertEqual(normalizer.hash_length, 10)
        
        # Test that normalization produces correct hash length
        result = normalizer.normalize("Test&nbsp;Title")
        self.assertRegex(result, r"^.*_[0-9a-f]{10}$")
    
    def test_get_normalizer_python_module(self):
        """Test that python module normalizer is created with correct hash_length."""
        config = DatasetConfig(
            name="Test",
            title_format="colon_separated",
            normalizer_type="python_module",
            hash_length=4,
        )
        normalizer = config.get_normalizer()
        
        self.assertIsInstance(normalizer, PythonModuleNormalizer)
        self.assertEqual(normalizer.hash_length, 4)
        
        # Test that normalization produces correct hash length
        result = normalizer.normalize("os.path.join")
        self.assertRegex(result, r"^os_path_join_[0-9a-f]{4}$")
    
    def test_normalizer_caching(self):
        """Test that normalizer is cached and reused."""
        config = DatasetConfig(
            name="Test",
            title_format="flat",
            normalizer_type="filesafe",
            hash_length=6,
        )
        
        normalizer1 = config.get_normalizer()
        normalizer2 = config.get_normalizer()
        
        # Should be the same object (cached)
        self.assertIs(normalizer1, normalizer2)
    
    def test_different_hash_lengths_produce_different_hashes(self):
        """Test that different hash_length configs produce different hash lengths."""
        config6 = DatasetConfig(
            name="Test6",
            title_format="flat",
            normalizer_type="filesafe",
            hash_length=6,
        )
        config8 = DatasetConfig(
            name="Test8",
            title_format="flat",
            normalizer_type="filesafe",
            hash_length=8,
        )
        
        norm6 = config6.get_normalizer()
        norm8 = config8.get_normalizer()
        
        result6 = norm6.normalize("test")
        result8 = norm8.normalize("test")
        
        # Base should be same, but hash lengths differ
        self.assertTrue(result6.startswith("test_"))
        self.assertTrue(result8.startswith("test_"))
        
        # Extract hash portions
        hash6 = result6.split("_")[-1]
        hash8 = result8.split("_")[-1]
        
        self.assertEqual(len(hash6), 6)
        self.assertEqual(len(hash8), 8)
    
    def test_hash_length_validation(self):
        """Test that invalid hash_length values are rejected."""
        with self.assertRaises(ValueError):
            DatasetConfig(
                name="Test",
                title_format="flat",
                normalizer_type="filesafe",
                hash_length=0,  # Too small
            )
        
        with self.assertRaises(ValueError):
            DatasetConfig(
                name="Test",
                title_format="flat",
                normalizer_type="filesafe",
                hash_length=33,  # Too large
            )


class TestDatasetConfigSerialization(unittest.TestCase):
    """Test that DatasetConfig properly serializes/deserializes."""
    
    def test_to_dict_excludes_normalizer(self):
        """Test that _normalizer is not serialized."""
        config = DatasetConfig(
            name="Test",
            title_format="flat",
            normalizer_type="filesafe",
            hash_length=6,
        )
        
        # Access normalizer to populate cache
        _ = config.get_normalizer()
        
        # Serialize to dict
        data = config.to_dict()
        
        # _normalizer should not be in dict
        self.assertNotIn("_normalizer", data)
        
        # But other fields should be present
        self.assertEqual(data["name"], "Test")
        self.assertEqual(data["normalizer_type"], "filesafe")
        self.assertEqual(data["hash_length"], 6)
    
    def test_save_and_load(self):
        """Test saving and loading config to/from JSON."""
        config = DatasetConfig(
            name="Test Dataset",
            title_format="hierarchical",
            normalizer_type="wiki",
            hash_length=8,
            path_separator="/",
            description="A test dataset"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            
            # Save
            config.save(path)
            
            # Verify file exists
            self.assertTrue(path.exists())
            
            # Load
            loaded = DatasetConfig.load(path)
            
            # Verify fields match
            self.assertEqual(loaded.name, config.name)
            self.assertEqual(loaded.title_format, config.title_format)
            self.assertEqual(loaded.normalizer_type, config.normalizer_type)
            self.assertEqual(loaded.hash_length, config.hash_length)
            self.assertEqual(loaded.path_separator, config.path_separator)
            self.assertEqual(loaded.description, config.description)
            
            # Verify normalizer works after loading
            normalizer = loaded.get_normalizer()
            result = normalizer.normalize("test")
            self.assertRegex(result, r"^test_[0-9a-f]{8}$")


class TestPredefinedConfigs(unittest.TestCase):
    """Test that predefined configs have correct normalizer types."""
    
    def test_wikipedia_config(self):
        """Test Wikipedia config uses passthrough normalizer."""
        self.assertEqual(WIKIPEDIA_CONFIG.normalizer_type, "passthrough")
        self.assertEqual(WIKIPEDIA_CONFIG.hash_length, 6)
        self.assertEqual(WIKIPEDIA_CONFIG.link_format, "markdown")
        
        # Verify normalizer works
        normalizer = WIKIPEDIA_CONFIG.get_normalizer()
        self.assertIsInstance(normalizer, PassthroughNormalizer)
    
    def test_thestack_config(self):
        """Test TheStack config uses python_module normalizer."""
        self.assertEqual(THESTACK_CONFIG.normalizer_type, "python_module")
        self.assertEqual(THESTACK_CONFIG.hash_length, 6)
        self.assertEqual(THESTACK_CONFIG.link_format, "python_import")
        
        # Verify normalizer works
        normalizer = THESTACK_CONFIG.get_normalizer()
        self.assertIsInstance(normalizer, PythonModuleNormalizer)
    
    def test_documentation_config(self):
        """Test Documentation config uses filesafe normalizer."""
        self.assertEqual(DOCUMENTATION_CONFIG.normalizer_type, "filesafe")
        self.assertEqual(DOCUMENTATION_CONFIG.hash_length, 6)
        
        # Verify normalizer works
        normalizer = DOCUMENTATION_CONFIG.get_normalizer()
        self.assertIsInstance(normalizer, FilesafeNormalizer)


class TestHashStrippingRegex(unittest.TestCase):
    """Test that hash stripping works with different hash lengths."""
    
    def test_hash_stripping_different_lengths(self):
        """Test regex patterns for different hash lengths work correctly."""
        test_cases = [
            (6, "Check out [Albert Einstein](albert_einstein_a1b2c3)"),
            (8, "Check out [Albert Einstein](albert_einstein_a1b2c3d4)"),
            (4, "Check out [Albert Einstein](albert_einstein_a1b2)"),
            (10, "Check out [Albert Einstein](albert_einstein_a1b2c3d4e5)"),
        ]
        
        for hash_length, content in test_cases:
            # Build pattern like pretokenize.py does
            hash_pattern = rf'(\]\(.*?)_[0-9a-f]{{{hash_length}}}(\))'
            
            # Strip hash
            result = re.sub(hash_pattern, r'\1\2', content)
            
            # Should have hash removed
            expected = "Check out [Albert Einstein](albert_einstein)"
            self.assertEqual(result, expected, 
                           f"Failed for hash_length={hash_length}")
    
    def test_hash_stripping_multiple_links(self):
        """Test hash stripping with multiple links in one document."""
        hash_length = 6
        content = """
        See [Foo](foo_abc123) and [Bar](bar_def456) for details.
        Also check [Baz](baz_789012).
        """
        
        hash_pattern = rf'(\]\(.*?)_[0-9a-f]{{{hash_length}}}(\))'
        result = re.sub(hash_pattern, r'\1\2', content)
        
        # All hashes should be stripped
        self.assertIn("[Foo](foo)", result)
        self.assertIn("[Bar](bar)", result)
        self.assertIn("[Baz](baz)", result)
        
        # No hashes should remain
        self.assertNotIn("abc123", result)
        self.assertNotIn("def456", result)
        self.assertNotIn("789012", result)
    
    def test_hash_stripping_preserves_non_hashes(self):
        """Test that hash stripping doesn't affect other underscores."""
        hash_length = 6
        content = "[Test_Name](test_name_with_underscores_abc123)"
        
        hash_pattern = rf'(\]\(.*?)_[0-9a-f]{{{hash_length}}}(\))'
        result = re.sub(hash_pattern, r'\1\2', content)
        
        # Only the hash at the end should be removed
        expected = "[Test_Name](test_name_with_underscores)"
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
