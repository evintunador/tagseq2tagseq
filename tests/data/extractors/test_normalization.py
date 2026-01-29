"""
Tests for link normalization logic.
"""
import unittest
from data.extractors.normalization import (
    HashingNormalizer, WikiTitleNormalizer, PythonModuleNormalizer
)


class TestHashingNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = HashingNormalizer()
    
    def test_basic_normalization(self):
        """Test basic normalization with spaces and mixed case."""
        result = self.normalizer.normalize("Test Title")
        self.assertTrue(result.startswith("test_title_"))
        self.assertRegex(result, r"_[a-f0-9]{6}$")
    
    def test_special_characters(self):
        """Test that special characters are replaced."""
        result = self.normalizer.normalize("Test@#$%Title")
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)
        self.assertNotIn("$", result)
        self.assertNotIn("%", result)
    
    def test_uniqueness_with_hash(self):
        """Test that different symbols produce different hashes."""
        # "A+B" normalizes to "a_b" and "A@B" also normalizes to "a_b"
        # But they should have different hashes because the hash is computed
        # from the canonical form (before symbol removal)
        result1 = self.normalizer.normalize("A+B")
        result2 = self.normalizer.normalize("A@B")
        
        # Results should be different because hashes differ
        self.assertNotEqual(result1, result2)
        
        # Both should start with "a_b_" (the normalized form)
        self.assertTrue(result1.startswith("a_b_"))
        self.assertTrue(result2.startswith("a_b_"))
    
    def test_length_limiting(self):
        """Test that very long strings are truncated."""
        long_string = "a" * 300
        result = self.normalizer.normalize(long_string)
        # Should be limited to max_length (193) + "_" + 6-char hash
        self.assertLessEqual(len(result), 193 + 1 + 6)
    
    def test_collapse_underscores(self):
        """Test that multiple underscores are collapsed."""
        result = self.normalizer.normalize("test___multiple___underscores")
        self.assertNotIn("___", result)
        self.assertNotIn("__", result)


class TestWikiTitleNormalizer(unittest.TestCase):
    def test_html_entities(self):
        """Test HTML entity decoding."""
        normalizer = WikiTitleNormalizer()
        result = normalizer.normalize("Test&nbsp;Title")
        # &nbsp; should be decoded (it becomes a space, then underscore)
        self.assertNotIn("&nbsp;", result)
        self.assertNotIn("nbsp", result)
    
    def test_ampersand_entity(self):
        """Test HTML ampersand entity."""
        normalizer = WikiTitleNormalizer()
        result = normalizer.normalize("A&amp;B")
        # &amp; should decode to & then become _
        self.assertNotIn("&amp;", result)
        self.assertNotIn("amp", result)
    
    def test_case_insensitive(self):
        """Test that normalization is case-insensitive."""
        normalizer = WikiTitleNormalizer()
        result1 = normalizer.normalize("Wikipedia")
        result2 = normalizer.normalize("wikipedia")
        result3 = normalizer.normalize("WIKIPEDIA")
        # All should produce same result
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)


class TestPythonModuleNormalizer(unittest.TestCase):
    def test_dots_to_underscores(self):
        """Test that dots are converted to underscores."""
        normalizer = PythonModuleNormalizer()
        result = normalizer.normalize("os.path.join")
        self.assertNotIn(".", result)
        self.assertTrue(result.startswith("os_path_join_"))
    
    def test_simple_module(self):
        """Test normalization of simple module name."""
        normalizer = PythonModuleNormalizer()
        result = normalizer.normalize("numpy")
        self.assertTrue(result.startswith("numpy_"))
        self.assertRegex(result, r"^numpy_[a-f0-9]{6}$")
    
    def test_nested_package(self):
        """Test normalization of nested package."""
        normalizer = PythonModuleNormalizer()
        result = normalizer.normalize("mypackage.submodule.function")
        self.assertNotIn(".", result)
        self.assertTrue(result.startswith("mypackage_submodule_function_"))


if __name__ == '__main__':
    unittest.main()
