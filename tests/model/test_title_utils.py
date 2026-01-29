"""
Unit tests for title normalization utilities.
"""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from experiments.dagseq2dagseq.model.title_utils import (
    normalize_title,
    generate_title_hash,
    create_filename,
    strip_hash,
    verify_title_hash,
)


class TestNormalizeTitle:
    """Tests for normalize_title function."""
    
    def test_basic_normalization(self):
        """Test basic title normalization."""
        assert normalize_title("Python Programming") == "python_programming"
        assert normalize_title("Hello World") == "hello_world"
    
    def test_special_characters(self):
        """Test that special characters are removed."""
        assert normalize_title("C++ Tutorial!") == "c_tutorial"
        assert normalize_title("Node.js Guide") == "nodejs_guide"
        assert normalize_title("Data (Science)") == "data_science"
    
    def test_multiple_spaces(self):
        """Test handling of multiple spaces."""
        assert normalize_title("Multiple   Spaces") == "multiple___spaces"
    
    def test_already_normalized(self):
        """Test that already normalized titles pass through."""
        assert normalize_title("already_normalized") == "already_normalized"
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert normalize_title("") == ""


class TestGenerateTitleHash:
    """Tests for generate_title_hash function."""
    
    def test_hash_length(self):
        """Test that hash is always 6 characters."""
        assert len(generate_title_hash("Python")) == 6
        assert len(generate_title_hash("A")) == 6
        assert len(generate_title_hash("Very Long Title With Many Words")) == 6
    
    def test_deterministic(self):
        """Test that hash is deterministic."""
        title = "Test Title"
        hash1 = generate_title_hash(title)
        hash2 = generate_title_hash(title)
        assert hash1 == hash2
    
    def test_different_titles_different_hashes(self):
        """Test that different titles produce different hashes (usually)."""
        hash1 = generate_title_hash("Python")
        hash2 = generate_title_hash("Java")
        # Note: Collision is theoretically possible but extremely unlikely
        assert hash1 != hash2
    
    def test_hash_is_hex(self):
        """Test that hash contains only hex characters."""
        hash_val = generate_title_hash("Test")
        assert all(c in "0123456789abcdef" for c in hash_val)


class TestCreateFilename:
    """Tests for create_filename function."""
    
    def test_basic_filename(self):
        """Test basic filename creation."""
        filename = create_filename("Python")
        assert filename.startswith("python_")
        assert len(filename) == len("python_") + 6
    
    def test_format(self):
        """Test filename format."""
        filename = create_filename("Test Title")
        parts = filename.rsplit("_", 1)
        assert len(parts) == 2
        assert parts[0] == "test_title"
        assert len(parts[1]) == 6
    
    def test_deterministic(self):
        """Test that filename generation is deterministic."""
        filename1 = create_filename("Test")
        filename2 = create_filename("Test")
        assert filename1 == filename2


class TestStripHash:
    """Tests for strip_hash function."""
    
    def test_strip_valid_hash(self):
        """Test stripping valid hash suffix."""
        assert strip_hash("python_a7f8c3") == "python"
        assert strip_hash("test_title_123abc") == "test_title"
    
    def test_no_hash(self):
        """Test that titles without hash are unchanged."""
        assert strip_hash("no_hash_here") == "no_hash_here"
        assert strip_hash("python") == "python"
    
    def test_multiple_underscores(self):
        """Test with multiple underscores."""
        assert strip_hash("test_title_name_abc123") == "test_title_name"
    
    def test_invalid_hash_not_stripped(self):
        """Test that invalid hash patterns are not stripped."""
        # Hash must be exactly 6 hex chars
        assert strip_hash("test_12345") == "test_12345"  # Only 5 chars
        assert strip_hash("test_1234567") == "test_1234567"  # 7 chars
        assert strip_hash("test_gggggg") == "test_gggggg"  # Non-hex chars


class TestVerifyTitleHash:
    """Tests for verify_title_hash function."""
    
    def test_valid_verification(self):
        """Test verifying a valid filename."""
        title = "Python"
        filename = create_filename(title)
        assert verify_title_hash(title, filename) is True
    
    def test_invalid_verification(self):
        """Test verifying an invalid filename."""
        assert verify_title_hash("Python", "python_wrong1") is False
        assert verify_title_hash("Python", "java_abc123") is False
    
    def test_case_sensitive(self):
        """Test that verification is case-sensitive (via normalization)."""
        # Both should normalize to same thing, but different original = different hash
        filename1 = create_filename("Python")
        filename2 = create_filename("PYTHON")
        # They might have different hashes because hash is based on original
        assert verify_title_hash("Python", filename1) is True
        assert verify_title_hash("PYTHON", filename2) is True


class TestRoundTrip:
    """Test round-trip conversions."""
    
    def test_create_and_strip(self):
        """Test creating filename and stripping hash."""
        title = "Test Title"
        filename = create_filename(title)
        normalized = strip_hash(filename)
        assert normalized == normalize_title(title)
    
    def test_verification_roundtrip(self):
        """Test that created filename verifies correctly."""
        titles = ["Python", "Machine Learning", "C++", "Hello World!"]
        for title in titles:
            filename = create_filename(title)
            assert verify_title_hash(title, filename) is True
