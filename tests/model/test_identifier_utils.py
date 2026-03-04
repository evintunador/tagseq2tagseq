"""
Unit tests for identifier normalization utilities.
"""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from experiments.tagseq2tagseq.model.identifier_utils import (
    normalize_identifier,
    generate_identifier_hash,
    create_normed_identifier,
    strip_hash,
    verify_identifier_hash,
)


class TestNormalizeIdentifier:
    """Tests for normalize_identifier function."""

    def test_basic_normalization(self):
        assert normalize_identifier("Python Programming") == "python_programming"
        assert normalize_identifier("Hello World") == "hello_world"

    def test_special_characters(self):
        assert normalize_identifier("C++ Tutorial!") == "c_tutorial"
        assert normalize_identifier("Node.js Guide") == "nodejs_guide"
        assert normalize_identifier("Data (Science)") == "data_science"

    def test_multiple_spaces(self):
        assert normalize_identifier("Multiple   Spaces") == "multiple___spaces"

    def test_already_normalized(self):
        assert normalize_identifier("already_normalized") == "already_normalized"

    def test_empty_string(self):
        assert normalize_identifier("") == ""


class TestGenerateIdentifierHash:
    """Tests for generate_identifier_hash function."""

    def test_hash_length(self):
        assert len(generate_identifier_hash("Python")) == 6
        assert len(generate_identifier_hash("A")) == 6
        assert len(generate_identifier_hash("Very Long Title With Many Words")) == 6

    def test_deterministic(self):
        identifier = "Test Title"
        assert generate_identifier_hash(identifier) == generate_identifier_hash(identifier)

    def test_different_identifiers_different_hashes(self):
        # Collision is theoretically possible but extremely unlikely
        assert generate_identifier_hash("Python") != generate_identifier_hash("Java")

    def test_hash_is_hex(self):
        hash_val = generate_identifier_hash("Test")
        assert all(c in "0123456789abcdef" for c in hash_val)


class TestCreateNormedIdentifier:
    """Tests for create_normed_identifier function."""

    def test_basic(self):
        normed = create_normed_identifier("Python")
        assert normed.startswith("python_")
        assert len(normed) == len("python_") + 6

    def test_format(self):
        normed = create_normed_identifier("Test Title")
        parts = normed.rsplit("_", 1)
        assert len(parts) == 2
        assert parts[0] == "test_title"
        assert len(parts[1]) == 6

    def test_deterministic(self):
        assert create_normed_identifier("Test") == create_normed_identifier("Test")


class TestStripHash:
    """Tests for strip_hash function."""

    def test_strip_valid_hash(self):
        assert strip_hash("python_a7f8c3") == "python"
        assert strip_hash("test_title_123abc") == "test_title"

    def test_no_hash(self):
        assert strip_hash("no_hash_here") == "no_hash_here"
        assert strip_hash("python") == "python"

    def test_multiple_underscores(self):
        assert strip_hash("test_title_name_abc123") == "test_title_name"

    def test_invalid_hash_not_stripped(self):
        assert strip_hash("test_12345") == "test_12345"    # 5 chars
        assert strip_hash("test_1234567") == "test_1234567"  # 7 chars
        assert strip_hash("test_gggggg") == "test_gggggg"  # non-hex


class TestVerifyIdentifierHash:
    """Tests for verify_identifier_hash function."""

    def test_valid(self):
        raw = "Python"
        normed = create_normed_identifier(raw)
        assert verify_identifier_hash(raw, normed) is True

    def test_invalid(self):
        assert verify_identifier_hash("Python", "python_wrong1") is False
        assert verify_identifier_hash("Python", "java_abc123") is False

    def test_case_sensitive_hash(self):
        # Hash is based on original raw_identifier, so different case → different hash
        normed1 = create_normed_identifier("Python")
        normed2 = create_normed_identifier("PYTHON")
        assert verify_identifier_hash("Python", normed1) is True
        assert verify_identifier_hash("PYTHON", normed2) is True


class TestRoundTrip:
    """Round-trip conversions."""

    def test_create_and_strip(self):
        raw = "Test Title"
        normed = create_normed_identifier(raw)
        stripped = strip_hash(normed)
        assert stripped == normalize_identifier(raw)

    def test_verification_roundtrip(self):
        identifiers = ["Python", "Machine Learning", "C++", "Hello World!"]
        for raw in identifiers:
            normed = create_normed_identifier(raw)
            assert verify_identifier_hash(raw, normed) is True
