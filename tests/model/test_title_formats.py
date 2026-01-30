"""
Tests for title formatting abstractions.
"""
import pytest
from model.title_formats import (
    FlatTitleFormatter,
    HierarchicalTitleFormatter,
    ColonSeparatedFormatter,
    set_default_formatter,
    get_default_formatter,
    normalize_title,
    create_filename,
    strip_hash,
)


class TestFlatTitleFormatter:
    """Tests for flat title formatter (Wikipedia-style)."""
    
    def test_normalize_basic(self):
        formatter = FlatTitleFormatter()
        assert formatter.normalize("Python Programming") == "python_programming"
        assert formatter.normalize("Machine Learning") == "machine_learning"
    
    def test_normalize_removes_special_chars(self):
        formatter = FlatTitleFormatter()
        assert formatter.normalize("C++ Programming!") == "c_programming"
        assert formatter.normalize("Data Science (2024)") == "data_science_2024"
    
    def test_normalize_removes_slashes(self):
        """Flat formatter should strip slashes (no hierarchy)."""
        formatter = FlatTitleFormatter()
        assert formatter.normalize("path/to/file") == "pathtofile"
    
    def test_add_hash(self):
        formatter = FlatTitleFormatter(hash_length=6)
        result = formatter.add_hash("python", "Python")
        assert result.startswith("python_")
        assert len(result) == len("python_") + 6
        # Hash should be hex
        assert all(c in '0123456789abcdef' for c in result.split('_')[1])
    
    def test_strip_hash(self):
        formatter = FlatTitleFormatter()
        assert formatter.strip_hash("python_a7f8c3") == "python"
        assert formatter.strip_hash("machine_learning_123456") == "machine_learning"
        # No hash to strip
        assert formatter.strip_hash("python") == "python"
    
    def test_create_link_target(self):
        formatter = FlatTitleFormatter()
        result = formatter.create_link_target("Python Programming")
        assert result.startswith("python_programming_")
        assert len(result) == len("python_programming_") + 6
    
    def test_validate_title(self):
        formatter = FlatTitleFormatter()
        assert formatter.validate_title("python")
        assert formatter.validate_title("python_programming")
        assert formatter.validate_title("python_a7f8c3")
        assert not formatter.validate_title("Python")  # uppercase
        assert not formatter.validate_title("python-programming")  # hyphen
        assert not formatter.validate_title("python/file")  # slash
    
    def test_hash_deterministic(self):
        """Same input should always produce same hash."""
        formatter = FlatTitleFormatter()
        hash1 = formatter.create_link_target("Python")
        hash2 = formatter.create_link_target("Python")
        assert hash1 == hash2


class TestHierarchicalTitleFormatter:
    """Tests for hierarchical title formatter (nested paths)."""
    
    def test_normalize_preserves_slashes(self):
        formatter = HierarchicalTitleFormatter()
        assert formatter.normalize("src/utils/helper.py") == "src/utils/helperpy"
        assert formatter.normalize("docs/api/reference.md") == "docs/api/referencemd"
    
    def test_normalize_basic(self):
        formatter = HierarchicalTitleFormatter()
        result = formatter.normalize("Docs/API Reference.md")
        # Lowercase, spaces to underscores, preserve /
        assert result == "docs/api_referencemd"
    
    def test_add_hash_with_path(self):
        formatter = HierarchicalTitleFormatter()
        result = formatter.add_hash("src/utils/helper", "src/utils/helper.py")
        assert result.startswith("src/utils/helper_")
        assert len(result) == len("src/utils/helper_") + 6
    
    def test_strip_hash_with_path(self):
        formatter = HierarchicalTitleFormatter()
        assert formatter.strip_hash("src/utils/helper_a7f8c3") == "src/utils/helper"
        assert formatter.strip_hash("docs/api/ref_123456") == "docs/api/ref"
    
    def test_create_link_target_with_path(self):
        formatter = HierarchicalTitleFormatter()
        result = formatter.create_link_target("src/utils/helper.py")
        assert "src/utils" in result
        assert result.endswith("_" + result.split('_')[-1])  # Has hash suffix
    
    def test_validate_title_with_path(self):
        formatter = HierarchicalTitleFormatter()
        assert formatter.validate_title("src/utils/helper")
        assert formatter.validate_title("src/utils/helper_a7f8c3")
        assert formatter.validate_title("docs/api/reference")
        assert not formatter.validate_title("Src/Utils")  # uppercase
        assert not formatter.validate_title("src\\utils")  # wrong separator
    
    def test_custom_separator(self):
        formatter = HierarchicalTitleFormatter(separator='.')
        normalized = formatter.normalize("com.example.MyClass")
        assert "." in normalized
        assert formatter.validate_title("com.example.myclass")


class TestColonSeparatedFormatter:
    """Tests for colon-separated formatter (GitHub-style)."""
    
    def test_normalize_with_colon_and_slash(self):
        formatter = ColonSeparatedFormatter()
        result = formatter.normalize("django_project:django/core/handlers.py")
        assert ":" in result
        assert "/" in result
        assert result == "django_project:django/core/handlerspy"
    
    def test_add_hash_preserves_structure(self):
        formatter = ColonSeparatedFormatter()
        result = formatter.add_hash("repo:path/file", "repo:path/file.py")
        assert result.startswith("repo:path/file_")
        assert ":" in result
        assert "/" in result
    
    def test_validate_colon_separated(self):
        formatter = ColonSeparatedFormatter()
        assert formatter.validate_title("repo:file")
        assert formatter.validate_title("repo:path/to/file")
        assert formatter.validate_title("repo:path/file_a7f8c3")
        assert not formatter.validate_title("no_colon_here")  # needs colon
        assert not formatter.validate_title("repo:Path")  # uppercase


class TestDefaultFormatter:
    """Tests for global default formatter system."""
    
    def test_default_is_flat(self):
        """By default should use flat formatter."""
        formatter = get_default_formatter()
        assert isinstance(formatter, FlatTitleFormatter)
    
    def test_set_and_get_default(self):
        """Should be able to set and retrieve default formatter."""
        original = get_default_formatter()
        
        try:
            # Set to hierarchical
            hierarchical = HierarchicalTitleFormatter()
            set_default_formatter(hierarchical)
            
            retrieved = get_default_formatter()
            assert retrieved is hierarchical
            
            # Convenience functions should use it
            result = normalize_title("src/utils/helper.py")
            assert "/" in result  # Hierarchical preserves slashes
        
        finally:
            # Restore original
            set_default_formatter(original)
    
    def test_convenience_functions_use_default(self):
        """Convenience functions should delegate to default formatter."""
        original = get_default_formatter()
        
        try:
            # Test with flat
            flat = FlatTitleFormatter()
            set_default_formatter(flat)
            assert normalize_title("Path/To/File") == "pathtofile"  # Strips slashes
            
            # Test with hierarchical
            hierarchical = HierarchicalTitleFormatter()
            set_default_formatter(hierarchical)
            assert "/" in normalize_title("Path/To/File")  # Preserves slashes
        
        finally:
            set_default_formatter(original)
    
    def test_create_filename_uses_default(self):
        """create_filename should use default formatter."""
        original = get_default_formatter()
        
        try:
            flat = FlatTitleFormatter()
            set_default_formatter(flat)
            
            result = create_filename("Test Title")
            assert result.startswith("test_title_")
            assert "_" in result
        
        finally:
            set_default_formatter(original)
    
    def test_strip_hash_uses_default(self):
        """strip_hash should use default formatter."""
        original = get_default_formatter()
        
        try:
            flat = FlatTitleFormatter()
            set_default_formatter(flat)
            
            assert strip_hash("test_title_a7f8c3") == "test_title"
        
        finally:
            set_default_formatter(original)


class TestCrossFormatter:
    """Tests comparing behavior across different formatters."""
    
    def test_all_formatters_are_deterministic(self):
        """All formatters should produce consistent hashes."""
        formatters = [
            FlatTitleFormatter(),
            HierarchicalTitleFormatter(),
            ColonSeparatedFormatter(),
        ]
        
        test_titles = [
            "Python",
            "path/to/file",
            "repo:path/to/file",
        ]
        
        for formatter in formatters:
            for title in test_titles:
                link1 = formatter.create_link_target(title)
                link2 = formatter.create_link_target(title)
                assert link1 == link2
    
    def test_formatters_handle_edge_cases(self):
        """All formatters should handle edge cases gracefully."""
        formatters = [
            FlatTitleFormatter(),
            HierarchicalTitleFormatter(),
            ColonSeparatedFormatter(),
        ]
        
        edge_cases = [
            "",
            "   ",
            "___",
            "!!!",
            "a",
        ]
        
        for formatter in formatters:
            for title in edge_cases:
                # Should not crash
                normalized = formatter.normalize(title)
                assert isinstance(normalized, str)
                
                # Should be able to add hash
                with_hash = formatter.add_hash(normalized, title)
                assert isinstance(with_hash, str)
    
    def test_strip_and_add_are_inverse(self):
        """strip_hash and add_hash should be inverse operations."""
        formatters = [
            FlatTitleFormatter(),
            HierarchicalTitleFormatter(),
            ColonSeparatedFormatter(),
        ]
        
        test_titles = [
            "Python",
            "src/utils/helper",
            "repo:path/file",
        ]
        
        for formatter in formatters:
            for title in test_titles:
                normalized = formatter.normalize(title)
                with_hash = formatter.add_hash(normalized, title)
                stripped = formatter.strip_hash(with_hash)
                assert stripped == normalized
