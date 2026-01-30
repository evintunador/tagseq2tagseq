"""
Title formatting abstractions for different dataset types.

Provides a protocol-based system for handling title normalization, link formatting,
and hash generation that can be customized per dataset (Wikipedia, GitHub, etc.).

This mirrors the abstraction pattern used in data/extractors/ but for the model's
title handling needs.
"""
import hashlib
import re
from typing import Protocol, Optional


class TitleFormatter(Protocol):
    """
    Protocol defining how titles are formatted, normalized, and linked.
    
    Different datasets have different conventions:
    - Wikipedia: flat titles like "Python_programming"
    - GitHub: hierarchical paths like "repo/src/utils/helper.py"
    - Academic papers: DOI-based or citation keys
    
    Implementations should be consistent with the data preprocessing pipeline.
    """
    
    def normalize(self, raw_title: str) -> str:
        """
        Normalize a raw title to its canonical form (without hash).
        
        This should match the normalization used during data preprocessing.
        
        Args:
            raw_title: The original, unnormalized title
            
        Returns:
            Normalized title string (no hash suffix)
            
        Examples:
            FlatFormatter: "Python Programming" -> "python_programming"
            HierarchicalFormatter: "repo/src/file.py" -> "repo/src/file.py"
        """
        ...
    
    def add_hash(self, normalized_title: str, original_title: str) -> str:
        """
        Add a hash suffix to a normalized title for uniqueness.
        
        Args:
            normalized_title: The already-normalized title
            original_title: The original title (used for hash generation)
            
        Returns:
            Title with hash suffix in format defined by implementation
            
        Examples:
            "python_programming" + hash("Python Programming") -> "python_programming_a7f8c3"
        """
        ...
    
    def strip_hash(self, title_with_hash: str) -> str:
        """
        Remove hash suffix from a normalized+hashed title.
        
        Args:
            title_with_hash: Title with hash suffix
            
        Returns:
            Title without hash suffix
            
        Examples:
            "python_programming_a7f8c3" -> "python_programming"
        """
        ...
    
    def create_link_target(self, raw_title: str) -> str:
        """
        Create a link target string from a raw title.
        
        This is what appears inside [text](...) markdown links.
        Typically: normalize(raw_title) + hash
        
        Args:
            raw_title: The title to link to
            
        Returns:
            Link target string
            
        Examples:
            "Python Programming" -> "python_programming_a7f8c3"
        """
        ...
    
    def validate_title(self, title: str) -> bool:
        """
        Check if a title is valid according to this formatter's rules.
        
        Args:
            title: Title to validate
            
        Returns:
            True if valid, False otherwise
        """
        ...


class FlatTitleFormatter:
    """
    Flat title formatter for Wikipedia-style datasets.
    
    Normalizes titles to lowercase with underscores, strips special characters.
    Suitable for single-level, human-readable titles without hierarchy.
    
    Examples:
        "Python Programming" -> "python_programming_a7f8c3"
        "Machine Learning" -> "machine_learning_b2e4d1"
    """
    
    def __init__(self, hash_length: int = 6):
        """
        Args:
            hash_length: Number of hex characters in hash suffix
        """
        self.hash_length = hash_length
    
    def normalize(self, raw_title: str) -> str:
        """Lowercase, replace spaces with underscores, strip special chars."""
        normalized = raw_title.lower()
        normalized = normalized.replace(' ', '_')
        # Keep only alphanumeric and underscores
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        return normalized
    
    def add_hash(self, normalized_title: str, original_title: str) -> str:
        """Add _XXXXXX hash suffix."""
        hash_obj = hashlib.md5(original_title.encode('utf-8'))
        hash_suffix = hash_obj.hexdigest()[:self.hash_length]
        return f"{normalized_title}_{hash_suffix}"
    
    def strip_hash(self, title_with_hash: str) -> str:
        """Remove _[0-9a-f]{6} suffix."""
        pattern = rf'_[0-9a-f]{{{self.hash_length}}}$'
        return re.sub(pattern, '', title_with_hash)
    
    def create_link_target(self, raw_title: str) -> str:
        """Normalize and add hash."""
        normalized = self.normalize(raw_title)
        return self.add_hash(normalized, raw_title)
    
    def validate_title(self, title: str) -> bool:
        """Check if title matches [a-z0-9_]+ pattern."""
        # Allow normalized titles with or without hash
        pattern = r'^[a-z0-9_]+(_[0-9a-f]{' + str(self.hash_length) + r'})?$'
        return bool(re.match(pattern, title))


class HierarchicalTitleFormatter:
    """
    Hierarchical title formatter for code/filesystem-style datasets.
    
    Preserves path separators (/) and allows hierarchical organization.
    Suitable for GitHub, documentation trees, or any nested structure.
    
    Examples:
        "repo/src/utils/helper.py" -> "repo/src/utils/helper.py_a7f8c3"
        "django/core/handlers.py" -> "django/core/handlers.py_b2e4d1"
    """
    
    def __init__(self, hash_length: int = 6, separator: str = '/'):
        """
        Args:
            hash_length: Number of hex characters in hash suffix
            separator: Path separator character (default: '/')
        """
        self.hash_length = hash_length
        self.separator = separator
    
    def normalize(self, raw_title: str) -> str:
        """
        Lowercase, replace spaces with underscores, preserve separators.
        
        Keeps path structure intact but normalizes within path components.
        """
        # Normalize but keep separators
        normalized = raw_title.lower()
        normalized = normalized.replace(' ', '_')
        # Keep alphanumeric, underscores, and separators
        allowed_pattern = f'[^a-z0-9_{re.escape(self.separator)}]'
        normalized = re.sub(allowed_pattern, '', normalized)
        return normalized
    
    def add_hash(self, normalized_title: str, original_title: str) -> str:
        """Add _XXXXXX hash suffix to end of path."""
        hash_obj = hashlib.md5(original_title.encode('utf-8'))
        hash_suffix = hash_obj.hexdigest()[:self.hash_length]
        return f"{normalized_title}_{hash_suffix}"
    
    def strip_hash(self, title_with_hash: str) -> str:
        """Remove _[0-9a-f]{6} suffix."""
        pattern = rf'_[0-9a-f]{{{self.hash_length}}}$'
        return re.sub(pattern, '', title_with_hash)
    
    def create_link_target(self, raw_title: str) -> str:
        """Normalize and add hash."""
        normalized = self.normalize(raw_title)
        return self.add_hash(normalized, raw_title)
    
    def validate_title(self, title: str) -> bool:
        """Check if title matches hierarchical pattern with optional hash."""
        # Allow paths with alphanumeric, underscores, separators, and optional hash
        sep = re.escape(self.separator)
        pattern = (
            rf'^[a-z0-9_{sep}]+'
            rf'(_[0-9a-f]{{{self.hash_length}}})?$'
        )
        return bool(re.match(pattern, title))


class ColonSeparatedFormatter(HierarchicalTitleFormatter):
    """
    Hierarchical formatter using colon separator.
    
    Useful for GitHub-style "repo:path/to/file" format where repo name
    is separated from path with a colon.
    
    Examples:
        "django_project_abc123:django/core/handlers.py" -> 
            "django_project_abc123:django/core/handlers.py_d4e5f6"
    """
    
    def __init__(self, hash_length: int = 6):
        # Allow both : and / as structural characters
        super().__init__(hash_length=hash_length, separator='/')
        self.repo_separator = ':'
    
    def normalize(self, raw_title: str) -> str:
        """Normalize but preserve both : and / separators."""
        normalized = raw_title.lower()
        normalized = normalized.replace(' ', '_')
        # Keep alphanumeric, underscores, colons, and forward slashes
        normalized = re.sub(r'[^a-z0-9_:/]', '', normalized)
        return normalized
    
    def validate_title(self, title: str) -> bool:
        """Check if title matches repo:path pattern with optional hash."""
        # Must contain at least one colon
        if ':' not in title:
            return False
        
        # Check overall pattern with optional hash
        pattern = (
            rf'^[a-z0-9_:]+[a-z0-9_:/]*'
            rf'(_[0-9a-f]{{{self.hash_length}}})?$'
        )
        return bool(re.match(pattern, title))


# Convenience functions that delegate to a global formatter
_default_formatter: TitleFormatter = FlatTitleFormatter()


def set_default_formatter(formatter: TitleFormatter) -> None:
    """
    Set the global default formatter used by convenience functions.
    
    This should typically be called once at application startup based on
    the dataset type being used.
    
    Args:
        formatter: TitleFormatter instance to use as default
        
    Example:
        >>> # For Wikipedia data
        >>> set_default_formatter(FlatTitleFormatter())
        >>> 
        >>> # For GitHub data
        >>> set_default_formatter(HierarchicalTitleFormatter())
    """
    global _default_formatter
    _default_formatter = formatter


def get_default_formatter() -> TitleFormatter:
    """Get the current default formatter."""
    return _default_formatter


# Backwards-compatible convenience functions (delegate to default formatter)
def normalize_title(title: str) -> str:
    """Normalize using default formatter."""
    return _default_formatter.normalize(title)


def generate_title_hash(title: str) -> str:
    """Generate hash using default formatter (for compatibility)."""
    normalized = _default_formatter.normalize(title)
    with_hash = _default_formatter.add_hash(normalized, title)
    # Extract just the hash part
    return with_hash[len(normalized)+1:] if '_' in with_hash else ''


def create_filename(title: str) -> str:
    """Create filename using default formatter."""
    return _default_formatter.create_link_target(title)


def strip_hash(title_with_hash: str) -> str:
    """Strip hash using default formatter."""
    return _default_formatter.strip_hash(title_with_hash)


def verify_title_hash(original_title: str, filename: str) -> bool:
    """Verify hash using default formatter."""
    expected = _default_formatter.create_link_target(original_title)
    return filename == expected
