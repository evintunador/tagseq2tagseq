"""
Identifier normalization utilities for creating filesystem-safe, collision-free names.

Core Purpose:
    Convert ANY string (titles, package names, URLs, etc.) into identifiers that are:
    - Filesystem-safe (no special characters)
    - Collision-free (unique hash suffix)
    - Human-readable (cleaned version of original)
    - Consistent (same input always produces same output)

Design Pattern:
    Template Method - Base class defines algorithm, subclasses customize steps.
    
The normalization algorithm:
    1. preprocess() - Domain-specific preprocessing (HTML decode, dot conversion, etc.)
    2. Canonicalization - Lowercase, space normalization
    3. Hash computation - On canonical form for collision resistance
    4. _clean() - Filesystem-safe character filtering
    5. Length limiting - Prevent filesystem issues
    6. Format: {readable_part}_{6char_hash}

This ensures "A+B" and "A-B" get different hashes but both become filesystem-safe.
"""
import re
import html
import hashlib
from .protocols import LinkNormalizer


class FilesafeNormalizer(LinkNormalizer):
    """
    Base normalizer using Template Method pattern.
    
    Provides the core normalization algorithm with hooks for domain-specific customization.
    Subclasses override preprocess() to handle their specific needs (HTML entities,
    module dots, etc.) without duplicating the core normalization logic.
    
    Examples:
        >>> normalizer = FilesafeNormalizer()
        >>> normalizer.normalize("Hello World")
        'hello_world_5eb63b'
        >>> normalizer.normalize("A+B")  # Different hash than "A-B"
        'a_b_4c614e'
    """
    
    def __init__(self, max_length: int = 193, hash_length: int = 6):
        """
        Args:
            max_length: Max length before hash suffix (default 193 for 200-char limit).
                        200 - 1 (underscore) - 6 (hash) = 193
            hash_length: Number of hex characters in hash suffix (default 6)
        """
        self.max_length = max_length
        self.hash_length = hash_length
    
    def normalize(self, text: str) -> str:
        """
        Template method defining the normalization algorithm.
        
        This is the public interface - don't override this unless you need to
        completely change the algorithm. Override preprocess() instead.
        
        Args:
            text: Raw string to normalize
        
        Returns:
            Normalized identifier: {readable_part}_{hash} where hash length is configurable
        """
        # Step 1: Domain-specific preprocessing (HOOK for subclasses)
        text = self.preprocess(text)
        
        # Step 2: Canonicalize (lowercase, normalize spaces)
        canonical = text.lower().strip().replace(' ', '_')
        
        # Step 3: Hash canonical form for collision resistance
        # This MUST happen before cleaning to preserve distinctions like "A+B" vs "A-B"
        hash_suffix = self._compute_hash(canonical)
        
        # Step 4: Clean for filesystem safety
        clean = self._clean(canonical)
        
        # Step 5: Limit length
        if len(clean) > self.max_length:
            clean = clean[:self.max_length]
        
        return f"{clean}_{hash_suffix}"
    
    def preprocess(self, text: str) -> str:
        """
        Hook for domain-specific preprocessing.
        
        Override this in subclasses to add custom preprocessing like:
        - HTML entity decoding (Wikipedia)
        - Dot-to-underscore conversion (Python modules)
        - Double-colon handling (Rust)
        - Bibtex key parsing (LaTeX)
        
        Args:
            text: Raw input text
        
        Returns:
            Preprocessed text (still may contain special characters)
        """
        return text  # Default: no preprocessing
    
    def _clean(self, text: str) -> str:
        """
        Filesystem-safe character filtering (shared across all normalizers).
        
        Keeps only: alphanumeric, hyphen, underscore
        Collapses multiple underscores to single
        Strips leading/trailing underscores
        
        Args:
            text: Canonical text to clean
        
        Returns:
            Filesystem-safe text
        """
        # Replace all non-alphanumeric (except - and _) with underscore
        clean = re.sub(r'[^a-z0-9\-_]', '_', text)
        # Collapse multiple underscores
        clean = re.sub(r'__+', '_', clean)
        # Strip leading/trailing underscores
        return clean.strip('_')
    
    def _compute_hash(self, text: str) -> str:
        """
        Compute MD5 hash for uniqueness with configurable length.
        
        Always hashes the canonical form (before cleaning) to maintain collision
        resistance. This ensures "A+B" and "A-B" get different identifiers even
        though both clean to "a_b".
        
        Args:
            text: Canonical text to hash
        
        Returns:
            hex hash with length specified by self.hash_length
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:self.hash_length]


class PassthroughNormalizer(LinkNormalizer):
    """
    No-op normalizer that returns links as-is.
    
    Used when source data is already normalized (e.g., Wikipedia markdown files
    created by dump_extractor.py which already normalizes filenames).
    
    Examples:
        >>> normalizer = PassthroughNormalizer()
        >>> normalizer.normalize("already_normalized_5eb63b")
        'already_normalized_5eb63b'
    """
    
    def normalize(self, link: str) -> str:
        """Return the link unchanged."""
        return link


class WikiTitleNormalizer(FilesafeNormalizer):
    """
    Wikipedia-specific normalizer for raw wiki titles.
    
    Handles Wikipedia's specific needs:
    - HTML entity decoding (&amp; → &, &quot; → ", etc.)
    - Preserves uniqueness for titles with special characters
    
    NOTE: Only use this for raw Wikipedia titles. For titles already processed
    by dump_extractor.py, use PassthroughNormalizer since normalization already happened.
    
    Examples:
        >>> normalizer = WikiTitleNormalizer()
        >>> normalizer.normalize("Albert Einstein")
        'albert_einstein_a1b2c3'
        >>> normalizer.normalize("C&amp;C++")  # Decodes HTML entity
        'c_c_4f3d2e'
    """
    
    def preprocess(self, text: str) -> str:
        """Decode HTML entities that appear in Wikipedia titles."""
        return html.unescape(text)


class PythonModuleNormalizer(FilesafeNormalizer):
    """
    Python module/package name normalizer.
    
    Handles Python-specific conventions:
    - Converts dots to underscores (package.module → package_module)
    - Preserves hierarchy information while being filesystem-safe
    
    Used by TheStack and other Python code datasets.
    
    Examples:
        >>> normalizer = PythonModuleNormalizer()
        >>> normalizer.normalize("numpy.array")
        'numpy_array_a1b2c3'
        >>> normalizer.normalize("foo.bar.baz")
        'foo_bar_baz_4f3d2e'
    """
    
    def preprocess(self, text: str) -> str:
        """Convert Python module dots to underscores."""
        return text.replace('.', '_')
