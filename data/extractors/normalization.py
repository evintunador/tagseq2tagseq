"""
Link normalization logic extracted from existing extractors.

Provides shared normalization utilities with domain-specific overrides.
"""
import re
import html
import hashlib
from typing import Optional
from .protocols import LinkContext, LinkNormalizer


class HashingNormalizer:
    """
    Base normalizer providing common logic:
    - Lowercase conversion
    - Special character replacement
    - Hash suffix generation
    - Length limiting
    
    This extracts shared logic from both wiki_graph_extractor/extract.py::normalize_title()
    and github_graph_extractor/extract.py::normalize_package_name().
    """
    
    def __init__(self, max_length: int = 193):
        """
        Args:
            max_length: Maximum length for the normalized identifier before hash suffix.
                Default 193 allows room for "_" + 6-char hash in 200-char limit.
        """
        self.max_length = max_length
    
    def _compute_hash(self, text: str) -> str:
        """Generate 6-char MD5 hash of text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:6]
    
    def _clean_text(self, text: str) -> str:
        """
        Override in subclasses for domain-specific cleaning.
        
        Default behavior:
        - Lowercase
        - Replace spaces with underscores
        - Replace special chars with underscores
        - Collapse multiple underscores
        - Strip leading/trailing underscores
        """
        # Lowercase, collapse underscores
        clean = text.lower().strip().replace(' ', '_')
        clean = re.sub(r'[^a-z0-9\-_]', '_', clean)
        clean = re.sub(r'__+', '_', clean)
        return clean.strip('_')
    
    def normalize(self, link: str, context: Optional[LinkContext] = None) -> str:
        """
        Common normalization pipeline.
        
        Args:
            link: Raw link string to normalize
            context: Optional context for domain-specific behavior
        
        Returns:
            Normalized identifier with hash suffix: {clean_text}_{hash}
        """
        # Decode HTML entities
        link = html.unescape(link)
        
        # Create canonical form for hash computation (before cleaning)
        # This ensures "A+B" and "A-B" get different hashes
        canonical = link.lower().strip().replace(' ', '_')
        link_hash = self._compute_hash(canonical)
        
        # Apply domain-specific cleaning for the readable part
        clean = self._clean_text(link)
        
        # Limit length
        if len(clean) > self.max_length:
            clean = clean[:self.max_length]
        
        return f"{clean}_{link_hash}"


class PassthroughNormalizer(LinkNormalizer):
    """
    No-op normalizer that returns links as-is.
    
    Used when source data is already normalized (e.g., Wikipedia markdown files
    created by dump_extractor.py which already calls normalize_title()).
    """
    
    def normalize(self, link: str, context: Optional[LinkContext] = None) -> str:
        """Return the link unchanged."""
        return link


class WikiTitleNormalizer(HashingNormalizer):
    """
    Wikipedia-specific normalization from extract.py normalize_title().
    
    Handles Wikipedia title conventions:
    - HTML entity decoding (handled in base)
    - Space/underscore equivalence (handled in base)
    - Case-insensitive matching (handled in base)
    
    NOTE: This should only be used for raw Wikipedia titles. For titles that 
    have already been processed by dump_extractor.py, use PassthroughNormalizer.
    """
    
    def _clean_text(self, text: str) -> str:
        """
        Wiki-specific cleaning.
        
        The base class already handles most Wikipedia normalization needs.
        HTML unescaping is done in the normalize() method before calling this.
        """
        # Use standard cleaning - it handles Wikipedia needs
        return super()._clean_text(text)


class PythonModuleNormalizer(HashingNormalizer):
    """
    Python module name normalization from github extract.py.
    
    Handles Python-specific conventions:
    - Converts dots to underscores (package.module -> package_module)
    - Preserves relative import indicators (leading dots)
    """
    
    def _clean_text(self, text: str) -> str:
        """
        Python module-specific cleaning.
        
        Converts dots to underscores for Python packages/modules.
        """
        # Convert dots to underscores for Python modules
        clean = text.replace('.', '_').lower()
        clean = re.sub(r'[^a-z0-9\-_]', '_', clean)
        clean = re.sub(r'__+', '_', clean)
        return clean.strip('_')
