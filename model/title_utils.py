"""
Title normalization utilities for DS2DS generation.

These utilities handle title normalization and hashing to match the format used
during training data preprocessing (data/pretokenize.py).
"""
import hashlib
import re


def normalize_title(title: str) -> str:
    """
    Normalize a title to match the training data format.
    
    Rules:
    - Lowercase the title
    - Replace spaces with underscores
    - Keep only alphanumeric characters and underscores
    
    Args:
        title: Original title string
        
    Returns:
        Normalized title (without hash suffix)
        
    Examples:
        >>> normalize_title("Python Programming")
        'python_programming'
        >>> normalize_title("C++ Tutorial!")
        'c_tutorial'
    """
    # Lowercase
    normalized = title.lower()
    # Replace spaces with underscores
    normalized = normalized.replace(' ', '_')
    # Keep only alphanumeric and underscores
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    return normalized


def generate_title_hash(title: str) -> str:
    """
    Generate a deterministic 6-character hex hash from a title.
    
    The hash is used to disambiguate titles with identical normalized forms.
    This must match the hashing scheme used during training data preprocessing.
    
    Args:
        title: Original title string (before normalization)
        
    Returns:
        6-character hex hash string
        
    Examples:
        >>> generate_title_hash("Python")
        'a7f8c3'  # Example output
    """
    # Use MD5 hash of the original title (before normalization)
    hash_obj = hashlib.md5(title.encode('utf-8'))
    # Take first 6 characters of hex digest
    return hash_obj.hexdigest()[:6]


def create_filename(title: str) -> str:
    """
    Create a filename from a title using normalization + hash suffix.
    
    This matches the format used in the training corpus: normalized_hash
    
    Args:
        title: Original title string
        
    Returns:
        Filename string in format "normalized_hash"
        
    Examples:
        >>> create_filename("New Topic")
        'new_topic_3a7f2c'  # Example output
    """
    normalized = normalize_title(title)
    hash_suffix = generate_title_hash(title)
    return f"{normalized}_{hash_suffix}"


def strip_hash(title_with_hash: str) -> str:
    """
    Remove the hash suffix from a normalized+hashed title.
    
    Removes the trailing _[0-9a-f]{6} pattern from a title string.
    
    Args:
        title_with_hash: Normalized title with hash suffix
        
    Returns:
        Normalized title without hash suffix
        
    Examples:
        >>> strip_hash("new_topic_3a7f2c")
        'new_topic'
        >>> strip_hash("no_hash_here")
        'no_hash_here'
    """
    # Remove _[0-9a-f]{6} suffix using regex
    return re.sub(r'_[0-9a-f]{6}$', '', title_with_hash)


def verify_title_hash(original_title: str, filename: str) -> bool:
    """
    Verify that a filename's hash matches the expected hash for a given title.
    
    This can be used to confirm that a normalized+hashed filename corresponds
    to a specific original title.
    
    Args:
        original_title: The original title to check
        filename: The normalized_hash filename to verify
        
    Returns:
        True if the hash in filename matches the expected hash for original_title
        
    Examples:
        >>> verify_title_hash("Python", "python_a7f8c3")
        True  # If "python_a7f8c3" is the correct filename for "Python"
    """
    expected_filename = create_filename(original_title)
    return filename == expected_filename
