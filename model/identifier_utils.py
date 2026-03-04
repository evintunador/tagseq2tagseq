"""
Identifier normalization utilities for TS2TS generation.

These utilities handle identifier normalization and hashing to match the format
used during training data preprocessing (data/pretokenize.py).
"""
import hashlib
import re


def normalize_identifier(raw_identifier: str) -> str:
    """
    Normalize an identifier to match the training data format.

    Rules:
    - Lowercase
    - Replace spaces with underscores
    - Keep only alphanumeric characters and underscores

    Args:
        raw_identifier: Original identifier string

    Returns:
        Normalized identifier (without hash suffix)

    Examples:
        >>> normalize_identifier("Python Programming")
        'python_programming'
        >>> normalize_identifier("C++ Tutorial!")
        'c_tutorial'
    """
    normalized = raw_identifier.lower()
    normalized = normalized.replace(' ', '_')
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    return normalized


def generate_identifier_hash(raw_identifier: str) -> str:
    """
    Generate a deterministic 6-character hex hash from an identifier.

    The hash is used to disambiguate identifiers with identical normalized forms.
    This must match the hashing scheme used during training data preprocessing.

    Args:
        raw_identifier: Original identifier string (before normalization)

    Returns:
        6-character hex hash string

    Examples:
        >>> generate_identifier_hash("Python")
        'a7f8c3'  # Example output
    """
    hash_obj = hashlib.md5(raw_identifier.encode('utf-8'))
    return hash_obj.hexdigest()[:6]


def create_normed_identifier(raw_identifier: str) -> str:
    """
    Create a normed_identifier from a raw identifier: normalized form + hash suffix.

    This matches the corpus storage format: normalized_hash

    Args:
        raw_identifier: Original identifier string

    Returns:
        normed_identifier string in format "normalized_hash"

    Examples:
        >>> create_normed_identifier("New Topic")
        'new_topic_3a7f2c'  # Example output
    """
    normalized = normalize_identifier(raw_identifier)
    hash_suffix = generate_identifier_hash(raw_identifier)
    return f"{normalized}_{hash_suffix}"


def strip_hash(normed_identifier: str) -> str:
    """
    Remove the hash suffix from a normalized+hashed identifier.

    Removes the trailing _[0-9a-f]{6} pattern.

    Args:
        normed_identifier: Normalized identifier with hash suffix

    Returns:
        Normalized identifier without hash suffix

    Examples:
        >>> strip_hash("new_topic_3a7f2c")
        'new_topic'
        >>> strip_hash("no_hash_here")
        'no_hash_here'
    """
    return re.sub(r'_[0-9a-f]{6}$', '', normed_identifier)


def verify_identifier_hash(raw_identifier: str, normed_identifier: str) -> bool:
    """
    Verify that a normed_identifier's hash matches the expected hash for a raw_identifier.

    Args:
        raw_identifier: The original identifier to check
        normed_identifier: The normalized_hash identifier to verify

    Returns:
        True if the hash in normed_identifier matches the expected hash for raw_identifier

    Examples:
        >>> verify_identifier_hash("Python", "python_a7f8c3")
        True  # If "python_a7f8c3" is the correct normed_identifier for "Python"
    """
    return normed_identifier == create_normed_identifier(raw_identifier)
