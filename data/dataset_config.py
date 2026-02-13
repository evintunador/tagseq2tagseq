"""
Dataset configuration for linking data preprocessing with model title formatting.

This module provides a unified way to specify dataset characteristics that affect
both data processing (pretokenization) and model behavior (title formatting, linking).
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Literal, Optional, Any, Type
import json

from data.extractors.normalization import (
    FilesafeNormalizer,
    PassthroughNormalizer,
    WikiTitleNormalizer,
    PythonModuleNormalizer,
    LinkNormalizer,
)
from model.title_formats import (
    FlatTitleFormatter,
    HierarchicalTitleFormatter,
    ColonSeparatedFormatter,
)


# Normalizer registry - add new normalizers here
NORMALIZER_MAP: dict[str, Type[LinkNormalizer]] = {
    'filesafe': FilesafeNormalizer,
    'passthrough': PassthroughNormalizer,
    'wiki': WikiTitleNormalizer,
    'python_module': PythonModuleNormalizer,
}


@dataclass
class DatasetConfig:
    """
    Configuration that defines dataset-specific conventions.
    
    This config should be consistent between:
    1. Data preprocessing (pretokenize.py)
    2. Model training/generation (title_formats.py)
    3. Graph building (extractors)
    
    Attributes:
        name: Human-readable dataset name
        title_format: Type of title formatting ('flat', 'hierarchical', 'colon_separated')
        link_format: Type of link in content ('markdown', 'python_import', 'latex_cite')
        normalizer_type: Type of normalizer to use ('filesafe', 'passthrough', 'wiki', 'python_module')
        hash_length: Number of hex characters in uniqueness hash
        path_separator: Character used for hierarchical paths (if applicable)
        description: Optional description of the dataset
        _normalizer: Cached normalizer instance (not serialized)
    """
    
    name: str
    title_format: Literal['flat', 'hierarchical', 'colon_separated']
    link_format: Literal['markdown', 'python_import'] = 'markdown'
    normalizer_type: Literal['filesafe', 'passthrough', 'wiki', 'python_module'] = 'filesafe'
    hash_length: int = 6
    path_separator: str = '/'
    description: Optional[str] = None
    _normalizer: Optional[Any] = field(default=None, init=False, repr=False, compare=False)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate hash_length
        if self.hash_length < 1 or self.hash_length > 32:
            raise ValueError(f"hash_length must be between 1 and 32, got {self.hash_length}")
        
        # Validate normalizer compatibility
        if self.normalizer_type == 'passthrough' and self.hash_length != 6:
            # PassthroughNormalizer assumes identifiers are already normalized with hashes
            # If using passthrough, we can't validate hash_length matches, so warn
            import logging
            logging.getLogger(__name__).warning(
                f"Using PassthroughNormalizer with hash_length={self.hash_length}. "
                "Ensure source data uses the same hash length."
            )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization (excludes _normalizer)."""
        data = asdict(self)
        # Remove the cached normalizer instance
        data.pop('_normalizer', None)
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DatasetConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'DatasetConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_formatter(self):
        """
        Create the appropriate TitleFormatter for this dataset.
        
        Returns:
            TitleFormatter instance configured for this dataset
        """
        if self.title_format == 'flat':
            return FlatTitleFormatter(hash_length=self.hash_length)
        elif self.title_format == 'hierarchical':
            return HierarchicalTitleFormatter(
                hash_length=self.hash_length,
                separator=self.path_separator
            )
        elif self.title_format == 'colon_separated':
            return ColonSeparatedFormatter(hash_length=self.hash_length)
        else:
            raise ValueError(f"Unknown title_format: {self.title_format}")
    
    def get_normalizer(self) -> LinkNormalizer:
        """
        Get the appropriate LinkNormalizer for this dataset.
        
        Caches the normalizer instance to avoid recreating it repeatedly.
        The normalizer is configured with the dataset's hash_length.
        
        Returns:
            LinkNormalizer instance configured for this dataset
        """
        if self._normalizer is not None:
            return self._normalizer
        
        # Look up normalizer class from registry
        normalizer_class = NORMALIZER_MAP.get(self.normalizer_type)
        if normalizer_class is None:
            raise ValueError(
                f"Unknown normalizer_type: {self.normalizer_type}. "
                f"Available types: {list(NORMALIZER_MAP.keys())}"
            )
        
        # Calculate max_length based on hash_length
        # Formula: 200 (total limit) - 1 (underscore) - hash_length = max_length
        max_length = 200 - 1 - self.hash_length
        
        # Instantiate normalizer with appropriate parameters
        if normalizer_class is PassthroughNormalizer:
            self._normalizer = normalizer_class()
        else:
            # FilesafeNormalizer and its subclasses take max_length and hash_length
            self._normalizer = normalizer_class(
                max_length=max_length,
                hash_length=self.hash_length
            )
        
        return self._normalizer


# Predefined configurations for common datasets
WIKIPEDIA_CONFIG = DatasetConfig(
    name="Wikipedia",
    title_format="flat",
    normalizer_type="passthrough",  # Wikipedia dump extractor pre-normalizes
    hash_length=6,
    description="Wikipedia articles with flat title structure"
)

THESTACK_CONFIG = DatasetConfig(
    name="TheStack",
    title_format="colon_separated",
    link_format="python_import",
    normalizer_type="python_module",
    hash_length=6,
    path_separator='/',
    description="TheStack code repositories with repo:path structure"
)

DOCUMENTATION_CONFIG = DatasetConfig(
    name="Documentation",
    title_format="hierarchical",
    normalizer_type="filesafe",
    hash_length=6,
    path_separator='/',
    description="Documentation with hierarchical path structure"
)


def save_config_to_pretokenized_dir(config: DatasetConfig, pretokenized_dir: Path) -> None:
    """
    Save dataset config alongside pretokenized data.
    
    This allows the model to automatically load the correct title formatter
    when loading a pretokenized dataset.
    
    Args:
        config: Dataset configuration
        pretokenized_dir: Directory containing pretokenized data
    """
    config_path = pretokenized_dir / "dataset_config.json"
    config.save(config_path)


def load_config_from_pretokenized_dir(pretokenized_dir: Path) -> DatasetConfig:
    """
    Load dataset config from pretokenized directory.
    
    Args:
        pretokenized_dir: Directory containing pretokenized data
        
    Returns:
        DatasetConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = pretokenized_dir / "dataset_config.json"
    return DatasetConfig.load(config_path)


def get_config_for_dataset_type(dataset_type: str) -> DatasetConfig:
    """
    Get predefined config for a known dataset type.
    
    Args:
        dataset_type: One of 'wikipedia', 'github', 'documentation'
        
    Returns:
        DatasetConfig instance
        
    Raises:
        ValueError: If dataset_type is not recognized
    """
    configs = {
        'wikipedia': WIKIPEDIA_CONFIG,
        'thestack': THESTACK_CONFIG,
        'documentation': DOCUMENTATION_CONFIG,
    }
    
    dataset_type = dataset_type.lower()
    if dataset_type not in configs:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            f"Known types: {list(configs.keys())}"
        )
    
    return configs[dataset_type]
