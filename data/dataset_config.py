"""
Dataset configuration for linking data preprocessing with model title formatting.

This module provides a unified way to specify dataset characteristics that affect
both data processing (pretokenization) and model behavior (title formatting, linking).
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional
import json


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
        title_strategy: How to extract titles from filepaths ('flat', 'hierarchical')
        hash_length: Number of hex characters in uniqueness hash
        path_separator: Character used for hierarchical paths (if applicable)
        description: Optional description of the dataset
    """
    
    name: str
    title_format: Literal['flat', 'hierarchical', 'colon_separated']
    title_strategy: Literal['flat', 'hierarchical']
    hash_length: int = 6
    path_separator: str = '/'
    description: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
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
        from model.title_formats import (
            FlatTitleFormatter,
            HierarchicalTitleFormatter,
            ColonSeparatedFormatter,
        )
        
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


# Predefined configurations for common datasets
WIKIPEDIA_CONFIG = DatasetConfig(
    name="Wikipedia",
    title_format="flat",
    title_strategy="flat",
    hash_length=6,
    description="Wikipedia articles with flat title structure"
)

GITHUB_CONFIG = DatasetConfig(
    name="GitHub",
    title_format="colon_separated",
    title_strategy="hierarchical",
    hash_length=6,
    path_separator='/',
    description="GitHub code repositories with repo:path structure"
)

DOCUMENTATION_CONFIG = DatasetConfig(
    name="Documentation",
    title_format="hierarchical",
    title_strategy="hierarchical",
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
        'github': GITHUB_CONFIG,
        'documentation': DOCUMENTATION_CONFIG,
    }
    
    dataset_type = dataset_type.lower()
    if dataset_type not in configs:
        raise ValueError(
            f"Unknown dataset_type: {dataset_type}. "
            f"Known types: {list(configs.keys())}"
        )
    
    return configs[dataset_type]
