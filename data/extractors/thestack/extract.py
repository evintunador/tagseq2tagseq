#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TheStack (GitHub) Data Extractor: Extract dependency links from repository code.
Processes Python repositories to extract import relationships as links.
"""
import re
import json
import os
from typing import Dict, List, Set

# Import shared normalizers
from data.extractors.normalization import PythonModuleNormalizer, FilesafeNormalizer

# Module-level normalizer instances (reused for all calls)
_module_normalizer = PythonModuleNormalizer()
_repo_normalizer = FilesafeNormalizer()  # Generic for repo names

# ======================================================================
# Main Processing Pipeline
# ======================================================================

def process_repository_data(repo_data: Dict) -> str:
    """
    Process a single repository's data into a normalized text format.
    For GitHub repositories, we focus on extracting import statements and
    converting them to linkable references.
    """
    content = repo_data.get("content", "")
    repo_name = repo_data.get("repo_name", "")
    file_path = repo_data.get("path", "")

    # Process the content to extract and normalize imports
    processed_content = extract_and_normalize_imports(content)

    # Add repository metadata as front matter
    header = f"# {repo_name}\n# Path: {file_path}\n\n"
    return header + processed_content

def extract_and_normalize_imports(content: str) -> str:
    """
    Extract Python import statements and convert them to normalized links.
    This creates a markdown-like format where imports become [link](normalized_link)
    """
    lines = content.split('\n')
    processed_lines = []

    for line in lines:
        # Find import statements
        import_matches = re.findall(r'\b(?:from\s+(\w+(?:\.\w+)*)|\bimport\s+(\w+(?:\.\w+)*))', line)

        if import_matches:
            # Replace import statements with normalized links
            for match in import_matches:
                # match is a tuple (from_module, import_module), one will be None
                module_name = match[0] or match[1]
                if module_name:
                    # Create a normalized link
                    normalized_link = normalize_package_name(module_name)
                    # Replace the import with a markdown link
                    line = re.sub(
                        r'\b(?:from\s+' + re.escape(module_name) + r'|\bimport\s+' + re.escape(module_name) + r')',
                        f'[{module_name}]({normalized_link})',
                        line
                    )

        processed_lines.append(line)

    return '\n'.join(processed_lines)

def normalize_package_name(package_name: str) -> str:
    """
    Normalize a Python package/module name for use in links.
    
    This is a convenience wrapper for backward compatibility with existing code.
    Uses PythonModuleNormalizer internally.
    
    Args:
        package_name: Python module or package name (e.g., "foo.bar.baz")
    
    Returns:
        Filesystem-safe identifier: {readable_part}_{6char_hash}
    
    Examples:
        >>> normalize_package_name("numpy.array")
        'numpy_array_a1b2c3'
        >>> normalize_package_name("my_package")
        'my_package_4f3d2e'
    """
    return _module_normalizer.normalize(package_name)


def normalize_repository_name(repo_name: str) -> str:
    """
    Normalize a GitHub repository name for use as a node identifier.
    
    This is a convenience wrapper for backward compatibility with existing code.
    Uses generic FilesafeNormalizer internally.
    
    Args:
        repo_name: Repository name (e.g., "owner/repo-name")
    
    Returns:
        Filesystem-safe identifier: {readable_part}_{6char_hash}
    
    Examples:
        >>> normalize_repository_name("numpy/numpy")
        'numpy_numpy_a1b2c3'
        >>> normalize_repository_name("microsoft/TypeScript")
        'microsoft_typescript_4f3d2e'
    """
    return _repo_normalizer.normalize(repo_name)

def extract_file_imports(content: str, file_path: str, repo_name: str) -> Set[str]:
    """
    Extract all imports from a file's content, focusing on intra-repository imports.
    Returns a set of imported module names that could be other files in the same repository.
    Handles both absolute and relative imports.
    """
    imports = set()

    # Patterns for different types of imports
    patterns = [
        # from module import ...
        r'^\s*from\s+([^\s;]+)\s+import',
        # import module
        r'^\s*import\s+([^\s;]+)',
        # from .module import ... (relative imports)
        r'^\s*from\s+(\.[^\s;]+)\s+import',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        for match in matches:
            module_name = match.strip()

            # Skip standard library and external packages
            if _is_potential_repo_file_import(module_name, file_path, repo_name):
                imports.add(module_name)

    return imports

def _is_potential_repo_file_import(module_name: str, file_path: str, repo_name: str) -> bool:
    """
    Determine if an import could be referencing another file in the same repository.
    """
    # Skip obvious standard library/external imports
    standard_libs = {
        'os', 'sys', 'json', 're', 'math', 'datetime', 'time', 'collections',
        'itertools', 'functools', 'typing', 'pathlib', 'subprocess', 'threading',
        'multiprocessing', 'asyncio', 'logging', 'unittest', 'pytest', 'numpy',
        'pandas', 'matplotlib', 'sklearn', 'tensorflow', 'torch', 'django',
        'flask', 'requests', 'urllib', 'http', 'xml', 'html', 'email', 'smtplib'
    }

    # For relative imports (starting with .)
    if module_name.startswith('.'):
        return True

    # For absolute imports, check if it looks like it could be a local module
    # Skip if it's a known external library
    base_module = module_name.split('.')[0]
    if base_module in standard_libs:
        return False

    # If the import looks like it could be part of this repository
    # (not starting with known external packages)
    return True

# ======================================================================
# Helper Functions
# ======================================================================

def is_python_file(file_path: str) -> bool:
    """Check if a file is a Python file based on extension."""
    return file_path.endswith(('.py', '.pyw'))

def is_likely_code_file(file_path: str) -> bool:
    """Check if a file is likely to contain code (not config/docs)."""
    code_extensions = {'.py', '.pyw', '.ipynb'}
    config_files = {'setup.py', 'requirements.txt', 'pyproject.toml', 'setup.cfg', '__init__.py'}

    _, ext = os.path.splitext(file_path)
    filename = os.path.basename(file_path)

    # Include Python files and some config files that might have dependencies
    return ext in code_extensions or filename in config_files
