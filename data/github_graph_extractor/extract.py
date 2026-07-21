#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GitHub Data Extractor: Extract dependency links from GitHub repository code.
Processes Python repositories to extract import relationships as links.
"""
import re
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

# Allow importing data.normalization when run standalone from this directory.
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.normalization import normalize_repo_name, normalize_package_name

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
    """Normalize a Python package/module name to a normed_identifier."""
    return normalize_package_name(package_name)

# ======================================================================
# Tree-sitter import extraction (build-time engine)
# ======================================================================
# The dependency graph shipped as the TheStack dataset is built with this
# extractor. It used to use hand-written regexes (see git history); those
# mangled multi-module lines (``import os, sys`` -> ``{'os,'}``, dropping
# ``sys``) and could not distinguish real imports from ones written inside
# docstrings/strings. We now enumerate import statements with the maintained
# tree-sitter Python grammar — the same engine the Go/Java/Rust/TS builders use,
# and the same one the harness oracle (data/graph_harness) grades against — while
# keeping the OUTPUT contract identical: a set of dotted module-name strings
# (absolute ``a.b.c`` or relative ``.foo`` / ``..pkg``) that the resolver
# (``build_graph_streaming._resolve_import_to_file``) maps to files. From-imports
# emit only the MODULE path (``from a.b import c`` -> ``a.b``), exactly as the
# regex did, so ``module -> file`` resolution and node-id format are unchanged.

class _PyImportParser:
    """Lazily-constructed tree-sitter Python parser for import extraction.

    Constructed once per worker process and reused (parser construction is the
    expensive part). Falls back to ``None`` availability if tree_sitter_python is
    not installed, in which case ``module_names`` raises and the caller uses the
    regex fallback.
    """

    _instance = None

    def __init__(self):
        import tree_sitter_python
        from tree_sitter import Language, Parser
        self._lang = Language(tree_sitter_python.language())
        self._parser = Parser(self._lang)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def module_names(self, content: str) -> Set[str]:
        """Return the set of dotted module names imported by *content*.

        - ``import a.b.c [as x], d.e`` -> ``{'a.b.c', 'd.e'}`` (aliases stripped,
          every comma-separated module captured).
        - ``from a.b import c [as d]`` -> ``{'a.b'}`` (module path only).
        - ``from . import x`` -> ``{'.'}``; ``from ..pkg import y`` -> ``{'..pkg'}``.
        - ``from __future__ import ...`` -> ``{'__future__'}`` (denylisted later).

        Imports inside strings/comments are ignored by the grammar. Returns the
        RAW module names; the stdlib/external denylist is applied by the caller.
        """
        src = content.encode("utf-8", errors="replace")
        tree = self._parser.parse(src)
        names: Set[str] = set()

        def text(node) -> str:
            return src[node.start_byte:node.end_byte].decode("utf-8", "replace")

        def walk(node):
            t = node.type
            if t == "import_statement":
                # import a.b.c [as x], d.e
                for child in node.named_children:
                    if child.type == "dotted_name":
                        names.add(text(child))
                    elif child.type == "aliased_import":
                        nm = child.child_by_field_name("name")
                        if nm is not None and nm.type == "dotted_name":
                            names.add(text(nm))
            elif t == "future_import_statement":
                # from __future__ import annotations  (distinct node type)
                names.add("__future__")
            elif t == "import_from_statement":
                mod = node.child_by_field_name("module_name")
                if mod is not None:
                    if mod.type == "dotted_name":
                        names.add(text(mod))          # absolute: a.b
                    elif mod.type == "relative_import":
                        names.add(text(mod))          # relative: '.', '.foo', '..pkg'
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return names


# Regex fallback (only used if tree_sitter_python is unavailable). Mirrors the
# tree-sitter contract closely: captures module paths, strips ``as`` aliases,
# and splits comma-separated ``import a, b``.
_IMPORT_LINE_RE = re.compile(r'^\s*import\s+(.+?)\s*(?:#.*)?$', re.MULTILINE)
_FROM_LINE_RE = re.compile(r'^\s*from\s+(\.*[\w.]*)\s+import\b', re.MULTILINE)


def _extract_module_names_regex(content: str) -> Set[str]:
    names: Set[str] = set()
    for m in _IMPORT_LINE_RE.finditer(content):
        for item in m.group(1).split(","):
            mod = item.strip().split(" as ")[0].strip()
            if mod and re.fullmatch(r"[\w.]+", mod):
                names.add(mod)
    for m in _FROM_LINE_RE.finditer(content):
        mod = m.group(1).strip()
        if mod:
            names.add(mod)
    return names


def extract_file_imports(content: str, file_path: str, repo_name: str) -> Set[str]:
    """
    Extract all imports from a file's content, focusing on intra-repository imports.
    Returns a set of imported module names that could be other files in the same repository.
    Handles both absolute and relative imports.

    Uses the tree-sitter Python grammar (build-time engine, shared with the other
    languages) to enumerate import statements robustly, falling back to a regex if
    tree_sitter_python is unavailable. The stdlib/external denylist is then applied
    so only plausibly-intra-repo module names survive.
    """
    try:
        module_names = _PyImportParser.get().module_names(content)
    except Exception:
        # tree_sitter unavailable or a parse error: fall back to regex so the
        # build never crashes on a single file.
        module_names = _extract_module_names_regex(content)

    imports = set()
    for module_name in module_names:
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

def normalize_repository_name(repo_name: str) -> str:
    """Normalize a GitHub repository name to a normed_identifier."""
    return normalize_repo_name(repo_name)

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
