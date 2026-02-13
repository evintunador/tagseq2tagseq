# Language-Agnostic Import Extraction Plan

## Problem Statement

Import/dependency extraction is currently Python-specific with hardcoded patterns and stdlib list:

**data/extractors/link_extractors.py**:
```python
class PythonImportExtractor(LinkExtractor):
    """Extracts Python import statements."""
    
    PATTERNS = [
        re.compile(r'^\s*from\s+([^\s;]+)\s+import', re.MULTILINE),
        re.compile(r'^\s*import\s+([^\s;]+)', re.MULTILINE),
    ]
    
    STDLIB = {
        'os', 'sys', 'json', 're', 'math', 'datetime', ...
    }
```

While we only need Python for now, we should design the abstraction to support future languages without major refactoring.

## Goals

1. Create language-agnostic base class for import extraction
2. Keep Python as the only implemented language for now
3. Design plugin architecture for future language support
4. Make stdlib filtering configurable per language/version
5. Prepare for future: JavaScript, Rust, Java, C++, Go, etc.

## Proposed Solution

### Create Language Plugin System

**File**: `data/extractors/import_extractors/__init__.py` (NEW DIRECTORY)

```python
"""
Language-agnostic import/dependency extraction.

This module provides a plugin system for extracting import statements
from source code in different programming languages. Each language
has its own extractor with language-specific patterns and rules.
"""

from .base import ImportExtractor, ImportInfo
from .python import PythonImportExtractor

__all__ = [
    'ImportExtractor',
    'ImportInfo',
    'PythonImportExtractor',
    'get_extractor_for_language',
]


def get_extractor_for_language(language: str, **kwargs) -> ImportExtractor:
    """
    Factory function to get the appropriate import extractor for a language.
    
    Args:
        language: Language name (e.g., 'python', 'javascript', 'rust')
        **kwargs: Language-specific configuration
    
    Returns:
        ImportExtractor instance for the specified language
    
    Raises:
        ValueError: If language is not supported
    
    Examples:
        >>> extractor = get_extractor_for_language('python', version='3.11')
        >>> imports = extractor.extract_imports(code, filename='example.py')
    """
    language = language.lower()
    
    if language in ('python', 'py'):
        return PythonImportExtractor(**kwargs)
    
    # Future languages:
    # elif language in ('javascript', 'js'):
    #     return JavaScriptImportExtractor(**kwargs)
    # elif language == 'rust':
    #     return RustImportExtractor(**kwargs)
    # elif language == 'java':
    #     return JavaImportExtractor(**kwargs)
    
    else:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages: python"
        )
```

**File**: `data/extractors/import_extractors/base.py` (NEW)

```python
"""Base classes and protocols for import extraction."""

from dataclasses import dataclass
from typing import Protocol, Set, Optional, List
from pathlib import Path


@dataclass
class ImportInfo:
    """
    Information about a single import statement.
    
    Attributes:
        module: The imported module/package name
        line_number: Line number where import appears (1-indexed)
        is_relative: Whether this is a relative import
        is_standard_lib: Whether this imports from standard library
        is_local: Whether this imports from the local project
        raw_statement: The original import statement text
    """
    module: str
    line_number: int
    is_relative: bool = False
    is_standard_lib: bool = False
    is_local: bool = True
    raw_statement: Optional[str] = None


class ImportExtractor(Protocol):
    """
    Protocol for language-specific import extractors.
    
    Each language implementation should provide:
    1. Pattern matching to find import statements
    2. Logic to extract module/package names
    3. Filtering for standard library vs local imports
    4. Resolution of relative imports to absolute paths
    """
    
    def extract_imports(
        self,
        content: str,
        filepath: Optional[str] = None
    ) -> Set[str]:
        """
        Extract import targets from source code.
        
        This is the main interface used by link extraction. It returns
        a simple set of module names that should be linked to.
        
        Args:
            content: Source code content
            filepath: Optional file path for relative import resolution
        
        Returns:
            Set of module names to link to (excludes stdlib, includes only local)
        """
        ...
    
    def extract_imports_detailed(
        self,
        content: str,
        filepath: Optional[str] = None
    ) -> List[ImportInfo]:
        """
        Extract detailed information about all imports.
        
        This provides more information for analysis and debugging.
        
        Args:
            content: Source code content
            filepath: Optional file path for relative import resolution
        
        Returns:
            List of ImportInfo objects with details about each import
        """
        ...
    
    def is_standard_library(self, module: str) -> bool:
        """
        Check if a module is part of the standard library.
        
        Args:
            module: Module name to check
        
        Returns:
            True if module is in standard library
        """
        ...
```

**File**: `data/extractors/import_extractors/python.py` (NEW)

Move and enhance current PythonImportExtractor:

```python
"""Python import extraction."""

import re
from typing import Set, List, Optional
from pathlib import Path

from .base import ImportExtractor, ImportInfo


class PythonImportExtractor(ImportExtractor):
    """
    Extracts Python import statements.
    
    Handles:
    - import module
    - from module import ...
    - from .module import ... (relative imports)
    
    Filters out standard library and popular third-party packages.
    """
    
    # Patterns for different import types
    PATTERNS = [
        # from module import ...
        (re.compile(r'^\s*from\s+([^\s;]+)\s+import', re.MULTILINE), 'from'),
        # import module (possibly with 'as')
        (re.compile(r'^\s*import\s+([^\s;]+?)(?:\s+as\s+\w+)?(?:\s*,|\s*$)', re.MULTILINE), 'import'),
    ]
    
    # Default Python 3.11 standard library
    # This should be loaded from a config file in production
    DEFAULT_STDLIB = {
        'os', 'sys', 'json', 're', 'math', 'datetime', 'time', 'collections',
        'itertools', 'functools', 'typing', 'pathlib', 'subprocess', 'threading',
        'multiprocessing', 'asyncio', 'logging', 'unittest', 'abc', 'argparse',
        'copy', 'enum', 'io', 'pickle', 'random', 'shutil', 'string', 'tempfile',
        'traceback', 'urllib', 'warnings', 'weakref', 'contextlib', 'dataclasses',
        'decimal', 'fractions', 'statistics', 'array', 'bisect', 'heapq', 'queue',
        'struct', 'codecs', 'encodings', 'locale', 'gettext', 'hashlib', 'hmac',
        'secrets', 'token', 'csv', 'configparser', 'tomllib', 'sqlite3',
        'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
        'email', 'smtplib', 'poplib', 'imaplib', 'http', 'urllib',
        'html', 'xml', 'webbrowser', 'socketserver', 'socket', 'ssl',
        'select', 'asynchat', 'asyncore', 'signal', 'mmap',
        'ctypes', 'platform', 'errno', 'pwd', 'grp',
        'pty', 'tty', 'termios', 'fcntl', 'pipes', 'resource',
        'syslog', 'curses', 'getopt', 'optparse',
        'imp', 'importlib', 'zipimport', 'pkgutil', 'modulefinder', 'runpy',
        'ast', 'symtable', 'token', 'keyword', 'tokenize', 'tabnanny', 'pyclbr',
        'py_compile', 'compileall', 'dis', 'pickletools', 'formatter',
        'msilib', 'msvcrt', 'winreg', 'winsound',
        'posix', 'pwd', 'grp', 'crypt', 'termios', 'tty', 'pty', 'fcntl',
        'pipes', 'resource', 'nis', 'syslog',
        'test', 'unittest', 'doctest', 'unittest.mock',
        # Built-in modules (not actual imports but sometimes appear)
        '__future__', '__main__', 'builtins',
    }
    
    # Popular third-party packages that should be filtered
    # This is separate from stdlib but also typically not local
    DEFAULT_THIRD_PARTY = {
        'pytest', 'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn',
        'tensorflow', 'torch', 'keras', 'django', 'flask', 'fastapi',
        'requests', 'aiohttp', 'httpx', 'beautifulsoup4', 'lxml',
        'pillow', 'opencv', 'pyyaml', 'toml', 'click', 'typer',
        'sqlalchemy', 'alembic', 'psycopg2', 'pymongo', 'redis',
        'celery', 'boto3', 'google-cloud', 'azure', 'pydantic',
        'attrs', 'cattrs', 'marshmallow', 'jsonschema',
    }
    
    def __init__(
        self,
        stdlib: Optional[Set[str]] = None,
        third_party: Optional[Set[str]] = None,
        include_third_party: bool = False,
        version: str = "3.11"
    ):
        """
        Initialize Python import extractor.
        
        Args:
            stdlib: Custom set of stdlib module names. If None, uses defaults.
            third_party: Custom set of third-party packages to filter.
            include_third_party: If True, don't filter third-party packages.
            version: Python version (for future version-specific stdlib).
        """
        self.stdlib = stdlib if stdlib is not None else self.DEFAULT_STDLIB.copy()
        self.third_party = third_party if third_party is not None else self.DEFAULT_THIRD_PARTY.copy()
        self.include_third_party = include_third_party
        self.version = version
    
    def extract_imports(
        self,
        content: str,
        filepath: Optional[str] = None
    ) -> Set[str]:
        """
        Extract Python import targets, filtering stdlib and third-party.
        
        Returns only local project imports that should be linked.
        """
        imports = set()
        
        for pattern, import_type in self.PATTERNS:
            matches = pattern.findall(content)
            for match in matches:
                module_name = match.strip()
                
                # Skip standard library
                if self.is_standard_library(module_name):
                    continue
                
                # Skip third-party unless explicitly included
                if not self.include_third_party and self.is_third_party(module_name):
                    continue
                
                # Keep relative imports (they're always local)
                if module_name.startswith('.'):
                    imports.add(module_name)
                else:
                    # For absolute imports, assume local if not filtered above
                    imports.add(module_name)
        
        return imports
    
    def extract_imports_detailed(
        self,
        content: str,
        filepath: Optional[str] = None
    ) -> List[ImportInfo]:
        """
        Extract detailed information about all Python imports.
        
        Includes line numbers and classification.
        """
        imports = []
        
        for line_num, line in enumerate(content.splitlines(), start=1):
            for pattern, import_type in self.PATTERNS:
                match = pattern.search(line)
                if match:
                    module_name = match.group(1).strip()
                    
                    is_relative = module_name.startswith('.')
                    is_stdlib = self.is_standard_library(module_name)
                    is_third_party = self.is_third_party(module_name)
                    is_local = not (is_stdlib or is_third_party)
                    
                    imports.append(ImportInfo(
                        module=module_name,
                        line_number=line_num,
                        is_relative=is_relative,
                        is_standard_lib=is_stdlib,
                        is_local=is_local,
                        raw_statement=line.strip()
                    ))
        
        return imports
    
    def is_standard_library(self, module: str) -> bool:
        """
        Check if module is in Python standard library.
        
        Args:
            module: Module name (might include dots like 'os.path')
        
        Returns:
            True if module is stdlib
        """
        # Strip leading dots from relative imports
        module = module.lstrip('.')
        
        # Get base module (before first dot)
        base_module = module.split('.')[0]
        
        return base_module in self.stdlib
    
    def is_third_party(self, module: str) -> bool:
        """
        Check if module is a known third-party package.
        
        Args:
            module: Module name
        
        Returns:
            True if module is third-party
        """
        module = module.lstrip('.')
        base_module = module.split('.')[0]
        return base_module in self.third_party
    
    @classmethod
    def from_config_file(cls, config_path: Path, **kwargs) -> 'PythonImportExtractor':
        """
        Load stdlib and third-party lists from config file.
        
        Config file should be JSON:
        {
            "stdlib": ["os", "sys", ...],
            "third_party": ["numpy", "pandas", ...]
        }
        
        Args:
            config_path: Path to JSON config file
            **kwargs: Additional arguments for constructor
        
        Returns:
            PythonImportExtractor with loaded config
        """
        import json
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            stdlib=set(config.get('stdlib', [])),
            third_party=set(config.get('third_party', [])),
            **kwargs
        )
```

### Update LinkExtractor to use ImportExtractor

**File**: `data/extractors/link_extractors.py`

Keep the original `PythonImportExtractor` as a wrapper for backwards compatibility:

```python
from typing import Set
from .protocols import LinkExtractor, LinkContext
from .import_extractors import PythonImportExtractor as _PythonImportExtractor


class PythonImportExtractor(LinkExtractor):
    """
    Wrapper around language-agnostic PythonImportExtractor.
    
    This maintains the LinkExtractor interface for graph building
    while delegating to the more detailed import_extractors system.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize with same arguments as import_extractors.PythonImportExtractor.
        
        Args:
            **kwargs: Passed to PythonImportExtractor
        """
        self._impl = _PythonImportExtractor(**kwargs)
    
    def extract_links(self, context: LinkContext) -> Set[str]:
        """
        Extract Python imports as links.
        
        Args:
            context: Context containing document and metadata
        
        Returns:
            Set of imported module names (local only, no stdlib)
        """
        return self._impl.extract_imports(
            content=context.document.content,
            filepath=context.document.metadata.get('filepath')
        )
```

### Add Configuration Files for Python Versions

**File**: `data/extractors/import_extractors/configs/python311_stdlib.json`

```json
{
  "version": "3.11",
  "stdlib": [
    "os", "sys", "json", "re", "math", "datetime", "time",
    ... (full list)
  ],
  "third_party": [
    "numpy", "pandas", "pytest", "django", "flask",
    ... (common packages to filter)
  ]
}
```

Add similar files for Python 3.10, 3.9, etc.

## Usage Examples

### Basic Usage (Current)

```python
from data.extractors.link_extractors import PythonImportExtractor

extractor = PythonImportExtractor()
context = LinkContext(document=doc, source_type="thestack")
imports = extractor.extract_links(context)
```

### Advanced Usage (New)

```python
from data.extractors.import_extractors import PythonImportExtractor

# Custom configuration
extractor = PythonImportExtractor(
    version="3.10",
    include_third_party=True  # Don't filter numpy, pandas, etc.
)

imports = extractor.extract_imports(code, filepath="src/main.py")

# Detailed analysis
import_info = extractor.extract_imports_detailed(code)
for imp in import_info:
    print(f"Line {imp.line_number}: import {imp.module} "
          f"(stdlib={imp.is_standard_lib}, local={imp.is_local})")
```

### Future Language Support

```python
from data.extractors.import_extractors import get_extractor_for_language

# When JavaScript support is added
extractor = get_extractor_for_language('javascript')
imports = extractor.extract_imports(js_code)

# When Rust support is added
extractor = get_extractor_for_language('rust')
imports = extractor.extract_imports(rust_code)
```

## Testing Requirements

### Unit Tests

**test_python_import_extractor.py**:

```python
def test_basic_imports():
    """Test extraction of basic import statements."""
    code = """
import os
import sys
from mypackage import foo
from .relative import bar
"""
    
    extractor = PythonImportExtractor()
    imports = extractor.extract_imports(code)
    
    # Should exclude stdlib (os, sys)
    # Should include local (mypackage, .relative)
    assert 'os' not in imports
    assert 'sys' not in imports
    assert 'mypackage' in imports
    assert '.relative' in imports


def test_stdlib_filtering():
    """Test that stdlib is filtered correctly."""
    code = "import os\nimport json\nimport mymodule"
    
    extractor = PythonImportExtractor()
    imports = extractor.extract_imports(code)
    
    assert imports == {'mymodule'}


def test_custom_stdlib():
    """Test with custom stdlib list."""
    code = "import mylib\nimport otherlib"
    
    # Treat 'mylib' as stdlib
    extractor = PythonImportExtractor(stdlib={'mylib'})
    imports = extractor.extract_imports(code)
    
    assert 'mylib' not in imports
    assert 'otherlib' in imports


def test_detailed_extraction():
    """Test detailed import information."""
    code = """import os
from mypackage import foo
import numpy as np
"""
    
    extractor = PythonImportExtractor()
    imports = extractor.extract_imports_detailed(code)
    
    assert len(imports) == 3
    assert imports[0].module == 'os'
    assert imports[0].is_standard_lib == True
    assert imports[1].module == 'mypackage'
    assert imports[1].is_local == True


def test_config_file_loading():
    """Test loading config from file."""
    import tempfile
    import json
    
    config = {
        "stdlib": ["os", "sys"],
        "third_party": ["numpy"]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    extractor = PythonImportExtractor.from_config_file(config_path)
    assert extractor.is_standard_library('os')
    assert extractor.is_third_party('numpy')
```

## Open Questions

1. **How to keep stdlib lists up-to-date?**
   - Generate from Python's `sys.stdlib_module_names`?
   - Maintain manually per version?
   - Ship with package and update periodically?
   - **Proposed**: Generate automatically during build, ship as JSON configs

2. **Should third-party filtering be opt-in or opt-out?**
   - Current: Opt-out (filter by default)
   - Alternative: Include all imports, let user decide what to link
   - **Proposed**: Keep current behavior (filter by default) but make configurable

3. **How to handle import resolution (e.g., relative imports)?**
   - `.module` needs to be resolved to absolute path
   - Requires knowledge of project structure
   - **Proposed**: For MVP, keep relative imports as-is; resolver is separate concern

4. **How to detect language from file extension?**
   - `.py` → Python
   - `.js` → JavaScript
   - `.rs` → Rust
   - **Proposed**: Add `get_language_from_extension()` utility function

5. **Should we parse AST for more accurate extraction?**
   - Regex is fast but can miss edge cases
   - AST parsing is slower but more accurate
   - **Proposed**: Start with regex, add AST option later if needed

## Implementation Steps

1. **Create directory structure**
   - `data/extractors/import_extractors/`
   - Add `__init__.py`, `base.py`, `python.py`

2. **Implement base classes**
   - `ImportExtractor` protocol
   - `ImportInfo` dataclass
   - `get_extractor_for_language()` factory

3. **Move and enhance Python extractor**
   - Copy current implementation to new location
   - Add detailed extraction
   - Add config file support

4. **Add configuration files**
   - Create `configs/` directory
   - Add Python 3.11 stdlib JSON
   - Add common third-party packages

5. **Update link_extractors.py**
   - Make `PythonImportExtractor` a thin wrapper
   - Maintain backwards compatibility

6. **Tests**
   - Unit tests for new system
   - Integration tests with graph building
   - Verify backwards compatibility

7. **Documentation**
   - Document new import_extractors module
   - Add examples for customization
   - Document how to add new languages

## Success Criteria

- [ ] `import_extractors` module created with proper structure
- [ ] Python extractor moved and enhanced
- [ ] Config files for Python stdlib/third-party
- [ ] Factory function for getting extractors by language
- [ ] Backwards compatibility maintained
- [ ] Tests pass for Python extraction
- [ ] Documentation explains how to add new languages
- [ ] No performance regression

## Future Work

Once the abstraction is in place, adding new languages is straightforward:

1. **JavaScript/TypeScript**
   - `import { foo } from 'bar'`
   - `const x = require('module')`
   - Filter node_modules

2. **Rust**
   - `use std::collections::HashMap;`
   - `use crate::module;`
   - Filter stdlib crates

3. **Java**
   - `import java.util.List;`
   - `import com.company.package.Class;`
   - Filter `java.*`, `javax.*`

4. **Go**
   - `import "fmt"`
   - `import "github.com/user/repo"`
   - Filter stdlib packages

Each just needs:
- Implement `ImportExtractor` protocol
- Define regex patterns or AST parser
- Provide stdlib/standard package list
- Add to factory function
