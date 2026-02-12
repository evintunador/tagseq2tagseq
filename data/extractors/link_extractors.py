"""
Link extraction implementations for different document types.

Extracts raw link targets from document content:
- MarkdownLinkExtractor: Extracts [text](target) links
- PythonImportExtractor: Extracts Python import statements
"""
import re
from typing import Set
from .protocols import LinkExtractor, LinkContext


class MarkdownLinkExtractor(LinkExtractor):
    """
    Extracts [text](target) markdown links.
    Skips image links ![text](target).
    
    Based on wiki_graph_extractor/build_graph.py::extract_links_worker().
    """
    
    # Pattern from build_graph.py line 46
    # Matches [text](target) but not ![text](target)
    # The negative lookbehind (?<!\!) ensures the [ is not preceded by !
    LINK_PATTERN = re.compile(r'(?<!\!)\[([^\]]*?)\]\((.*?)\)')
    
    def extract_links(self, context: LinkContext) -> Set[str]:
        """
        Extract markdown link targets.
        
        Args:
            context: Context containing document and metadata
        
        Returns:
            Set of link targets (the part in parentheses)
        """
        from urllib.parse import unquote
        
        # Find all matches - group(2) contains the URL part
        matches = self.LINK_PATTERN.finditer(context.document.content)
        links = [match.group(2) for match in matches]
        
        # Unquote any URL encoding
        # This handles cases where links might have encoded characters
        return {unquote(link) for link in links}


class PythonImportExtractor(LinkExtractor):
    """
    Extracts Python import statements.
    
    Based on github_graph_extractor/extract.py::extract_file_imports().
    
    Handles:
    - import module
    - from module import ...
    - from .module import ... (relative imports)
    
    Filters out standard library imports.
    """
    
    # Patterns for different import types
    PATTERNS = [
        # from module import ...
        re.compile(r'^\s*from\s+([^\s;]+)\s+import', re.MULTILINE),
        # import module
        re.compile(r'^\s*import\s+([^\s;]+)', re.MULTILINE),
        # from .module import ... (relative imports)
        re.compile(r'^\s*from\s+(\.[^\s;]+)\s+import', re.MULTILINE),
    ]
    
    # Standard library modules to skip (from extract.py lines 114-120)
    # We filter these out because we only want intra-repository dependencies
    STDLIB = {
        'os', 'sys', 'json', 're', 'math', 'datetime', 'time', 'collections',
        'itertools', 'functools', 'typing', 'pathlib', 'subprocess', 'threading',
        'multiprocessing', 'asyncio', 'logging', 'unittest', 'pytest', 'numpy',
        'pandas', 'matplotlib', 'sklearn', 'tensorflow', 'torch', 'django',
        'flask', 'requests', 'urllib', 'http', 'xml', 'html', 'email', 'smtplib'
    }
    
    def extract_links(self, context: LinkContext) -> Set[str]:
        """
        Extract Python imports.
        
        Args:
            context: Context containing document and metadata
        
        Returns:
            Set of imported module names
        """
        imports = set()
        
        for pattern in self.PATTERNS:
            matches = pattern.findall(context.document.content)
            for match in matches:
                module_name = match.strip()
                
                # Skip standard library
                if self._is_stdlib(module_name):
                    continue
                
                # Keep relative imports (start with .)
                if module_name.startswith('.'):
                    imports.add(module_name)
                else:
                    # For absolute imports, check if it's not stdlib
                    imports.add(module_name)
        
        return imports
    
    def _is_stdlib(self, module_name: str) -> bool:
        """
        Check if module is standard library.
        
        Args:
            module_name: Module name to check
        
        Returns:
            True if module is in standard library
        """
        # Get base module name (before first dot)
        base_module = module_name.lstrip('.').split('.')[0]
        return base_module in self.STDLIB
