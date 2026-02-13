# LaTeX Citation Support Plan (Future Work)

## Status

**This is a FUTURE feature.** This plan outlines what will be needed when we add LaTeX support, but it is NOT to be implemented now. This document exists to:

1. Keep LaTeX requirements in mind during current architecture decisions
2. Ensure abstractions we build now will support LaTeX later
3. Provide a roadmap when we're ready to add LaTeX

## Overview

LaTeX support will enable the model to:
- Process academic papers and documents
- Extract citation relationships (`\cite{key}`, `\citep{key}`, etc.)
- Link between papers via citations
- Potentially handle includes (`\input{file}`, `\include{chapter}`)

## Current Preparation

### Already Planned in DatasetConfig

**data/dataset_config.py** already mentions LaTeX:

```python
link_format: Literal['markdown', 'python_import', 'latex_cite']  # latex_cite mentioned but not implemented
```

### Predefined Config Exists

```python
LATEX_CONFIG = DatasetConfig(
    name="LaTeX",
    title_format="hierarchical",
    link_format="latex_cite",  # NOT IMPLEMENTED YET
    file_extension=".tex",
    hash_length=6,
    path_separator='/',
    description="LaTeX documents with citations (future support)"
)
```

## What Needs to Be Implemented

### 1. LaTeX Link Extractor

**File**: `data/extractors/link_extractors.py` (ADD NEW CLASS)

```python
class LaTeXCiteExtractor(LinkExtractor):
    """
    Extracts LaTeX citation keys from \cite commands.
    
    Handles:
    - \cite{key}
    - \citep{key}
    - \citet{key}
    - \cite{key1,key2,key3}  # Multiple citations
    - \cite[prefix]{key}     # With optional prefix
    - \cite[prefix][suffix]{key}  # With prefix and suffix
    - \nocite{key}
    
    Also optionally handles:
    - \bibliography{bibfile}  # Link to .bib files
    - \input{file}            # File includes
    - \include{chapter}       # Chapter includes
    """
    
    # Citation commands to extract
    CITE_PATTERNS = [
        # \cite{key}
        re.compile(r'\\cite\{([^}]+)\}'),
        # \citep{key}, \citet{key}, etc.
        re.compile(r'\\cite[pt]\{([^}]+)\}'),
        # \cite[prefix]{key}
        re.compile(r'\\cite\[[^\]]*\]\{([^}]+)\}'),
        # \cite[prefix][suffix]{key}
        re.compile(r'\\cite\[[^\]]*\]\[[^\]]*\]\{([^}]+)\}'),
        # \nocite{key}
        re.compile(r'\\nocite\{([^}]+)\}'),
    ]
    
    # File includes (optional)
    INCLUDE_PATTERNS = [
        re.compile(r'\\input\{([^}]+)\}'),
        re.compile(r'\\include\{([^}]+)\}'),
    ]
    
    def __init__(
        self,
        extract_includes: bool = False,
        extract_bibliography: bool = False
    ):
        """
        Args:
            extract_includes: If True, extract \input and \include as links
            extract_bibliography: If True, extract \bibliography files as links
        """
        self.extract_includes = extract_includes
        self.extract_bibliography = extract_bibliography
    
    def extract_links(self, context: LinkContext) -> Set[str]:
        """Extract citation keys from LaTeX content."""
        citations = set()
        
        # Extract citation keys
        for pattern in self.CITE_PATTERNS:
            matches = pattern.finditer(context.document.content)
            for match in matches:
                # Citation can be comma-separated: \cite{key1,key2,key3}
                keys = match.group(1).split(',')
                for key in keys:
                    key = key.strip()
                    if key:
                        citations.add(key)
        
        # Optionally extract includes
        if self.extract_includes:
            for pattern in self.INCLUDE_PATTERNS:
                matches = pattern.finditer(context.document.content)
                for match in matches:
                    filename = match.group(1).strip()
                    # Add .tex extension if not present
                    if not filename.endswith('.tex'):
                        filename = filename + '.tex'
                    citations.add(filename)
        
        return citations
```

### 2. LaTeX Normalizer

**File**: `data/extractors/normalization.py` (ADD NEW CLASS)

```python
class LaTeXCiteNormalizer(FilesafeNormalizer):
    """
    Normalizer for LaTeX citation keys.
    
    BibTeX keys have conventions:
    - Often format: Author2023word
    - Can contain: letters, numbers, hyphens, underscores
    - Sometimes colons: Smith:2023:ML
    
    Examples:
        >>> normalizer = LaTeXCiteNormalizer()
        >>> normalizer.normalize("Smith2023:ML")
        'smith2023_ml_a1b2c3'
        >>> normalizer.normalize("Author:2023:DeepLearning")
        'author_2023_deeplearning_d4e5f6'
    """
    
    def preprocess(self, text: str) -> str:
        """
        LaTeX-specific preprocessing.
        
        - Convert colons to underscores (common in BibTeX keys)
        - Keep hyphens initially (will be converted by parent class)
        """
        # Convert colons to underscores
        text = text.replace(':', '_')
        return text
```

### 3. LaTeX Tokenized Link Detector

**File**: `model/graph_traversal/link_detectors.py` (ADD NEW CLASS)

```python
class LaTeXCiteDetector(TokenizedLinkDetector):
    """
    Detects LaTeX citation commands in tokenized content.
    
    Looks for token sequences matching:
    - '\cite' token (or '\citep', '\citet')
    - '{' token
    - Citation key tokens
    - '}' token
    
    More complex than markdown because:
    - Multiple citation commands (\cite, \citep, \citet, etc.)
    - Multiple keys in one command: \cite{key1,key2}
    - Optional arguments: \cite[prefix]{key}
    """
    
    uses_outgoing_titles = True  # Like Python, we'll use graph's resolution
    
    def __init__(
        self,
        tokenizer_config: TokenizerConfig,
        cite_variants: Optional[List[str]] = None
    ):
        """
        Args:
            tokenizer_config: Configuration containing token IDs
            cite_variants: Citation commands to detect (default: ['cite', 'citep', 'citet'])
        """
        if cite_variants is None:
            cite_variants = ['cite', 'citep', 'citet', 'nocite']
        
        # We need to discover token IDs for LaTeX commands
        # This is complex because '\cite' might be multiple tokens
        # Likely: '\' + 'cite'
        self.tokenizer_config = tokenizer_config
        self.cite_variants = cite_variants
        
        # TODO: Discover token IDs for LaTeX commands
        # This requires tokenizing "\cite", "\citep", etc.
        # and storing the token sequences
        
        logger.warning("LaTeXCiteDetector is not fully implemented")
    
    def detect_links(
        self,
        input_ids: torch.Tensor,
        tokenizer_decode_fn: Callable[[List[int]], str]
    ) -> List[LinkInfo]:
        """
        Detect LaTeX citation positions.
        
        For MVP, this would need to:
        1. Find token sequences matching \cite{...}
        2. Extract the range of tokens inside braces
        3. Decode to get citation key
        4. Match against doc_spans.outgoing_titles
        
        This is MORE complex than Python or Markdown because:
        - LaTeX commands are likely multi-token
        - Need to handle optional arguments: \cite[pre]{key}
        - Need to handle multiple keys: \cite{key1,key2}
        """
        raise NotImplementedError(
            "LaTeX citation detection in tokenized content is not yet implemented. "
            "This requires detecting multi-token sequences like '\\cite{...}' "
            "and parsing optional arguments."
        )
```

### 4. BibTeX Parser (Optional)

If we want to link TO papers (not just citations), we need to parse `.bib` files:

**File**: `data/extractors/bibtex_parser.py` (NEW)

```python
"""
BibTeX parser for extracting citation metadata.

This allows building a graph where:
- Papers are nodes
- Citations create edges
- Paper metadata (title, author, year) is stored

Optional feature for richer dataset.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BibEntry:
    """A single BibTeX entry."""
    key: str
    entry_type: str  # article, book, inproceedings, etc.
    fields: Dict[str, str]  # title, author, year, etc.


class BibTeXParser:
    """
    Simple BibTeX parser.
    
    Extracts citation keys and metadata from .bib files.
    """
    
    # Pattern: @article{key, fields...}
    ENTRY_PATTERN = re.compile(
        r'@(\w+)\s*\{\s*([^,]+)\s*,(.*?)\n\}',
        re.DOTALL | re.IGNORECASE
    )
    
    def parse(self, content: str) -> List[BibEntry]:
        """
        Parse BibTeX content into entries.
        
        Args:
            content: .bib file content
        
        Returns:
            List of BibEntry objects
        """
        entries = []
        
        for match in self.ENTRY_PATTERN.finditer(content):
            entry_type = match.group(1).lower()
            key = match.group(2).strip()
            fields_str = match.group(3)
            
            fields = self._parse_fields(fields_str)
            
            entries.append(BibEntry(
                key=key,
                entry_type=entry_type,
                fields=fields
            ))
        
        return entries
    
    def _parse_fields(self, fields_str: str) -> Dict[str, str]:
        """Parse field = {value} pairs."""
        fields = {}
        
        # Pattern: field = {value} or field = "value"
        field_pattern = re.compile(r'(\w+)\s*=\s*[{"](.*?)[}"]', re.DOTALL)
        
        for match in field_pattern.finditer(fields_str):
            field_name = match.group(1).lower()
            field_value = match.group(2).strip()
            fields[field_name] = field_value
        
        return fields
```

## Architecture Considerations

### How Citations Differ from Code Imports

| Aspect | Code Imports | LaTeX Citations |
|--------|--------------|-----------------|
| Target resolution | Filesystem (module.py) | BibTeX key (abstract) |
| Targets exist | Yes (must have file) | Maybe (cite without bib entry) |
| Multiple per statement | Rare | Common (\cite{a,b,c}) |
| Optional args | No | Yes (\cite[prefix]{key}) |
| Bidirectional | Yes (imports create deps) | Sort of (citations are references) |

### Graph Structure for Papers

Two approaches:

**Approach 1: LaTeX files as nodes**
- Nodes are .tex files
- Edges are citations between files
- Citation keys resolved via .bib files
- Simpler but less semantic

**Approach 2: Papers as nodes**
- Nodes are papers (from .bib entries)
- .tex files reference papers
- Edges are citation relationships
- More semantic but more complex

**Recommendation**: Start with Approach 1, can migrate to Approach 2 later.

## Dataset Preparation

### Example Dataset Structure

```
papers/
├── paper1.tex
├── paper2.tex
├── paper3.tex
└── references.bib
```

### Graph Building Process

1. **Extract .tex files**
   - Use `FileSource` with `.tex` extension
   - Extract citation keys with `LaTeXCiteExtractor`

2. **Parse .bib file**
   - Extract citation keys and metadata
   - Create mapping: key → paper title/identifier

3. **Build graph**
   - Nodes: .tex files (or papers if using Approach 2)
   - Edges: Citations between papers
   - Normalize citation keys with `LaTeXCiteNormalizer`

4. **Pretokenize**
   - Hash stripping needs to handle LaTeX format:
   - `\cite{key_abc123}` → `\cite{key}`

## Testing Strategy

When implementing, test with:

1. **Simple LaTeX document**
   - Single citation: `\cite{Smith2023}`
   - Multiple citations: `\cite{Smith2023,Jones2022}`
   - Citation with prefix: `\cite[see][]{Smith2023}`

2. **Real paper dataset**
   - arXiv papers with .tex source
   - Papers with .bib files
   - Multi-file papers (\include, \input)

3. **Edge cases**
   - Missing citations (cited but not in .bib)
   - Unused entries (in .bib but not cited)
   - Circular citations
   - Self-citations

## Integration with Current System

### What Already Works

✅ `DatasetConfig` has `link_format='latex_cite'`
✅ `LATEX_CONFIG` predefined
✅ `file_extension='.tex'` supported by `FileSource`
✅ Normalization system can handle LaTeX keys

### What Needs Implementation

❌ `LaTeXCiteExtractor` class
❌ `LaTeXCiteNormalizer` class (optional, `FilesafeNormalizer` might work)
❌ `LaTeXCiteDetector` for tokenized detection
❌ BibTeX parser (if using Approach 2)
❌ Hash stripping for LaTeX format in pretokenize
❌ Tests for LaTeX support

### Estimated Effort

- **LaTeX Link Extractor**: 4-6 hours
  - Regex patterns for citations
  - Handling optional arguments
  - Multiple citation support
  - Tests

- **LaTeX Normalizer**: 2-4 hours
  - Preprocessing logic
  - Tests
  - Integration with existing system

- **Tokenized Detector**: 8-12 hours
  - Multi-token sequence detection
  - Handling `\cite` variants
  - Optional argument parsing
  - Tests

- **BibTeX Parser**: 4-8 hours (OPTIONAL)
  - Entry parsing
  - Field extraction
  - Tests

- **Integration & Testing**: 8-12 hours
  - Hash stripping for LaTeX
  - End-to-end tests
  - Dataset preparation
  - Documentation

**Total**: ~26-42 hours (without BibTeX parser)

## Open Questions for Future Implementation

1. **Should we parse .bib files or just use citation keys as-is?**
   - Parsing allows richer metadata
   - Not parsing is simpler
   - Can we infer paper identity from just citation keys?

2. **How to handle multi-file LaTeX projects?**
   - Main.tex includes chapter1.tex, chapter2.tex
   - Do we treat each file separately or merge?
   - How does this affect citation resolution?

3. **What about arxiv identifiers?**
   - Many papers cite by arXiv ID: `\cite{arxiv:2301.12345}`
   - Should we resolve these to actual papers?
   - Or treat as opaque keys?

4. **Do we need LaTeX → Markdown conversion?**
   - For training, do we convert .tex to .md?
   - Or train on raw .tex?
   - Markdown might be cleaner for the model

5. **How to handle LaTeX math?**
   - Math notation is important in papers
   - Do we keep `$$...$$` or convert to text?
   - Special tokens for math?

## Success Criteria (When Implemented)

- [ ] Can extract citations from .tex files
- [ ] Citations link to target papers/keys
- [ ] Graph built from citation relationships
- [ ] Pretokenization cleans citation hashes
- [ ] Tokenized detector finds citations during training
- [ ] Tests pass for LaTeX support
- [ ] Documentation explains LaTeX dataset preparation
- [ ] Example dataset demonstrates full pipeline

## Related Work

Consider looking at:
- **arXiv bulk data**: LaTeX source available
- **S2ORC**: Semantic Scholar Open Research Corpus (has citations)
- **CiteSeer**: Citation network dataset
- **Paper citation graphs**: Many existing datasets

These could provide ready-made datasets once LaTeX support is implemented.
