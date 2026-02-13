# File Extension Abstraction Plan

## Problem Statement

File extensions are hardcoded in multiple places:

1. **data/extractors/sources.py:40** - `.md` extension hardcoded in MarkdownFileSource
2. **data/extractors/wikipedia/dump_extractor.py:418** - Generates `.md` files
3. Assumption that all text files are markdown

This prevents supporting:
- Python files (`.py`) for TheStack dataset
- LaTeX files (`.tex`) for future academic paper support
- Mixed datasets with multiple file types

## Goals

1. Make file extensions configurable per dataset
2. Support multiple file types in the same dataset
3. Maintain backwards compatibility with Wikipedia `.md` files
4. Set up infrastructure for future LaTeX support

## Proposed Solution

### Add extension field to DatasetConfig

**File**: `data/dataset_config.py`

```python
@dataclass
class DatasetConfig:
    """Configuration that defines dataset-specific conventions."""
    
    name: str
    title_format: Literal['flat', 'hierarchical', 'colon_separated']
    link_format: Literal['markdown', 'python_import'] = 'markdown'
    file_extension: str = '.md'  # NEW FIELD
    hash_length: int = 6
    path_separator: str = '/'
    description: Optional[str] = None
```

Update predefined configs:

```python
WIKIPEDIA_CONFIG = DatasetConfig(
    name="Wikipedia",
    title_format="flat",
    file_extension=".md",
    hash_length=6,
    description="Wikipedia articles with flat title structure"
)

GITHUB_CONFIG = DatasetConfig(
    name="GitHub",
    title_format="colon_separated",
    link_format="python_import",
    file_extension=".py",  # Python files
    hash_length=6,
    path_separator='/',
    description="GitHub code repositories with repo:path structure"
)

LATEX_CONFIG = DatasetConfig(
    name="LaTeX",
    title_format="hierarchical",
    link_format="latex_cite",  # Will be implemented later
    file_extension=".tex",
    hash_length=6,
    path_separator='/',
    description="LaTeX documents with citations (future support)"
)
```

### Update MarkdownFileSource to be FileSource

**File**: `data/extractors/sources.py`

**Current**:
```python
class MarkdownFileSource(ContentSource):
    """Reads .md files from a directory."""
    
    def __init__(self, input_dir: Path, recursive: bool = True):
        self.input_dir = Path(input_dir)
        self.recursive = recursive
        # ...
    
    def iter_documents(self) -> Iterator[Document]:
        pattern = '**/*.md' if self.recursive else '*.md'
        # ...
```

**Proposed**:
```python
class FileSource(ContentSource):
    """
    Reads files with a specific extension from a directory.
    
    Replaces the old MarkdownFileSource with a more general implementation
    that works for any text file type (.md, .py, .tex, etc.).
    """
    
    def __init__(
        self,
        input_dir: Path,
        extension: str = '.md',
        recursive: bool = True,
        encoding: str = 'utf-8'
    ):
        """
        Args:
            input_dir: Directory containing files
            extension: File extension to search for (e.g., '.md', '.py', '.tex')
            recursive: If True, search subdirectories recursively
            encoding: Text encoding for reading files
        """
        self.input_dir = Path(input_dir)
        self.extension = extension if extension.startswith('.') else f'.{extension}'
        self.recursive = recursive
        self.encoding = encoding
        
        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
    
    def iter_documents(self) -> Iterator[Document]:
        """
        Yield Document for each file with the specified extension.
        
        The identifier is the filename without extension.
        The filepath is stored in metadata for reference.
        """
        pattern = f'**/*{self.extension}' if self.recursive else f'*{self.extension}'
        
        for filepath in self.input_dir.glob(pattern):
            try:
                with open(filepath, 'r', encoding=self.encoding) as f:
                    content = f.read()
            except (IOError, UnicodeDecodeError) as e:
                # Skip files that can't be read
                logger.warning(f"Could not read {filepath}: {e}")
                continue
            
            # Title from filename (without extension)
            # For Wikipedia .md files, the filename is already normalized by dump_extractor.py
            # For Python .py files, the filename is the module name
            # For LaTeX .tex files, the filename is the document name
            identifier = filepath.stem
            
            yield Document(
                identifier=identifier,
                normalized_identifier=identifier,  # Will be normalized by graph builder
                content=content,
                metadata={
                    'filepath': str(filepath),
                    'extension': self.extension
                }
            )


# Backwards compatibility alias
MarkdownFileSource = FileSource  # For existing code that uses this name
```

### Update dump_extractor to use configurable extension

**File**: `data/extractors/wikipedia/dump_extractor.py`

**Current (line ~418)**:
```python
output_filename = normalize_title(title) + '.md'
```

**Proposed**:
```python
def extract_wikipedia_dump(
    dump_path: Path,
    output_dir: Path,
    max_workers: int = 4,
    file_extension: str = '.md'  # NEW PARAMETER
):
    """
    Extract Wikipedia dump to individual files.
    
    Args:
        dump_path: Path to Wikipedia Cirrus JSON dump
        output_dir: Directory to write extracted files
        max_workers: Number of parallel workers
        file_extension: Extension for output files (default: '.md')
    """
    # ...
    output_filename = normalize_title(title) + file_extension
    # ...
```

**Note**: For Wikipedia, we'll always use `.md` since we're converting wikitext to markdown.
This parameter is mainly for consistency and future flexibility.

### Update CLI to use DatasetConfig

**File**: `data/extractors/cli.py`

```python
def main():
    # ...
    
    # Wiki subcommand
    wiki_parser = subparsers.add_parser(
        "wiki", 
        help="Extract Wikipedia link graph",
        description="Build a link graph from Wikipedia files"
    )
    wiki_parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing files from dump_extractor.py"
    )
    wiki_parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output JSONL file for graph"
    )
    wiki_parser.add_argument(
        "--extension",
        type=str,
        default=".md",
        help="File extension to search for (default: .md)"
    )
    # ...
    
    args = parser.parse_args()
    
    if args.source == "wiki":
        from data.extractors.wikipedia.builder import build_wiki_graph
        from data.dataset_config import WIKIPEDIA_CONFIG
        
        # Allow override of extension via CLI
        config = WIKIPEDIA_CONFIG
        if args.extension != ".md":
            logger.info(f"Using custom extension: {args.extension}")
            config = DatasetConfig(
                name=config.name,
                title_format=config.title_format,
                link_format=config.link_format,
                file_extension=args.extension,
                hash_length=config.hash_length,
                path_separator=config.path_separator,
                description=config.description
            )
        
        graph = build_wiki_graph(
            args.input_dir,
            dataset_config=config  # Pass config instead of hardcoded values
        )
        # ...
```

### Update graph builders to use extension

**File**: `data/extractors/wikipedia/builder.py` and similar

```python
def build_wiki_graph(
    input_dir: Path,
    dataset_config: DatasetConfig
) -> dict:
    """
    Build Wikipedia graph from extracted files.
    
    Args:
        input_dir: Directory containing files
        dataset_config: Dataset configuration including file extension
    
    Returns:
        Dictionary representing the graph
    """
    from data.extractors.sources import FileSource
    from data.extractors.link_extractors import MarkdownLinkExtractor
    from data.extractors.normalization import PassthroughNormalizer
    from data.extractors.graph_builder import build_graph
    
    # Create source with configurable extension
    source = FileSource(
        input_dir=input_dir,
        extension=dataset_config.file_extension,
        recursive=True
    )
    
    # Rest of the implementation...
    extractor = MarkdownLinkExtractor()
    normalizer = PassthroughNormalizer()
    
    return build_graph(
        source=source,
        extractor=extractor,
        normalizer=normalizer,
        source_type=dataset_config.name
    )
```

## Testing Requirements

### Unit Tests

**test_file_source.py**:

```python
import tempfile
from pathlib import Path
from data.extractors.sources import FileSource

def test_markdown_files():
    """Test with .md files (existing behavior)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test files
        (tmpdir / "doc1.md").write_text("# Doc 1")
        (tmpdir / "doc2.md").write_text("# Doc 2")
        (tmpdir / "other.txt").write_text("Other")
        
        source = FileSource(tmpdir, extension='.md')
        docs = list(source.iter_documents())
        
        assert len(docs) == 2
        assert set(d.identifier for d in docs) == {'doc1', 'doc2'}

def test_python_files():
    """Test with .py files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        (tmpdir / "module1.py").write_text("import os")
        (tmpdir / "module2.py").write_text("import sys")
        
        source = FileSource(tmpdir, extension='.py')
        docs = list(source.iter_documents())
        
        assert len(docs) == 2
        assert set(d.identifier for d in docs) == {'module1', 'module2'}

def test_extension_normalization():
    """Test that extension with/without leading dot works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test.tex").write_text("LaTeX content")
        
        # Both should work
        source1 = FileSource(tmpdir, extension='.tex')
        source2 = FileSource(tmpdir, extension='tex')
        
        assert len(list(source1.iter_documents())) == 1
        assert len(list(source2.iter_documents())) == 1

def test_backwards_compatibility():
    """Test that MarkdownFileSource alias still works."""
    from data.extractors.sources import MarkdownFileSource
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "test.md").write_text("Content")
        
        # Old name should still work
        source = MarkdownFileSource(tmpdir)
        docs = list(source.iter_documents())
        assert len(docs) == 1
```

### Integration Tests

1. Extract Wikipedia dump to `.md` files
2. Build graph using FileSource with `.md` extension
3. Verify graph is identical to previous implementation

4. Create test Python dataset with `.py` files
5. Build graph using FileSource with `.py` extension
6. Verify imports are extracted correctly

## Implementation Steps

1. **Add `file_extension` to DatasetConfig**
   - Update dataclass
   - Update predefined configs
   - Add tests

2. **Rename and enhance MarkdownFileSource → FileSource**
   - Add extension parameter
   - Keep backwards-compatibility alias
   - Add tests

3. **Update dump_extractor (optional)**
   - Add extension parameter
   - Document that Wikipedia always uses `.md`

4. **Update CLI**
   - Pass DatasetConfig to builders
   - Allow extension override if needed

5. **Update graph builders**
   - Use config.file_extension
   - Test with both `.md` and `.py`

6. **Documentation**
   - Update README with extension info
   - Document how to add new file types
   - Add examples for Python and future LaTeX

## Backwards Compatibility

- `MarkdownFileSource` becomes an alias for `FileSource`
- Default extension is still `.md`
- Existing code continues to work
- Wikipedia dataset generation unchanged

## Success Criteria

- [ ] DatasetConfig has file_extension field
- [ ] FileSource works with any extension
- [ ] MarkdownFileSource alias exists for compatibility
- [ ] Tests pass for .md, .py extensions
- [ ] Wikipedia extraction still produces .md files
- [ ] Python dataset can use .py files
- [ ] Documentation updated
- [ ] No hardcoded `.md` assumptions in ContentSource

## Future Extensions

1. **Multiple extensions per dataset**
   - Some datasets might have mixed file types
   - `file_extensions: List[str]` instead of `file_extension: str`

2. **Extension-specific content parsing**
   - Different handling for code vs text files
   - Language-specific syntax highlighting
   - Comment extraction from code

3. **Automatic extension detection**
   - Scan directory and detect file types
   - Build unified graph across multiple types
   - Useful for documentation + code repositories
