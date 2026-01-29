# Graph Extractors Framework

Shared abstractions for building document graphs from various sources (Wikipedia, GitHub, LaTeX, etc.).

## Architecture

The framework provides pluggable components that can be mixed and matched:

- **LinkExtractor**: Extract raw link targets from document content
- **LinkNormalizer**: Normalize links to canonical, filesystem-safe identifiers
- **ContentSource**: Iterate over documents from any source
- **GraphBuilder**: Core graph construction logic

## Quick Start

### Using Existing Extractors

**Wikipedia:**
```bash
# Using new unified CLI
python -m data.extractors.cli wiki extracted_wiki/ -o wiki_graph.jsonl

# Or using existing CLI (backward compatible)
python data/wiki_graph_extractor/build_graph.py extracted_wiki/ -o wiki_graph.jsonl
```

**GitHub:**
```bash
# Using new unified CLI
python -m data.extractors.cli github repos.jsonl -o github_graph.jsonl

# Or using existing CLI (backward compatible)
python data/github_graph_extractor/build_graph_streaming.py repos.jsonl -o github_graph.jsonl
```

### Creating New Extractors

The framework makes it easy to create extractors for new document types. Here's a complete example for LaTeX citations:

```python
from pathlib import Path
import re

from data.extractors.graph_builder import GraphBuilder
from data.extractors.sources import MarkdownFileSource
from data.extractors.normalization import HashingNormalizer
from data.extractors.protocols import LinkExtractor, LinkContext


class LaTeXCitationExtractor:
    """Extract LaTeX citation commands."""
    
    CITE_PATTERN = re.compile(r'\\cite\{([^}]+)\}')
    
    def extract_links(self, content, context):
        """Find all \cite{key} references."""
        citations = self.CITE_PATTERN.findall(content)
        # Handle multiple citations in one command: \cite{key1,key2}
        all_keys = set()
        for citation in citations:
            keys = citation.split(',')
            all_keys.update(key.strip() for key in keys)
        return all_keys


# Use it with GraphBuilder
builder = GraphBuilder(
    source=MarkdownFileSource(Path("papers/")),
    link_extractor=LaTeXCitationExtractor(),
    normalizer=HashingNormalizer(),
    source_type="latex",
)

graph = builder.build_graph(Path("citations.jsonl"))
print(f"Built citation graph with {len(graph)} papers")
```

## Components

### LinkExtractor Protocol

Extracts raw link targets from document content.

**Implementations:**
- `MarkdownLinkExtractor`: Extracts `[text](target)` links
- `PythonImportExtractor`: Extracts Python `import` and `from...import` statements

**Interface:**
```python
class LinkExtractor(Protocol):
    def extract_links(self, content: str, context: LinkContext) -> Set[str]:
        """Returns set of raw link targets found in content."""
```

### LinkNormalizer Protocol

Normalizes raw links into canonical identifiers suitable for filenames.

**Implementations:**
- `HashingNormalizer`: Base class with common normalization logic
- `WikiTitleNormalizer`: Wikipedia-specific title normalization
- `PythonModuleNormalizer`: Python module name normalization

**Interface:**
```python
class LinkNormalizer(Protocol):
    def normalize(self, link: str, context: LinkContext) -> str:
        """Returns normalized, filesystem-safe identifier."""
```

**Key features:**
- Converts to lowercase
- Replaces special characters with underscores
- Appends 6-character hash for uniqueness
- Limits length to prevent filesystem issues

### ContentSource Protocol

Provides iterator over documents from a data source.

**Implementations:**
- `MarkdownFileSource`: Reads `.md` files from directory
- `JSONLSource`: Streams records from JSON Lines files

**Interface:**
```python
class ContentSource(Protocol):
    def iter_documents(self) -> Iterator[Document]:
        """Yields Document objects."""
```

### GraphBuilder

Core graph construction algorithm:

1. Extract links from all documents
2. Normalize identifiers and link targets
3. Build bidirectional graph (outgoing + incoming links)
4. Write to JSONL format

**Usage:**
```python
from data.extractors.graph_builder import GraphBuilder
from data.extractors.sources import MarkdownFileSource
from data.extractors.link_extractors import MarkdownLinkExtractor
from data.extractors.normalization import WikiTitleNormalizer

builder = GraphBuilder(
    source=MarkdownFileSource(input_dir),
    link_extractor=MarkdownLinkExtractor(),
    normalizer=WikiTitleNormalizer(),
    source_type="wiki",
    show_progress=True,
)

graph = builder.build_graph(output_path)
```

## Output Format

All extractors produce JSONL (JSON Lines) format with one node per line:

```json
{
  "title": "normalized_identifier_hash123",
  "char_count": 1234,
  "outgoing": ["link1", "link2"],
  "incoming": ["backlink1"]
}
```

- `title`: Normalized identifier with hash suffix
- `char_count`: Number of characters in document content
- `outgoing`: Links from this document to others
- `incoming`: Links from other documents to this one

## Testing

Run the test suite:

```bash
# Run all extractor tests
pytest tests/data/extractors/

# Run specific test file
pytest tests/data/extractors/test_normalization.py

# Run with verbose output
pytest tests/data/extractors/ -v
```

## Extending the Framework

### Adding a New Link Type

1. **Create a LinkExtractor** for your link format:

```python
class MyLinkExtractor:
    def extract_links(self, content: str, context: LinkContext) -> Set[str]:
        # Your extraction logic here
        return set_of_links
```

2. **Optionally create a custom normalizer** if you need domain-specific normalization:

```python
class MyNormalizer(HashingNormalizer):
    def _clean_text(self, text: str) -> str:
        # Custom cleaning logic
        return cleaned_text
```

3. **Compose with GraphBuilder**:

```python
builder = GraphBuilder(
    source=your_source,
    link_extractor=MyLinkExtractor(),
    normalizer=MyNormalizer(),
    source_type="mytype",
)
```

### Adding a New Data Source

Create a class implementing the `ContentSource` protocol:

```python
class MySource:
    def iter_documents(self) -> Iterator[Document]:
        for item in my_data_source:
            yield Document(
                identifier=item.id,
                content=item.text,
                metadata={"extra": item.metadata}
            )
```

## Migration from Old Code

The refactoring maintains backward compatibility:

- **Old CLIs still work**: `build_graph.py` and `build_graph_streaming.py` are thin wrappers
- **New unified CLI**: `python -m data.extractors.cli` provides consistent interface
- **Existing tests**: Old test files continue to work unchanged
- **Incremental migration**: Complex logic (like GitHub import resolution) can be moved incrementally

## Design Principles

1. **Separation of Concerns**: Link extraction, normalization, and graph building are independent
2. **Pluggability**: Components can be mixed and matched
3. **Testability**: Each component can be tested in isolation
4. **Reusability**: Common logic (normalization, graph building) is shared
5. **Extensibility**: Easy to add new link types and data sources

## Future Work

Potential enhancements:

- GitHub import resolution logic in framework
- Statistics and visualization support
- Parallel processing support
- Streaming support for very large datasets
- More link extractors (Rust imports, JavaScript requires, etc.)
- More normalizers (URL normalization, etc.)
