"""
Content source implementations for different data formats.

Provides iterators over documents from various sources:
- FileSource: Read text files from directory (supports any extension)
- JSONLSource: Stream JSON Lines files
- TheStackJSONLSource: Specialized source for TheStack dataset
"""
import json
from pathlib import Path
from typing import Iterator, Optional, List
from .protocols import Document, ContentSource


class FileSource(ContentSource):
    """
    Reads text files from a directory with specified extension.
    
    Used by graph extractors to read text files (markdown, Python, LaTeX, etc.).
    The extension parameter determines which files to read.
    """
    
    def __init__(self, input_dir: Path, extension: str = '.md', recursive: bool = True):
        """
        Args:
            input_dir: Directory containing text files
            extension: File extension to match (e.g., '.md', '.py', '.tex')
            recursive: If True, search subdirectories recursively
        """
        self.input_dir = Path(input_dir)
        self.extension = extension if extension.startswith('.') else f'.{extension}'
        self.recursive = recursive
        
        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
    
    def iter_documents(self) -> Iterator[Document]:
        """
        Yield Document for each file matching the extension.
        
        The identifier is the filename without extension.
        The filepath is stored in metadata for reference.
        """
        pattern = f'**/*{self.extension}' if self.recursive else f'*{self.extension}'
        
        for filepath in self.input_dir.glob(pattern):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except (IOError, UnicodeDecodeError) as e:
                # Skip files that can't be read
                continue
            
            # Identifier from filename (without extension)
            identifier = filepath.stem
            
            yield Document(
                identifier=identifier,
                normalized_identifier=identifier,  # Already normalized in filename
                content=content,
                metadata={'filepath': str(filepath)}
            )


# Backward compatibility alias
MarkdownFileSource = FileSource


class JSONLSource(ContentSource):
    """
    Streams records from JSONL (JSON Lines) file.
    
    Used by GitHub graph extractor to process repository data.
    Each line in the file should be a valid JSON object.
    """
    
    def __init__(
        self,
        input_file: Path,
        identifier_field: str,
        content_field: str = "content",
        additional_fields: Optional[List[str]] = None
    ):
        """
        Args:
            input_file: Path to JSONL file
            identifier_field: JSON field to use as document identifier
            content_field: JSON field containing document content
            additional_fields: Additional JSON fields to include in metadata
        """
        self.input_file = Path(input_file)
        self.identifier_field = identifier_field
        self.content_field = content_field
        self.additional_fields = additional_fields or []
        
        if not self.input_file.exists():
            raise ValueError(f"Input file does not exist: {self.input_file}")
    
    def iter_documents(self) -> Iterator[Document]:
        """
        Yield Document for each JSONL line.
        
        Skips:
        - Empty lines
        - Invalid JSON
        - Records missing identifier field
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Skip malformed JSON lines
                    continue
                
                identifier = data.get(self.identifier_field)
                content = data.get(self.content_field, "")
                
                if not identifier:
                    # Skip records without identifier
                    continue
                
                # Collect additional metadata fields
                metadata = {
                    k: data.get(k) 
                    for k in self.additional_fields
                }
                # Add line number for debugging
                metadata['line_number'] = line_num
                
                # For JSONL, identifier needs normalization
                # We'll let the normalizer handle this in the graph builder
                yield Document(
                    identifier=identifier,
                    normalized_identifier=identifier,  # Will be normalized by graph builder
                    content=content or "",
                    metadata=metadata
                )


class TheStackJSONLSource(ContentSource):
    """
    Specialized source for TheStack dataset that constructs repo:path identifiers.
    
    Used by TheStack graph extractor to read repository data and construct
    identifiers in the format "repo_name:file_path" to match graph titles.
    """
    
    def __init__(
        self,
        input_file: Path,
        repo_field: str = "max_stars_repo_name",
        path_field: str = "max_stars_repo_path",
        content_field: str = "content"
    ):
        """
        Args:
            input_file: Path to JSONL file
            repo_field: JSON field containing repository name
            path_field: JSON field containing file path
            content_field: JSON field containing file content
        """
        self.input_file = Path(input_file)
        self.repo_field = repo_field
        self.path_field = path_field
        self.content_field = content_field
        
        if not self.input_file.exists():
            raise ValueError(f"Input file does not exist: {self.input_file}")
    
    def iter_documents(self) -> Iterator[Document]:
        """
        Yield Document for each JSONL line with repo:path identifier.
        
        The identifier is constructed as "{repo}:{path}" to match the format
        used by the GitHub graph builder.
        """
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                repo_name = data.get(self.repo_field)
                file_path = data.get(self.path_field)
                content = data.get(self.content_field, "")
                
                if not repo_name or not file_path:
                    continue
                
                # Construct identifier as repo:path
                identifier = f"{repo_name}:{file_path}"
                
                metadata = {
                    'repo_name': repo_name,
                    'file_path': file_path,
                    'line_number': line_num
                }
                
                # For GitHub, identifier needs normalization
                yield Document(
                    identifier=identifier,
                    normalized_identifier=identifier,  # Will be normalized by graph builder
                    content=content or "",
                    metadata=metadata
                )
