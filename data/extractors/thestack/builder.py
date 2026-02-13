"""
TheStack intra-repository dependency graph builder using shared framework.

This module extends the generic GraphBuilder with TheStack-specific logic:
- Groups files by repository
- Resolves imports to actual files within the same repository
- Only creates links between files in the same repo (intra-repo dependencies)
- Filters to keep only files with 2+ connections
"""
from pathlib import Path
from typing import Dict, Set, Optional
import logging
import os

from data.extractors.graph_builder import GraphBuilder, GraphNode
from data.extractors.sources import JSONLSource
from data.extractors.link_extractors import PythonImportExtractor
from data.dataset_config import DatasetConfig, THESTACK_CONFIG
from .extract import extract_file_imports, normalize_repository_name


logger = logging.getLogger(__name__)


def _path_to_module_name(file_path: str) -> str:
    """Convert file path to Python module name."""
    module_path = file_path[:-3] if file_path.endswith(".py") else file_path
    return module_path.replace("/", ".").replace("\\", ".").strip(".")


def _resolve_import_to_file(
    imported_module: str, 
    module_to_file: Dict[str, str], 
    importing_file: str
) -> Optional[str]:
    """
    Resolve an import statement to an actual file path within the repository.
    
    Handles:
    - Relative imports (starting with .)
    - Absolute imports matching module names
    - Submodule imports
    """
    # Handle relative imports
    if imported_module.startswith("."):
        dot_count = len(imported_module) - len(imported_module.lstrip("."))
        base_module = imported_module[dot_count:]

        import_dir = os.path.dirname(importing_file)
        for _ in range(dot_count - 1):
            import_dir = os.path.dirname(import_dir)

        if base_module:
            full_module_path = os.path.join(import_dir, base_module.replace(".", "/"))
        else:
            full_module_path = import_dir

        full_module_path = full_module_path.rstrip("/")

        # Find matching file
        for file_path in module_to_file.values():
            if file_path == full_module_path + ".py" or file_path.startswith(full_module_path + "/"):
                return file_path
        return None

    # Handle absolute imports - exact match
    if imported_module in module_to_file:
        return module_to_file[imported_module]

    # Handle submodule imports (e.g., "foo" matches "foo.bar" or "foo.bar" matches "foo")
    for module_name, file_path in module_to_file.items():
        if module_name.startswith(imported_module + "."):
            return file_path
        if imported_module.startswith(module_name + "."):
            return file_path

    return None


class TheStackGraphBuilder(GraphBuilder):
    """
    Extended graph builder for TheStack with intra-repository import resolution.
    
    TheStack has additional complexity:
    - Needs to group files by repository
    - Must resolve imports to actual files within repo
    - Only creates links between files in the same repository
    - Filters nodes with <2 links
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build_graph(self, output_path: Path) -> Dict[str, GraphNode]:
        """
        Build intra-repository dependency graph.
        
        This overrides the base implementation to:
        1. Group files by repository
        2. Resolve imports to actual files within each repo
        3. Only create edges between files in the same repo
        4. Filter to keep only files with 2+ connections
        
        Args:
            output_path: Where to write filtered graph
        
        Returns:
            Filtered graph with only nodes having 2+ intra-repo links
        """
        logger.info("Building TheStack intra-repository dependency graph...")
        
        # Phase 1: Group files by repository and extract imports
        repo_files = {}  # repo_name -> [(file_path, content, imports)]
        
        for doc in self.source.iter_documents():
            repo_name = doc.metadata.get("max_stars_repo_name", "")
            file_path = doc.identifier
            content = doc.content
            
            if not repo_name or not file_path:
                continue
            
            # Extract imports using extract.py logic
            imported_modules = extract_file_imports(content, file_path, repo_name)
            
            repo_files.setdefault(repo_name, []).append((
                file_path,
                content,
                imported_modules
            ))
        
        logger.info(f"Grouped {sum(len(files) for files in repo_files.values())} files "
                   f"into {len(repo_files)} repositories")
        
        # Phase 2: Build graph per repository with import resolution
        final_graph = {}
        repos_with_links = 0
        
        for repo_name, files in repo_files.items():
            # Skip repos with only 1 file (can't have intra-repo links)
            if len(files) < 2:
                continue
            
            # Build module -> file mapping for this repo
            module_to_file = {}
            file_to_imports = {}
            norm_repo = normalize_repository_name(repo_name)
            
            for file_path, content, imported_modules in files:
                file_to_imports[file_path] = imported_modules
                module_name = _path_to_module_name(file_path)
                module_to_file[module_name] = file_path
                
                # Also map __init__.py to parent module
                if file_path.endswith("/__init__.py"):
                    parent_module = _path_to_module_name(file_path[:-12])
                    if parent_module:
                        module_to_file[parent_module] = file_path
            
            # Create nodes for this repo
            repo_graph = {}
            path_to_key = {}
            
            for file_path, content, _ in files:
                # Use repo:path as the key format
                k = f"{norm_repo}:{file_path}"
                path_to_key[file_path] = k
                node = GraphNode(
                    identifier=f"{repo_name}:{file_path}",  # Original repo:path
                    normalized_identifier=k,  # Normalized repo:path
                    char_count=len(content)
                )
                # Include metadata
                node.metadata['max_stars_repo_name'] = repo_name
                node.metadata['max_stars_repo_path'] = file_path
                repo_graph[k] = node
            
            # Resolve imports to actual files and create outgoing edges
            for file_path, imported_modules in file_to_imports.items():
                outgoing_links = set()
                for imported_module in imported_modules:
                    target_file = _resolve_import_to_file(
                        imported_module, 
                        module_to_file, 
                        file_path
                    )
                    # Only add link if target exists and is different from source
                    if target_file and target_file != file_path:
                        # Convert to full repo:path format
                        target_key = path_to_key.get(target_file)
                        if target_key:
                            outgoing_links.add(target_key)
                
                if outgoing_links:
                    repo_graph[path_to_key[file_path]].outgoing = outgoing_links
            
            # Compute incoming edges
            for source_key, node in repo_graph.items():
                for target_key in node.outgoing:
                    # target_key is now already in full repo:path format
                    if target_key in repo_graph:
                        repo_graph[target_key].incoming.add(source_key)
            
            # Check if this repo has any links
            if any(node.outgoing or node.incoming for node in repo_graph.values()):
                repos_with_links += 1
                final_graph.update(repo_graph)
        
        logger.info(f"Graph complete: {len(final_graph)} nodes from {repos_with_links} repos "
                   f"with intra-repository dependencies")
        
        # Phase 3: Filter to keep only nodes with 2+ total links
        nodes_before = len(final_graph)
        filtered = {
            k: v for k, v in final_graph.items()
            if (len(v.outgoing) + len(v.incoming)) >= 2
        }
        nodes_after = len(filtered)
        
        logger.info(
            f"Filtered graph: {nodes_before} -> {nodes_after} nodes "
            f"({nodes_after/nodes_before*100:.1f}% kept with 2+ links)"
        )
        
        # Write filtered graph
        self._write_graph(filtered, output_path)
        
        return filtered


def build_thestack_graph(
    input_file: Path,
    output_path: Path,
    show_progress: bool = True,
    dataset_config: Optional[DatasetConfig] = None,
) -> Dict[str, GraphNode]:
    """
    Build TheStack intra-repository dependency graph.
    
    This processes a JSONL file from download_sample.py and builds
    a graph showing file-to-file dependencies within repositories.
    
    Args:
        input_file: JSONL file from download_sample.py
        output_path: Path for output graph.jsonl
        show_progress: Show progress bars
        dataset_config: Optional DatasetConfig to use. If None, uses THESTACK_CONFIG.
                       The normalizer will be obtained from this config.
    
    Returns:
        Dictionary of graph nodes (filtered to 2+ links)
    
    Example:
        >>> from pathlib import Path
        >>> graph = build_thestack_graph(
        ...     input_file=Path("sample_100k.jsonl"),
        ...     output_path=Path("thestack_graph.jsonl")
        ... )
        >>> print(f"Built graph with {len(graph)} nodes")
    """
    # Use default TheStack config if not provided
    if dataset_config is None:
        dataset_config = THESTACK_CONFIG
    
    # Get normalizer from config (ensures hash_length is consistent)
    normalizer = dataset_config.get_normalizer()
    
    builder = TheStackGraphBuilder(
        source=JSONLSource(
            input_file,
            identifier_field="max_stars_repo_path",
            additional_fields=["max_stars_repo_name"]
        ),
        link_extractor=PythonImportExtractor(),
        normalizer=normalizer,
        source_type="thestack",
        show_progress=show_progress,
    )
    
    graph = builder.build_graph(output_path)
    
    # Save dataset config alongside graph
    config_path = output_path.parent / "dataset_config.json"
    dataset_config.save(config_path)
    
    return graph


# Alias for backward compatibility
build_github_graph = build_thestack_graph
