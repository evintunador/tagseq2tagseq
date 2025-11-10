import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from gpt_lab.data_sources.catalog_utils import BinaryShardIO

logger = logging.getLogger(__name__)


class GraphIndex:
    """
    Loads and provides access to the pre-tokenized graph dataset metadata.
    This includes the graph structure (nodes and edges) and the location
    of each document's tokens within the binary shards.
    """

    def __init__(self, run_directory: Path):
        """
        Initializes the index by loading metadata.json and tokenized_graph.jsonl.

        Args:
            run_directory: The path to the specific experiment run directory
                           containing the metadata, graph, and shard files.
        """
        self.run_directory = Path(run_directory)
        if not self.run_directory.is_dir():
            raise FileNotFoundError(f"Run directory not found: {self.run_directory}")

        # Load dataset-wide metadata
        metadata_path = self.run_directory / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.run_directory}")
        
        logger.info(f"Loading graph index from {self.run_directory}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.token_dtype = np.dtype(self.metadata["dtype_str"])
        self.shard_filenames = [self.run_directory / fname for fname in self.metadata["shard_filenames"]]

        # Load the graph structure and token location data
        graph_path = self.run_directory / "tokenized_graph.jsonl"
        if not graph_path.exists():
            raise FileNotFoundError(f"tokenized_graph.jsonl not found in {self.run_directory}")
            
        self.nodes: Dict[str, Dict[str, Any]] = {}
        with open(graph_path, "r", encoding="utf-8") as f:
            for line in f:
                node_data = json.loads(line)
                title = node_data["title"]
                self.nodes[title] = node_data
        
        logger.info(f"Graph index loaded. Found {len(self.nodes):,} nodes across {len(self.shard_filenames)} shards.")

    def get_node(self, title: str) -> Optional[Dict[str, Any]]:
        """Returns the full data for a given node (title)."""
        return self.nodes.get(title)

    def get_outgoing_links(self, title: str) -> List[str]:
        """Returns the list of titles that the given title links to."""
        node = self.get_node(title)
        return node.get("outgoing", []) if node else []

    def get_incoming_links(self, title: str) -> List[str]:
        """Returns the list of titles that link to the given title."""
        node = self.get_node(title)
        return node.get("incoming", []) if node else []

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, title: str) -> bool:
        return title in self.nodes

    def get_all_titles(self) -> List[str]:
        """Returns a list of all document titles in the graph."""
        return list(self.nodes.keys())


class PretokShardedBackend:
    """
    Provides access to the tokenized document data stored in binary shards.
    Uses memory-mapping for efficient, on-demand data retrieval.
    """

    def __init__(self, index: GraphIndex):
        """
        Initializes the backend with a GraphIndex.

        Args:
            index: A fully initialized GraphIndex object.
        """
        self.index = index
        self._memmaps: Dict[int, np.memmap] = {}
        logger.info("PretokShardedBackend initialized.")

    def _get_memmap(self, shard_idx: int) -> np.memmap:
        """
        Lazily opens and caches a memory-map for a given shard index.
        """
        if shard_idx not in self._memmaps:
            shard_path = self.index.shard_filenames[shard_idx]
            if not shard_path.exists():
                raise FileNotFoundError(f"Shard file not found: {shard_path}")
            
            # The BinaryShardIO methods handle reading the header to determine
            # the correct dtype and offset. We pass our expected dtype for validation.
            self._memmaps[shard_idx] = BinaryShardIO.read_datafile_tokens_memmap(
                shard_path, dtype=self.index.token_dtype
            )
        return self._memmaps[shard_idx]

    def get_tokens(self, title: str) -> Optional[np.ndarray]:
        """
        Retrieves the token array for a given document title.

        Args:
            title: The title of the document.

        Returns:
            A numpy array of tokens, or None if the title is not found.
        """
        node_data = self.index.get_node(title)
        if not node_data:
            logger.warning(f"Title '{title}' not found in graph index.")
            return None

        shard_idx = node_data["tok_shard_idx"]
        offset_bytes = node_data["tok_offset_bytes"]
        num_tokens = node_data["tok_len"]
        
        # Calculate the start index in terms of tokens, not bytes
        # The offset from the file start must account for the header
        token_start_idx = (offset_bytes - (256 * 4)) // self.index.token_dtype.itemsize

        memmap = self._get_memmap(shard_idx)
        
        # Slice the token array from the memory-mapped file
        tokens = memmap[token_start_idx : token_start_idx + num_tokens]
        
        # It's good practice to ensure the length matches, as a sanity check.
        if len(tokens) != num_tokens:
             logger.error(
                 f"Token length mismatch for '{title}'. Expected {num_tokens}, got {len(tokens)}. "
                 f"Shard: {shard_idx}, Byte Offset: {offset_bytes}"
             )
             return None

        return tokens

    def close(self):
        """Closes all open memory-mapped files."""
        for mm in self._memmaps.values():
            # The memmap object in numpy doesn't have a close method itself.
            # The underlying file handle is closed when the object is garbage collected.
            # To be explicit, we can access the underlying base object if it's a file buffer.
            if hasattr(mm, '_mmap'):
                 mm._mmap.close()
        self._memmaps.clear()
        logger.info("All memory-mapped files have been closed.")
