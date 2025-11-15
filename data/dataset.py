import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from gpt_lab.data_sources.catalog_utils import BinaryShardIO

logger = logging.getLogger(__name__)


class GraphIndex:
    """
    Loads and provides access to the pre-tokenized graph dataset metadata.
    This includes the graph structure (nodes and edges) and the location
    of each document's tokens within the binary shards.

    In addition to exposing raw node dictionaries, the index maintains a
    stable mapping between human-readable document titles and compact
    integer ids so that higher-level components can work with integer
    representations of the graph.
    """

    def __init__(self, run_directory: Path):
        """
        Initializes the index by loading ``metadata.json`` and
        ``tokenized_graph.jsonl``.

        The initializer also performs basic validation to ensure that each
        node contains the fields required for locating its tokens in the
        binary shards (``tok_shard_idx``, ``tok_offset_bytes``, ``tok_len``)
        and constructs title/id lookup tables.

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
        self.shard_filenames = [
            self.run_directory / fname for fname in self.metadata["shard_filenames"]
        ]

        # Load the graph structure and token location data
        graph_path = self.run_directory / "tokenized_graph.jsonl"
        if not graph_path.exists():
            raise FileNotFoundError(
                f"tokenized_graph.jsonl not found in {self.run_directory}"
            )

        self.nodes: Dict[str, Dict[str, Any]] = {}
        with open(graph_path, "r", encoding="utf-8") as f:
            for line in f:
                node_data = json.loads(line)
                title = node_data["title"]
                self.nodes[title] = node_data

        # Create mappings between titles and integer ids, and validate the
        # presence and types of token-location fields required by the
        # pre-tokenized backend.
        self._title_to_id: Dict[str, int] = {}
        self._id_to_title: List[str] = []

        required_fields = ("tok_shard_idx", "tok_offset_bytes", "tok_len")

        for idx, (title, node) in enumerate(self.nodes.items()):
            # Validate that required token-location metadata is present.
            missing = [field for field in required_fields if field not in node]
            if missing:
                raise ValueError(
                    f"Node '{title}' is missing required token metadata fields: {missing}"
                )

            # Validate basic types for token-location metadata.
            shard_idx = node["tok_shard_idx"]
            offset_bytes = node["tok_offset_bytes"]
            tok_len = node["tok_len"]

            if not isinstance(shard_idx, int):
                raise ValueError(
                    f"Node '{title}' has invalid tok_shard_idx={shard_idx!r}; "
                    "expected an integer."
                )
            if not isinstance(offset_bytes, int):
                raise ValueError(
                    f"Node '{title}' has invalid tok_offset_bytes={offset_bytes!r}; "
                    "expected an integer."
                )
            if not isinstance(tok_len, int):
                raise ValueError(
                    f"Node '{title}' has invalid tok_len={tok_len!r}; expected an integer."
                )

            self._title_to_id[title] = idx
            self._id_to_title.append(title)

        logger.info(
            f"Graph index loaded. Found {len(self.nodes):,} nodes across "
            f"{len(self.shard_filenames)} shards."
        )

    def get_node(self, title: str) -> Optional[Dict[str, Any]]:
        """Returns the full metadata dictionary for a given node title."""
        return self.nodes.get(title)

    def get_outgoing_links(self, title: str) -> List[str]:
        """Returns the list of titles that the given title links to."""
        node = self.get_node(title)
        return node.get("outgoing", []) if node else []

    def get_incoming_links(self, title: str) -> List[str]:
        """Returns the list of titles that link to the given title."""
        node = self.get_node(title)
        return node.get("incoming", []) if node else []

    def get_id(self, title: str) -> int:
        """
        Returns the integer id corresponding to ``title``.

        Raises:
            KeyError: If the title is not present in the index.
        """
        try:
            return self._title_to_id[title]
        except KeyError as exc:
            raise KeyError(f"Unknown document title: {title!r}") from exc

    def get_title(self, doc_id: int) -> str:
        """
        Returns the document title associated with ``doc_id``.

        Raises:
            IndexError: If ``doc_id`` is out of range.
        """
        if doc_id < 0 or doc_id >= len(self._id_to_title):
            raise IndexError(f"Document id out of range: {doc_id}")
        return self._id_to_title[doc_id]

    def neighbors_out(self, doc_id: int) -> List[int]:
        """
        Returns the ids of all outgoing neighbors from the node ``doc_id``.

        Unknown neighbor titles (if any) are silently skipped.
        """
        title = self.get_title(doc_id)
        node = self.get_node(title)
        if not node:
            return []

        neighbor_ids: List[int] = []
        for neighbor_title in node.get("outgoing", []):
            doc_idx = self._title_to_id.get(neighbor_title)
            if doc_idx is not None:
                neighbor_ids.append(doc_idx)
        return neighbor_ids

    def neighbors_in(self, doc_id: int) -> List[int]:
        """
        Returns the ids of all incoming neighbors to the node ``doc_id``.

        Unknown neighbor titles (if any) are silently skipped.
        """
        title = self.get_title(doc_id)
        node = self.get_node(title)
        if not node:
            return []

        neighbor_ids: List[int] = []
        for neighbor_title in node.get("incoming", []):
            doc_idx = self._title_to_id.get(neighbor_title)
            if doc_idx is not None:
                neighbor_ids.append(doc_idx)
        return neighbor_ids

    def get_token_len(self, doc_id: int) -> int:
        """
        Returns the precomputed token length (``tok_len``) for the document
        identified by ``doc_id``.
        """
        title = self.get_title(doc_id)
        node = self.get_node(title)
        if not node:
            raise KeyError(f"No node metadata found for document id {doc_id}")
        return int(node["tok_len"])

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

    The backend is intentionally minimal and does not implement any graph
    traversal or batching logic. It focuses solely on translating node
    metadata into slices of the underlying binary shard files.
    """

    def __init__(self, index: GraphIndex):
        """
        Initializes the backend with a ``GraphIndex``.

        Args:
            index: A fully initialized ``GraphIndex`` object.
        """
        self.index = index
        # Expose the underlying token dtype as a stable attribute so that
        # callers can configure downstream models or buffers accordingly.
        self.dtype = index.token_dtype
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
            logger.info(f"Opening memmap for shard index {shard_idx}: {shard_path}")
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
            A numpy array of tokens, or ``None`` if the title is not found.
        """
        node_data = self.index.get_node(title)
        if not node_data:
            logger.warning(f"Title '{title}' not found in graph index.")
            return None

        shard_idx = node_data["tok_shard_idx"]
        offset_bytes = node_data["tok_offset_bytes"]
        num_tokens = node_data["tok_len"]

        # Calculate the start index in terms of tokens, not bytes.
        # The offset from the file start must account for the header.
        token_start_idx = (
            offset_bytes - (256 * 4)
        ) // self.index.token_dtype.itemsize

        memmap = self._get_memmap(shard_idx)

        # Slice the token array from the memory-mapped file.
        tokens = memmap[token_start_idx : token_start_idx + num_tokens]

        # It's good practice to ensure the length matches, as a sanity check.
        if len(tokens) != num_tokens:
            logger.error(
                f"Token length mismatch for '{title}'. Expected {num_tokens}, got {len(tokens)}. "
                f"Shard: {shard_idx}, Byte Offset: {offset_bytes}"
            )
            return None

        return tokens

    def get_tokens_by_id(self, doc_id: int) -> Optional[np.ndarray]:
        """
        Retrieves the token array for a document identified by integer id.

        This is a thin convenience wrapper that converts the id to a title via
        the associated ``GraphIndex`` and then delegates to ``get_tokens`` to
        preserve the existing behavior and validation logic.
        """
        title = self.index.get_title(doc_id)
        return self.get_tokens(title)

    def close(self):
        """Closes all open memory-mapped files."""
        for shard_idx, mm in list(self._memmaps.items()):
            # The memmap object in numpy doesn't have a close method itself.
            # The underlying file handle is closed when the object is garbage collected.
            # To be explicit, we can access the underlying base object if it's a file buffer.
            if hasattr(mm, "_mmap"):
                mm._mmap.close()
            logger.info(f"Closed memmap for shard index {shard_idx}")
        self._memmaps.clear()
        logger.info("All memory-mapped files have been closed.")
