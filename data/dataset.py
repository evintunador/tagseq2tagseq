import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tunalab.pretokenized_data.shard_io import BinaryShardIO

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
        and constructs normed_identifier/id lookup tables.

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
                normed_id = node_data["normed_identifier"]
                self.nodes[normed_id] = node_data

        # Create mappings between normed_identifiers and integer ids, and
        # validate the presence and types of token-location fields required
        # by the pre-tokenized backend.
        self._normed_to_id: Dict[str, int] = {}
        self._id_to_normed: List[str] = []

        required_fields = ("tok_shard_idx", "tok_offset_bytes", "tok_len")

        for idx, (normed_id, node) in enumerate(self.nodes.items()):
            # Validate that required token-location metadata is present.
            missing = [field for field in required_fields if field not in node]
            if missing:
                raise ValueError(
                    f"Node '{normed_id}' is missing required token metadata fields: {missing}"
                )

            # Validate basic types for token-location metadata.
            shard_idx = node["tok_shard_idx"]
            offset_bytes = node["tok_offset_bytes"]
            tok_len = node["tok_len"]

            if not isinstance(shard_idx, int):
                raise ValueError(
                    f"Node '{normed_id}' has invalid tok_shard_idx={shard_idx!r}; "
                    "expected an integer."
                )
            if not isinstance(offset_bytes, int):
                raise ValueError(
                    f"Node '{normed_id}' has invalid tok_offset_bytes={offset_bytes!r}; "
                    "expected an integer."
                )
            if not isinstance(tok_len, int):
                raise ValueError(
                    f"Node '{normed_id}' has invalid tok_len={tok_len!r}; expected an integer."
                )

            self._normed_to_id[normed_id] = idx
            self._id_to_normed.append(normed_id)

        logger.info(
            f"Graph index loaded. Found {len(self.nodes):,} nodes across "
            f"{len(self.shard_filenames)} shards."
        )

    def get_node(self, normed_identifier: str) -> Optional[Dict[str, Any]]:
        """Returns the full metadata dictionary for a given normed_identifier."""
        return self.nodes.get(normed_identifier)

    def get_raw_identifier(self, normed_identifier: str) -> Optional[str]:
        """Returns the human-readable raw identifier for a node."""
        node = self.get_node(normed_identifier)
        return node.get("raw_identifier") if node else None

    def get_outgoing_links(self, normed_identifier: str) -> List[str]:
        """Returns the normed_identifiers that the given node links to."""
        node = self.get_node(normed_identifier)
        return node.get("outgoing", []) if node else []

    def get_incoming_links(self, normed_identifier: str) -> List[str]:
        """Returns the normed_identifiers that link to the given node."""
        node = self.get_node(normed_identifier)
        return node.get("incoming", []) if node else []

    def get_id(self, normed_identifier: str) -> int:
        """
        Returns the integer id corresponding to ``normed_identifier``.

        Raises:
            KeyError: If the identifier is not present in the index.
        """
        try:
            return self._normed_to_id[normed_identifier]
        except KeyError as exc:
            raise KeyError(f"Unknown normed_identifier: {normed_identifier!r}") from exc

    def get_normed_identifier(self, doc_id: int) -> str:
        """
        Returns the normed_identifier associated with ``doc_id``.

        Raises:
            IndexError: If ``doc_id`` is out of range.
        """
        if doc_id < 0 or doc_id >= len(self._id_to_normed):
            raise IndexError(f"Document id out of range: {doc_id}")
        return self._id_to_normed[doc_id]

    def neighbors_out(self, doc_id: int) -> List[int]:
        """
        Returns the ids of all outgoing neighbors from the node ``doc_id``.

        Unknown neighbors (if any) are silently skipped.
        """
        normed_id = self.get_normed_identifier(doc_id)
        node = self.get_node(normed_id)
        if not node:
            return []

        neighbor_ids: List[int] = []
        for neighbor_normed in node.get("outgoing", []):
            doc_idx = self._normed_to_id.get(neighbor_normed)
            if doc_idx is not None:
                neighbor_ids.append(doc_idx)
        return neighbor_ids

    def neighbors_in(self, doc_id: int) -> List[int]:
        """
        Returns the ids of all incoming neighbors to the node ``doc_id``.

        Unknown neighbors (if any) are silently skipped.
        """
        normed_id = self.get_normed_identifier(doc_id)
        node = self.get_node(normed_id)
        if not node:
            return []

        neighbor_ids: List[int] = []
        for neighbor_normed in node.get("incoming", []):
            doc_idx = self._normed_to_id.get(neighbor_normed)
            if doc_idx is not None:
                neighbor_ids.append(doc_idx)
        return neighbor_ids

    def get_token_len(self, doc_id: int) -> int:
        """
        Returns the precomputed token length (``tok_len``) for the document
        identified by ``doc_id``.
        """
        normed_id = self.get_normed_identifier(doc_id)
        node = self.get_node(normed_id)
        if not node:
            raise KeyError(f"No node metadata found for document id {doc_id}")
        return int(node["tok_len"])

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, normed_identifier: str) -> bool:
        return normed_identifier in self.nodes

    def get_split(self, normed_identifier: str) -> Optional[str]:
        """
        Returns the split name (``"train"``, ``"val_community"``,
        ``"val_random"``) for a node, or ``None`` if no split annotation
        is present.
        """
        node = self.get_node(normed_identifier)
        if node is None:
            return None
        return node.get("split")

    def get_split_ids(self, split_name: str) -> List[int]:
        """
        Returns the integer ids of all nodes belonging to *split_name*.

        Nodes without a ``"split"`` field are excluded.
        """
        return [
            self._normed_to_id[nid]
            for nid, node in self.nodes.items()
            if node.get("split") == split_name
        ]

    def get_all_normed_identifiers(self) -> List[str]:
        """Returns a list of all normed_identifiers in the graph."""
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

    def get_tokens(self, normed_identifier: str) -> Optional[np.ndarray]:
        """
        Retrieves the token array for a given normed_identifier.

        Args:
            normed_identifier: The normed identifier of the document.

        Returns:
            A numpy array of tokens, or ``None`` if the identifier is not found.
        """
        node_data = self.index.get_node(normed_identifier)
        if not node_data:
            logger.warning(f"Identifier '{normed_identifier}' not found in graph index.")
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
                f"Token length mismatch for '{normed_identifier}'. Expected {num_tokens}, got {len(tokens)}. "
                f"Shard: {shard_idx}, Byte Offset: {offset_bytes}"
            )
            return None

        return tokens

    def get_tokens_by_id(self, doc_id: int) -> Optional[np.ndarray]:
        """
        Retrieves the token array for a document identified by integer id.

        This is a thin convenience wrapper that converts the id to a normed_identifier via
        the associated ``GraphIndex`` and then delegates to ``get_tokens`` to
        preserve the existing behavior and validation logic.
        """
        normed_id = self.index.get_normed_identifier(doc_id)
        return self.get_tokens(normed_id)

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
