import json
import pytest
import numpy as np
from pathlib import Path

from experiments.dagseq2dagseq.data.dataset import GraphIndex, PretokShardedBackend


@pytest.fixture(scope="module")
def dummy_run_directory(tmpdir_factory):
    """
    Creates a temporary directory with a dummy pre-tokenized dataset
    for testing purposes. This fixture has a 'module' scope, so it's
    created once per test module run.
    """
    run_dir = Path(tmpdir_factory.mktemp("dummy_run"))

    # 1. Create metadata.json
    metadata = {
        "tokenizer": "dummy_tokenizer",
        "dtype_str": "uint16",
        "shard_filenames": ["shard_000000.bin"]
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # 2. Create a binary shard file
    # Node A: [0, 1, 2, 3, 4]
    # Node B: [10, 11, 12]
    # Node C: [20, 21, 22, 23, 24, 25]
    tokens_a = np.array([0, 1, 2, 3, 4], dtype=np.uint16)
    tokens_b = np.array([10, 11, 12], dtype=np.uint16)
    tokens_c = np.array([20, 21, 22, 23, 24, 25], dtype=np.uint16)
    
    all_tokens = np.concatenate([tokens_a, tokens_b, tokens_c])
    
    shard_path = run_dir / "shard_000000.bin"
    
    # We write a dummy header and then the tokens, similar to the pretokenize script.
    # The BinaryShardIO reader will correctly skip the header.
    header = np.zeros(256, dtype=np.int32)
    header[0] = 11041999  # Magic number
    header[1] = 1         # Version
    header[2] = len(all_tokens) # Total tokens
    header[3] = np.dtype(metadata['dtype_str']).itemsize

    with open(shard_path, "wb") as f:
        f.write(header.tobytes())
        f.write(all_tokens.tobytes())

    # 3. Create tokenized_graph.jsonl
    header_size_bytes = 256 * 4
    dtype_size_bytes = 2

    graph_data = [
        {
            "title": "Node A", "outgoing": ["Node B"], "incoming": [],
            "tok_shard_idx": 0,
            "tok_offset_bytes": header_size_bytes + (0 * dtype_size_bytes),
            "tok_len": len(tokens_a)
        },
        {
            "title": "Node B", "outgoing": ["Node C"], "incoming": ["Node A"],
            "tok_shard_idx": 0,
            "tok_offset_bytes": header_size_bytes + (len(tokens_a) * dtype_size_bytes),
            "tok_len": len(tokens_b)
        },
        {
            "title": "Node C", "outgoing": [], "incoming": ["Node B"],
            "tok_shard_idx": 0,
            "tok_offset_bytes": header_size_bytes + ((len(tokens_a) + len(tokens_b)) * dtype_size_bytes),
            "tok_len": len(tokens_c)
        }
    ]
    with open(run_dir / "tokenized_graph.jsonl", "w") as f:
        for entry in graph_data:
            f.write(json.dumps(entry) + "\n")
            
    return run_dir


def test_graph_index_loading(dummy_run_directory):
    """Tests that GraphIndex loads data correctly."""
    index = GraphIndex(dummy_run_directory)

    assert len(index) == 3
    assert "Node A" in index
    assert "Node D" not in index
    assert index.token_dtype == np.uint16
    assert len(index.shard_filenames) == 1
    assert index.shard_filenames[0].name == "shard_000000.bin"

    node_b = index.get_node("Node B")
    assert node_b is not None
    assert node_b["title"] == "Node B"
    assert node_b["outgoing"] == ["Node C"]
    assert node_b["incoming"] == ["Node A"]
    assert node_b["tok_len"] == 3

    assert index.get_all_titles() == ["Node A", "Node B", "Node C"]

    # Basic id mapping sanity checks
    node_a_id = index.get_id("Node A")
    node_b_id = index.get_id("Node B")
    node_c_id = index.get_id("Node C")

    assert index.get_title(node_a_id) == "Node A"
    assert index.get_title(node_b_id) == "Node B"
    assert index.get_title(node_c_id) == "Node C"

    # Neighbor helpers should reflect the simple chain A -> B -> C
    assert index.neighbors_out(node_a_id) == [node_b_id]
    assert index.neighbors_out(node_b_id) == [node_c_id]
    assert index.neighbors_out(node_c_id) == []

    assert index.neighbors_in(node_a_id) == []
    assert index.neighbors_in(node_b_id) == [node_a_id]
    assert index.neighbors_in(node_c_id) == [node_b_id]

    # Token lengths
    assert index.get_token_len(node_a_id) == 5
    assert index.get_token_len(node_b_id) == 3
    assert index.get_token_len(node_c_id) == 6


def test_graph_index_file_not_found():
    """Tests that GraphIndex raises errors for missing files."""
    with pytest.raises(FileNotFoundError):
        GraphIndex(Path("/tmp/non_existent_dir"))
    
    with pytest.raises(FileNotFoundError, match="metadata.json not found"):
        GraphIndex(Path(".")) # Assuming metadata.json is not in root


def test_graph_index_invalid_token_metadata_missing_field(tmp_path):
    """GraphIndex should raise a clear error if required token metadata is missing."""
    run_dir = tmp_path / "run_missing_field"
    run_dir.mkdir()

    metadata = {
        "tokenizer": "dummy_tokenizer",
        "dtype_str": "uint16",
        "shard_filenames": ["shard_000000.bin"],
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # tokenized_graph.jsonl with a node missing tok_offset_bytes
    bad_node = {
        "title": "Bad Node",
        "outgoing": [],
        "incoming": [],
        "tok_shard_idx": 0,
        # "tok_offset_bytes" missing on purpose
        "tok_len": 5,
    }
    with open(run_dir / "tokenized_graph.jsonl", "w") as f:
        f.write(json.dumps(bad_node) + "\n")

    with pytest.raises(ValueError, match="Bad Node"):
        GraphIndex(run_dir)


def test_graph_index_invalid_token_metadata_types(tmp_path):
    """GraphIndex should raise a clear error if token metadata has wrong types."""
    run_dir = tmp_path / "run_bad_types"
    run_dir.mkdir()

    metadata = {
        "tokenizer": "dummy_tokenizer",
        "dtype_str": "uint16",
        "shard_filenames": ["shard_000000.bin"],
    }
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # tokenized_graph.jsonl with wrong types for tok_len
    bad_node = {
        "title": "Bad Node",
        "outgoing": [],
        "incoming": [],
        "tok_shard_idx": 0,
        "tok_offset_bytes": 256 * 4,
        "tok_len": "5",  # wrong type on purpose
    }
    with open(run_dir / "tokenized_graph.jsonl", "w") as f:
        f.write(json.dumps(bad_node) + "\n")

    with pytest.raises(ValueError, match="Bad Node"):
        GraphIndex(run_dir)

def test_pretok_sharded_backend_initialization(dummy_run_directory):
    """Tests that the backend initializes correctly."""
    index = GraphIndex(dummy_run_directory)
    backend = PretokShardedBackend(index)
    assert backend.index is index
    assert not backend._memmaps # Memmaps should be lazily loaded
    assert backend.dtype == np.uint16


def test_pretok_sharded_backend_get_tokens(dummy_run_directory):
    """Tests retrieving token data from the sharded backend."""
    index = GraphIndex(dummy_run_directory)
    backend = PretokShardedBackend(index)

    # Test retrieving each node's tokens
    tokens_a = backend.get_tokens("Node A")
    assert tokens_a is not None
    assert np.array_equal(tokens_a, np.array([0, 1, 2, 3, 4], dtype=np.uint16))

    tokens_b = backend.get_tokens("Node B")
    assert tokens_b is not None
    assert np.array_equal(tokens_b, np.array([10, 11, 12], dtype=np.uint16))

    tokens_c = backend.get_tokens("Node C")
    assert tokens_c is not None
    assert np.array_equal(tokens_c, np.array([20, 21, 22, 23, 24, 25], dtype=np.uint16))

    # Test lazy loading of memmap
    assert len(backend._memmaps) == 1
    # Requesting again should not create a new memmap
    backend.get_tokens("Node A")
    assert len(backend._memmaps) == 1

    # Test non-existent node
    tokens_d = backend.get_tokens("Node D")
    assert tokens_d is None

    backend.close()
    assert len(backend._memmaps) == 0


def test_pretok_sharded_backend_get_tokens_by_id(dummy_run_directory):
    """Tests retrieving token data by integer document id."""
    index = GraphIndex(dummy_run_directory)
    backend = PretokShardedBackend(index)

    node_a_id = index.get_id("Node A")
    node_c_id = index.get_id("Node C")

    tokens_a = backend.get_tokens_by_id(node_a_id)
    assert tokens_a is not None
    assert np.array_equal(tokens_a, np.array([0, 1, 2, 3, 4], dtype=np.uint16))

    tokens_c = backend.get_tokens_by_id(node_c_id)
    assert tokens_c is not None
    assert np.array_equal(tokens_c, np.array([20, 21, 22, 23, 24, 25], dtype=np.uint16))

    backend.close()
