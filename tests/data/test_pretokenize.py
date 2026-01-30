import json
import pytest
import numpy as np
import tiktoken
from pathlib import Path
import argparse
import logging

from tunalab.reproducibility import ReproducibilityManager
from tunalab.pretokenized_data.shard_io import BinaryShardIO

from data.pretokenize import run_preprocessing, load_custom_tokenizer


# Disable excessive logging from the script during tests
logging.disable(logging.INFO)


@pytest.fixture(scope="module")
def dummy_pretokenize_input(tmpdir_factory):
    """
    Creates a temporary directory with dummy markdown files and a graph.jsonl
    to serve as input for the pre-tokenization script.
    """
    input_dir = Path(tmpdir_factory.mktemp("pretokenize_input"))
    
    # Create dummy markdown files
    md_content = {
        "Doc_A": "This is the first document.",
        "Doc_B": "Here is the second one.",
        "Doc_C": "And a third, slightly longer document."
    }
    for title, content in md_content.items():
        with open(input_dir / f"{title}.md", "w") as f:
            f.write(f"# {title}\n{content}")

    # Create dummy graph.jsonl (titles must match filenames without .md extension)
    graph_data = [
        {"title": "Doc_A", "outgoing": ["Doc_B"], "incoming": []},
        {"title": "Doc_B", "outgoing": [], "incoming": ["Doc_A"]},
        {"title": "Doc_C", "outgoing": [], "incoming": []},
    ]
    graph_file_path = input_dir / "graph.jsonl"
    with open(graph_file_path, "w") as f:
        for entry in graph_data:
            f.write(json.dumps(entry) + "\n")
            
    return input_dir, graph_file_path


def test_pretokenize_script_with_tiktoken(dummy_pretokenize_input, tmpdir_factory):
    """
    Tests the main logic of the pre-tokenization script with a standard tiktoken tokenizer.
    """
    input_dir, graph_file_path = dummy_pretokenize_input
    runs_dir = Path(tmpdir_factory.mktemp("pretokenize_runs"))

    # 1. Setup arguments for the script's main function
    args = argparse.Namespace(
        input_dir=input_dir,
        graph_file=graph_file_path,
        runs_dir=runs_dir,
        tokenizer_file=None,
        tokenizer_name="gpt2",
        shard_size_gb=0.01,  # Small shard size to ensure it gets created
        processes=1,
        quiet=True
    )
    
    # Use the real ReproducibilityManager to mimic the actual script execution
    with ReproducibilityManager(output_dir=str(runs_dir), is_main_process=True) as rep:
        # 2. Run the preprocessing logic
        run_preprocessing(args, rep)
        
        # 3. Validate the output
        output_dir = Path(rep.output_dir)
        assert output_dir.is_dir()

        # Check for essential files
        metadata_path = output_dir / "metadata.json"
        graph_path = output_dir / "tokenized_graph.jsonl"
        shard_path = output_dir / "shard_000000.bin"
        
        assert metadata_path.exists()
        assert graph_path.exists()
        assert shard_path.exists()

        # Validate metadata.json
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert metadata["tokenizer"] == "gpt2"
        assert metadata["dtype_str"] == "uint16"
        assert metadata["shard_filenames"] == ["shard_000000.bin"]

        # Validate tokenized_graph.jsonl
        enc = tiktoken.get_encoding("gpt2")
        expected_tokens = {
            "Doc_A": enc.encode("# Doc_A\nThis is the first document."),
            "Doc_B": enc.encode("# Doc_B\nHere is the second one."),
            "Doc_C": enc.encode("# Doc_C\nAnd a third, slightly longer document.")
        }

        with open(graph_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3
        
        graph_nodes = {json.loads(line)['title']: json.loads(line) for line in lines}
        assert graph_nodes["Doc_A"]["tok_len"] == len(expected_tokens["Doc_A"])
        assert graph_nodes["Doc_B"]["tok_len"] == len(expected_tokens["Doc_B"])
        assert graph_nodes["Doc_C"]["tok_len"] == len(expected_tokens["Doc_C"])
        assert graph_nodes["Doc_A"]["tok_shard_idx"] == 0

        # Validate shard content
        all_read_tokens = BinaryShardIO.read_datafile_tokens_memmap(shard_path, dtype=np.uint16)
        
        expected_all_tokens = []
        for title in ["Doc_A", "Doc_B", "Doc_C"]: # Assumes this processing order
            expected_all_tokens.extend(expected_tokens[title])
            
        # Note: The actual order of processing can vary with multiprocessing.
        # For a robust test with multiple workers, we'd need to read the offsets
        # from the graph.jsonl and check slices. For a single worker, this is fine.
        if args.processes == 1:
            doc_a_info = graph_nodes["Doc_A"]
            offset_a = (doc_a_info["tok_offset_bytes"] - 1024) // 2
            len_a = doc_a_info["tok_len"]
            assert np.array_equal(all_read_tokens[offset_a:offset_a+len_a], expected_tokens["Doc_A"])
            
            doc_b_info = graph_nodes["Doc_B"]
            offset_b = (doc_b_info["tok_offset_bytes"] - 1024) // 2
            len_b = doc_b_info["tok_len"]
            assert np.array_equal(all_read_tokens[offset_b:offset_b+len_b], expected_tokens["Doc_B"])

# Re-enable logging
logging.disable(logging.NOTSET)
