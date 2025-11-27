"""
DAGWiki Pre-tokenizer: A tool to pre-tokenize a graph dataset of Markdown files
into sharded binary files for efficient loading during model training.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import re
import sys
from functools import partial
from pathlib import Path
from queue import Empty
from time import sleep, time
from typing import Callable, List, Optional
import pickle

import numpy as np
import tiktoken
from tqdm import tqdm

from gpt_lab.data_sources.catalog_utils import BinaryShardIO
from gpt_lab.distributed import is_main
from gpt_lab.reproducibility import ReproducibilityManager
from gpt_lab.logger import setup_experiment_logging

logger = logging.getLogger(__name__)


def load_custom_tokenizer(tokenizer_path: Path):
    """Loads a custom tokenizer from a .pkl file."""
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Custom tokenizer file not found: {tokenizer_path}")
    
    logger.info(f"Loading custom tokenizer from: {tokenizer_path}")
    with open(tokenizer_path, 'rb') as f:
        tokenizer_data = pickle.load(f)
    
    enc = tiktoken.Encoding(
        name=tokenizer_path.stem,
        pat_str=tokenizer_data["pat_str"],
        mergeable_ranks=tokenizer_data["mergeable_ranks"],
        special_tokens=tokenizer_data.get("special_tokens", {})
    )
    return enc


def tokenize_worker(
    filepath: str,
    queue: mp.Queue,
    encode_fn: Callable[[str], List[int]],
    dtype: np.dtype,
):
    """
    Reads a single markdown file, extracts its title, tokenizes its content,
    and puts the result onto the queue.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            title_line = f.readline()
            if not title_line.startswith('# '):
                logger.warning(f"Skipping {filepath}: does not have a title header.")
                return
            title = os.path.splitext(os.path.basename(filepath))[0]
            content = f.read()
            
        # Clean hashes from links in the text to avoid polluting the model with implementation details.
        # We want [Link](Title) instead of [Link](Title_123456).
        # The hash is defined as exactly 6 hex characters at the end of the target.
        # Match pattern: ](target_hash) -> ](target)
        # Regex explanation:
        #   (\]\(.*?)   -> Group 1: Capture "](" and the start of the target
        #   _[0-9a-f]{6} -> Match underscore followed by 6 hex chars (the hash)
        #   (\))        -> Group 2: Capture the closing parenthesis
        content = re.sub(r'(\]\(.*?)_[0-9a-f]{6}(\))', r'\1\2', content)

        tokens = encode_fn(content)
        tokens_np = np.asarray(tokens, dtype=dtype)
        queue.put((title, tokens_np))

    except Exception as e:
        logger.error(f"Could not process file {filepath}: {e}")


def writer_process(
    queue: mp.Queue,
    output_dir: Path,
    graph_data: dict,
    metadata: dict,
    shard_size_gb: float,
    total_files: int,
):
    """
    Consumes tokenized data from the queue and writes it to sharded binary files.
    Also generates the final tokenized_graph.jsonl and metadata.json.
    """
    shard_size_bytes = int(shard_size_gb * (1024**3))
    token_metadata = {}
    shard_filenames = []
    processed_count = 0
    
    pbar = tqdm(total=total_files, desc="Processing files", unit="file")

    shard_idx = 0
    current_shard_file = None
    current_shard_offset = 0
    
    try:
        while True:
            try:
                # Wait for an item, but with a timeout to check for the sentinel
                # This helps prevent hanging if the producers die unexpectedly
                title, tokens = queue.get(timeout=10)
                
                if title is None: # Sentinel value
                    logger.info("Writer process received sentinel. Finalizing...")
                    break

                # If current shard is full or doesn't exist, create a new one
                if current_shard_file is None or (current_shard_offset + tokens.nbytes) > shard_size_bytes:
                    if current_shard_file:
                        finalize_shard(current_shard_file, current_shard_offset, metadata["dtype_str"])
                    
                    shard_filename = output_dir / f"shard_{shard_idx:06d}.bin"
                    shard_filenames.append(shard_filename.name)
                    logger.info(f"Creating new shard: {shard_filename}")
                    
                    current_shard_file = open(shard_filename, "wb")
                    # Write a placeholder header, we'll fill it in at the end
                    current_shard_file.write(np.zeros(256, dtype=np.int32).tobytes())
                    current_shard_offset = 256 * 4 # Start after the header
                    shard_idx += 1

                # Write tokens and record metadata
                start_offset = current_shard_offset
                current_shard_file.write(tokens.tobytes())
                current_shard_offset += tokens.nbytes
                
                token_metadata[title] = {
                    "tok_shard_idx": shard_idx - 1,
                    "tok_offset_bytes": start_offset,
                    "tok_len": len(tokens),
                }
                
                processed_count += 1
                pbar.update(1)

            except Empty:
                logger.info("Queue is empty, waiting for more items...")
                # This is a simple check to see if all work is done.
                # A more robust implementation might use a separate signal.
                if pbar.n >= total_files:
                    logger.warning("Queue empty and all files processed. Exiting writer loop.")
                    break
                sleep(1)

    finally:
        if current_shard_file:
            finalize_shard(current_shard_file, current_shard_offset, metadata["dtype_str"])
        pbar.close()

    # --- Finalization ---
    logger.info("Aggregating final graph data...")
    final_graph_data = []
    for title, data in tqdm(graph_data.items(), desc="Merging graph data"):
        if title in token_metadata:
            data.update(token_metadata[title])
            final_graph_data.append(data)
        else:
            logger.warning(f"Title '{title}' from graph.jsonl not found in tokenized files. Excluding.")
            
    # Write tokenized_graph.jsonl
    output_graph_file = output_dir / "tokenized_graph.jsonl"
    logger.info(f"Writing tokenized graph to {output_graph_file}...")
    with open(output_graph_file, "w", encoding="utf-8") as f:
        for item in final_graph_data:
            f.write(json.dumps(item) + "\n")
            
    # Write metadata.json
    metadata["shard_filenames"] = shard_filenames
    output_metadata_file = output_dir / "metadata.json"
    logger.info(f"Writing metadata to {output_metadata_file}...")
    with open(output_metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("Pre-tokenization complete.")


def finalize_shard(file_handle, total_bytes: int, dtype_str: str):
    """Writes the final header to a shard file and closes it."""
    file_handle.seek(0)
    dtype = np.dtype(dtype_str)
    token_count = (total_bytes - 256 * 4) // dtype.itemsize
    
    header = np.zeros(256, dtype=np.int32)
    header[0] = 11041999  # Magic number
    header[1] = 1         # Version
    header[2] = token_count
    header[3] = dtype.itemsize
    
    file_handle.write(header.tobytes())
    file_handle.close()
    logger.info(f"Finalized shard {file_handle.name} with {token_count:,} tokens.")


# ===========================================================================
# Main Execution
# ===========================================================================

def run_preprocessing(args, rep: ReproducibilityManager):
    """The main logic of the preprocessing script, wrapped to be called by the manager."""
    
    # --- Setup Logging ---
    # The ReproducibilityManager gives us a unique output directory.
    # We set up our logging to go there.
    if rep.output_dir:
        setup_experiment_logging(rep.output_dir, rank=0, is_main_process=True)

    # Log a structured snapshot of the reproducibility context
    logger.info(
        "System Information",
        extra={
            "git_info": rep.get_git_info(),
            "software_environment": rep.software_environment,
            "runtime_environment": rep.runtime_environment,
            "run_invocation": rep.run_invocation,
        },
    )

    # --- Tokenizer Setup ---
    # As requested, the core logic uses a callable. 
    # Prioritize loading a custom tokenizer if a file is provided.
    try:
        if args.tokenizer_file:
            enc = load_custom_tokenizer(args.tokenizer_file)
            tokenizer_name = args.tokenizer_file.stem
        else:
            logger.info(f"Loading standard tiktoken tokenizer: {args.tokenizer_name}")
            enc = tiktoken.get_encoding(args.tokenizer_name)
            tokenizer_name = args.tokenizer_name
        
        encode_fn = enc.encode
        vocab_size = enc.n_vocab
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return

    token_dtype = BinaryShardIO.pick_token_dtype(vocab_size)
    logger.info(f"Using tokenizer '{tokenizer_name}' with vocab size {vocab_size}. Selected token dtype: {token_dtype.__name__}")

    # --- Load Graph Data ---
    logger.info(f"Loading graph data from {args.graph_file}...")
    try:
        with open(args.graph_file, "r", encoding="utf-8") as f:
            # Create a dictionary from the generator for the writer process
            lines = f.readlines()
            graph_data = {json.loads(line)['title']: json.loads(line) for line in lines}
    except Exception as e:
        logger.error(f"Failed to load graph file: {e}")
        return
    logger.info(f"Loaded {len(graph_data):,} nodes from graph file.")

    # --- File Discovery ---
    logger.info("Discovering markdown files...")
    md_files = sorted(list(args.input_dir.rglob("*.md")))
    if not md_files:
        logger.error("No markdown files found in the input directory.")
        return
    logger.info(f"Found {len(md_files)} markdown files to process.")

    # --- Multiprocessing Setup ---
    manager = mp.Manager()
    queue = manager.Queue()

    dataset_metadata = {
        "tokenizer": tokenizer_name,
        "dtype_str": token_dtype.__name__,
        "shard_filenames": [], # Will be populated by the writer
    }
    
    # --- Start Processes ---
    # The writer process gets the unique output directory from the ReproducibilityManager
    writer = mp.Process(
        target=writer_process,
        args=(
            queue,
            Path(rep.output_dir), # Use the manager's unique output directory
            graph_data,
            dataset_metadata,
            args.shard_size_gb,
            len(md_files)
        ),
    )
    writer.start()

    with mp.Pool(processes=args.processes) as pool:
        worker_fn = partial(
            tokenize_worker,
            queue=queue,
            encode_fn=encode_fn,
            dtype=token_dtype,
        )
        # Using imap_unordered for potentially better performance as results are processed as they complete
        # A simple pool.map is also fine and was used before.
        list(tqdm(pool.imap_unordered(worker_fn, md_files), total=len(md_files), desc="Tokenizing"))
    
    # --- Signal writer to finish and wait ---
    queue.put((None, None)) # Sentinel value
    logger.info("All files sent to workers. Waiting for writer to finish...")
    writer.join()
    logger.info("All processes finished.")


def main():
    parser = argparse.ArgumentParser(
        prog="DAGWiki Pre-tokenizer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the extracted Markdown files."
    )
    parser.add_argument(
        "graph_file",
        type=Path,
        help="Path to the graph.jsonl file."
    )
    parser.add_argument(
        "-o", "--runs-dir",
        type=Path,
        required=True,
        help="Root directory to store experiment runs. A unique sub-directory will be created here."
    )
    parser.add_argument(
        "--tokenizer-file",
        type=Path,
        default=None,
        help="Path to a custom .pkl tokenizer file. Overrides --tokenizer-name."
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="Name of the tiktoken tokenizer to use if --tokenizer-file is not provided (e.g., 'gpt2', 'cl100k_base')."
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=2.0,
        help="Target size for each binary shard in gigabytes."
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help=f"Number of worker processes to use (default: {max(1, mp.cpu_count() - 1)})."
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress reporting and info messages."
    )
    args = parser.parse_args()

    # Basic logging setup for messages that happen before the ReproducibilityManager takes over
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # The ReproducibilityManager will handle creating a unique output directory,
    # capturing git state, and setting up file-based logging for the run.
    with ReproducibilityManager(
        output_dir=str(args.runs_dir), 
        is_main_process=True
    ) as rep:
        run_preprocessing(args, rep)


if __name__ == "__main__":
    main()
