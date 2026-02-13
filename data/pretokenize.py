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
from functools import partial
from pathlib import Path
from queue import Empty
from time import sleep
from typing import Callable, List, Literal
import pickle

import numpy as np
import tiktoken
from tqdm import tqdm

from tunalab.pretokenized_data.shard_io import BinaryShardIO
from tunalab.reproducibility import ReproducibilityManager
from tunalab import tracking

from .dataset_config import DatasetConfig, save_config_to_pretokenized_dir


logger = logging.getLogger(__name__)


# Title extraction strategies
def extract_title_flat(filepath: str, input_dir: Path) -> str:
    """Extract title as basename only (for flat datasets like Wikipedia)."""
    return os.path.splitext(os.path.basename(filepath))[0]


def extract_title_hierarchical(filepath: str, input_dir: Path) -> str:
    """Extract title as relative path from input_dir (for nested datasets like GitHub)."""
    relative_path = os.path.relpath(filepath, input_dir)
    return os.path.splitext(relative_path)[0]


def build_source_id_to_title_map(graph_data: dict) -> dict:
    """
    Build a mapping from source_identifier to normalized title.
    
    The graph stores both the normalized title and the original source_identifier
    in metadata. This creates a reverse lookup.
    
    Args:
        graph_data: Dictionary of graph nodes keyed by normalized title
    
    Returns:
        Dictionary mapping source_identifier -> normalized title
    """
    source_id_to_title = {}
    
    for title, node_data in graph_data.items():
        source_id = node_data.get('source_identifier')
        if source_id:
            source_id_to_title[source_id] = title
        else:
            logger.warning(f"Graph node '{title}' has no source_identifier in metadata")
    
    return source_id_to_title


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
    doc_tuple: tuple,  # (source_identifier, content)
    queue: mp.Queue,
    encode_fn: Callable[[str], List[int]],
    dtype: np.dtype,
    source_id_to_title: dict,
    hash_length: int,
):
    """
    Tokenizes a document and puts the result onto the queue.
    
    Args:
        doc_tuple: (source_identifier, content) tuple
        queue: Multiprocessing queue for results
        encode_fn: Tokenization function
        dtype: NumPy dtype for tokens
        source_id_to_title: Mapping from source identifiers to normalized titles
        hash_length: Length of hash suffix to strip from links
    """
    source_id, content = doc_tuple
    
    try:
        # Look up the normalized title from the graph
        title = source_id_to_title.get(source_id)
        if title is None:
            logger.warning(f"Source identifier '{source_id}' not found in graph, skipping")
            return
            
        # Clean hashes from links in the text to avoid polluting the model with implementation details.
        # We want [Link](Title) instead of [Link](Title_123456).
        # The hash is defined as exactly hash_length hex characters at the end of the target.
        # Match pattern: ](target_hash) -> ](target)
        # Regex explanation:
        #   (\]\(.*?)   -> Group 1: Capture "](" and the start of the target
        #   _[0-9a-f]{N} -> Match underscore followed by N hex chars (the hash)
        #   (\))        -> Group 2: Capture the closing parenthesis
        hash_pattern = rf'(\]\(.*?)_[0-9a-f]{{{hash_length}}}(\))'
        content = re.sub(hash_pattern, r'\1\2', content)

        tokens = encode_fn(content)
        tokens_np = np.asarray(tokens, dtype=dtype)
        queue.put((title, tokens_np))

    except Exception as e:
        logger.error(f"Could not process document '{source_id}': {e}")


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
    
    # Write dataset_config.json for model to know how to handle titles
    # This is returned from the writer_process indirectly via the parent function
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
        log_dir = os.path.join(rep.output_dir, "logs")
        tracking.init(log_dir, rank=0)

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

    # --- Create Content Source ---
    logger.info(f"Creating content source (type: {args.source_type})...")
    
    if args.source_type == "markdown":
        from data.extractors.sources import MarkdownFileSource
        source = MarkdownFileSource(args.input_dir)
    elif args.source_type == "jsonl":
        from data.extractors.sources import JSONLSource
        if not args.input_file:
            logger.error("--input-file required for jsonl source type")
            return
        source = JSONLSource(
            args.input_file,
            identifier_field=args.identifier_field,
            content_field=args.content_field,
            additional_fields=args.additional_fields or []
        )
    elif args.source_type == "thestack":
        from data.extractors.sources import TheStackJSONLSource
        if not args.input_file:
            logger.error("--input-file required for thestack source type")
            return
        source = TheStackJSONLSource(
            args.input_file,
            repo_field=args.repo_field or "max_stars_repo_name",
            path_field=args.path_field or "max_stars_repo_path",
            content_field=args.content_field
        )
    else:
        logger.error(f"Unknown source type: {args.source_type}")
        return
    
    # --- Load documents from source ---
    logger.info("Loading documents from source...")
    documents = []
    for doc in source.iter_documents():
        documents.append((doc.identifier, doc.content))
    logger.info(f"Loaded {len(documents)} documents from source")
    
    # --- Build source_id to title mapping from graph ---
    logger.info("Building source_id to title mapping from graph...")
    source_id_to_title = build_source_id_to_title_map(graph_data)
    logger.info(f"Graph contains {len(source_id_to_title)} source_identifier mappings")
    
    # Check how many documents can be mapped
    mappable = sum(1 for source_id, _ in documents if source_id in source_id_to_title)
    logger.info(f"Can map {mappable}/{len(documents)} documents to graph titles")
    
    if mappable < len(documents):
        unmapped = len(documents) - mappable
        logger.warning(f"{unmapped} documents not found in graph and will be skipped")

    # --- Create Dataset Config Early ---
    # We need this early to get hash_length for tokenization
    sample_titles = list(graph_data.keys())[:10] if graph_data else []
    
    # Auto-detect title format from actual graph titles
    if any(':' in title for title in sample_titles):
        title_format = 'colon_separated'  # GitHub style: repo:path
    elif any('/' in title for title in sample_titles):
        title_format = 'hierarchical'  # Nested paths
    else:
        title_format = 'flat'  # Simple names
    
    # Auto-detect link format from source type
    if args.source_type == 'thestack':
        link_format = 'python_import'
        normalizer_type = 'python_module'
    else:
        link_format = 'markdown'
        normalizer_type = 'passthrough'  # Assume pre-normalized for markdown sources
    
    dataset_config = DatasetConfig(
        name=args.dataset_name or f"Dataset from {args.input_dir.name if args.input_dir else args.input_file.name}",
        title_format=title_format,
        link_format=link_format,
        normalizer_type=normalizer_type,
        hash_length=6,  # TODO: Make this configurable via CLI
        description=f"Pretokenized from {args.source_type} source"
    )
    logger.info(f"Dataset configuration: {dataset_config.name} (title: {title_format}, links: {link_format}, hash: {dataset_config.hash_length})")

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
            len(documents)
        ),
    )
    writer.start()

    with mp.Pool(processes=args.processes) as pool:
        worker_fn = partial(
            tokenize_worker,
            queue=queue,
            encode_fn=encode_fn,
            dtype=token_dtype,
            source_id_to_title=source_id_to_title,
            hash_length=dataset_config.hash_length,
        )
        # Using imap_unordered for potentially better performance as results are processed as they complete
        list(tqdm(pool.imap_unordered(worker_fn, documents), total=len(documents), desc="Tokenizing"))
    
    # --- Signal writer to finish and wait ---
    queue.put((None, None)) # Sentinel value
    logger.info("All files sent to workers. Waiting for writer to finish...")
    writer.join()
    logger.info("All processes finished.")
    
    # Save dataset config to output directory
    save_config_to_pretokenized_dir(dataset_config, Path(rep.output_dir))
    logger.info(f"Saved dataset configuration to {rep.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        prog="DAGWiki Pre-tokenizer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs='?',
        help="Directory containing source files (for markdown source type)."
    )
    parser.add_argument(
        "graph_file",
        type=Path,
        help="Path to the graph.jsonl file."
    )
    parser.add_argument(
        "--source-type",
        type=str,
        choices=['markdown', 'jsonl', 'thestack'],
        default='markdown',
        help="Type of content source: 'markdown' (.md files), 'jsonl' (generic JSON Lines), or 'thestack' (TheStack repository data)."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Input file path (required for jsonl source type)."
    )
    parser.add_argument(
        "--identifier-field",
        type=str,
        default="identifier",
        help="JSON field to use as document identifier (for jsonl source)."
    )
    parser.add_argument(
        "--content-field",
        type=str,
        default="content",
        help="JSON field containing document content (for jsonl/thestack source)."
    )
    parser.add_argument(
        "--repo-field",
        type=str,
        help="JSON field containing repository name (for thestack source, default: max_stars_repo_name)."
    )
    parser.add_argument(
        "--path-field",
        type=str,
        help="JSON field containing file path (for thestack source, default: max_stars_repo_path)."
    )
    parser.add_argument(
        "--additional-fields",
        type=str,
        nargs='*',
        help="Additional JSON fields to include in metadata (for jsonl source)."
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
        "--dataset-name",
        type=str,
        default=None,
        help="Optional name for the dataset (saved in config)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress reporting and info messages."
    )
    args = parser.parse_args()

    # Validate source-type specific arguments
    if args.source_type == 'markdown':
        if not args.input_dir:
            parser.error("input_dir is required for markdown source type")
    elif args.source_type in ['jsonl', 'thestack']:
        if not args.input_file:
            parser.error(f"--input-file is required for {args.source_type} source type")

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
