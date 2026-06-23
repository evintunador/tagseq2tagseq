"""
Pre-tokenizer: Converts a graph dataset into sharded binary token files for
efficient loading during model training.

The content source is pluggable — pass any DocumentSource (see
data/document_sources.py) to run_preprocessing. The default CLI entry
point handles Wikipedia markdown directories; see pretokenize_stack.py
for The Stack variant.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
from time import sleep
import pickle

import numpy as np
import tiktoken
from tqdm import tqdm

from tunalab.pretokenized_data.shard_io import BinaryShardIO
from tunalab.reproducibility import ReproducibilityManager
from tunalab import tracking


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


# Per-worker globals, populated by _worker_init via the Pool initializer.
# The result queue MUST reach workers by fork-inheritance (initializer args),
# never by pickling through imap — a plain mp.Queue is not picklable
# (`RuntimeError: Queue objects should only be shared ... through inheritance`),
# which is why these are module globals rather than partial() arguments.
_WORKER_QUEUE = None
_WORKER_ENCODE_FN = None
_WORKER_DTYPE = None


def _worker_init(queue, encode_fn, dtype):
    """Pool initializer: stash the (inherited) queue/encoder/dtype in globals."""
    global _WORKER_QUEUE, _WORKER_ENCODE_FN, _WORKER_DTYPE
    _WORKER_QUEUE = queue
    _WORKER_ENCODE_FN = encode_fn
    _WORKER_DTYPE = dtype


def tokenize_worker(record: tuple):
    """
    Tokenizes a (normed_id, content_str) record and puts the result onto the
    module-global result queue (set by _worker_init).

    Content pre-processing (e.g. hash-stripping for Wikipedia) is the
    responsibility of the DocumentSource, not this worker.
    """
    try:
        normed_id, content = record
        tokens = _WORKER_ENCODE_FN(content)
        tokens_np = np.asarray(tokens, dtype=_WORKER_DTYPE)
        _WORKER_QUEUE.put((normed_id, tokens_np))
    except Exception as e:
        logger.error(f"Could not process record '{record[0] if record else '?'}': {e}")


def _load_split_lookup(splits_file: Path) -> dict:
    """Build a normed_id -> split_name lookup from a splits.json file."""
    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
    lookup = {}
    for split_name, split_info in splits_data["splits"].items():
        for nid in split_info["ids"]:
            lookup[nid] = split_name
    return lookup


def writer_process(
    queue: mp.Queue,
    output_dir: Path,
    graph_data: dict,
    metadata: dict,
    shard_size_gb: float,
    total_files: int,
    split_lookup: dict | None = None,
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
                normed_id, tokens = queue.get(timeout=10)

                if normed_id is None: # Sentinel value
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

                token_metadata[normed_id] = {
                    "tok_shard_idx": shard_idx - 1,
                    "tok_offset_bytes": start_offset,
                    "tok_len": len(tokens),
                }
                
                processed_count += 1
                pbar.update(1)

            except Empty:
                # Drain strictly until the explicit sentinel — never exit on a
                # count heuristic. The old `pbar.n >= total_files` early-exit
                # could stop draining while a worker was still mid-`put`, which
                # blocked that worker and in turn deadlocked the pool's join.
                # The producer side guarantees a sentinel is sent only after all
                # workers have finished and flushed (see run_preprocessing), so
                # waiting here is bounded and safe.
                logger.info("Queue is empty, waiting for more items...")
                sleep(1)

    finally:
        if current_shard_file:
            finalize_shard(current_shard_file, current_shard_offset, metadata["dtype_str"])
        pbar.close()

    # --- Finalization ---
    logger.info("Aggregating final graph data...")
    final_graph_data = []
    for normed_id, data in tqdm(graph_data.items(), desc="Merging graph data"):
        if normed_id in token_metadata:
            data.update(token_metadata[normed_id])
            if split_lookup is not None:
                data["split"] = split_lookup.get(normed_id)
            final_graph_data.append(data)
        else:
            logger.warning(f"normed_identifier '{normed_id}' from graph.jsonl not found in tokenized files. Excluding.")
            
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

def run_preprocessing(args, rep: ReproducibilityManager, source=None):
    """
    Core pre-tokenization logic.

    Args:
        args: Parsed CLI arguments.
        rep: ReproducibilityManager instance.
        source: Any iterable of (normed_id, content_str) pairs that also
            supports len(). If None, falls back to a MarkdownDirectorySource
            built from args.input_dir (original Wikipedia behaviour).
    """
    
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
            graph_data = {json.loads(line)['normed_identifier']: json.loads(line) for line in lines}
    except Exception as e:
        logger.error(f"Failed to load graph file: {e}")
        return
    logger.info(f"Loaded {len(graph_data):,} nodes from graph file.")

    # --- Content Source ---
    if source is None:
        from data.document_sources import MarkdownDirectorySource
        source = MarkdownDirectorySource(args.input_dir)
    if len(source) == 0:
        logger.error("Source contains no documents.")
        return
    logger.info(f"Source has {len(source):,} documents to process.")

    # --- Load Splits (optional) ---
    split_lookup = None
    splits_file = getattr(args, "splits_file", None)
    if splits_file and Path(splits_file).exists():
        logger.info(f"Loading split assignments from {splits_file}...")
        split_lookup = _load_split_lookup(Path(splits_file))
        logger.info(f"Loaded split assignments for {len(split_lookup):,} nodes.")

    # --- Multiprocessing Setup ---
    # Use a plain mp.Queue, NOT mp.Manager().Queue(). The Manager spins up a
    # server process whose per-connection socket threads (seen blocked in
    # `unix_stream_data_wait`) do not shut down cleanly, which pinned the writer
    # child alive after it finished and deadlocked the pool/writer join at the
    # end of a full run — leaving the dataset without metadata.json and the
    # pipeline stuck before the split step. A plain mp.Queue has no such server.
    queue: mp.Queue = mp.Queue()

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
            Path(rep.output_dir),
            graph_data,
            dataset_metadata,
            args.shard_size_gb,
            len(source),
            split_lookup,
        ),
    )
    writer.start()

    # The queue, encoder and dtype reach workers by fork-inheritance via the
    # Pool initializer — NOT as pickled imap arguments (a plain mp.Queue can't be
    # pickled). Workers then receive only the record through imap_unordered.
    #
    # Teardown ordering matters with a plain mp.Queue: the writer drains the
    # queue CONCURRENTLY while workers run, so by the time imap is exhausted the
    # workers have flushed their puts. close()+join() then reaps them cleanly;
    # terminate() in finally is a hard backstop so a stray worker can never wedge
    # the join (the deadlock that previously left runs hung after tokenizing).
    pool = mp.Pool(
        processes=args.processes,
        initializer=_worker_init,
        initargs=(queue, encode_fn, token_dtype),
    )
    try:
        list(tqdm(pool.imap_unordered(tokenize_worker, source),
                  total=len(source), desc="Tokenizing"))
        pool.close()
        pool.join()
    finally:
        pool.terminate()

    # --- Signal writer to finish and wait ---
    queue.put((None, None)) # Sentinel value
    logger.info("All files sent to workers. Waiting for writer to finish...")
    writer.join()
    logger.info("All processes finished.")

    # Release the queue's own feeder thread / pipe so this process can exit
    # promptly instead of lingering on interpreter shutdown.
    queue.close()
    queue.join_thread()


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
        "--splits-file",
        type=Path,
        default=None,
        help="Path to splits.json produced by the graph splitter. When provided, "
             "each node in tokenized_graph.jsonl is annotated with a 'split' field.",
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
