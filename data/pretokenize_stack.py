"""
The Stack Pre-tokenizer: Pre-tokenize a Stack JSONL dataset into sharded
binary files using the dependency graph produced by
data/github_graph_extractor/build_graph_streaming.py.

Usage:
    python -m data.pretokenize_stack \\
        data/github_graph_extractor/sample_1M.jsonl \\
        data/github_graph_extractor/graph.jsonl \\
        -o runs/stack_pretokenized \\
        -p 60
"""
import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path

from tunalab.reproducibility import ReproducibilityManager

from data.document_sources import StackJSONLSource
from data.pretokenize import run_preprocessing


def main():
    parser = argparse.ArgumentParser(
        prog="Stack Pre-tokenizer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "jsonl_file",
        type=Path,
        help="Path to the downloaded Stack JSONL file (e.g. sample_1M.jsonl).",
    )
    parser.add_argument(
        "graph_file",
        type=Path,
        help="Path to the graph.jsonl produced by build_graph_streaming.py.",
    )
    parser.add_argument(
        "-o", "--runs-dir",
        type=Path,
        required=True,
        help="Root directory to store experiment runs. A unique sub-directory will be created here.",
    )
    parser.add_argument(
        "--tokenizer-file",
        type=Path,
        default=None,
        help="Path to a custom .pkl tokenizer file. Overrides --tokenizer-name.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="gpt2",
        help="tiktoken tokenizer name if --tokenizer-file is not provided (default: gpt2).",
    )
    parser.add_argument(
        "--shard-size-gb",
        type=float,
        default=2.0,
        help="Target size for each binary shard in gigabytes (default: 2.0).",
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=max(1, mp.cpu_count() - 1),
        help=f"Number of worker processes (default: {max(1, mp.cpu_count() - 1)}).",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress reporting and info messages.",
    )
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Load graph titles into a set for O(1) lookup during JSONL scan.
    logger.info(f"Loading graph titles from {args.graph_file} ...")
    with open(args.graph_file, "r", encoding="utf-8") as f:
        graph_titles = {json.loads(line)["normed_identifier"] for line in f}
    logger.info(f"Loaded {len(graph_titles):,} graph nodes.")

    source = StackJSONLSource(args.jsonl_file, graph_titles)

    with ReproducibilityManager(output_dir=str(args.runs_dir), is_main_process=True) as rep:
        run_preprocessing(args, rep, source)


if __name__ == "__main__":
    main()
