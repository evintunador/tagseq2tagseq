"""
Go Pre-tokenizer: pre-tokenize the Go package corpus into sharded binary files
using the package graph produced by
data/go_graph_extractor/build_go_graph.py.

Unlike pretokenize_stack (which reads raw file records), the Go builder already
grouped files into package nodes and emitted content.jsonl, so this wrapper reads
that content directly via GoPackageContentSource (same pattern as arxiv/fineweb).

Usage:
    python -m data.pretokenize_go \\
        data/go_graph_extractor/content.jsonl \\
        data/go_graph_extractor/graph.jsonl \\
        -o runs/go_pretokenized \\
        -p 60
"""
import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path

from tunalab.reproducibility import ReproducibilityManager

from data.document_sources import GoPackageContentSource
from data.pretokenize import run_preprocessing


def main():
    parser = argparse.ArgumentParser(
        prog="Go Pre-tokenizer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "content_file",
        type=Path,
        help="Path to content.jsonl produced by build_go_graph.py.",
    )
    parser.add_argument(
        "graph_file",
        type=Path,
        help="Path to graph.jsonl produced by build_go_graph.py.",
    )
    parser.add_argument(
        "-o", "--runs-dir",
        type=Path,
        required=True,
        help="Root directory to store experiment runs.",
    )
    parser.add_argument("--tokenizer-file", type=Path, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--shard-size-gb", type=float, default=2.0)
    parser.add_argument(
        "-p", "--processes", type=int,
        default=max(1, mp.cpu_count() - 1),
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading graph node ids from {args.graph_file} ...")
    with open(args.graph_file, "r", encoding="utf-8") as f:
        graph_normed_ids = {json.loads(line)["normed_identifier"] for line in f}
    logger.info(f"Loaded {len(graph_normed_ids):,} package nodes.")

    source = GoPackageContentSource(args.content_file, graph_normed_ids)

    with ReproducibilityManager(output_dir=str(args.runs_dir), is_main_process=True) as rep:
        run_preprocessing(args, rep, source)


if __name__ == "__main__":
    main()
