"""
ArXiv (unarXive) Pre-tokenizer: Pre-tokenize the ArXiv dataset extracted by
data/arxiv_graph_extractor/extract.py into sharded binary files.

Usage:
    python -m data.pretokenize_arxiv \\
        /fss-data/.../graphs/<arxiv_run_dir>/content.jsonl \\
        /fss-data/.../graphs/<arxiv_run_dir>/graph.jsonl \\
        -o /fss-data/.../pretokenized_datasets/arxiv \\
        -p 60

The content/graph inputs come from an extraction run dir produced by
data/arxiv_graph_extractor/extract.py (each run is its own directory under
graphs/, e.g. graphs/arxiv_run_legacy_20260616/).
"""
import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path

from tunalab.reproducibility import ReproducibilityManager

from data.document_sources import ArxivUnarxiveSource
from data.pretokenize import run_preprocessing


def main():
    parser = argparse.ArgumentParser(
        prog="ArXiv Pre-tokenizer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "content_jsonl",
        type=Path,
        help="Path to the content.jsonl produced by arxiv_graph_extractor/extract.py.",
    )
    parser.add_argument(
        "graph_file",
        type=Path,
        help="Path to the graph.jsonl produced by arxiv_graph_extractor/extract.py.",
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

    logger.info(f"Loading graph node ids from {args.graph_file} ...")
    with open(args.graph_file, "r", encoding="utf-8") as f:
        graph_normed_ids = {json.loads(line)["normed_identifier"] for line in f}
    logger.info(f"Loaded {len(graph_normed_ids):,} graph nodes.")

    source = ArxivUnarxiveSource(args.content_jsonl, graph_normed_ids)

    with ReproducibilityManager(output_dir=str(args.runs_dir), is_main_process=True) as rep:
        run_preprocessing(args, rep, source)


if __name__ == "__main__":
    main()
