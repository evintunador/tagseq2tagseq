"""
Java Pre-tokenizer: pre-tokenize the Java file corpus into sharded binary files
using the FQN-keyed graph produced by
data/java_graph_extractor/build_java_graph.py.

Thin wrapper (arxiv/fineweb/go pattern): the builder already emitted content.jsonl
(one {"normed_identifier": <FQN>, "content"} per file node), so this reads it via
the generic content source and calls the shared preprocessing core.

Usage:
    python -m data.pretokenize_java \\
        graphs/java/content.jsonl graphs/java/graph.jsonl \\
        -o runs/java_pretokenized -p 60
"""
import argparse
import json
import logging
import multiprocessing as mp
from pathlib import Path

from tunalab.reproducibility import ReproducibilityManager

from data.document_sources import ContentJsonlSource
from data.pretokenize import run_preprocessing


def main():
    parser = argparse.ArgumentParser(
        prog="Java Pre-tokenizer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("content_file", type=Path,
                        help="content.jsonl from build_java_graph.py")
    parser.add_argument("graph_file", type=Path,
                        help="graph.jsonl from build_java_graph.py")
    parser.add_argument("-o", "--runs-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-file", type=Path, default=None)
    parser.add_argument("--tokenizer-name", type=str, default="gpt2")
    parser.add_argument("--shard-size-gb", type=float, default=2.0)
    parser.add_argument("-p", "--processes", type=int,
                        default=max(1, mp.cpu_count() - 1))
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
    logger.info(f"Loaded {len(graph_normed_ids):,} file nodes.")

    source = ContentJsonlSource(args.content_file, graph_normed_ids)

    with ReproducibilityManager(output_dir=str(args.runs_dir), is_main_process=True) as rep:
        run_preprocessing(args, rep, source)


if __name__ == "__main__":
    main()
