#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAGWikiGraphBuilder: A tool to build a link graph from a directory of 
Markdown files produced by DAGWikiExtractor.

This is now a thin wrapper around the shared graph building framework.
"""
import argparse
import logging
from pathlib import Path

from .builder import build_wiki_graph


def main():
    parser = argparse.ArgumentParser(
        prog="DAGWikiGraphBuilder",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing the extracted Markdown files."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file for the graph (default: 'graph.jsonl' inside the input directory)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of articles to process (for testing) - NOTE: Not yet implemented in new framework"
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=None,
        help="Number of worker processes to use - NOTE: Not yet implemented in new framework"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress reporting"
    )
    args = parser.parse_args()

    # Default output location
    if args.output is None:
        args.output = str(Path(args.input_dir) / 'graph.jsonl')

    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    if args.limit is not None:
        logging.warning("--limit option not yet implemented in new framework, processing all files")
    
    if args.processes is not None:
        logging.warning("--processes option not yet implemented in new framework")

    # Delegate to new builder
    build_wiki_graph(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        show_progress=not args.quiet,
    )


if __name__ == '__main__':
    main()
