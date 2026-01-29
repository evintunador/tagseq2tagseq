#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GitHub Graph Builder (Streaming Version).

This is now a thin wrapper around the shared graph building framework.
Complex streaming and import resolution logic is in builder.py.
"""
import argparse
import logging
from pathlib import Path

from .builder import build_github_graph


def main():
    parser = argparse.ArgumentParser(
        prog="GitHubGraphBuilderStreaming",
        description="Memory-efficient streaming graph builder for large GitHub datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", help="JSONL file containing sampled repository data.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file for the graph (default: '<input>_streaming_graph.jsonl')",
    )
    parser.add_argument("--batch-size", type=int, default=50000, 
                       help="Number of files to process per batch - NOTE: Not yet implemented in new framework")
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=None,
        help="Number of worker processes - NOTE: Not yet implemented in new framework",
    )
    parser.add_argument("--buckets", type=int, default=256, 
                       help="Number of repo hash buckets - NOTE: Not yet implemented in new framework")
    parser.add_argument("--bucket-workers", type=int, default=None, 
                       help="Workers for bucket processing - NOTE: Not yet implemented in new framework")
    parser.add_argument("--no-stats", action="store_true", 
                       help="Skip statistics computation - NOTE: Not yet implemented in new framework")
    parser.add_argument("--no-plots", action="store_true", 
                       help="Skip plot generation - NOTE: Not yet implemented in new framework")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress reporting")
    
    args = parser.parse_args()

    if args.output is None:
        input_dir = Path(args.input_file).parent
        base_name = Path(args.input_file).stem
        args.output = str(input_dir / f"{base_name}_streaming_graph.jsonl")

    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    
    # Warn about unimplemented options
    if args.batch_size != 50000:
        logging.warning("--batch-size option not yet implemented in new framework")
    if args.processes is not None:
        logging.warning("--processes option not yet implemented in new framework")
    if args.buckets != 256:
        logging.warning("--buckets option not yet implemented in new framework")
    if args.bucket_workers is not None:
        logging.warning("--bucket-workers option not yet implemented in new framework")
    if args.no_stats:
        logging.warning("--no-stats option not yet implemented in new framework")
    if args.no_plots:
        logging.warning("--no-plots option not yet implemented in new framework")

    # Delegate to new builder
    build_github_graph(
        input_file=Path(args.input_file),
        output_path=Path(args.output),
        show_progress=not args.quiet,
    )


if __name__ == "__main__":
    main()
