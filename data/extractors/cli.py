#!/usr/bin/env python
"""
Unified CLI for graph extraction from multiple sources.

Usage:
    python -m data.extractors.cli wiki <input_dir> -o graph.jsonl
    python -m data.extractors.cli github <input_file> -o graph.jsonl
"""
import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="graph-extract",
        description="Extract document graphs from various sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="source", required=True, help="Data source type")
    
    # Wiki subcommand
    wiki_parser = subparsers.add_parser(
        "wiki", 
        help="Extract Wikipedia link graph",
        description="Build a link graph from Wikipedia markdown files"
    )
    wiki_parser.add_argument(
        "input_dir", 
        type=Path, 
        help="Directory containing .md files from dump_extractor.py"
    )
    wiki_parser.add_argument(
        "-o", "--output", 
        type=Path, 
        required=True, 
        help="Output path for graph.jsonl"
    )
    wiki_parser.add_argument(
        "-q", "--quiet", 
        action="store_true",
        help="Suppress progress bars"
    )
    
    # GitHub subcommand
    github_parser = subparsers.add_parser(
        "github", 
        help="Extract GitHub dependency graph",
        description="Build intra-repository dependency graph from GitHub data"
    )
    github_parser.add_argument(
        "input_file", 
        type=Path, 
        help="Input JSONL file from download_sample.py"
    )
    github_parser.add_argument(
        "-o", "--output", 
        type=Path, 
        required=True, 
        help="Output path for graph.jsonl"
    )
    github_parser.add_argument(
        "-q", "--quiet", 
        action="store_true",
        help="Suppress progress bars"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=log_level, 
        format='%(levelname)s: %(message)s'
    )
    
    # Route to appropriate builder
    try:
        if args.source == "wiki":
            from data.wiki_graph_extractor.builder import build_wiki_graph
            
            if not args.input_dir.exists():
                logging.error(f"Input directory does not exist: {args.input_dir}")
                sys.exit(1)
            
            logging.info(f"Building Wikipedia graph from {args.input_dir}")
            graph = build_wiki_graph(
                input_dir=args.input_dir, 
                output_path=args.output, 
                show_progress=not args.quiet
            )
            logging.info(f"Built graph with {len(graph)} nodes -> {args.output}")
        
        elif args.source == "github":
            from data.github_graph_extractor.builder import build_github_graph
            
            if not args.input_file.exists():
                logging.error(f"Input file does not exist: {args.input_file}")
                sys.exit(1)
            
            logging.info(f"Building GitHub graph from {args.input_file}")
            graph = build_github_graph(
                input_file=args.input_file, 
                output_path=args.output, 
                show_progress=not args.quiet
            )
            logging.info(f"Built graph with {len(graph)} nodes -> {args.output}")
        
    except Exception as e:
        logging.error(f"Graph building failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
