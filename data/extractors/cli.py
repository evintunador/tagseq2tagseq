#!/usr/bin/env python
"""
Unified CLI for graph extraction from multiple sources.

Usage:
    python -m data.extractors.cli wikipedia <input_dir> -o graph.jsonl
    python -m data.extractors.cli thestack <input_file> -o graph.jsonl
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
    
    # thestack subcommand
    thestack_parser = subparsers.add_parser(
        "thestack", 
        help="Extract thestack dependency graph",
        description="Build intra-repository dependency graph from thestack data"
    )
    thestack_parser.add_argument(
        "input_file", 
        type=Path, 
        help="Input JSONL file from download_sample.py"
    )
    thestack_parser.add_argument(
        "-o", "--output", 
        type=Path, 
        required=True, 
        help="Output path for graph.jsonl"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(levelname)s: %(message)s'
    )
    
    # Route to appropriate builder
    try:
        if args.source == "wiki":
            from data.extractors.wikipedia.builder import build_wiki_graph
            
            if not args.input_dir.exists():
                logging.error(f"Input directory does not exist: {args.input_dir}")
                sys.exit(1)
            
            logging.info(f"Building Wikipedia graph from {args.input_dir}")
            graph = build_wiki_graph(
                input_dir=args.input_dir, 
                output_path=args.output
            )
            logging.info(f"Built graph with {len(graph)} nodes -> {args.output}")
        
        elif args.source == "thestack":
            from data.extractors.thestack.builder import build_thestack_graph
            
            if not args.input_file.exists():
                logging.error(f"Input file does not exist: {args.input_file}")
                sys.exit(1)
            
            logging.info(f"Building thestack graph from {args.input_file}")
            graph = build_thestack_graph(
                input_file=args.input_file, 
                output_path=args.output
            )
            logging.info(f"Built graph with {len(graph)} nodes -> {args.output}")
        
    except Exception as e:
        logging.error(f"Graph building failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
