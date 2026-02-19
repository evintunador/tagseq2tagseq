#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAGWikiGraphBuilder: A tool to build a link graph from a directory of 
Markdown files produced by DAGWikiExtractor.
"""
import argparse
import glob
import json
import logging
import os
import re
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import islice
from tqdm import tqdm
from timeit import default_timer

import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ===========================================================================
# Worker Process Function
# ===========================================================================

def extract_links_worker(filepath):
    """
    Reads a single markdown file, extracts its title from the filename,
    and all outgoing links from the content.
    Links are standard markdown [text](link) format.
    """
    try:
        # Title is derived from the filename (minus extension)
        filename = os.path.basename(filepath)
        title = os.path.splitext(filename)[0]

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            char_count = len(content)
            
            # This regex finds links but avoids image links ![...](...)
            # It captures the link target from [text](target)
            links = re.findall(r'\[[^!\]]*?\]\((.*?)\)', content)
            
            # The link target is the title we want to link to
            # Since extract.py now writes normalized targets, we just use them as is
            # (except for potentially needing to unquote if there are URL encodings left, 
            # though our strict normalization mostly avoids them)
            from urllib.parse import unquote
            outgoing_links = {unquote(link) for link in links}

            return (title, list(outgoing_links), char_count)
    except Exception as e:
        logging.warning(f"Could not process file {filepath}: {e}")
        return None

# ===========================================================================
# Statistics Generation
# ===========================================================================

def compute_and_save_stats(graph_data, jsonl_output_path):
    logging.info("Computing graph statistics...")
    
    # Build NetworkX graph for easy metric calculation
    G = nx.DiGraph()
    
    # Add all nodes first
    for title in graph_data:
        G.add_node(title)
        
    # Add edges (only considering links within the set of files we processed)
    edge_count = 0
    for source, data in graph_data.items():
        for target in data['outgoing']:
            if target in graph_data:
                G.add_edge(source, target)
                edge_count += 1
                
    logging.info(f"Graph built in memory with {G.number_of_nodes()} nodes and {G.number_of_edges()} internal edges.")

    # Basic Stats
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_directed": True,
        "avg_clustering_coefficient": 0.0 # Placeholder, costly to compute for large graphs
    }
    
    # Connectivity
    # Note: Strongly/Weakly connected components can be slow on very large graphs, 
    # but usually manageable for Wikipedia subsets.
    try:
        stats["strongly_connected_components"] = nx.number_strongly_connected_components(G)
        stats["weakly_connected_components"] = nx.number_weakly_connected_components(G)
    except Exception as e:
        logging.warning(f"Could not compute connectivity stats: {e}")
    
    # Degree stats
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    if in_degrees:
        stats["avg_in_degree"] = float(np.mean(in_degrees))
        stats["max_in_degree"] = int(np.max(in_degrees))
        stats["median_in_degree"] = float(np.median(in_degrees))
        
    if out_degrees:
        stats["avg_out_degree"] = float(np.mean(out_degrees))
        stats["max_out_degree"] = int(np.max(out_degrees))
        stats["median_out_degree"] = float(np.median(out_degrees))
    
    # Save text stats
    base_path = os.path.splitext(jsonl_output_path)[0]
    stats_path = f"{base_path}_stats.json"
    try:
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Saved stats to {stats_path}")
    except Exception as e:
        logging.error(f"Failed to save stats json: {e}")
    
    # Plots
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Out-degree histogram
        ax1.hist(out_degrees, bins=50, log=True, color='blue', alpha=0.7)
        ax1.set_title("Out-degree Distribution (Log Scale)")
        ax1.set_xlabel("Number of Outgoing Links")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        
        # In-degree histogram
        ax2.hist(in_degrees, bins=50, log=True, color='green', alpha=0.7)
        ax2.set_title("In-degree Distribution (Log Scale)")
        ax2.set_xlabel("Number of Incoming Links")
        ax2.set_ylabel("Frequency")
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.tight_layout()
        plot_path = f"{base_path}_degree_dist.png"
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved degree distribution plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to save plots: {e}")

# ===========================================================================
# Main Execution
# ===========================================================================

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
        help="Limit the number of articles to process (for testing)"
    )
    parser.add_argument(
        "-p", "--processes",
        type=int,
        default=cpu_count() - 1,
        help=f"Number of worker processes to use (default: {cpu_count() - 1})"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress reporting"
    )
    args = parser.parse_args()

    # If no output file is specified, default to graph.jsonl inside the input dir
    if args.output is None:
        args.output = os.path.join(args.input_dir, 'graph.jsonl')

    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        return

    # --- File Discovery ---
    logging.info("Discovering markdown files...")
    # Using an iterator for memory efficiency with large numbers of files
    md_files_iterator = glob.iglob(os.path.join(args.input_dir, '**', '*.md'), recursive=True)
    
    # Apply limit if specified
    if args.limit:
        md_files_iterator = islice(md_files_iterator, args.limit)
    
    # We need a list for the progress bar, so we realize the iterator here
    md_files = list(md_files_iterator)

    if not md_files:
        logging.error("No markdown files found in the input directory.")
        return
        
    logging.info(f"Found {len(md_files)} markdown files to process.")

    # --- Parallel Link Extraction ---
    logging.info(f"Starting link extraction with {args.processes} workers...")
    start_time = default_timer()
    
    with Pool(processes=args.processes) as pool:
        # The main process will read the file and put articles into the pool
        results_iterator = pool.imap_unordered(extract_links_worker, md_files, chunksize=100)
        
        progress_bar = tqdm(
            results_iterator,
            total=len(md_files),
            desc="Extracting links",
            unit=" files",
            disable=args.quiet
        )
        
        # Filter out None results from files that couldn't be processed
        link_data = [result for result in progress_bar if result]

    duration = default_timer() - start_time
    logging.info(f"Link extraction finished in {duration:.2f}s.")

    # --- Graph Aggregation ---
    logging.info("Aggregating link data into a graph...")
    graph = {}

    # First pass: add all nodes and their outgoing links
    for title, outgoing_links, char_count in link_data:
        if title not in graph:
            graph[title] = {'outgoing': [], 'incoming': [], 'char_count': 0}
        # We use a set to avoid duplicate links from the same article
        graph[title]['outgoing'].extend(outgoing_links)
        graph[title]['char_count'] = char_count
    
    # Second pass: build the incoming links
    for source_title, data in graph.items():
        for target_title in data['outgoing']:
            if target_title in graph:
                graph[target_title]['incoming'].append(source_title)
            # Optional: handle dangling links (links to pages not in the dump)
            # else:
            #     logging.debug(f"Dangling link found from '{source_title}' to '{target_title}'")

    # --- Sort and Write Output ---
    logging.info(f"Sorting and writing graph to {args.output}...")
    
    # Sort the graph keys (article titles) alphabetically
    sorted_titles = sorted(graph.keys())
    
    with open(args.output, 'w', encoding='utf-8') as f:
        progress_bar = tqdm(
            sorted_titles,
            desc="Writing JSONL",
            unit=" nodes",
            disable=args.quiet
        )
        for title in progress_bar:
            node_data = {
                'title': title,
                'char_count': graph[title]['char_count'],
                'outgoing': sorted(list(set(graph[title]['outgoing']))),
                'incoming': sorted(list(set(graph[title]['incoming'])))
            }
            f.write(json.dumps(node_data) + '\n')
            
    logging.info("Graph construction complete.")

    # --- Compute and Save Stats ---
    compute_and_save_stats(graph, args.output)


if __name__ == '__main__':
    main()
