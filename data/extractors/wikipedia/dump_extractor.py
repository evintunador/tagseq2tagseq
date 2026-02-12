#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DAGWikiExtractor: A streamlined tool to extract and clean text from 
Wikipedia Cirrus dumps, converting articles into individual Markdown files
with preserved internal links.
"""
import argparse
import gzip
import json
import logging
import os
from timeit import default_timer
from multiprocessing import Pool, cpu_count
from functools import partial
import glob
from tqdm import tqdm

from data.extractors.wikipedia.extract import process_wikitext, normalize_title

# ===========================================================================
# Worker Process Function
# ===========================================================================

def process_article_worker(article_data, output_dir, write_title: bool):
    """
    A single worker's task: process one article's text and save it
    into a subdirectory based on the first letter of its title.
    """
    title = None # Initialize for robust error logging
    try:
        title, source_text, page_id = article_data
        
        # Process the raw wikitext through our cleaning pipeline
        final_text = process_wikitext(source_text)
        
        # Generate a safe, normalized filename for the article
        output_filename = normalize_title(title) + '.md'
        
        # Determine the subdirectory and handle problematic filenames
        if output_filename and output_filename[0].isalnum():
            # Use the first letter for standard articles
            first_char = output_filename[0].upper()
        else:
            # Use a catch-all for others
            first_char = '_'
            # If the original filename was invalid/empty after normalization, use the page ID as a fallback
            if not output_filename or output_filename == '.md':
                output_filename = f"{page_id}.md"

        # Create the subdirectory if it doesn't exist. This is safe for multiprocessing.
        subdir_path = os.path.join(output_dir, first_char)
        os.makedirs(subdir_path, exist_ok=True)
        
        # Construct the final path and save the file
        output_path = os.path.join(subdir_path, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            if write_title:
                out_file.write(f"# {title}\n\n")
            out_file.write(final_text)
        return 1 # Return 1 for success
    except Exception as e:
        # Safely log the title if it was assigned
        article_title = title if title else "Unknown Article"
        logging.error(f"Error processing article: {article_title} - {e}")
        return 0 # Return 0 for failure

# ===========================================================================
# Helper Functions
# ===========================================================================

def read_articles(file_handle, limit):
    """
    A generator that reads the dump file line by line, yielding article data.
    """
    count = 0
    for line in file_handle:
        try:
            article = json.loads(line)
            if article.get('namespace') == 0 and 'source_text' in article:
                yield (article['title'], article['source_text'], article.get('page_id', 0))
                count += 1
                if limit and count >= limit:
                    return
        except json.JSONDecodeError:
            continue

# ===========================================================================
# Core Processing Logic
# ===========================================================================

def process_dump(input_file, output_dir, limit=None, process_count=None, write_title: bool = False):
    """
    Reads a Wikipedia Cirrus dump, processes each article in parallel, 
    and saves it as a Markdown file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info(f"Starting extraction from '{input_file}'...")
    logging.info(f"Using {process_count} worker processes.")
    start_time = default_timer()
    articles_processed = 0

    # The main process will read the file and put articles into the pool
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        # Create a pool of worker processes
        with Pool(processes=process_count) as pool:
            # Create a partial function with the output_dir already filled in
            worker = partial(process_article_worker, output_dir=output_dir, write_title=write_title)
            
            # Use imap_unordered for memory efficiency.
            article_iterator = read_articles(f, limit)
            
            # Since we don't know the total number of articles beforehand, tqdm
            # will show progress as iterations/sec without a percentage bar.
            results_iterator = pool.imap_unordered(worker, article_iterator, chunksize=100)
            progress_bar = tqdm(
                results_iterator,
                desc=f"Processing {os.path.basename(input_file)}",
                unit=" articles",
                total=limit # Provide total if limit is known for a better bar
            )
            
            for result in progress_bar:
                articles_processed += result
    
    duration = default_timer() - start_time
    # Avoid division by zero if the process was very fast or processed nothing
    rate = (articles_processed / duration) if duration > 0 else 0
    logging.info(
        f"Finished processing {articles_processed} articles in {duration:.2f}s "
        f"({rate:.2f} art/s)"
    )

# ===========================================================================
# Main Execution
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="DAGWikiExtractor",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs", 
        nargs='+',
        help="One or more input Wikipedia Cirrus dump files (.json.gz) or directories."
    )
    parser.add_argument(
        "-o", "--output", 
        default="output",
        help="Directory for extracted Markdown files (default: 'output')"
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
        "--write_title",
        action="store_true",
        default=False,
        help="If set, include the Wikipedia page title in each output markdown file.",
    )
    parser.add_argument(
        "-q", "--quiet", 
        action="store_true", 
        help="Suppress progress reporting"
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # --- File Discovery and Sorting ---
    dump_files = []
    for path in args.inputs:
        if os.path.isdir(path):
            # Use glob to find all matching files recursively
            found_files = glob.glob(os.path.join(path, '**', '*.json.gz'), recursive=True)
            dump_files.extend(found_files)
        elif os.path.isfile(path):
            dump_files.append(path)
        else:
            logging.warning(f"Input path is not a valid file or directory, skipping: {path}")
    
    # Sort files alphabetically, which works for date-stamped filenames
    dump_files.sort()

    if not dump_files:
        logging.error("No input files found. Aborting.")
        return

    logging.info(f"Found {len(dump_files)} dump file(s) to process in order:")
    for f in dump_files:
        logging.info(f"  - {os.path.basename(f)}")
    # --- End File Discovery ---

    for input_file in dump_files:
        process_dump(input_file, args.output, args.limit, args.processes, args.write_title)

if __name__ == '__main__':
    main()
