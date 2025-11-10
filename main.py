import argparse
import logging
import random
from pathlib import Path

from dataset import GraphIndex, PretokShardedBackend


logger = logging.getLogger(__name__)


def main():
    """
    Main function to demonstrate loading and using the pre-tokenized graph dataset.
    This is a temporary placeholder to demonstrate the use of whatever components are
    finished until the time when all components are finished and actual experiment is
    ready to be written & run.
    """
    parser = argparse.ArgumentParser(
        description="Load and inspect a pre-tokenized DAGWiki dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to the pre-tokenized dataset run directory (e.g., 'raw_data/pretokenized/2025-11-09...')."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting a sample document to display."
    )
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        logger.error(f"Dataset directory not found: {args.dataset_dir}")
        return

    try:
        # 1. Initialize the GraphIndex
        logger.info("Initializing GraphIndex...")
        graph_index = GraphIndex(args.dataset_dir)
        logger.info(f"Successfully loaded graph with {len(graph_index)} nodes.")

        # 2. Initialize the data backend
        logger.info("Initializing PretokShardedBackend...")
        backend = PretokShardedBackend(graph_index)

        # 3. Demonstrate usage
        logger.info("\n--- Demonstrating data access ---")
        
        # Get a list of all titles and select a random one
        all_titles = graph_index.get_all_titles()
        if not all_titles:
            logger.warning("Graph contains no nodes.")
            return
            
        random.seed(args.seed)
        sample_title = random.choice(all_titles)
        
        logger.info(f"Fetching random sample document: '{sample_title}'")

        # Retrieve token data for the sample title
        tokens = backend.get_tokens(sample_title)
        
        if tokens is not None:
            node_info = graph_index.get_node(sample_title)
            print(f"\n--- Sample Document ---")
            print(f"Title:          {node_info.get('title')}")
            print(f"Character Count: {node_info.get('char_count', 'N/A')}")
            print(f"Token Count:    {len(tokens)}")
            print(f"Token dtype:    {tokens.dtype}")
            print(f"First 10 tokens: {tokens[:10]}")
            print(f"Outgoing Links:  {len(node_info.get('outgoing', []))}")
            print(f"  -> {node_info.get('outgoing', [])[:5]}") # Print first 5 links
            print(f"Incoming Links:  {len(node_info.get('incoming', []))}")
            print(f"  -> {node_info.get('incoming', [])[:5]}") # Print first 5 links
            print("-----------------------\n")
        else:
            logger.error(f"Failed to retrieve tokens for '{sample_title}'.")

    except FileNotFoundError as e:
        logger.error(f"A required dataset file was not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        if 'backend' in locals():
            backend.close()


if __name__ == "__main__":
    main()
