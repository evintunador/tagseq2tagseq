import os
import random
import argparse

def estimate_tokens_in_directory(directory, sample_size=10_000, chars_per_token=4):
    """
    Estimates the total number of tokens in a directory of text files,
    searching recursively through all subdirectories.
    """
    print(f"Analyzing directory: {directory}")

    # Recursively find all Markdown files in the directory
    all_file_paths = []
    try:
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.md'):
                    all_file_paths.append(os.path.join(root, filename))
        
        if not all_file_paths:
            print("Error: No .md files found in the directory or its subdirectories.")
            return
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory}'")
        return

    total_file_count = len(all_file_paths)
    print(f"Found {total_file_count} total articles.")

    # Determine the sample of files to analyze
    if total_file_count > sample_size:
        print(f"Taking a random sample of {sample_size} files for analysis...")
        files_to_sample = random.sample(all_file_paths, sample_size)
    else:
        print("Analyzing all files...")
        files_to_sample = all_file_paths

    # Calculate total characters in the sampled files
    total_chars_in_sample = 0
    actual_sample_size = len(files_to_sample)
    for filepath in files_to_sample:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                total_chars_in_sample += len(f.read())
        except Exception as e:
            print(f"Warning: Could not read file {filepath}. Skipping. Error: {e}")

    if actual_sample_size == 0:
        print("Error: No files were successfully sampled.")
        return

    # Calculate averages and extrapolate
    avg_chars_per_file = total_chars_in_sample / actual_sample_size
    estimated_total_chars = avg_chars_per_file * total_file_count
    estimated_total_tokens = estimated_total_chars / chars_per_token

    # --- Present the results ---
    print("\n--- Token Estimate ---")
    print(f"Average characters per article (from sample): {avg_chars_per_file:,.0f}")
    print(f"Estimated total characters in dataset: {estimated_total_chars:,.0f}")
    print(f"Using a heuristic of {chars_per_token} characters per token...")
    print(f"\nEstimated total tokens: {estimated_total_tokens:,.0f}")

    # For context, convert to millions/billions
    if estimated_total_tokens > 1_000_000_000:
        print(f"Which is approximately {estimated_total_tokens / 1_000_000_000:.2f} billion tokens.")
    elif estimated_total_tokens > 1_000_000:
        print(f"Which is approximately {estimated_total_tokens / 1_000_000:.2f} million tokens.")

def main():
    parser = argparse.ArgumentParser(
        description="Estimate the total number of tokens in a directory of text files.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "directory",
        help="The directory containing the .md files to analyze (e.g., 'output/')."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of files to sample for the estimation (default: 1000)."
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="The heuristic for characters per token (default: 4.0)."
    )
    args = parser.parse_args()

    estimate_tokens_in_directory(args.directory, args.sample_size, args.chars_per_token)

if __name__ == '__main__':
    main()
