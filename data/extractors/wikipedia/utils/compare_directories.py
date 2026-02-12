#!/usr/bin/env python3
import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

# ANSI color codes
BOLD = '\033[1m'
RESET = '\033[0m'
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'

def compare_directories(dir1, dir2, limit_diffs=10, context=3, max_checks=None):
    """
    Compares files in two directories.
    
    Args:
        dir1: Baseline directory
        dir2: New directory
        limit_diffs: Stop after finding this many files with differences
        context: Number of context lines for diff
        max_checks: Maximum number of files to inspect (to avoid infinite loops if no diffs found)
    """
    
    print(f"Comparing {BOLD}{dir1}{RESET} and {BOLD}{dir2}{RESET}...")
    
    # Find all common markdown files
    files1 = set(str(p.relative_to(dir1)) for p in Path(dir1).rglob('*.md'))
    files2 = set(str(p.relative_to(dir2)) for p in Path(dir2).rglob('*.md'))
    
    common_files = list(files1.intersection(files2))
    
    if not common_files:
        print(f"{RED}No common markdown files found between the directories.{RESET}")
        return

    print(f"Found {len(common_files)} common markdown files.")
    
    # Shuffle to get a random sample
    random.shuffle(common_files)
    
    diffs_found = 0
    files_checked = 0
    
    # Path to the single-file compare script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compare_script = os.path.join(script_dir, 'compare_markdown.py')
    
    for rel_path in common_files:
        if max_checks and files_checked >= max_checks:
            print(f"\n{YELLOW}Reached maximum check limit of {max_checks} files.{RESET}")
            break
            
        file1_path = os.path.join(dir1, rel_path)
        file2_path = os.path.join(dir2, rel_path)
        
        # Run diff command to check if files are different
        # We use diff -q first to be fast
        try:
            # Returns 0 if same, 1 if different
            result = subprocess.run(
                ['diff', '-q', file1_path, file2_path], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        except FileNotFoundError:
             # Fallback if 'diff' is not available (e.g. windows without tools), just run python script
             result = None

        files_checked += 1

        if result is None or result.returncode != 0:
            # Files are different (or we couldn't check easily), so run the visual comparison
            print(f"\n{BOLD}Difference found in: {rel_path}{RESET}")
            
            # Call the existing python comparison tool
            subprocess.run(
                [sys.executable, compare_script, file1_path, file2_path, '--context', str(context)]
            )
            
            diffs_found += 1
            if diffs_found >= limit_diffs:
                print(f"\n{GREEN}Reached limit of {limit_diffs} files with differences.{RESET}")
                break
        else:
            # Files are identical, skip silently (or print a dot?)
            pass

    if diffs_found == 0:
        print(f"\n{GREEN}Checked {files_checked} files. No differences found.{RESET}")
    else:
        print(f"\nFound differences in {diffs_found} out of {files_checked} files checked.")

def main():
    parser = argparse.ArgumentParser(
        description="Compare random subset of files between two directories until differences are found."
    )
    parser.add_argument("dir1", help="Path to the first directory (baseline)")
    parser.add_argument("dir2", help="Path to the second directory (new)")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Stop after showing diffs for this many files (default: 10)")
    parser.add_argument("--max-checks", type=int, default=1000, help="Maximum number of files to check (default: 1000)")
    parser.add_argument("--context", "-c", type=int, default=3, help="Number of context lines to show in diff")

    args = parser.parse_args()
    
    compare_directories(args.dir1, args.dir2, args.limit, args.context, args.max_checks)

if __name__ == "__main__":
    main()

