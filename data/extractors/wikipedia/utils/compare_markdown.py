#!/usr/bin/env python3
import argparse
import difflib
import sys
import os

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

def colorize_diff(diff_lines):
    for line in diff_lines:
        if line.startswith('---') or line.startswith('+++'):
            yield f"{BOLD}{line.strip()}{RESET}"
        elif line.startswith('@@'):
            yield f"{CYAN}{line.strip()}{RESET}"
        elif line.startswith('-'):
            yield f"{RED}{line.rstrip()}{RESET}"
        elif line.startswith('+'):
            yield f"{GREEN}{line.rstrip()}{RESET}"
        else:
            yield line.rstrip()

def compare_files(file1_path, file2_path, context=3):
    if not os.path.exists(file1_path):
        print(f"Error: File not found: {file1_path}")
        return
    if not os.path.exists(file2_path):
        print(f"Error: File not found: {file2_path}")
        return

    try:
        with open(file1_path, 'r', encoding='utf-8') as f1, \
             open(file2_path, 'r', encoding='utf-8') as f2:
            f1_lines = f1.readlines()
            f2_lines = f2.readlines()
    except UnicodeDecodeError:
        print("Error: One of the files is not a valid text file (utf-8 decode failed).")
        return

    diff = difflib.unified_diff(
        f1_lines, 
        f2_lines, 
        fromfile=f"a/{os.path.basename(file1_path)}", 
        tofile=f"b/{os.path.basename(file2_path)}",
        n=context
    )

    diff_list = list(diff)
    
    if not diff_list:
        print(f"{GREEN}No differences found.{RESET}")
        return

    print(f"Showing differences between {BOLD}{file1_path}{RESET} and {BOLD}{file2_path}{RESET}:\n")
    
    for line in colorize_diff(diff_list):
        print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretty print a diff between two files.")
    parser.add_argument("file1", help="Path to the first file (original)")
    parser.add_argument("file2", help="Path to the second file (modified)")
    parser.add_argument("--context", "-c", type=int, default=3, help="Number of context lines to show")

    args = parser.parse_args()
    
    compare_files(args.file1, args.file2, args.context)

