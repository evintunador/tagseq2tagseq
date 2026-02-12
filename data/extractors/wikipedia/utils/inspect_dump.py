import gzip
import json
import os
import sys

def inspect_cirrus_dump(file_path, target_title=None):
    """
    Reads and prints the full source_text of a specific article (or the first one).
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    found = False
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Check if this is the article we want
                if 'title' in data and 'source_text' in data:
                    if target_title and data['title'] != target_title:
                        continue
                    
                    print("-" * 80)
                    print(f"Title: {data.get('title')}")
                    print("-" * 80)
                    print(data.get('source_text'))
                    print("-" * 80)
                    found = True
                    break
            except json.JSONDecodeError:
                continue
    
    if target_title and not found:
        print(f"Article '{target_title}' not found.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inspect_dump.py <dump_path> [article_title]")
        sys.exit(1)
        
    dump_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else None
    
    inspect_cirrus_dump(dump_path, title)
