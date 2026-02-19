#!/usr/bin/env python3
"""
Simple Wikipedia Cirrus Dump Downloader (No external dependencies)

Downloads Wikimedia Cirrus dumps using only Python standard library.
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
import time
import bz2
import gzip
import re
from html.parser import HTMLParser
from urllib.parse import quote


class DownloadProgressBar:
    """Simple progress bar for downloads."""

    def __init__(self, total_size, desc="Downloading"):
        self.total_size = total_size
        self.desc = desc
        self.downloaded = 0
        self.start_time = time.time()

    def update(self, chunk_size):
        self.downloaded += chunk_size
        elapsed = time.time() - self.start_time
        speed = self.downloaded / elapsed if elapsed > 0 else 0
        speed_mb = speed / (1024 * 1024)

        if self.total_size > 0:
            percent = (self.downloaded / self.total_size) * 100
            # Calculate ETA
            if self.downloaded > 0:
                total_time = elapsed * self.total_size / self.downloaded
                eta = total_time - elapsed
                eta_str = f"{eta:.0f}s" if eta < 60 else f"{eta/60:.1f}m"
            else:
                eta_str = "--"

            # Progress bar
            bar_width = 40
            filled = int(bar_width * self.downloaded / self.total_size)
            bar = "█" * filled + "░" * (bar_width - filled)

            print(f"{self.desc}: [{bar}] {percent:5.1f}% {self.downloaded/1024/1024:05.1f}MB/{self.total_size/1024/1024:4.1f}MB {speed_mb:4.1f}MB/s ETA:{eta_str}",
                  end='\r', flush=True)
        else:
            # Unknown size - just show bytes downloaded
            mb_downloaded = self.downloaded / (1024 * 1024)
            print(f"{self.desc}: {mb_downloaded:8.1f}MB downloaded {speed_mb:5.1f}MB/s",
                  end='\r', flush=True)


def download_dump(url, output_path):
    """Download a single dump file with progress."""
    try:
        # Get file info
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))

        # Download with progress
        progress = DownloadProgressBar(total_size, os.path.basename(output_path))

        def update_progress(chunk):
            progress.update(len(chunk))

        # Create custom opener with progress callback
        class ProgressReporter:
            def __init__(self, progress_bar):
                self.progress_bar = progress_bar

            def __call__(self, block_num, block_size, total_size):
                self.progress_bar.update(block_size)

        # Download the file
        urllib.request.urlretrieve(url, output_path, ProgressReporter(progress))
        print()  # New line after progress bar
        return True

    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        return False


class DirectoryListingParser(HTMLParser):
    """Parse HTML directory listing to extract file links."""
    
    def __init__(self):
        super().__init__()
        self.files = []
        self.directories = []
    
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for attr_name, attr_value in attrs:
                if attr_name == 'href' and attr_value:
                    # Skip parent directory links
                    if attr_value not in ['../', './']:
                        if attr_value.endswith('/'):
                            # It's a directory
                            dir_name = attr_value.rstrip('/')
                            if dir_name:
                                self.directories.append(dir_name)
                        else:
                            # It's a file
                            self.files.append(attr_value)


def find_latest_date(base_url):
    """Find the latest available date from the cirrus_search_index directory listing."""
    try:
        print(f"Checking available dates at: {base_url}")
        with urllib.request.urlopen(base_url) as response:
            html_content = response.read().decode('utf-8')
        
        # Parse HTML to extract directory links
        parser = DirectoryListingParser()
        parser.feed(html_content)
        
        # Filter for date-like directories (YYYYMMDD format)
        date_pattern = re.compile(r'^\d{8}$')
        dates = [d for d in parser.directories if date_pattern.match(d)]
        
        if not dates:
            raise ValueError(f"No date directories found at {base_url}")
        
        # Sort dates and get the latest
        dates.sort()
        latest_date = dates[-1]
        
        print(f"Found {len(dates)} available date(s), using latest: {latest_date}")
        return latest_date
        
    except urllib.error.HTTPError as e:
        raise ValueError(f"Failed to access directory listing at {base_url}: {e}")
    except Exception as e:
        raise ValueError(f"Error finding latest date: {e}")


def cleanup_old_dumps(output_dir, confirm=True):
    """Remove old dump files matching the pattern *wiki-*-cirrussearch-content.json.gz"""
    pattern = "*wiki-*-cirrussearch-content.json.gz"
    dump_files = list(output_dir.glob(pattern))
    
    if not dump_files:
        print("No old dump files found to clean.")
        return True
    
    print(f"Found {len(dump_files)} old dump file(s) to clean:")
    for f in dump_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    if confirm:
        response = input("\nDelete these files? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return False
    
    for f in dump_files:
        try:
            f.unlink()
            print(f"✓ Deleted {f.name}")
        except Exception as e:
            print(f"✗ Error deleting {f.name}: {e}")
            return False
    
    print("Cleanup completed.")
    return True


def list_available_indexes(base_url):
    """List available index_name directories for a given date."""
    try:
        with urllib.request.urlopen(base_url) as response:
            html_content = response.read().decode('utf-8')
        
        parser = DirectoryListingParser()
        parser.feed(html_content)
        
        # Filter for directories starting with "index_name="
        indexes = [d for d in parser.directories if d.startswith('index_name=')]
        
        # If no directories found, try parsing links differently
        # Sometimes directories are listed as links without trailing /
        if not indexes:
            # Look for any links that contain "index_name="
            all_items = parser.files + parser.directories
            indexes = [item for item in all_items if 'index_name=' in item]
            # Remove file extensions to get directory names
            indexes = [idx.replace('.json.bz2', '') for idx in indexes]
            indexes = list(set([idx for idx in indexes if idx.startswith('index_name=')]))
        
        # Also try a simple regex search as fallback
        if not indexes:
            # Look for href="index_name=..." patterns
            pattern = r'href=["\']?([^"\'>\s]*index_name=[^"\'>\s/]+)'
            matches = re.findall(pattern, html_content)
            indexes = list(set([m for m in matches if m.startswith('index_name=')]))
        
        return indexes
    except Exception as e:
        return []


def discover_shard_files(base_url, lang, date):
    """Discover all shard files for a given language and date from the new directory structure."""
    # Construct the subdirectory path: index_name={lang}wiki_content/
    if lang == 'simple':
        subdir = f"index_name=simplewiki_content"
    else:
        subdir = f"index_name={lang}wiki_content"
    
    # Try without URL encoding first (some servers handle = in paths)
    # If that fails, we'll try with encoding
    dir_url = f"{base_url}{subdir}/"
    
    print(f"Discovering shard files from: {dir_url}")
    
    try:
        # Fetch directory listing HTML
        with urllib.request.urlopen(dir_url) as response:
            html_content = response.read().decode('utf-8')
        
        # Parse HTML to extract file links
        parser = DirectoryListingParser()
        parser.feed(html_content)
        
        # Filter for .json.bz2 files and sort them
        shard_files = sorted([f for f in parser.files if f.endswith('.json.bz2')])
        
        if not shard_files:
            # Try to list available indexes to help user
            available_indexes = list_available_indexes(base_url)
            error_msg = f"No .json.bz2 shard files found in {dir_url}"
            if available_indexes:
                error_msg += f"\n\nAvailable indexes for this date:\n"
                for idx in sorted(available_indexes)[:10]:  # Show first 10
                    error_msg += f"  - {idx}\n"
                if len(available_indexes) > 10:
                    error_msg += f"  ... and {len(available_indexes) - 10} more\n"
            raise ValueError(error_msg)
        
        # Construct full URLs
        shard_urls = [f"{base_url}{subdir}/{f}" for f in shard_files]
        
        print(f"Found {len(shard_urls)} shard file(s)")
        return shard_urls
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Try with URL encoding as fallback
            encoded_subdir = quote(subdir, safe='/')
            encoded_dir_url = f"{base_url}{encoded_subdir}/"
            try:
                print(f"Trying with URL encoding: {encoded_dir_url}")
                with urllib.request.urlopen(encoded_dir_url) as response:
                    html_content = response.read().decode('utf-8')
                
                parser = DirectoryListingParser()
                parser.feed(html_content)
                shard_files = sorted([f for f in parser.files if f.endswith('.json.bz2')])
                
                if shard_files:
                    shard_urls = [f"{base_url}{encoded_subdir}/{f}" for f in shard_files]
                    print(f"Found {len(shard_urls)} shard file(s)")
                    return shard_urls
            except:
                pass
            
            # If both attempts failed, list available indexes to help user
            print(f"\nAttempting to list available indexes from date directory...")
            available_indexes = list_available_indexes(base_url)
            error_msg = (
                f"Failed to access directory listing at {dir_url}: {e}\n"
                f"This might mean:\n"
                f"  1. The language '{lang}' might not be available for this date\n"
                f"  2. The index name format might be different\n"
            )
            if available_indexes:
                error_msg += f"\nAvailable indexes for date {date}:\n"
                for idx in sorted(available_indexes)[:20]:  # Show first 20
                    error_msg += f"  - {idx}\n"
                if len(available_indexes) > 20:
                    error_msg += f"  ... and {len(available_indexes) - 20} more\n"
                # Check if there's a similar index name
                if lang == 'simple':
                    similar = [idx for idx in available_indexes if 'simple' in idx.lower()]
                    if similar:
                        error_msg += f"\nNote: Found indexes containing 'simple':\n"
                        for idx in similar[:5]:
                            error_msg += f"  - {idx}\n"
                    else:
                        error_msg += f"\nNote: No indexes found containing 'simple'. "
                        error_msg += f"Available indexes are listed above.\n"
            else:
                error_msg += f"\nCould not automatically list available indexes.\n"
                error_msg += f"Please check manually at: {base_url}\n"
                error_msg += f"Look for directories starting with 'index_name='"
            raise ValueError(error_msg)
        else:
            raise ValueError(f"Failed to access directory listing at {dir_url}: {e}")
    except Exception as e:
        raise ValueError(f"Error discovering shard files: {e}")


def download_and_concatenate_shards(shard_urls, output_path):
    """Download all shards, decompress bz2, concatenate, and compress as gzip."""
    temp_dir = output_path.parent / f".temp_{output_path.stem}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        downloaded_shards = []
        
        # Download all shards
        print(f"\nDownloading {len(shard_urls)} shard file(s)...")
        for i, shard_url in enumerate(shard_urls, 1):
            shard_filename = os.path.basename(shard_url)
            temp_shard_path = temp_dir / shard_filename
            
            print(f"\n[{i}/{len(shard_urls)}] Downloading {shard_filename}...")
            if not download_dump(shard_url, temp_shard_path):
                raise Exception(f"Failed to download {shard_url}")
            
            downloaded_shards.append(temp_shard_path)
        
        # Decompress, concatenate, and compress
        print(f"\nConcatenating and compressing shards...")
        total_size = 0
        
        with gzip.open(output_path, 'wb') as outfile:
            for i, shard_path in enumerate(downloaded_shards, 1):
                print(f"[{i}/{len(downloaded_shards)}] Processing {shard_path.name}...", end='\r')
                
                with bz2.open(shard_path, 'rb') as bz2_file:
                    while True:
                        chunk = bz2_file.read(1024 * 1024)  # Read 1MB chunks
                        if not chunk:
                            break
                        outfile.write(chunk)
                        total_size += len(chunk)
        
        print(f"\n✓ Successfully created {output_path.name} ({total_size / (1024*1024):.1f} MB uncompressed)")
        
        # Clean up temporary files
        print("Cleaning up temporary files...")
        for shard_path in downloaded_shards:
            shard_path.unlink()
        temp_dir.rmdir()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing shards: {e}")
        # Clean up on error
        if temp_dir.exists():
            for shard_path in downloaded_shards:
                if shard_path.exists():
                    shard_path.unlink()
            try:
                temp_dir.rmdir()
            except:
                pass
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simple downloader for Wikimedia Cirrus dumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download English Wikipedia (recommended)
  python download_wiki_dumps_simple.py --lang en

  # Download Simple English Wikipedia (smaller, good for testing)
  python download_wiki_dumps_simple.py --lang simple

  # Download to custom directory
  python download_wiki_dumps_simple.py --lang en --output-dir ./my_dumps

  # Clean old dumps before downloading
  python download_wiki_dumps_simple.py --lang en --clean
        """
    )

    parser.add_argument(
        '--lang', '-l',
        required=True,
        help='Language code to download (e.g., en, simple, fr, de)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('../wiki_dumps'),  # Relative to wiki_graph_extractor directory
        help='Directory to save downloaded dumps (default: ../wiki_dumps)'
    )

    parser.add_argument(
        '--date',
        default=None,
        help='Dump date in YYYYMMDD format (default: automatically use latest available date)'
    )

    parser.add_argument(
        '--clean', '-c',
        action='store_true',
        help='Clean old dump files before downloading'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt when using --clean'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cleanup old dumps if requested
    if args.clean:
        print("Cleaning old dump files...")
        if not cleanup_old_dumps(args.output_dir, confirm=not args.yes):
            sys.exit(1)
        print()

    # Determine the date to use
    base_index_url = "https://dumps.wikimedia.org/other/cirrus_search_index/"
    if args.date:
        dump_date = args.date
        print(f"Using specified date: {dump_date}")
    else:
        try:
            dump_date = find_latest_date(base_index_url)
        except Exception as e:
            print(f"✗ Error finding latest date: {e}")
            print("Please specify a date manually with --date YYYYMMDD")
            sys.exit(1)

    # Construct filename (maintain same format for compatibility)
    if args.lang == 'simple':
        filename = f"simplewiki-{dump_date}-cirrussearch-content.json.gz"
    else:
        filename = f"{args.lang}wiki-{dump_date}-cirrussearch-content.json.gz"

    output_path = args.output_dir / filename

    # Use new base URL
    base_url = f"https://dumps.wikimedia.org/other/cirrus_search_index/{dump_date}/"

    print(f"Target output: {filename}")
    print(f"Output path: {output_path}")
    print()

    try:
        # Discover shard files
        shard_urls = discover_shard_files(base_url, args.lang, dump_date)
        
        # Download and concatenate shards
        if download_and_concatenate_shards(shard_urls, output_path):
            print(f"\n✓ Successfully downloaded and processed {filename}")

            # Show next steps
            print(f"\nNext steps:")
            print(f"1. Extract articles: python dump_extractor.py {output_path}")
            print(f"2. For testing, limit to first 1000 articles: python dump_extractor.py {output_path} --limit 1000")
            print(f"3. Build graph: python build_graph.py output/")
            print(f"4. Pre-tokenize: python -m ../data.pretokenize graph.jsonl output/ --output-dir ../pretokenized/")
        else:
            print(f"✗ Failed to download {filename}")
            sys.exit(1)
            
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
