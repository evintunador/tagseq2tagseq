#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GitHub Data Sampler: Download and sample Python repositories from The Stack dataset.
Streams N items from bigcode/the-stack-dedup (Python only) and writes JSONL efficiently.

Optimizations for large runs (1M–10M+):
- Constant memory streaming (no accumulation)
- Chunked/buffered writes (no per-item flush)
- Optional gzip output to reduce disk + often improve end-to-end throughput
- Optional fast JSON serialization via orjson (falls back to stdlib json)
"""

import argparse
import gzip
import logging
import os
import time
from itertools import islice
from typing import Tuple

from datasets import load_dataset
from tqdm import tqdm

# Optional faster JSON
try:
    import orjson  # type: ignore

    def json_dumps(obj) -> str:
        return orjson.dumps(obj).decode("utf-8")

    JSON_BACKEND = "orjson"
except Exception:
    import json

    def json_dumps(obj) -> str:
        return json.dumps(obj, ensure_ascii=False)

    JSON_BACKEND = "stdlib"


def setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def open_output(path: str, gzip_enabled: bool, text_buffer_bytes: int) -> Tuple[object, bool]:
    """
    Returns (file_handle, is_binary).
    - If gzip_enabled or path ends with .gz: open gzip in binary mode.
    - Else open plain text file with large buffering.
    """
    if gzip_enabled or path.endswith(".gz"):
        return gzip.open(path, "wb", compresslevel=6), True
    return open(path, "w", encoding="utf-8", buffering=max(1, text_buffer_bytes)), False


def load_stack_python_stream(token_enabled: bool = True):
    """
    Load The Stack (dedup) Python stream. Handles datasets auth arg differences.
    """
    try:
        return load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
            token=token_enabled,  # newer datasets versions
        )
    except TypeError:
        return load_dataset(
            "bigcode/the-stack-dedup",
            data_dir="data/python",
            split="train",
            streaming=True,
            use_auth_token=token_enabled,  # older datasets versions
        )


def download_sample(
    output_path: str,
    limit: int,
    *,
    gzip_enabled: bool,
    buffer_lines: int,
    flush_every_lines: int,
    max_content_chars: int,
    text_buffer_mb: int,
    verbose: bool,
):
    setup_logging(verbose)

    logging.info(f"JSON backend: {JSON_BACKEND}")
    logging.info(f"Output: {output_path} (gzip={gzip_enabled})")
    logging.info(
        f"Settings: limit={limit:,}, buffer_lines={buffer_lines:,}, flush_every_lines={flush_every_lines:,}, "
        f"max_content_chars={max_content_chars}, text_buffer_mb={text_buffer_mb}"
    )

    print("Loading bigcode/the-stack-dedup dataset (Python repositories only)...")

    ds = load_stack_python_stream(token_enabled=True)

    f, is_binary = open_output(output_path, gzip_enabled=gzip_enabled, text_buffer_bytes=text_buffer_mb * 1024 * 1024)

    print(f"Streaming + writing {limit:,} items to {output_path}...")
    start = time.time()

    line_buffer = []
    written = 0
    errors = 0

    def write_buffer(force_flush: bool = False):
        nonlocal line_buffer, f
        if not line_buffer:
            return
        chunk = "\n".join(line_buffer) + "\n"
        if is_binary:
            f.write(chunk.encode("utf-8"))
        else:
            f.write(chunk)
        line_buffer.clear()
        if force_flush:
            f.flush()

    try:
        with tqdm(total=limit, desc="Downloading", unit="items") as pbar:
            for item in islice(ds, limit):
                try:
                    if not isinstance(item, dict):
                        errors += 1
                        pbar.update(1)
                        continue

                    # Optional: truncate very large content to prevent gigantic output files.
                    # Set --max-content-chars 0 to disable truncation.
                    if max_content_chars > 0 and "content" in item and item["content"]:
                        c = item["content"]
                        if isinstance(c, str) and len(c) > max_content_chars:
                            item["content"] = c[:max_content_chars] + "...[truncated]"

                    line_buffer.append(json_dumps(item))
                    written += 1
                    pbar.update(1)

                    if len(line_buffer) >= buffer_lines:
                        write_buffer(force_flush=False)

                    if flush_every_lines > 0 and written % flush_every_lines == 0:
                        write_buffer(force_flush=True)

                except Exception as e:
                    errors += 1
                    logging.debug(f"Error serializing/writing item #{written+1}: {e}")

            # final flush
            write_buffer(force_flush=True)

    finally:
        try:
            f.close()
        except Exception:
            pass

    elapsed = time.time() - start
    rate = written / elapsed if elapsed > 0 else 0.0

    size_str = "unknown"
    try:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        size_str = f"{size_mb:,.1f} MB"
    except Exception:
        pass

    print(f"Done. Wrote {written:,} items (errors: {errors:,}) in {elapsed:,.1f}s ({rate:,.1f} items/s). Size: {size_str}")


def main():
    parser = argparse.ArgumentParser(
        prog="GitHubDataSampler",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file (default auto-named). Use .gz extension or --gzip to compress.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Number of items to sample (default: 100,000)",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Write gzip-compressed JSONL (.jsonl.gz). Recommended for 1M+.",
    )
    parser.add_argument(
        "--buffer-lines",
        type=int,
        default=10_000,
        help="Number of JSON lines to buffer before writing a chunk (default: 10,000).",
    )
    parser.add_argument(
        "--flush-every-lines",
        type=int,
        default=200_000,
        help="Force a flush every N items (default: 200,000). 0 disables.",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=50_000,
        help="Truncate 'content' to this many chars. Set to 0 to disable truncation.",
    )
    parser.add_argument(
        "--text-buffer-mb",
        type=int,
        default=8,
        help="OS/file buffer size for non-gzip output in MB (default: 8).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        raise SystemExit("Error: --limit must be positive")

    # Auto-generate output filename if not specified
    if args.output is None:
        if args.limit >= 10_000_000:
            args.output = "sample_10M.jsonl.gz" if args.gzip else "sample_10M.jsonl"
        elif args.limit >= 1_000_000:
            args.output = "sample_1M.jsonl.gz" if args.gzip else "sample_1M.jsonl"
        elif args.limit >= 100_000:
            args.output = "sample_100k.jsonl.gz" if args.gzip else "sample_100k.jsonl"
        elif args.limit >= 10_000:
            args.output = "sample_10k.jsonl.gz" if args.gzip else "sample_10k.jsonl"
        else:
            args.output = f"sample_{args.limit}.jsonl.gz" if args.gzip else f"sample_{args.limit}.jsonl"

    gzip_enabled = args.gzip or str(args.output).endswith(".gz")

    download_sample(
        output_path=args.output,
        limit=args.limit,
        gzip_enabled=gzip_enabled,
        buffer_lines=max(1, args.buffer_lines),
        flush_every_lines=max(0, args.flush_every_lines),
        max_content_chars=max(0, args.max_content_chars),
        text_buffer_mb=max(1, args.text_buffer_mb),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
