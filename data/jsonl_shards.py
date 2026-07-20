"""Iterate JSON records from a single .jsonl file OR a directory of .jsonl shards.

The full-subset downloader (data/stack_sharded_download.py) writes one JSONL per
parquet shard into a directory, whereas the 2M-sample downloaders write a single
JSONL. The graph builders accept either: pass a file path (read that file) or a
directory (read every *.jsonl in it, sorted). Repo grouping happens after the full
read, so a repo whose files span multiple shards is still assembled correctly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


def iter_jsonl_records(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a .jsonl file or a dir of .jsonl shards.

    Malformed lines are skipped. A directory is read as sorted ``*.jsonl`` (the
    ``.done`` markers the downloader writes are ignored — they aren't ``.jsonl``).
    """
    path = Path(path)
    if path.is_dir():
        shards = sorted(path.glob("*.jsonl"))
        if not shards:
            raise FileNotFoundError(f"no *.jsonl shards in {path}")
        logger.info("reading %d JSONL shards from %s", len(shards), path)
        for shard in shards:
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
