"""
DocumentSource implementations for the pre-tokenization pipeline.

Each source is an iterable of (normed_id, content_str) pairs. New dataset
types (Stack v2, ArXiv, etc.) add a class here without touching the
shared sharding/writing infrastructure in pretokenize.py.
"""
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


class MarkdownDirectorySource:
    """
    Yields (normed_id, content) for every .md file under input_dir.

    normed_id is the filename stem (no extension). The raw identifier marker
    written as the last line by dump_extractor.py is stripped from content.
    """

    def __init__(self, input_dir: Path):
        self._files = sorted(input_dir.rglob("*.md"))

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        for filepath in self._files:
            try:
                content = filepath.read_text(encoding="utf-8")
                normed_id = filepath.stem
                # Strip raw identifier marker from last line (written by dump_extractor.py)
                lines = content.rsplit('\n', 1)
                content = lines[0] if len(lines) > 1 else content
                yield normed_id, content
            except Exception as e:
                logger.error(f"Could not read {filepath}: {e}")


def _normalize_repo_name(repo_name: str) -> str:
    """
    Canonical normalization for GitHub repository names.

    Mirrors github_graph_extractor.extract.normalize_repository_name
    exactly so that titles produced here match those in graph.jsonl.
    Pure function with no external deps — duplicated intentionally to
    avoid a cross-module dependency on a script that must be run from
    its own directory.
    """
    clean = repo_name.replace("/", "_").replace("-", "_").lower()
    clean = re.sub(r"[^a-z0-9\-_]", "_", clean)
    clean = re.sub(r"__+", "_", clean)
    clean = clean.strip("_")
    h = hashlib.md5(clean.encode("utf-8")).hexdigest()[:6]
    return f"{clean}_{h}"


class StackJSONLSource:
    """
    Yields (normed_id, content) for Python files in a The Stack JSONL dump
    that appear in the pre-built dependency graph.

    Compatible with both the-stack-dedup (v1) and the-stack-v2 record
    formats — both use 'max_stars_repo_name' / 'max_stars_repo_path'
    for repository metadata and 'content' for file content.

    Args:
        jsonl_path: Path to the downloaded JSONL file (e.g. sample_1M.jsonl).
        graph_normed_ids: Set of normed_identifier strings from graph.jsonl.
            Only records whose reconstructed normed_id appears here are yielded.
    """

    def __init__(self, jsonl_path: Path, graph_normed_ids: set[str]):
        self._jsonl_path = jsonl_path
        self._graph_normed_ids = graph_normed_ids

    def __len__(self) -> int:
        # Every graph node should have a matching JSONL record; use the
        # graph size as the expected count. The writer handles the actual
        # sentinel-based termination so minor mismatches are fine.
        return len(self._graph_normed_ids)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        with open(self._jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                repo = record.get("max_stars_repo_name", "")
                path = record.get("max_stars_repo_path", "")
                content = record.get("content", "")
                if not (repo and path and content):
                    continue
                normed_id = f"{_normalize_repo_name(repo)}:{path}"
                if normed_id in self._graph_normed_ids:
                    yield normed_id, content
