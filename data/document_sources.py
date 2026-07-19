"""
DocumentSource implementations for the pre-tokenization pipeline.

Each source is an iterable of (normed_id, content_str) pairs. New dataset
types (Stack v2, ArXiv, etc.) add a class here without touching the
shared sharding/writing infrastructure in pretokenize.py.
"""
import json
import logging
from pathlib import Path
from typing import Iterator

from data.normalization import normalize_repo_name

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
                normed_id = f"{normalize_repo_name(repo)}:{path}"
                if normed_id in self._graph_normed_ids:
                    yield normed_id, content


class ArxivUnarxiveSource:
    """
    Yields (normed_id, content) for ArXiv papers from a pre-built content JSONL.

    Unlike StackJSONLSource, the heavy lifting (normed_id computation, body
    rehydration into LaTeX, citation rewriting) already happened in the ArXiv
    extractor (data/arxiv_graph_extractor/extract.py), which emits a content
    JSONL of one ``{"normed_identifier", "content"}`` object per line alongside
    graph.jsonl. This source is therefore a thin reader: it yields each record
    whose normed_id appears in the graph.

    Args:
        content_jsonl: Path to the extractor's content JSONL.
        graph_normed_ids: Set of normed_identifier strings from graph.jsonl.
            Only records whose normed_id appears here are yielded (keeps the
            tokenized corpus aligned with the graph after any split/filter).
    """

    def __init__(self, content_jsonl: Path, graph_normed_ids: set[str]):
        self._content_jsonl = content_jsonl
        self._graph_normed_ids = graph_normed_ids

    def __len__(self) -> int:
        # As with StackJSONLSource, the graph size is the expected count; the
        # writer's sentinel handles the actual termination so minor skew is fine.
        return len(self._graph_normed_ids)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        with open(self._content_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                normed_id = record.get("normed_identifier", "")
                content = record.get("content", "")
                if not (normed_id and content):
                    continue
                if normed_id in self._graph_normed_ids:
                    yield normed_id, content


class FineWebSource:
    """
    Yields (normed_id, content) for FineWeb docs from a pre-built content JSONL.

    Identical reader contract to ArxivUnarxiveSource: the builder
    (data/fineweb_graph_extractor/build_fineweb.py) already streamed FineWeb,
    assigned normed ids, and wrote content.jsonl + an edgeless graph.jsonl. This
    source is a thin reader that yields each record whose normed_id is in the
    graph (keeps the tokenized corpus aligned with the graph after any filter).

    Args:
        content_jsonl: Path to the builder's content JSONL.
        graph_normed_ids: Set of normed_identifier strings from graph.jsonl.
    """

    def __init__(self, content_jsonl: Path, graph_normed_ids: set[str]):
        self._content_jsonl = content_jsonl
        self._graph_normed_ids = graph_normed_ids

    def __len__(self) -> int:
        return len(self._graph_normed_ids)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        with open(self._content_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                normed_id = record.get("normed_identifier", "")
                content = record.get("content", "")
                if not (normed_id and content):
                    continue
                if normed_id in self._graph_normed_ids:
                    yield normed_id, content


class ContentJsonlSource:
    """
    Generic thin reader over a builder's ``content.jsonl`` (the arxiv/fineweb
    pattern), yielding (normed_id, content) for records whose normed_id is in the
    graph. Used by the code-language builders (Go package nodes, Java file nodes)
    whose extractor already assigned each node its normed_id and wrote content
    alongside graph.jsonl. Keeps the tokenized corpus aligned after splits.

    Args:
        content_jsonl: Path to the builder's content JSONL.
        graph_normed_ids: Set of normed_identifier strings (import path / FQN).
    """

    def __init__(self, content_jsonl: Path, graph_normed_ids: set[str]):
        self._content_jsonl = content_jsonl
        self._graph_normed_ids = graph_normed_ids

    def __len__(self) -> int:
        return len(self._graph_normed_ids)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        with open(self._content_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                normed_id = record.get("normed_identifier", "")
                content = record.get("content", "")
                if not (normed_id and content):
                    continue
                if normed_id in self._graph_normed_ids:
                    yield normed_id, content


# Backward-compatible alias: the Go pipeline originally named this source after
# its package-node use; it is a generic content.jsonl reader (also used by Java).
GoPackageContentSource = ContentJsonlSource
