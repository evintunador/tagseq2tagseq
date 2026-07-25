"""Tier C — hard dedup gate vs the training corpus. CPU-only.

Policy (2026-07-24): a repo lives in the training dataset OR the benchmark,
never both. Two passes:
  1. repo-name intersection — benchmark examples whose repo appears in the
     training corpus are EXCLUDED.
  2. file-hash — among survivors, normalized-content SHA1 of each primary/aux
     file is matched against training file hashes to catch cross-repo
     copy-pastes (vendored deps, renamed forks). Primary-file match drops the
     example; aux-only match drops that aux doc (and the example if no aux
     remains).

Training-side inputs come from the per-language raw shards
(/fss-data/.../raw/<lang>/shards/*.jsonl: max_stars_repo_name + content) or,
where raw shards are absent, the built corpus's tokenized_graph.jsonl
raw_identifier ("repo:path") for repo names (hash pass unavailable there —
reported as such). Results feed the port build (exclusion) and the harness
report (verification that the shipped port has zero overlap).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from .schema import CrossDocExample

logger = logging.getLogger(__name__)


def _norm_content(text: str) -> bytes:
    """Whitespace-normalized bytes for copy-paste hashing (per-line rstrip,
    blank lines dropped) — robust to EOL/trailing-space churn between hosts."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln).encode("utf-8", "replace")


def file_hash(text: str) -> str:
    return hashlib.sha1(_norm_content(text)).hexdigest()


def load_training_repos_from_graph(tokenized_graph_jsonl: Path) -> Set[str]:
    """Repo names from a built corpus's tokenized_graph.jsonl raw_identifier
    ("repo/name:path" → "repo/name"). Lines without ':' contribute nothing."""
    repos: Set[str] = set()
    with open(tokenized_graph_jsonl) as f:
        for line in f:
            rid = json.loads(line).get("raw_identifier") or ""
            if ":" in rid:
                repos.add(rid.split(":", 1)[0])
    return repos


def iter_training_shards(shard_dir: Path) -> Iterable[Tuple[str, str]]:
    """Yield (repo_name, content) from raw downloader shards."""
    for shard in sorted(shard_dir.glob("*.jsonl")):
        with open(shard) as f:
            for line in f:
                rec = json.loads(line)
                yield rec.get("max_stars_repo_name", ""), rec.get("content", "")


@dataclass
class DedupReport:
    port: str
    n_examples: int
    n_repo_overlap_dropped: int = 0
    n_hash_dropped: int = 0
    n_aux_docs_hash_dropped: int = 0
    n_survivors: int = 0
    hash_pass_available: bool = True
    overlapping_repos: List[str] = field(default_factory=list)


def run_dedup(
    port_name: str,
    examples: List[CrossDocExample],
    training_repos: Set[str],
    training_hashes: Optional[Set[str]] = None,
) -> Tuple[List[CrossDocExample], DedupReport]:
    """Apply the hard dedup policy; return (surviving examples, report)."""
    rep = DedupReport(port=port_name, n_examples=len(examples),
                      hash_pass_available=training_hashes is not None)
    survivors: List[CrossDocExample] = []
    overlap: Set[str] = set()

    for ex in examples:
        if ex.repo in training_repos:
            rep.n_repo_overlap_dropped += 1
            overlap.add(ex.repo)
            continue
        if training_hashes is not None:
            if file_hash(ex.context) in training_hashes:
                rep.n_hash_dropped += 1
                continue
            kept_aux = tuple(d for d in ex.aux
                             if file_hash(d.content) not in training_hashes)
            n_dropped_aux = len(ex.aux) - len(kept_aux)
            rep.n_aux_docs_hash_dropped += n_dropped_aux
            if n_dropped_aux and not kept_aux:
                rep.n_hash_dropped += 1
                continue
            if n_dropped_aux:
                ex = CrossDocExample(repo=ex.repo, file_path=ex.file_path,
                                     context=ex.context, target=ex.target,
                                     aux=kept_aux, meta=ex.meta)
        survivors.append(ex)

    rep.n_survivors = len(survivors)
    rep.overlapping_repos = sorted(overlap)
    return survivors, rep
