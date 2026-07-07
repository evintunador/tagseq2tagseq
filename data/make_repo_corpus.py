"""
make_repo_corpus.py — carve a single-repo corpus dir out of a code dataset.

Link resolution for code models matches a bare relative import path
(``utils/helpers.py``) against a corpus keyed by that same bare path (via the
detector's ``index_doc_span``, which strips the repo prefix). Bare paths are only
unambiguous WITHIN one repo — across the full multi-repo dataset many repos share
paths like ``setup.py``. So a code corpus must contain exactly one repo.

This script produces such a corpus cheaply: like ``split_graph.py``, it writes a
directory containing only a filtered ``tokenized_graph.jsonl`` plus a
``metadata.json`` whose ``shard_filenames`` are ABSOLUTE paths to the parent's
shards. The token shards are shared, not copied — the output dir is a few hundred
KB regardless of repo size.

Corpus identifiers are of the form ``owner/repo:path/to/file.py``; a node belongs
to ``--repo`` when the part before the first ``:`` equals it exactly (so
``repoA`` does not match ``repoA-fork``).

Usage:
    python data/make_repo_corpus.py \\
        --dataset-dir /path/to/pretokenized_datasets/thestack \\
        --repo "000alen/Phaedra" \\
        [--output-dir /path/to/out]     # default: <dataset-dir>/repos/<safe_repo>

Point ``generate.py --dataset`` (or the eval annotator corpus) at the resulting
directory.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _safe_repo_name(repo: str) -> str:
    """Filesystem-safe directory name for a repo id (``owner/repo`` -> ``owner_repo``)."""
    return repo.replace("/", "_").replace(":", "_")


def _repo_of(raw_identifier: str) -> str:
    """Return the repo prefix (part before the first ``:``) of a raw_identifier."""
    return raw_identifier.split(":", 1)[0]


def make_repo_corpus(dataset_dir: Path, repo: str, output_dir: Path) -> int:
    """Write a single-repo corpus dir. Returns the number of nodes written.

    Reads the parent's tokenized_graph.jsonl + metadata.json, keeps only nodes
    whose raw_identifier repo prefix == ``repo``, rewrites edges to stay within
    the repo, and writes metadata.json with absolute shard paths (shards shared).
    """
    graph_path = dataset_dir / "tokenized_graph.jsonl"
    meta_path = dataset_dir / "metadata.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"tokenized_graph.jsonl not found in {dataset_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {dataset_dir}")

    with open(meta_path, encoding="utf-8") as f:
        parent_metadata = json.load(f)

    # Keep only this repo's nodes; collect their normed_identifiers for edge filtering.
    kept: List[dict] = []
    kept_nids = set()
    with open(graph_path, encoding="utf-8") as f:
        for line in f:
            node = json.loads(line)
            raw = node.get("raw_identifier", "")
            if _repo_of(raw) == repo:
                kept.append(node)
                kept_nids.add(node["normed_identifier"])

    if not kept:
        raise ValueError(
            f"No nodes found for repo {repo!r} in {dataset_dir}. "
            f"Identifiers are of the form 'owner/repo:path'; check the repo name."
        )

    # Resolve shards to absolute paths (shared, not copied) — same convention as
    # split_graph.write_splits.
    abs_shards = [
        str((dataset_dir / fname).resolve())
        for fname in parent_metadata.get("shard_filenames", [])
    ]
    repo_metadata = {**parent_metadata, "shard_filenames": abs_shards}

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "tokenized_graph.jsonl", "w", encoding="utf-8") as f:
        for node in kept:
            node = dict(node)
            node["outgoing"] = [nid for nid in node.get("outgoing", []) if nid in kept_nids]
            node["incoming"] = [nid for nid in node.get("incoming", []) if nid in kept_nids]
            f.write(json.dumps(node) + "\n")
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(repo_metadata, f, ensure_ascii=False)

    logger.info("Wrote %d nodes for repo %s -> %s", len(kept), repo, output_dir)
    return len(kept)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    parser = argparse.ArgumentParser(
        description="Carve a single-repo corpus dir out of a multi-repo code dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to the parent pretokenized dataset directory (e.g. .../thestack).",
    )
    parser.add_argument(
        "--repo", required=True,
        help="Repo id to extract, e.g. '000alen/Phaedra' (part before the ':').",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Default: <dataset-dir>/repos/<owner_repo>.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else dataset_dir / "repos" / _safe_repo_name(args.repo)
    )
    n = make_repo_corpus(dataset_dir, args.repo, output_dir)
    print(f"Wrote {n} nodes for repo {args.repo!r} to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
