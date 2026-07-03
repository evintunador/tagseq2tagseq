"""
data/merge_datasets.py — union N pretokenized dataset dirs into one artifact.

Each input is a self-contained pretokenized dataset (as produced by
data/pretokenize.py and friends): a metadata.json, a tokenized_graph.jsonl, and
one or more shard_*.bin token files. This script merges several of them into a
single directory that is byte-compatible with data.dataset.GraphIndex, so every
downstream stage (split_graph.py, epoch_precompute.py, training) runs on the
merged dir unchanged.

Why this works without re-tokenizing anything:

  * Token offsets are shard-relative. ``tok_offset_bytes`` is a byte offset into
    a node's OWN shard file (past the 1024-byte / 256*int32 header). If we keep
    each source shard intact and only renumber ``tok_shard_idx``, every offset
    stays valid — no shard byte-rewriting.
  * Cross-dataset edges are already latent. build_graph.py keeps every
    ``outgoing`` target, including ones that dangle within a single source; only
    ``incoming`` is a reverse index restricted to existing nodes. So merging is:
    union the nodes, then recompute ``incoming`` over the union — cross-source
    links light up for free. (GraphIndex.neighbors_out already skips unknown
    targets, so leftover dangling ``outgoing`` entries are harmless.)

Collisions: when the same normed_identifier appears in more than one input, the
higher-priority source wins (see --priority). The losing node's shard bytes stay
on disk as harmless dead space; only its graph entry is dropped.

Provenance: every merged node gets a ``source`` field set to its input's tag.
This drives source-stratified splitting (split_graph.py --stratify-by-source)
and, eventually, source-aware pack partitioning for a mixed
wiki+thestack+arxiv corpus.

Usage:
    python data/merge_datasets.py \\
        --inputs enwiki=/fss-data/.../enwiki simplewiki=/fss-data/.../simplewiki \\
        --output /fss-data/.../pretokenized_datasets/wiki_merged \\
        --priority enwiki,simplewiki \\
        --shard-mode hardlink
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

SHARD_MODES = ("hardlink", "symlink", "copy")


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

def _parse_inputs(pairs: List[str]) -> List[Tuple[str, Path]]:
    """Parse ``tag=dir`` CLI pairs into an ordered [(tag, Path), ...] list.

    Order is preserved and used as the default collision priority. Duplicate
    tags are rejected (the tag is the provenance label and must be unique).
    """
    inputs: List[Tuple[str, Path]] = []
    seen: set[str] = set()
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"--inputs entry {pair!r} must be of the form tag=dir"
            )
        tag, dir_str = pair.split("=", 1)
        tag = tag.strip()
        if not tag:
            raise ValueError(f"--inputs entry {pair!r} has an empty tag")
        if tag in seen:
            raise ValueError(f"Duplicate input tag {tag!r}")
        seen.add(tag)
        inputs.append((tag, Path(dir_str.strip())))
    if not inputs:
        raise ValueError("No --inputs provided")
    return inputs


def _resolve_priority(priority_arg: str | None, tags: List[str]) -> List[str]:
    """Return the tag order (highest priority first) for collision resolution.

    Defaults to input order. When --priority is given it must be a permutation
    of the input tags (every tag present exactly once) so the ordering is
    unambiguous.
    """
    if not priority_arg:
        return list(tags)
    order = [t.strip() for t in priority_arg.split(",") if t.strip()]
    if sorted(order) != sorted(tags):
        raise ValueError(
            f"--priority {order} must be a permutation of the input tags {tags}"
        )
    return order


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_dataset(dataset_dir: Path) -> Tuple[dict, List[dict]]:
    """Load (metadata, nodes) from a pretokenized dataset directory."""
    meta_path = dataset_dir / "metadata.json"
    graph_path = dataset_dir / "tokenized_graph.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {dataset_dir}")
    if not graph_path.exists():
        raise FileNotFoundError(f"tokenized_graph.jsonl not found in {dataset_dir}")

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    nodes: List[dict] = []
    with open(graph_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                nodes.append(json.loads(line))
    return metadata, nodes


# ---------------------------------------------------------------------------
# Shard materialization
# ---------------------------------------------------------------------------

def _place_shard(src: Path, dst: Path, mode: str) -> None:
    """Materialize a single shard file at ``dst`` referencing ``src``.

    hardlink → falls back to copy across filesystems (EXDEV).
    symlink   → absolute symlink to the source.
    copy      → physical byte copy.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError as e:
            # EXDEV: cross-device link not permitted → fall back to a copy.
            logger.warning("hardlink %s → %s failed (%s); copying instead", src, dst, e)
            shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown shard mode: {mode!r}")


def _materialize_shards(
    inputs: List[Tuple[str, Path]],
    input_metas: Dict[str, dict],
    output_dir: Path,
    mode: str,
) -> Tuple[Dict[str, Dict[int, int]], List[str]]:
    """Materialize every input's shards into ``output_dir`` with global names.

    Shards are laid out in INPUT order (not priority order) so the on-disk
    numbering is stable/readable; collision resolution happens separately at the
    node level and does not depend on shard placement.

    Returns:
        shard_remap: tag -> {local_shard_idx -> global_shard_idx}
        shard_filenames: ordered list of global shard basenames for metadata
    """
    shard_remap: Dict[str, Dict[int, int]] = {}
    shard_filenames: List[str] = []
    global_idx = 0

    for tag, src_dir in inputs:
        local_names = input_metas[tag]["shard_filenames"]
        remap: Dict[int, int] = {}
        for local_idx, local_name in enumerate(local_names):
            src = src_dir / local_name
            if not src.exists():
                raise FileNotFoundError(f"Shard {src} referenced by {tag} not found")
            global_name = f"shard_{global_idx:06d}.bin"
            _place_shard(src, output_dir / global_name, mode)
            remap[local_idx] = global_idx
            shard_filenames.append(global_name)
            global_idx += 1
        shard_remap[tag] = remap
        logger.info("Materialized %d shard(s) from %s (%s)", len(local_names), tag, mode)

    return shard_remap, shard_filenames


# ---------------------------------------------------------------------------
# Node union + edge recompute
# ---------------------------------------------------------------------------

def merge_nodes(
    inputs: List[Tuple[str, Path]],
    input_nodes: Dict[str, List[dict]],
    priority: List[str],
    shard_remap: Dict[str, Dict[int, int]],
) -> Tuple[List[dict], Dict[str, int]]:
    """Union nodes across inputs by priority and recompute incoming edges.

    Iterates sources highest-priority first; a normed_identifier already claimed
    by a higher-priority source is skipped (that source wins the collision).
    Each surviving node gets its ``tok_shard_idx`` remapped to the global index
    and a ``source`` provenance field stamped on.

    Returns:
        merged: list of node dicts (insertion order = priority, then file order)
        collisions: tag -> count of nodes dropped because a higher-priority
            source already had that id
    """
    merged: Dict[str, dict] = {}
    collisions: Dict[str, int] = {tag: 0 for tag, _ in inputs}

    for tag in priority:
        remap = shard_remap[tag]
        for node in input_nodes[tag]:
            nid = node["normed_identifier"]
            if nid in merged:
                collisions[tag] += 1
                continue
            node = dict(node)  # don't mutate the loaded copy
            local_shard = node["tok_shard_idx"]
            if local_shard not in remap:
                raise ValueError(
                    f"Node {nid!r} from {tag} has tok_shard_idx={local_shard} "
                    f"outside that source's shard range"
                )
            node["tok_shard_idx"] = remap[local_shard]
            node["source"] = tag
            merged[nid] = node

    # Recompute incoming over the union. Clear first so we never double-count or
    # carry stale within-source reverse edges; dangling outgoing targets (not in
    # the union) are left in place — harmless, GraphIndex skips them.
    for node in merged.values():
        node["incoming"] = []
    for src_nid, node in merged.items():
        for tgt_nid in node.get("outgoing", []):
            tgt = merged.get(tgt_nid)
            if tgt is not None:
                tgt["incoming"].append(src_nid)

    # Sort each incoming list for determinism (matches build_graph.py's sorted output).
    for node in merged.values():
        node["incoming"].sort()

    return list(merged.values()), collisions


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def merge_datasets(
    inputs: List[Tuple[str, Path]],
    output_dir: Path,
    priority: List[str],
    shard_mode: str,
) -> dict:
    """Merge the given inputs into ``output_dir``. Returns the written metadata."""
    input_metas: Dict[str, dict] = {}
    input_nodes: Dict[str, List[dict]] = {}
    for tag, src_dir in inputs:
        meta, nodes = _load_dataset(src_dir)
        input_metas[tag] = meta
        input_nodes[tag] = nodes
        logger.info("Loaded %s: %d nodes, %d shard(s)", tag, len(nodes),
                    len(meta.get("shard_filenames", [])))

    # --- Homogeneity check (hard fail): tokenizer + dtype must match ---------
    # Token ids from different tokenizers/dtypes are incomparable; a merged
    # corpus with mixed encodings is silently corrupt, so this must abort.
    tokenizers = {input_metas[t]["tokenizer"] for t, _ in inputs}
    dtypes = {input_metas[t]["dtype_str"] for t, _ in inputs}
    if len(tokenizers) != 1:
        raise ValueError(f"Inputs use different tokenizers: {tokenizers}")
    if len(dtypes) != 1:
        raise ValueError(f"Inputs use different token dtypes: {dtypes}")
    tokenizer = tokenizers.pop()
    dtype_str = dtypes.pop()

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_remap, shard_filenames = _materialize_shards(
        inputs, input_metas, output_dir, shard_mode
    )

    merged_nodes, collisions = merge_nodes(inputs, input_nodes, priority, shard_remap)

    # --- Per-source counts (post-collision) for the manifest -----------------
    source_counts: Dict[str, int] = {tag: 0 for tag, _ in inputs}
    for node in merged_nodes:
        source_counts[node["source"]] += 1

    # --- Write tokenized_graph.jsonl -----------------------------------------
    graph_path = output_dir / "tokenized_graph.jsonl"
    with open(graph_path, "w", encoding="utf-8") as f:
        for node in merged_nodes:
            f.write(json.dumps(node) + "\n")

    # --- Write metadata.json -------------------------------------------------
    metadata = {
        "tokenizer": tokenizer,
        "dtype_str": dtype_str,
        "shard_filenames": shard_filenames,
        "merged_from": {
            "priority": priority,
            "shard_mode": shard_mode,
            "sources": [
                {
                    "tag": tag,
                    "dir": str(src_dir.resolve()),
                    "nodes_in": len(input_nodes[tag]),
                    "nodes_kept": source_counts[tag],
                    "nodes_dropped_to_collision": collisions[tag],
                }
                for tag, src_dir in inputs
            ],
            "total_nodes": len(merged_nodes),
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    total_dropped = sum(collisions.values())
    logger.info(
        "Merged %d nodes across %d sources → %s (%d dropped to collisions)",
        len(merged_nodes), len(inputs), output_dir, total_dropped,
    )
    for tag, _ in inputs:
        logger.info(
            "  %-16s kept=%d dropped=%d",
            tag, source_counts[tag], collisions[tag],
        )
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True, metavar="TAG=DIR",
        help="Input datasets as tag=dir pairs. The tag becomes each node's "
             "'source' provenance field and (by default) the collision priority.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for the merged dataset.",
    )
    parser.add_argument(
        "--priority", default=None,
        help="Comma-separated tag order, highest priority first (wins id "
             "collisions). Must be a permutation of the input tags. "
             "Defaults to --inputs order.",
    )
    parser.add_argument(
        "--shard-mode", default="hardlink", choices=SHARD_MODES,
        help="How to reference source shards in the merged dir (default: hardlink).",
    )
    args = parser.parse_args()

    inputs = _parse_inputs(args.inputs)
    tags = [tag for tag, _ in inputs]
    priority = _resolve_priority(args.priority, tags)

    merge_datasets(
        inputs=inputs,
        output_dir=args.output,
        priority=priority,
        shard_mode=args.shard_mode,
    )


if __name__ == "__main__":
    main()
