"""
data/merge_packs.py — balance + remap N per-source precomputed schedules into a
single multi-dataset epoch, then re-bucket the union for density-balanced DDP.

Motivation
----------
Each source (wiki/thestack/arxiv/fineweb) is precomputed INDEPENDENTLY by
precompute_epochs.py, because EpochPrecomputer bakes in one link_detector +
layout_policy per run and dispatches its worker-partitioner off node 0's
identifier — neither survives a heterogeneous merged graph.  Since we train on
WITHIN-source packs (a pack never mixes sources), a per-source schedule over
that source's own ``splits/train`` graph produces byte-identical packs to what a
merged-graph run would, so we reuse the existing schedules verbatim.

This tool assembles them into one trainable epoch:

  1. Balance   — select a target number of packs per source, drawn EVENLY across
                 that source's density buckets, so the subsample preserves the
                 source's kv_block_count distribution (not a first-N / density-
                 core-biased cut).
  2. Remap     — a pack's ``doc_ids`` index into its source's splits/train graph.
                 Rewrite them to index into the MERGED train graph, matching by
                 ``normed_identifier`` (doc_id == line index in
                 tokenized_graph.jsonl, so both maps are built by a cheap linear
                 read — no GraphIndex construction).  ``link_target_doc_ids`` are
                 remapped the same way.
  3. Concat    — union all remapped records, assign globally-unique pack_ids.
  4. Re-bucket — recompute equal-count quantile buckets over the UNION via
                 _assign_buckets, because bucket B in one source is not the same
                 density as bucket B in another; DDP load-balancing needs buckets
                 that are quantiles over the packs actually trained.

The merged train graph (whose doc_ids the output references) must be produced by
merging each source's ``splits/train`` dir with data/merge_datasets.py FIRST, so
split membership is PRESERVED (never recomputed — recomputing would reshuffle
train/val/test and leak held-out nodes into training packs).

Usage
-----
    python data/merge_packs.py \\
        --merged-train-dir /fss-data/.../pretokenized_datasets/merged_all/splits/train \\
        --source wiki=/fss-data/.../wiki_merged/splits/train=/fss-data/.../schedules/wiki_merged_bfs/epoch_0=all \\
        --source stack=/fss-data/.../thestack/splits/train=/fss-data/.../schedules/thestack_bfs/epoch_0=all \\
        --source arxiv=/fss-data/.../arxiv/splits/train=/fss-data/.../schedules/arxiv_bfs/epoch_0=152600 \\
        --source fineweb=/fss-data/.../fineweb_39b/splits/train=/fss-data/.../schedules/fineweb_bfs/epoch_0=100300 \\
        --output /fss-data/.../pretokenized_datasets/merged_all/epoch_0 \\
        --n-buckets 32 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq

from data.epoch_precompute import (
    PackRecord,
    _assign_buckets,
    _records_to_table,
    _table_to_records,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# doc_id <-> normed_identifier maps (doc_id == line index; no GraphIndex load)
# ---------------------------------------------------------------------------

def _read_normed_ids(graph_dir: Path) -> List[str]:
    """Return normed_identifiers in doc_id order (line order of the jsonl)."""
    path = graph_dir / "tokenized_graph.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"tokenized_graph.jsonl not found in {graph_dir}")
    ids: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["normed_identifier"])
    return ids


def _build_remap(source_train_dir: Path, merged_nid_to_id: Dict[str, int]) -> Dict[int, int]:
    """source_train_doc_id -> merged_train_doc_id, via normed_identifier."""
    source_ids = _read_normed_ids(source_train_dir)
    remap: Dict[int, int] = {}
    missing = 0
    for src_doc_id, nid in enumerate(source_ids):
        merged_id = merged_nid_to_id.get(nid)
        if merged_id is None:
            missing += 1
            continue
        remap[src_doc_id] = merged_id
    if missing:
        # A source-train node absent from the merged-train graph means the merge
        # dropped it (id collision) — every pack referencing it would orphan.
        raise ValueError(
            f"{missing} nodes in {source_train_dir} are absent from the merged "
            f"train graph (collision drop?). Packs cannot be safely remapped."
        )
    return remap


# ---------------------------------------------------------------------------
# Balancing: select target packs evenly across a source's density buckets
# ---------------------------------------------------------------------------

def _select_balanced(records: List[PackRecord], target: int, seed: int) -> List[PackRecord]:
    """Select ``target`` records spread evenly across their bucket_ids.

    Preserves the source's density distribution: within each bucket we shuffle
    and take a proportional share.  ``target`` >= len(records) returns all.
    """
    if target >= len(records):
        return list(records)

    by_bucket: Dict[int, List[PackRecord]] = {}
    for r in records:
        by_bucket.setdefault(r.bucket_id, []).append(r)

    rng = random.Random(seed)
    for b in by_bucket:
        rng.shuffle(by_bucket[b])

    n_total = len(records)
    # Largest-remainder apportionment: each bucket's base quota is floor of its
    # proportional share; leftover picks go to the largest fractional remainders
    # (capped by what each bucket actually has).
    taken: Dict[int, int] = {}
    remainders: List[Tuple[float, int]] = []
    for b, recs in by_bucket.items():
        exact = target * len(recs) / n_total
        base = min(int(exact), len(recs))
        taken[b] = base
        remainders.append((exact - int(exact), b))

    shortfall = target - sum(taken.values())
    # Buckets with unpicked records, largest fractional remainder first.
    remainders.sort(reverse=True)
    for _, b in remainders:
        if shortfall <= 0:
            break
        room = len(by_bucket[b]) - taken[b]
        add = min(room, shortfall)
        taken[b] += add
        shortfall -= add

    selected: List[PackRecord] = []
    for b, recs in by_bucket.items():
        selected.extend(recs[:taken[b]])
    return selected


# ---------------------------------------------------------------------------
# Source spec parsing: tag=train_dir=schedule_dir=target
# ---------------------------------------------------------------------------

def _parse_source(spec: str) -> Tuple[str, Path, Path, str]:
    parts = spec.split("=")
    if len(parts) != 4:
        raise ValueError(
            f"--source {spec!r} must be tag=train_dir=schedule_epoch_dir=target "
            f"(target is an int or 'all')"
        )
    tag, train_dir, sched_dir, target = parts
    return tag.strip(), Path(train_dir.strip()), Path(sched_dir.strip()), target.strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def merge_packs(
    merged_train_dir: Path,
    sources: List[Tuple[str, Path, Path, str]],
    output_dir: Path,
    n_buckets: int,
    seed: int,
    token_budget: int,
) -> dict:
    logger.info("Reading merged train graph node ids from %s", merged_train_dir)
    merged_ids = _read_normed_ids(merged_train_dir)
    merged_nid_to_id = {nid: i for i, nid in enumerate(merged_ids)}
    logger.info("Merged train graph: %d nodes", len(merged_ids))

    all_records: List[PackRecord] = []
    manifest_sources = []
    pack_id_base = 0

    for tag, train_dir, sched_dir, target in sources:
        parquet = sched_dir / "packs.parquet"
        table = pq.read_table(str(parquet))
        recs = _table_to_records(table)

        # Auto-read the layout this source's packs were budgeted under. The
        # schedule metadata is the source of truth: build_packed_batch re-applies
        # the layout's prefix/suffix at materialization, and the sampler budgeted
        # each pack to token_budget INCLUDING that prefix, so a mismatched layout
        # yields T != token_budget and crashes _materialize. Stamping the actual
        # per-source layout lets one BucketedPackDataset materialize a mixed corpus.
        sched_meta = json.loads((sched_dir / "metadata.json").read_text())
        source_layout = sched_meta.get("layout_policy", "")
        logger.info("%s: loaded %d packs from %s (layout=%r)",
                    tag, len(recs), parquet, source_layout)

        # Balance
        if target == "all":
            chosen = recs
        else:
            chosen = _select_balanced(recs, int(target), seed)
        logger.info("%s: selected %d / %d packs", tag, len(chosen), len(recs))

        # Remap doc_ids source-train -> merged-train
        remap = _build_remap(train_dir, merged_nid_to_id)
        for r in chosen:
            r.doc_ids = [remap[d] for d in r.doc_ids]
            r.link_target_doc_ids = [[remap[d] for d in tgts]
                                     for tgts in r.link_target_doc_ids]
            r.pack_id = pack_id_base
            r.layout_name = source_layout
            pack_id_base += 1
        all_records.extend(chosen)

        manifest_sources.append({
            "tag": tag,
            "train_dir": str(train_dir),
            "schedule_dir": str(sched_dir),
            "layout_policy": source_layout,
            "packs_available": len(recs),
            "packs_selected": len(chosen),
            "target": target,
            "tokens_est": len(chosen) * token_budget,
        })

    logger.info("Union: %d packs across %d sources", len(all_records), len(sources))

    # Re-bucket over the union so buckets are quantiles over trained packs.
    _assign_buckets(all_records, n_buckets)
    all_records.sort(key=lambda r: (r.bucket_id, r.pack_id))

    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(_records_to_table(all_records),
                   str(output_dir / "packs.parquet"), compression="snappy")

    metadata = {
        "n_buckets": n_buckets,
        "n_packs": len(all_records),
        "token_budget": token_budget,
        "strategy": "merged",
        "seed": seed,
        "epoch_idx": 0,
        "kv_method": "analytical",
        "link_detector": "per_source",
        "layout_policy": "per_source",
        "merged_train_dir": str(merged_train_dir),
        "sources": manifest_sources,
        "total_tokens_est": len(all_records) * token_budget,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Wrote %d packs (%d buckets) → %s (~%.2fB tokens)",
                len(all_records), n_buckets, output_dir,
                len(all_records) * token_budget / 1e9)
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged-train-dir", required=True, type=Path,
                   help="Merged train graph (produced by merge_datasets over each "
                        "source's splits/train). Output doc_ids index into this.")
    p.add_argument("--source", action="append", required=True, dest="sources",
                   metavar="TAG=TRAIN_DIR=SCHED_EPOCH_DIR=TARGET",
                   help="Repeatable. TARGET is an int (#packs) or 'all'.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output epoch dir (writes packs.parquet + metadata.json).")
    p.add_argument("--n-buckets", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-budget", type=int, default=32768)
    args = p.parse_args()

    sources = [_parse_source(s) for s in args.sources]
    merge_packs(args.merged_train_dir, sources, args.output,
                args.n_buckets, args.seed, args.token_budget)


if __name__ == "__main__":
    main()
