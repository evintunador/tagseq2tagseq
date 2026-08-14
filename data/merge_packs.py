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


def _read_ids_and_sources(graph_dir: Path) -> Tuple[List[str], List[str]]:
    """Return (normed_identifiers, sources) in doc_id order. ``source`` is the
    provenance tag merge_datasets stamps on each merged node (empty string if a
    graph predates the field). Used to detect collision HIJACKS — an id whose
    winning merged node belongs to a DIFFERENT source than the one being
    remapped (e.g. kotlin & java both emit bare FQN class names, java wins)."""
    path = graph_dir / "tokenized_graph.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"tokenized_graph.jsonl not found in {graph_dir}")
    ids: List[str] = []
    sources: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                ids.append(d["normed_identifier"])
                sources.append(d.get("source", ""))
    return ids, sources


def _build_remap(source_train_dir: Path, tag: str,
                 merged_nid_to_id: Dict[str, int],
                 merged_id_to_source: Dict[int, str]) -> Tuple[Dict[int, int], int]:
    """source_train_doc_id -> merged_train_doc_id, via normed_identifier.

    Returns ``(remap, n_missing)``. A source-train node is UNMAPPABLE if either:
      (a) its normed_identifier is absent from the merged train graph, OR
      (b) it is a collision HIJACK — the id IS present but the winning merged
          node belongs to a DIFFERENT source (``merged_id_to_source[mid] != tag``).
    Case (b) is the dangerous one: two sources emit the same normed_identifier
    (kotlin & java both use bare fully-qualified class names with no repo prefix),
    merge_datasets keeps the higher-priority source's node, so a naive id lookup
    would silently resolve this source's doc to the OTHER source's tokens —
    corrupting the cross-doc signal. Both cases get NO remap entry; the caller
    drops any pack whose core docs reference them and prunes dead cross-doc
    grants. Tolerate + log rather than raise: disjoint-namespace sources
    (wiki/arxiv/…) map 100% while FQN-colliding code langs lose ~0.06%."""
    source_ids = _read_normed_ids(source_train_dir)
    remap: Dict[int, int] = {}
    missing = 0
    for src_doc_id, nid in enumerate(source_ids):
        merged_id = merged_nid_to_id.get(nid)
        if merged_id is None or merged_id_to_source.get(merged_id, tag) != tag:
            missing += 1
            continue
        remap[src_doc_id] = merged_id
    return remap, missing


def _remap_pack(r: PackRecord, remap: Dict[int, int]) -> bool:
    """Remap a pack's doc_ids + link targets in place, source->merged.

    Returns True if the pack survives, False if it must be DROPPED because one of
    its core ``doc_ids`` was a collision-dropped node (its tokens have no merged
    doc_id, so BucketedPackDataset can't materialize it). Dead cross-doc grants
    (targets that were dropped, or already point out-of-split) are silently
    pruned — a grant that can't resolve simply doesn't fire, same as val/OOC
    targets. Grant list structure (per-doc inner lists) is preserved."""
    if any(d not in remap for d in r.doc_ids):
        return False
    r.doc_ids = [remap[d] for d in r.doc_ids]
    r.link_target_doc_ids = [[remap[d] for d in tgts if d in remap]
                             for tgts in r.link_target_doc_ids]
    return True


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

def _parse_source(spec: str) -> Tuple[str, Path, List[Path], str]:
    parts = spec.split("=")
    if len(parts) != 4:
        raise ValueError(
            f"--source {spec!r} must be tag=train_dir=schedule_epoch_dir(s)=target "
            f"(schedule field may be a COMMA-SEPARATED list of epoch dirs to union "
            f"for multi-epoch balancing; target is an int or 'all')"
        )
    tag, train_dir, sched_field, target = parts
    # Multi-epoch: comma-separated distinct-seed epoch dirs are UNIONED (each epoch
    # is a different packing of the same graph → new co-occurrence structure, not a
    # replay). Balancing then selects `target` across the union.
    sched_dirs = [Path(p.strip()) for p in sched_field.split(",") if p.strip()]
    return tag.strip(), Path(train_dir.strip()), sched_dirs, target.strip()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def merge_packs(
    merged_train_dir: Path,
    sources: List[Tuple[str, Path, List[Path], str]],
    output_dir: Path,
    n_buckets: int,
    seed: int,
    token_budget: int,
) -> dict:
    logger.info("Reading merged train graph node ids from %s", merged_train_dir)
    merged_ids, merged_srcs = _read_ids_and_sources(merged_train_dir)
    merged_nid_to_id = {nid: i for i, nid in enumerate(merged_ids)}
    merged_id_to_source = {i: s for i, s in enumerate(merged_srcs)}
    logger.info("Merged train graph: %d nodes", len(merged_ids))

    all_records: List[PackRecord] = []
    manifest_sources = []
    pack_id_base = 0

    for tag, train_dir, sched_dirs, target in sources:
        # Union packs across one-or-more (distinct-seed) epoch dirs. Each epoch is
        # a different packing of the same graph (new neighborhoods co-packed), so
        # unioning multiplies the available packs for multi-epoch balancing. All
        # epochs of a source share the same layout_policy (read from the first).
        recs = []
        for sd in sched_dirs:
            ep_recs = _table_to_records(pq.read_table(str(sd / "packs.parquet")))
            # Stamp the epoch index this pack was BUDGETED under so the loader
            # replays the SAME stochastic prefix coin-flip (which is salted by
            # epoch: _include_prefix hashes id:epoch). Prefer the schedule's own
            # epoch_idx from metadata; fall back to parsing epoch_N from the dir
            # name. Without this, a unioned epoch_1 pack materialized under the
            # merged loader's epoch 0 flips different docs' prefixes → T!=budget.
            _sm = json.loads((sd / "metadata.json").read_text())
            _ep = _sm.get("epoch_idx")
            if _ep is None:
                _name = sd.name  # e.g. "epoch_3"
                _ep = int(_name.split("_")[-1]) if _name.startswith("epoch_") else 0
            for r in ep_recs:
                r.layout_epoch = int(_ep)
            recs.extend(ep_recs)
        sched_meta = json.loads((sched_dirs[0] / "metadata.json").read_text())
        source_layout = sched_meta.get("layout_policy", "")
        if len(sched_dirs) > 1:
            logger.info("%s: unioned %d packs across %d epochs (%s) (layout=%r)",
                        tag, len(recs), len(sched_dirs),
                        ",".join(p.name for p in sched_dirs), source_layout)
        else:
            logger.info("%s: loaded %d packs from %s (layout=%r)",
                        tag, len(recs), sched_dirs[0].name, source_layout)

        # Balance across the union. NOTE: bucket_id is a per-epoch density quantile
        # with the same n_buckets, so pooling epochs into one bucket_id space keeps
        # the density-stratified selection meaningful; pack_id collides across
        # epochs but is reassigned uniquely below, so the collision is harmless.
        if target == "all":
            chosen = recs
        else:
            chosen = _select_balanced(recs, int(target), seed)
        logger.info("%s: selected %d / %d packs", tag, len(chosen), len(recs))

        # Remap doc_ids source-train -> merged-train (tolerant of collision drops
        # AND collision HIJACKS: an id owned by another source in the merged graph)
        remap, n_missing = _build_remap(train_dir, tag, merged_nid_to_id,
                                        merged_id_to_source)
        if n_missing:
            logger.warning("%s: %d source-train nodes absent from merged graph "
                           "(id-collision losers) — packs referencing them as "
                           "CORE docs will be dropped; dead grants pruned.",
                           tag, n_missing)
        kept: List[PackRecord] = []
        dropped_packs = 0
        for r in chosen:
            if not _remap_pack(r, remap):
                dropped_packs += 1
                continue
            r.pack_id = pack_id_base
            r.layout_name = source_layout
            pack_id_base += 1
            kept.append(r)
        if dropped_packs:
            logger.warning("%s: dropped %d / %d selected packs (core doc was a "
                           "collision loser); kept %d.",
                           tag, dropped_packs, len(chosen), len(kept))
        all_records.extend(kept)

        manifest_sources.append({
            "tag": tag,
            "train_dir": str(train_dir),
            "schedule_dirs": [str(p) for p in sched_dirs],
            "n_epochs_unioned": len(sched_dirs),
            "layout_policy": source_layout,
            "packs_available": len(recs),
            "packs_selected": len(chosen),
            "packs_kept": len(kept),
            "packs_dropped_collision": dropped_packs,
            "nodes_missing_collision": n_missing,
            "target": target,
            "tokens_est": len(kept) * token_budget,
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
                   metavar="TAG=TRAIN_DIR=SCHED_EPOCH_DIR(S)=TARGET",
                   help="Repeatable. SCHED_EPOCH_DIR(S) may be a comma-separated "
                        "list of distinct-seed epoch dirs to UNION (multi-epoch "
                        "balancing). TARGET is an int (#packs) or 'all'.")
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
