"""Generate per-step pack fixtures from the thestack BFS epoch_0 schedule.

Walks the exact rank-0 / world_size-8 pack sequence and saves a set of
representative packs as reusable .pt fixtures for kernel correctness tests.

The saved fixtures use the same schema as tests/fixtures/real_packs/ so
load_fixture_batch() in the harness loads them without modification.

Usage:
    python scripts/generate_thestack_fixtures.py

Outputs:
    tests/fixtures/thestack_packs/zero_<N>.pt   — sparse-bucket packs where
        triton_v18 backward returns all-zero gradients (vs non-zero in flex)
    tests/fixtures/thestack_packs/nan_0.pt      — the pack at step 98 where
        triton_v18 backward produces a NaN gradient in dQ (flex is finite)
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.thestack_nan_probe import (
    load_bucket_lists,
    iter_rank_packs,
    pack_to_mask_inputs,
)
from data.bucketed_pack_dataset import _make_bucket_sequence
from data.epoch_precompute import _table_to_records

PARQUET   = "schedules/thestack_bfs/epoch_0/packs.parquet"
RANK      = 0
WORLD_SIZE = 8
MAX_GRANTS = 256
N_BUCKETS  = 32
EPOCH_SEED = 0

# Steps confirmed to produce all-zero triton_v18 backward (from job 41830)
ZERO_STEPS = [64, 67, 80, 83, 84, 85, 88, 89, 90, 97]
# Step confirmed to produce NaN in dQ for triton_v18 (from job 41830)
NAN_STEPS  = [98]

OUT_DIR = Path("tests/fixtures/thestack_packs")


def pack_record_to_fixture(record, step: int, label: str, kv_block_count: int) -> dict:
    """Convert a PackRecord to the fixture dict schema used by load_fixture_batch."""
    pos = 0
    doc_spans = []
    for i, eff_len in enumerate(record.effective_lens):
        doc_spans.append({
            "doc_id":    record.doc_ids[i],
            "start":     pos,
            "end":       pos + eff_len,
        })
        pos += eff_len

    # Store link_to_target with raw GraphIndex doc IDs as targets (same as
    # simplewiki fixtures) so load_fixture_batch can remap them uniformly.
    link_to_target: Dict[int, List[int]] = {}
    for link_pos, targets in zip(record.link_end_positions, record.link_target_doc_ids):
        raw_targets = [t for t in targets if t in set(record.doc_ids)]
        if raw_targets:
            link_to_target[int(link_pos)] = raw_targets

    n_grants = sum(len(v) for v in link_to_target.values())

    return {
        "doc_spans":      doc_spans,
        "link_to_target": {str(k): v for k, v in link_to_target.items()},
        "n_grants":       n_grants,
        "kv_block_count": kv_block_count,
        "seq_len":        pos,
        "density_label":  label,
        "dataset":        "thestack",
        "source_step":    step,
        "rank":           RANK,
        "world_size":     WORLD_SIZE,
        "token_budget":   32768,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading parquet: {PARQUET}")
    bucket_lists = load_bucket_lists(PARQUET)
    total_packs = sum(len(v) for v in bucket_lists.values())
    print(f"  {total_packs} packs across {len(bucket_lists)} buckets")
    print(f"  rank={RANK}  world_size={WORLD_SIZE}  max_grants={MAX_GRANTS}")

    target_steps = set(ZERO_STEPS + NAN_STEPS)
    end_step = max(target_steps)

    saved = {}
    for step, bucket, pack in iter_rank_packs(
        bucket_lists, N_BUCKETS, WORLD_SIZE, RANK,
        start_step=0, end_step=end_step, epoch_seed=EPOCH_SEED,
    ):
        if step not in target_steps:
            continue

        if step in NAN_STEPS:
            label = f"nan_{NAN_STEPS.index(step)}"
        else:
            label = f"zero_{ZERO_STEPS.index(step)}"

        fixture = pack_record_to_fixture(pack, step, label, pack.kv_block_count)
        path = OUT_DIR / f"{label}.pt"
        torch.save(fixture, str(path))
        saved[label] = (step, pack.kv_block_count, fixture["n_grants"], fixture["seq_len"])
        print(f"  saved {label}.pt  (step={step}, kv_block_count={pack.kv_block_count}, "
              f"n_grants={fixture['n_grants']}, seq_len={fixture['seq_len']})")

    print(f"\nSaved {len(saved)} fixtures to {OUT_DIR}/")
    print("\nPaste these into THESTACK_FIXTURES in attention_harness.py:")
    for label, (step, kv, grants, seqlen) in sorted(saved.items()):
        n_docs = None  # not stored; compute from pack if needed
        print(f'    FixtureMeta(label="{label}", n_grants={grants}, max_grants={MAX_GRANTS}, '
              f'kv_block_count={kv}, seq_len={seqlen}, n_docs=0, '
              f'file_label="{label}", fixture_dir="thestack"),')


if __name__ == "__main__":
    main()
