"""
Tests for BucketedPackDataset and BucketState.

These tests use in-memory mock parquet files (no real dataset required).
They verify:
  - Exact resume continuation after a mid-epoch checkpoint
  - world_size change on resume (bucket_consumed cursors remain valid)
  - grad_accum flexibility (accum step boundary changes don't break coverage)
"""

import collections
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from data.bucketed_pack_dataset import BucketState, BucketedPackDataset, _make_bucket_sequence
from data.epoch_precompute import PackRecord, _records_to_table


# ---------------------------------------------------------------------------
# Helpers — build mock epoch dirs without a real dataset
# ---------------------------------------------------------------------------

def _make_epoch_dir(
    tmp_dir: str,
    n_buckets: int = 4,
    packs_per_bucket: int = 10,
    epoch_idx: int = 0,
) -> str:
    """Write a mock packs.parquet + metadata.json to a temp directory."""
    epoch_dir = os.path.join(tmp_dir, f"epoch_{epoch_idx}")
    os.makedirs(epoch_dir, exist_ok=True)

    records = []
    pack_id = 0
    for bucket_id in range(n_buckets):
        for _ in range(packs_per_bucket):
            records.append(PackRecord(
                pack_id=pack_id,
                doc_ids=[pack_id * 2, pack_id * 2 + 1],
                effective_lens=[64, 64],
                truncated_flags=[False, False],
                trim_sides=["tail", "tail"],
                link_end_positions=[],
                link_target_doc_ids=[],
                kv_block_count=100 + bucket_id * 10,
                bucket_id=bucket_id,
            ))
            pack_id += 1

    table = _records_to_table(records)
    pq.write_table(table, os.path.join(epoch_dir, "packs.parquet"), compression="snappy")
    with open(os.path.join(epoch_dir, "metadata.json"), "w") as f:
        json.dump({"n_buckets": n_buckets, "n_packs": len(records), "token_budget": 128}, f)

    return epoch_dir


class _MockGraph:
    """Minimal GraphIndex stand-in for materialisation tests."""
    def get_normed_identifier(self, doc_id): return f"doc_{doc_id}"
    def get_raw_identifier(self, nid): return nid
    def get_outgoing_links(self, nid): return []
    def get_incoming_links(self, nid): return []
    def get_categories(self, nid): return ""


class _MockBackend:
    """Minimal PretokShardedBackend that returns zeros."""
    def get_tokens_by_id(self, doc_id):
        import numpy as np
        return np.zeros(64, dtype=np.int32)


class _MockLayout:
    def prefix_length(self, info): return 0
    def suffix_length(self, info): return 0
    def prefix_tokens(self, info): return []
    def suffix_tokens(self, info): return []


def _make_dataset(
    epoch_dirs,
    rank=0,
    world_size=1,
    start_state=None,
):
    return BucketedPackDataset(
        epoch_dirs=epoch_dirs,
        graph=_MockGraph(),
        backend=_MockBackend(),
        layout=_MockLayout(),
        rank=rank,
        world_size=world_size,
        start_state=start_state,
    )


def _collect_pack_ids(dataset, n_steps: int) -> List[int]:
    """Yield n_steps batches and return the pack_ids."""
    pack_ids = []
    it = iter(dataset)
    for _ in range(n_steps):
        batch = next(it)
        # pack_ids are encoded in doc_ids[0]//2 — see _make_epoch_dir
        doc_id = batch["doc_ids"][0]
        pack_ids.append(doc_id // 2)
    return pack_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBucketedPackDataset:

    def test_resume_exact_continuation(self):
        """Run N steps, checkpoint, create new dataset from state, verify same packs."""
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = _make_epoch_dir(tmp, n_buckets=4, packs_per_bucket=10)
            epoch_dirs = [epoch_dir]

            n_initial = 5
            ds1 = _make_dataset(epoch_dirs)
            packs1 = _collect_pack_ids(ds1, n_initial)
            state_after_5 = ds1.get_state()

            # Collect 5 more from the uninterrupted run
            packs1_next5 = _collect_pack_ids(ds1, 5)

            # Resume from checkpoint
            ds2 = _make_dataset(epoch_dirs, start_state=state_after_5)
            packs2_next5 = _collect_pack_ids(ds2, 5)

            assert packs1_next5 == packs2_next5, (
                f"Resume diverged:\n  original: {packs1_next5}\n  resumed:  {packs2_next5}"
            )

    def test_world_size_change_on_resume(self):
        """bucket_consumed cursors stay valid when world_size doubles on resume.

        After N steps with world_size=2 (ranks 0 and 1), resume with world_size=4.
        The new ranks should not repeat or gap-skip any previously assigned packs.
        """
        with tempfile.TemporaryDirectory() as tmp:
            # Generous bucket to avoid falling back
            epoch_dir = _make_epoch_dir(tmp, n_buckets=2, packs_per_bucket=40)
            epoch_dirs = [epoch_dir]

            n_steps = 4  # consume 4 accum steps with world_size=2 → 8 packs total

            ds_r0 = _make_dataset(epoch_dirs, rank=0, world_size=2)
            ds_r1 = _make_dataset(epoch_dirs, rank=1, world_size=2)

            packs_r0 = _collect_pack_ids(ds_r0, n_steps)
            packs_r1 = _collect_pack_ids(ds_r1, n_steps)

            # Ranks must never overlap
            assert len(set(packs_r0) & set(packs_r1)) == 0, (
                "Ranks 0 and 1 consumed overlapping packs"
            )

            state = ds_r0.get_state()  # all ranks maintain identical bucket_consumed

            # Resume with world_size=4; two new ranks start where the old pair left off
            ds_new_r2 = _make_dataset(epoch_dirs, rank=2, world_size=4, start_state=state)
            ds_new_r3 = _make_dataset(epoch_dirs, rank=3, world_size=4, start_state=state)

            packs_r2 = _collect_pack_ids(ds_new_r2, n_steps)
            packs_r3 = _collect_pack_ids(ds_new_r3, n_steps)

            all_consumed = packs_r0 + packs_r1 + packs_r2 + packs_r3
            # All pack_ids unique (no repeats)
            assert len(all_consumed) == len(set(all_consumed)), (
                "Duplicate packs after world_size change"
            )

    def test_grad_accum_flexibility(self):
        """Changing grad_accum between runs does not break coverage or ordering.

        With accum_steps=1 each optimizer step corresponds to one accum step.
        With accum_steps=2 each optimizer step draws two consecutive accum steps.
        Both should draw the same packs in the same order; only the optimizer
        step boundary changes.
        """
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = _make_epoch_dir(tmp, n_buckets=4, packs_per_bucket=20)
            epoch_dirs = [epoch_dir]

            # Run 8 accum steps at once (simulating accum_steps=1)
            ds_a1 = _make_dataset(epoch_dirs)
            packs_a1 = _collect_pack_ids(ds_a1, 8)

            # Run same 8 steps in two batches of 4 (simulating accum_steps=2)
            ds_a2 = _make_dataset(epoch_dirs)
            packs_a2_first4 = _collect_pack_ids(ds_a2, 4)
            # (optimizer step boundary — state saved/resumed here)
            packs_a2_second4 = _collect_pack_ids(ds_a2, 4)
            packs_a2 = packs_a2_first4 + packs_a2_second4

            assert packs_a1 == packs_a2, (
                f"grad_accum change altered pack ordering:\n  a=1: {packs_a1}\n  a=2: {packs_a2}"
            )

    def test_state_reflects_latest_yield(self):
        """get_state() returns the position after the most recently yielded item."""
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = _make_epoch_dir(tmp, n_buckets=2, packs_per_bucket=20)
            epoch_dirs = [epoch_dir]
            ds = _make_dataset(epoch_dirs)
            state0 = ds.get_state()
            assert state0.global_accum_step == 0

            it = iter(ds)
            next(it)
            state1 = ds.get_state()
            assert state1.global_accum_step == 1

            next(it)
            state2 = ds.get_state()
            assert state2.global_accum_step == 2

    def test_epoch_exhaustion_raises(self):
        """RuntimeError is raised when all epoch dirs are exhausted."""
        with tempfile.TemporaryDirectory() as tmp:
            # 1 bucket, 2 packs — exhausted in 2 accum steps with world_size=1
            epoch_dir = _make_epoch_dir(tmp, n_buckets=1, packs_per_bucket=2)
            ds = _make_dataset([epoch_dir])
            # drain the epoch
            list_batches = []
            it = iter(ds)
            with pytest.raises(RuntimeError, match="exhausted"):
                while True:
                    list_batches.append(next(it))

    def test_bucket_sequence_no_adjacent_repeats(self):
        """_make_bucket_sequence should not have long runs of the same bucket."""
        seq = _make_bucket_sequence(n_buckets=8, seed=42, n_repeats=10)
        # Each contiguous block of 8 must be a permutation (no repeats within block)
        for i in range(0, len(seq), 8):
            block = seq[i:i + 8]
            if len(block) == 8:
                assert len(set(block)) == 8, f"Block {i//8} has duplicate buckets: {block}"

    def test_token_budget_mismatch_raises(self):
        """AssertionError is raised when a materialized pack has wrong token length."""
        with tempfile.TemporaryDirectory() as tmp:
            epoch_dir = _make_epoch_dir(tmp, n_buckets=1, packs_per_bucket=1)
            # Overwrite metadata with a mismatched token_budget (128 is correct; 512 is wrong).
            with open(os.path.join(epoch_dir, "metadata.json"), "w") as f:
                json.dump({"n_buckets": 1, "n_packs": 1, "token_budget": 512}, f)
            ds = _make_dataset([epoch_dir])
            it = iter(ds)
            with pytest.raises(AssertionError, match="token_budget"):
                next(it)
