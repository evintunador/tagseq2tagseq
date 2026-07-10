"""
tests/data/test_merge_packs.py — unit tests for data/merge_packs.py.

Covers the pure-logic pieces that don't need real datasets on disk:
  - _select_balanced: exact target count, even spread across density buckets,
    edge cases (target >= available, target of 1)
  - _parse_source: tag=train_dir=schedule_dir=target spec parsing + validation
"""
from collections import Counter

import pytest

from data.epoch_precompute import PackRecord
from data.merge_packs import _select_balanced, _parse_source


def _make_records(n, n_buckets, seed=0):
    """n records spread over n_buckets via a deterministic pseudo-shuffle."""
    recs = []
    for i in range(n):
        recs.append(PackRecord(
            pack_id=i, doc_ids=[i], effective_lens=[8], truncated_flags=[False],
            trim_sides=["tail"], link_end_positions=[], link_target_doc_ids=[],
            kv_block_count=i, bucket_id=(i * 7 + 3) % n_buckets,
        ))
    return recs


# ---------------------------------------------------------------------------
# _select_balanced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target", [1, 250, 499, 500, 501])
def test_select_balanced_exact_count(target):
    recs = _make_records(500, n_buckets=32)
    selected = _select_balanced(recs, target, seed=42)
    assert len(selected) == min(target, len(recs))


def test_select_balanced_returns_all_when_target_exceeds():
    recs = _make_records(100, n_buckets=8)
    assert len(_select_balanced(recs, 999, seed=42)) == 100


def test_select_balanced_spreads_across_buckets():
    # A 50% draw should take ~half of each bucket, not drain a few buckets.
    recs = _make_records(800, n_buckets=16)
    selected = _select_balanced(recs, 400, seed=42)
    avail = Counter(r.bucket_id for r in recs)
    got = Counter(r.bucket_id for r in selected)
    ratios = [got[b] / avail[b] for b in avail]
    # Every bucket contributes proportionally (allow rounding slack from
    # largest-remainder apportionment on small buckets).
    assert min(ratios) >= 0.3
    assert max(ratios) <= 0.7


def test_select_balanced_deterministic():
    recs = _make_records(300, n_buckets=8)
    a = {r.pack_id for r in _select_balanced(recs, 150, seed=42)}
    b = {r.pack_id for r in _select_balanced(recs, 150, seed=42)}
    assert a == b


# ---------------------------------------------------------------------------
# _parse_source
# ---------------------------------------------------------------------------

def test_parse_source_valid():
    tag, train_dir, sched_dir, target = _parse_source(
        "arxiv=/data/arxiv/splits/train=/sched/arxiv_bfs/epoch_0=152600"
    )
    assert tag == "arxiv"
    assert str(train_dir) == "/data/arxiv/splits/train"
    assert str(sched_dir) == "/sched/arxiv_bfs/epoch_0"
    assert target == "152600"


def test_parse_source_all_target():
    _, _, _, target = _parse_source("wiki=/a=/b=all")
    assert target == "all"


def test_parse_source_wrong_field_count_raises():
    with pytest.raises(ValueError, match="tag=train_dir"):
        _parse_source("wiki=/a=/b")
