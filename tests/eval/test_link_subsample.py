"""Tests for the graph-sparsity link-subsampling instrument.

Covers eval/scoring.py::subsample_link_to_target + hash_of_edges — the
mask-time edge-dropout knob behind the graph-sparsity scaling law
(memory [[graph-sparsity-scaling-law]]).
"""

import numpy as np
import pytest

from eval.scoring import subsample_link_to_target, hash_of_edges


def _make_map(n_src=10, deg=4, seed=1):
    """A grant map with n_src sources each linking to `deg` targets."""
    rng = np.random.RandomState(seed)
    l2t = {}
    for s in range(n_src):
        pos = 100 * (s + 1)  # arbitrary distinct link positions
        l2t[pos] = sorted(int(x) for x in rng.choice(1000, size=deg, replace=False))
    return l2t


def _count_edges(l2t):
    return sum(len(v) for v in l2t.values())


# --------------------------------------------------------------------------- #
# Endpoint identity — the load-bearing property.
# --------------------------------------------------------------------------- #

def test_keep_1_is_identity():
    l2t = _make_map()
    out = subsample_link_to_target(l2t, 1.0, seed=0)
    assert out == l2t


def test_keep_1_returns_a_copy_not_the_input():
    l2t = _make_map()
    out = subsample_link_to_target(l2t, 1.0, seed=0)
    out[next(iter(out))].append(-999)
    # mutating the output must not touch the input
    assert all(-999 not in v for v in l2t.values())


def test_keep_0_is_empty_equiv_doc_causal():
    l2t = _make_map()
    for mode in ("edge", "node"):
        assert subsample_link_to_target(l2t, 0.0, seed=0, mode=mode) == {}


def test_empty_input_stays_empty():
    assert subsample_link_to_target({}, 0.5, seed=0) == {}


# --------------------------------------------------------------------------- #
# Edge mode: uniform thinning, correct counts, determinism.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("keep", [0.25, 0.5, 0.75])
def test_edge_keeps_expected_count(keep):
    l2t = _make_map(n_src=20, deg=5)          # 100 edges
    n_in = _count_edges(l2t)
    out = subsample_link_to_target(l2t, keep, seed=0, mode="edge")
    assert _count_edges(out) == round(keep * n_in)


def test_edge_is_a_subset_of_input():
    l2t = _make_map()
    out = subsample_link_to_target(l2t, 0.5, seed=3, mode="edge")
    for pos, tgts in out.items():
        assert pos in l2t
        for t in tgts:
            assert t in l2t[pos]
        # no duplicates introduced
        assert len(tgts) == len(set(tgts))


def test_edge_deterministic_same_seed():
    l2t = _make_map()
    a = subsample_link_to_target(l2t, 0.5, seed=7, mode="edge")
    b = subsample_link_to_target(l2t, 0.5, seed=7, mode="edge")
    assert a == b


def test_edge_different_seed_differs():
    l2t = _make_map(n_src=30, deg=6)
    a = subsample_link_to_target(l2t, 0.5, seed=1, mode="edge")
    b = subsample_link_to_target(l2t, 0.5, seed=2, mode="edge")
    assert a != b


def test_edge_density_is_monotone_in_keep():
    l2t = _make_map(n_src=40, deg=8)
    counts = [
        _count_edges(subsample_link_to_target(l2t, k, seed=0, mode="edge"))
        for k in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert counts == sorted(counts)
    assert counts[0] == 0
    assert counts[-1] == _count_edges(l2t)


def test_edge_nested_across_datasets_independent_of_insertion_order():
    # Same edge set, different dict insertion order → identical subsample
    # (selection is over the canonical sorted edge list).
    l2t = _make_map()
    shuffled = dict(reversed(list(l2t.items())))
    a = subsample_link_to_target(l2t, 0.5, seed=5, mode="edge")
    b = subsample_link_to_target(shuffled, 0.5, seed=5, mode="edge")
    assert a == b


# --------------------------------------------------------------------------- #
# Node mode: whole-target dropout.
# --------------------------------------------------------------------------- #

def test_node_keeps_only_surviving_targets():
    # one hub target shared by many sources; node-drop is all-or-nothing on it
    l2t = {10: [1, 2], 20: [1, 3], 30: [1, 4]}   # target 1 is a hub (in-deg 3)
    # keep 3/4 targets → drop exactly one distinct target
    out = subsample_link_to_target(l2t, 0.75, seed=0, mode="node")
    kept_targets = {t for v in out.values() for t in v}
    all_targets = {1, 2, 3, 4}
    assert len(kept_targets) == 3
    assert kept_targets < all_targets
    # every surviving edge points at a kept target
    for v in out.values():
        assert all(t in kept_targets for t in v)


def test_node_vs_edge_differ():
    l2t = _make_map(n_src=25, deg=5)
    e = subsample_link_to_target(l2t, 0.5, seed=0, mode="edge")
    n = subsample_link_to_target(l2t, 0.5, seed=0, mode="node")
    assert e != n


def test_bad_args_raise():
    l2t = _make_map()
    with pytest.raises(ValueError):
        subsample_link_to_target(l2t, 1.5, seed=0)
    with pytest.raises(ValueError):
        subsample_link_to_target(l2t, 0.5, seed=0, mode="bogus")


# --------------------------------------------------------------------------- #
# Fingerprint: stable + process-independent.
# --------------------------------------------------------------------------- #

def test_hash_of_edges_stable():
    edges = [(1, 2), (3, 4), (5, 6)]
    assert hash_of_edges(edges) == hash_of_edges(list(edges))


def test_hash_of_edges_order_sensitive():
    assert hash_of_edges([(1, 2), (3, 4)]) != hash_of_edges([(3, 4), (1, 2)])
