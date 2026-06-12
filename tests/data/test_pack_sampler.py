import random
from dataclasses import dataclass
from typing import Dict, List

import pytest

from data.pack_sampler import (
    DocPlacement,
    PackBatchSampler,
)
from data.traversal import RandomSelectionStrategy


@dataclass
class DummyGraph:
    """Minimal graph + token-length stub compatible with PackBatchSampler."""

    token_lens: Dict[int, int]
    outgoing: Dict[int, List[int]]
    incoming: Dict[int, List[int]]

    def __len__(self) -> int:
        return len(self.token_lens)

    def get_token_len(self, doc_id: int) -> int:
        return self.token_lens[doc_id]

    def neighbors_out(self, doc_id: int) -> List[int]:
        return self.outgoing.get(doc_id, [])

    def neighbors_in(self, doc_id: int) -> List[int]:
        return self.incoming.get(doc_id, [])

    def get_normed_identifier(self, doc_id: int) -> str:
        return str(doc_id)

    def get_raw_identifier(self, normed_identifier: str) -> str:
        return normed_identifier

    def get_outgoing_links(self, normed_identifier: str) -> list:
        return []

    def get_incoming_links(self, normed_identifier: str) -> list:
        return []


class _FixedSeedRng:
    """Deterministic RNG stub that always returns the same value for randrange."""

    def __init__(self, value: int = 0) -> None:
        self.value = value

    def randrange(self, n: int) -> int:  # type: ignore[override]
        return self.value % n

    def random(self) -> float:  # type: ignore[override]
        # Used by some traversal strategies; we keep it deterministic.
        return 0.0


def test_pack_batch_sampler_basic_properties():
    """Sampler should respect token budget and avoid duplicates within a pack."""
    graph = DummyGraph(
        token_lens={0: 4, 1: 6, 2: 3},
        outgoing={0: [], 1: [], 2: []},
        incoming={0: [], 1: [], 2: []},
    )

    sampler = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: RandomSelectionStrategy(),
        token_budget=10,
        doc_budget=None,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        seed=123,
    )

    pack = next(iter(sampler))
    assert pack, "Expected a non-empty pack"

    doc_ids = [p.doc_id for p in pack]
    assert len(doc_ids) == len(set(doc_ids)), "Doc ids must be unique within a pack"

    total_tokens = sum(p.effective_len for p in pack)
    assert 0 < total_tokens <= 10

    # Per-doc lengths should not exceed their full token lengths.
    for p in pack:
        assert p.effective_len <= graph.get_token_len(p.doc_id)
        assert p.doc_trim_side == "tail"


def test_compute_budgeted_length_truncate_vs_skip():
    """Internal per-doc budgeting should distinguish 'truncate' and 'skip'."""
    graph = DummyGraph(
        token_lens={0: 10},
        outgoing={0: []},
        incoming={0: []},
    )

    # Truncate policy
    sampler_trunc = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: RandomSelectionStrategy(),
        token_budget=20,
        doc_budget=5,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        seed=0,
    )
    effective_len, truncated = sampler_trunc._compute_budgeted_length(10)
    assert effective_len == 5
    assert truncated is True

    # Skip policy
    sampler_skip = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: RandomSelectionStrategy(),
        token_budget=20,
        doc_budget=5,
        overflow_policy="skip",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        seed=0,
    )
    assert sampler_skip._compute_budgeted_length(10) is None


def test_order_placements_prefer_targets_first():
    """Targets-first ordering should prefer link targets before linkers."""
    # 0 -> 1 -> 2 in the graph; targets-first should order [2, 1, 0].
    graph = DummyGraph(
        token_lens={0: 1, 1: 1, 2: 1},
        outgoing={0: [1], 1: [2], 2: []},
        incoming={0: [], 1: [0], 2: [1]},
    )

    sampler = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: RandomSelectionStrategy(),
        token_budget=10,
        order_mode="prefer_targets_first",
        doc_budget=None,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        seed=0,
    )

    placements = [
        DocPlacement(doc_id=0, effective_len=1, truncated=False, doc_trim_side="tail"),
        DocPlacement(doc_id=1, effective_len=1, truncated=False, doc_trim_side="tail"),
        DocPlacement(doc_id=2, effective_len=1, truncated=False, doc_trim_side="tail"),
    ]

    ordered = sampler._order_placements(placements)
    assert [p.doc_id for p in ordered] == [2, 1, 0]


def test_order_placements_keeps_components_contiguous():
    """prefer_targets_first must keep each connected component contiguous.

    Two independent chains (10->11->12 and 20->21) are packed together. The
    targets-first sort reverses each chain internally, but a component's docs
    must never be interleaved with another component's — the doc_concatenated
    mask kernel requires each component to be a single contiguous run.
    """
    graph = DummyGraph(
        token_lens={d: 1 for d in (10, 11, 12, 20, 21)},
        outgoing={10: [11], 11: [12], 12: [], 20: [21], 21: []},
        incoming={11: [10], 12: [11], 21: [20]},
    )

    sampler = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: RandomSelectionStrategy(),
        token_budget=10,
        order_mode="prefer_targets_first",
        doc_budget=None,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        seed=0,
    )

    # Deliberately interleave the two components in insertion order.
    placements = [
        DocPlacement(10, 1, False, "tail", component_id=0),
        DocPlacement(20, 1, False, "tail", component_id=1),
        DocPlacement(11, 1, False, "tail", component_id=0),
        DocPlacement(21, 1, False, "tail", component_id=1),
        DocPlacement(12, 1, False, "tail", component_id=0),
    ]

    ordered = sampler._order_placements(placements)

    # Each component appears as one contiguous block; component 0 first since
    # its first doc was inserted before component 1's.
    comp_seq = [p.component_id for p in ordered]
    assert comp_seq == [0, 0, 0, 1, 1]
    # Within each component, targets precede linkers.
    assert [p.doc_id for p in ordered] == [12, 11, 10, 21, 20]


def _component_sampler(graph):
    return PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: RandomSelectionStrategy(),
        token_budget=10,
        doc_budget=None,
        overflow_policy="truncate",
        seed=0,
    )


def test_assign_components_groups_by_connectivity():
    """_assign_components labels weakly-connected components of the induced graph."""
    # Two disconnected chains: {10->11->12} and {20->21}.
    graph = DummyGraph(
        token_lens={d: 1 for d in (10, 11, 12, 20, 21)},
        outgoing={10: [11], 11: [12], 12: [], 20: [21], 21: []},
        incoming={11: [10], 12: [11], 21: [20]},
    )
    sampler = _component_sampler(graph)

    placements = [DocPlacement(d, 1, False, "tail") for d in (10, 11, 12, 20, 21)]
    sampler._assign_components(placements)
    cid = {p.doc_id: p.component_id for p in placements}

    # The two chains are distinct components; members of a chain share an id.
    assert cid[10] == cid[11] == cid[12]
    assert cid[20] == cid[21]
    assert cid[10] != cid[20]
    # Numbered by first appearance.
    assert cid[10] == 0 and cid[20] == 1


def test_assign_components_robust_to_traversal_restart():
    """Regression: a single seed-and-grow call that spans DISCONNECTED docs
    (because the traversal strategy restarts its frontier when exhausted) must
    still yield one component PER connected sub-graph — not one merged blob.

    Previously component_id was a per-_seed_and_grow_subgraph counter, so a
    frontier restart merged unrelated docs into one component, causing
    doc_concatenated to fuse unrelated repos into a single super-document.
    """
    # Three singleton docs with NO edges between them — fully disconnected.
    graph = DummyGraph(
        token_lens={0: 1, 1: 1, 2: 1},
        outgoing={0: [], 1: [], 2: []},
        incoming={0: [], 1: [], 2: []},
    )
    sampler = _component_sampler(graph)

    placements = [DocPlacement(d, 1, False, "tail") for d in (0, 1, 2)]
    sampler._assign_components(placements)
    comp_ids = [p.component_id for p in placements]

    # Each disconnected doc is its own component.
    assert len(set(comp_ids)) == 3
    assert sorted(comp_ids) == [0, 1, 2]


def test_pack_level_truncation_head_vs_tail():
    """Pack-level truncation should trim from the configured end of the pack."""
    graph = DummyGraph(
        token_lens={0: 5, 1: 5, 2: 5},
        outgoing={0: [], 1: [], 2: []},
        incoming={0: [], 1: [], 2: []},
    )

    # Head-trimming sampler
    sampler_head = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: RandomSelectionStrategy(),
        token_budget=11,
        doc_budget=None,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        seed=0,
    )

    placements_head = [
        DocPlacement(doc_id=0, effective_len=5, truncated=False, doc_trim_side="tail"),
        DocPlacement(doc_id=1, effective_len=5, truncated=False, doc_trim_side="tail"),
        DocPlacement(doc_id=2, effective_len=5, truncated=False, doc_trim_side="tail"),
    ]
    trimmed_head = sampler_head._apply_pack_truncation(placements_head, total_tokens=15)

    total_tokens_head = sum(p.effective_len for p in trimmed_head)
    assert total_tokens_head == 11

    truncated_indices_head = [i for i, p in enumerate(trimmed_head) if p.truncated]
    # Truncated docs should form a prefix of the pack under head trimming.
    assert truncated_indices_head == list(range(len(truncated_indices_head)))

    # Tail-trimming sampler
    sampler_tail = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: RandomSelectionStrategy(),
        token_budget=11,
        doc_budget=None,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="tail",
        seed=0,
    )

    placements_tail = [
        DocPlacement(doc_id=0, effective_len=5, truncated=False, doc_trim_side="tail"),
        DocPlacement(doc_id=1, effective_len=5, truncated=False, doc_trim_side="tail"),
        DocPlacement(doc_id=2, effective_len=5, truncated=False, doc_trim_side="tail"),
    ]
    trimmed_tail = sampler_tail._apply_pack_truncation(placements_tail, total_tokens=15)

    total_tokens_tail = sum(p.effective_len for p in trimmed_tail)
    assert total_tokens_tail == 11

    truncated_indices_tail = [i for i, p in enumerate(trimmed_tail) if p.truncated]
    # Truncated docs should form a suffix of the pack under tail trimming.
    assert truncated_indices_tail == list(
        range(len(trimmed_tail) - len(truncated_indices_tail), len(trimmed_tail))
    )


class _ChainStrategy:
    """
    Deterministic traversal strategy that walks along outgoing edges in a chain.

    On each step it follows the first outgoing neighbor of the last document in
    the local component's history, or stays on the same node if there are no
    outgoing neighbors. The sampler is still responsible for de-duplicating
    document ids at the pack level.
    """

    def reset_for_new_pack(self, graph, rng, first_doc_id) -> None:  # type: ignore[override]
        del rng
        self._graph = graph
        self._start = first_doc_id

    def propose_next(self, graph, rng, current_doc_ids):  # type: ignore[override]
        del rng
        last = current_doc_ids[-1] if current_doc_ids else self._start
        neighbors = self._graph.neighbors_out(last)
        return neighbors[0] if neighbors else last


def test_iter_chain_no_truncation():
    """A simple chain walk should include each doc once with full length."""
    graph = DummyGraph(
        token_lens={0: 3, 1: 3, 2: 3},
        outgoing={0: [1], 1: [2], 2: []},
        incoming={0: [], 1: [0], 2: [1]},
    )

    sampler = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: _ChainStrategy(),
        token_budget=9,  # exactly enough for all three docs
        doc_budget=None,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        seed=0,
    )
    sampler._rng = _FixedSeedRng(0)  # always pick doc 0 as the seed

    pack = next(iter(sampler))
    doc_ids = [p.doc_id for p in pack]
    assert doc_ids == [0, 1, 2]
    assert all(p.effective_len == 3 for p in pack)
    assert all(not p.truncated for p in pack)


def test_iter_respects_doc_budget_with_truncate():
    """Iterating the sampler should apply doc-level truncation via doc_budget."""
    graph = DummyGraph(
        token_lens={0: 10, 1: 3},
        outgoing={0: [1], 1: []},
        incoming={0: [], 1: [0]},
    )

    sampler = PackBatchSampler(
        graph=graph,
        strategy_factory=lambda: _ChainStrategy(),
        token_budget=20,
        doc_budget=5,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        seed=0,
    )
    sampler._rng = _FixedSeedRng(0)  # ensure doc 0 is the seed

    pack = next(iter(sampler))
    doc_ids = [p.doc_id for p in pack]
    assert doc_ids == [0, 1]

    lengths = {p.doc_id: p.effective_len for p in pack}
    truncated_flags = {p.doc_id: p.truncated for p in pack}

    # Doc 0 is truncated down to the doc_budget; doc 1 fits in full.
    assert lengths[0] == 5
    assert truncated_flags[0] is True

    assert lengths[1] == 3
    assert truncated_flags[1] is False


