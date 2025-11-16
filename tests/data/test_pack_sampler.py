import random
from dataclasses import dataclass
from typing import Dict, List

import pytest

from experiments.dagseq2dagseq.data.pack_sampler import (
    DocPlacement,
    PackBatchSampler,
)
from experiments.dagseq2dagseq.data.traversal import RandomSelectionStrategy


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


