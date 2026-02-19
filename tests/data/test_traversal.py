import random
from dataclasses import dataclass
from typing import Dict, List

import pytest

from experiments.dagseq2dagseq.data.traversal import (
    BFSStrategy,
    CompositeTraversalStrategy,
    DFSStrategy,
    RandomSelectionStrategy,
    RandomWalkStrategy,
    TraversalStrategy,
)


@dataclass
class SimpleGraph:
    """A minimal in-memory graph for testing traversal strategies."""

    outgoing: Dict[int, List[int]]
    incoming: Dict[int, List[int]]

    def __len__(self) -> int:
        return len(self.outgoing)

    def neighbors_out(self, doc_id: int) -> List[int]:
        return self.outgoing.get(doc_id, [])

    def neighbors_in(self, doc_id: int) -> List[int]:
        return self.incoming.get(doc_id, [])


def test_random_selection_strategy_basic():
    graph = SimpleGraph(
        outgoing={0: [], 1: [], 2: []},
        incoming={0: [], 1: [], 2: []},
    )
    rng = random.Random(123)
    strategy = RandomSelectionStrategy()

    # Should work even if reset is not called, but we call it to follow the contract.
    strategy.reset_for_new_pack(graph, rng, first_doc_id=0)

    for _ in range(20):
        doc_id = strategy.propose_next(graph, rng, current_pack_ids=[])
        assert 0 <= doc_id < len(graph)


def test_random_walk_strategy_follows_edges_and_teleports():
    # Simple chain: 0 -> 1 -> 2, no incoming edges used.
    outgoing = {0: [1], 1: [2], 2: []}
    incoming = {0: [], 1: [0], 2: [1]}
    graph = SimpleGraph(outgoing=outgoing, incoming=incoming)
    rng = random.Random(42)

    strategy = RandomWalkStrategy(edge_mode="outgoing", restart_prob=0.0)
    strategy.reset_for_new_pack(graph, rng, first_doc_id=0)

    # First two steps should deterministically follow the chain 0 -> 1 -> 2.
    step1 = strategy.propose_next(graph, rng, current_pack_ids=[0])
    step2 = strategy.propose_next(graph, rng, current_pack_ids=[0, step1])
    assert step1 == 1
    assert step2 == 2

    # Now at node 2 with no outgoing neighbors; we should teleport somewhere in [0, 2].
    step3 = strategy.propose_next(graph, rng, current_pack_ids=[0, step1, step2])
    assert 0 <= step3 < len(graph)


def test_bfs_strategy_order_and_restart():
    # Chain 0 -> 1 -> 2 -> 3
    outgoing = {0: [1], 1: [2], 2: [3], 3: []}
    incoming = {0: [], 1: [0], 2: [1], 3: [2]}
    graph = SimpleGraph(outgoing=outgoing, incoming=incoming)
    rng = random.Random(7)

    strategy = BFSStrategy(edge_mode="outgoing")
    strategy.reset_for_new_pack(graph, rng, first_doc_id=0)

    seen = [strategy.propose_next(graph, rng, current_pack_ids=[0])]
    for _ in range(3):
        seen.append(strategy.propose_next(graph, rng, current_pack_ids=seen))

    # For a simple chain, BFS should visit nodes in linear order.
    assert seen[:4] == [0, 1, 2, 3]

    # After exhausting the component, BFS should still return valid ids.
    next_id = strategy.propose_next(graph, rng, current_pack_ids=seen)
    assert 0 <= next_id < len(graph)


def test_dfs_strategy_order_and_restart():
    # Chain 0 -> 1 -> 2 -> 3
    outgoing = {0: [1], 1: [2], 2: [3], 3: []}
    incoming = {0: [], 1: [0], 2: [1], 3: [2]}
    graph = SimpleGraph(outgoing=outgoing, incoming=incoming)
    rng = random.Random(13)

    strategy = DFSStrategy(edge_mode="outgoing")
    strategy.reset_for_new_pack(graph, rng, first_doc_id=0)

    seen = [strategy.propose_next(graph, rng, current_pack_ids=[0])]
    for _ in range(3):
        seen.append(strategy.propose_next(graph, rng, current_pack_ids=seen))

    # For a simple chain, DFS should also walk straight down the chain.
    assert seen[:4] == [0, 1, 2, 3]

    next_id = strategy.propose_next(graph, rng, current_pack_ids=seen)
    assert 0 <= next_id < len(graph)


class _ConstantStrategy:
    """Test helper that always proposes the same id."""

    def __init__(self, constant_id: int) -> None:
        self.constant_id = constant_id

    def reset_for_new_pack(self, graph, rng, first_doc_id) -> None:  # type: ignore[override]
        del graph, rng, first_doc_id

    def propose_next(self, graph, rng, current_pack_ids):  # type: ignore[override]
        del graph, rng, current_pack_ids
        return self.constant_id


def test_composite_traversal_strategy_alternate_mode():
    graph = SimpleGraph(outgoing={0: [], 1: []}, incoming={0: [], 1: []})
    rng = random.Random(99)

    s0: TraversalStrategy = _ConstantStrategy(0)  # type: ignore[assignment]
    s1: TraversalStrategy = _ConstantStrategy(1)  # type: ignore[assignment]

    composite = CompositeTraversalStrategy(
        strategies=[(s0, 1.0), (s1, 1.0)],
        mode="alternate",
    )
    composite.reset_for_new_pack(graph, rng, first_doc_id=0)

    ids = [
        composite.propose_next(graph, rng, current_pack_ids=[]),
        composite.propose_next(graph, rng, current_pack_ids=[]),
        composite.propose_next(graph, rng, current_pack_ids=[]),
        composite.propose_next(graph, rng, current_pack_ids=[]),
    ]
    assert ids == [0, 1, 0, 1]


def test_composite_traversal_strategy_per_step_random_weights():
    graph = SimpleGraph(outgoing={0: [], 1: []}, incoming={0: [], 1: []})
    rng = random.Random(1234)

    s0: TraversalStrategy = _ConstantStrategy(0)  # type: ignore[assignment]
    s1: TraversalStrategy = _ConstantStrategy(1)  # type: ignore[assignment]

    # Weight s1 heavily so that it is virtually always chosen.
    composite = CompositeTraversalStrategy(
        strategies=[(s0, 0.0), (s1, 10.0)],
        mode="per_step_random",
    )
    composite.reset_for_new_pack(graph, rng, first_doc_id=0)

    ids = [composite.propose_next(graph, rng, current_pack_ids=[]) for _ in range(10)]
    assert all(i == 1 for i in ids)


