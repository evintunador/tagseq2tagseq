import logging
import random
from collections import deque
from typing import Deque, List, Optional, Protocol, Set, Tuple

from .dataset import GraphIndex


logger = logging.getLogger(__name__)


class TraversalStrategy(Protocol):
    """
    Interface for graph traversal strategies used by higher-level pack samplers.

    A traversal strategy defines *how* to move around the graph when growing a
    single pack, but it does **not** enforce token budgets, maximum pack size,
    or deduplication. The underlying graph may contain cycles and is not
    assumed to be acyclic.

    The typical calling pattern from a pack sampler is:

    1. Choose a starting document id for a new pack.
    2. Call ``reset_for_new_pack`` exactly once to initialize the strategy.
    3. Call ``propose_next`` zero or more times to obtain candidate doc ids.

    The sampler is responsible for rejecting duplicates, enforcing budgets, and
    deciding when to stop growing the pack.
    """

    def reset_for_new_pack(
        self,
        graph: GraphIndex,
        rng: random.Random,
        first_doc_id: int,
    ) -> None:
        """
        Prepare the strategy to grow a brand new pack starting from ``first_doc_id``.

        This method is called exactly once per pack, before any calls to
        ``propose_next`` for that pack.

        Args:
            graph: The immutable graph index used for all traversal decisions.
            rng: A ``random.Random`` instance to use for any stochastic choices.
            first_doc_id: The document id chosen by the sampler as the first
                element in the new pack.
        """

    def propose_next(
        self,
        graph: GraphIndex,
        rng: random.Random,
        current_pack_ids: List[int],
    ) -> int:
        """
        Propose the next document id to add to the current pack.

        This method may be called many times per pack. The strategy may use
        ``current_pack_ids`` to inform its behavior (for example, treating the
        last id as the current position in a random walk), but it is **not**
        required to avoid proposing ids that already appear in
        ``current_pack_ids``.

        The pack sampler will handle:
        - Deduplication within a pack.
        - Budget enforcement (tokens, maximum pack length, etc.).
        - Any retry or fallback logic if the proposed id is rejected.

        Args:
            graph: The immutable graph index used for all traversal decisions.
            rng: A ``random.Random`` instance to use for any stochastic choices.
            current_pack_ids: The sequence of document ids already accepted
                into the current pack, in order.

        Returns:
            The id of the next document the strategy suggests adding.
        """


class RandomSelectionStrategy:
    """
    A simple traversal strategy that proposes uniformly random document ids.

    This strategy **ignores the graph structure entirely**: each call to
    ``propose_next`` samples a document id uniformly from ``[0, len(graph))``
    using the provided RNG. It is therefore useful as a baseline or for
    debugging higher-level components.

    Notes:
        - ``reset_for_new_pack`` is called once per pack and records the
          current graph size.
        - ``propose_next`` may be called many times per pack and is allowed to
          propose ids that already appear in ``current_pack_ids``; the pack
          sampler is responsible for deduplication and budget enforcement.
    """

    def __init__(self) -> None:
        self._num_nodes: Optional[int] = None

    def reset_for_new_pack(
        self,
        graph: GraphIndex,
        rng: random.Random,  # noqa: ARG002 - rng reserved for consistency with interface
        first_doc_id: int,  # noqa: ARG002 - not used by this strategy
    ) -> None:
        """
        Record the number of nodes in the graph for subsequent random draws.

        Args:
            graph: The graph index that will be traversed.
            rng: Unused; included for interface compatibility.
            first_doc_id: Unused; this strategy does not treat the starting
                document specially.
        """
        del rng, first_doc_id  # make it explicit that these are intentionally unused
        self._num_nodes = len(graph)

    def propose_next(
        self,
        graph: GraphIndex,
        rng: random.Random,
        current_pack_ids: List[int],  # noqa: ARG002 - not used by this strategy
    ) -> int:
        """
        Propose a uniformly random document id from the entire graph.

        Args:
            graph: The graph index that will be traversed.
            rng: Source of randomness for the selection.
            current_pack_ids: Unused; this strategy ignores pack history.

        Returns:
            A random integer in ``[0, len(graph))``.
        """
        del current_pack_ids

        # ``reset_for_new_pack`` should have been called before the first
        # proposal, but we defensively handle the case where it was not.
        num_nodes = self._num_nodes if self._num_nodes is not None else len(graph)
        if num_nodes == 0:
            raise RuntimeError("RandomSelectionStrategy cannot operate on an empty graph.")

        return rng.randrange(num_nodes)


class RandomWalkStrategy:
    """
    A traversal strategy that performs a (potentially restarting) random walk
    over the graph.

    The walk maintains an internal ``_current_doc_id`` and, on each step,
    either:

    - Teleports to a random node with probability ``restart_prob``, or
    - Follows one of the neighbors of the current node according to
      ``edge_mode`` and, when applicable, the ``(w_in, w_out)`` weights.

    Notes:
        - The underlying graph may contain cycles; this strategy does not track
          visited nodes and can revisit the same document many times.
        - The pack sampler is responsible for deduplication, budgets, and for
          deciding when to stop growing a pack.
    """

    def __init__(
        self,
        edge_mode: str = "outgoing",
        w_in: float = 0.5,
        w_out: float = 0.5,
        restart_prob: float = 0.0,
    ) -> None:
        """
        Args:
            edge_mode: Which edges to follow. One of ``{"incoming", "outgoing", "both"}``.
            w_in: When ``edge_mode == "both"``, relative weight for following
                incoming edges.
            w_out: When ``edge_mode == "both"``, relative weight for following
                outgoing edges.
            restart_prob: Probability in ``[0, 1]`` of teleporting to a random
                node instead of following neighbors at each step.
        """
        if edge_mode not in {"incoming", "outgoing", "both"}:
            raise ValueError(f"Invalid edge_mode={edge_mode!r}. Must be 'incoming', 'outgoing', or 'both'.")
        if not (0.0 <= restart_prob <= 1.0):
            raise ValueError(f"restart_prob must be in [0, 1], got {restart_prob}.")

        self.edge_mode = edge_mode
        self.w_in = float(w_in)
        self.w_out = float(w_out)
        self.restart_prob = float(restart_prob)

        self._current_doc_id: Optional[int] = None

    def reset_for_new_pack(
        self,
        graph: GraphIndex,  # noqa: ARG002 - kept for interface symmetry; not used directly here
        rng: random.Random,  # noqa: ARG002
        first_doc_id: int,
    ) -> None:
        """
        Initialize the walk to start from ``first_doc_id`` for a new pack.

        Args:
            graph: Unused; the walk only needs the starting id here.
            rng: Unused in the reset step; randomness is used in ``propose_next``.
            first_doc_id: The document id chosen by the sampler as the first
                element in the new pack.
        """
        del graph, rng
        self._current_doc_id = first_doc_id

    def _teleport(self, graph: GraphIndex, rng: random.Random) -> int:
        num_nodes = len(graph)
        if num_nodes == 0:
            raise RuntimeError("RandomWalkStrategy cannot operate on an empty graph.")
        self._current_doc_id = rng.randrange(num_nodes)
        return self._current_doc_id

    def _choose_neighbors(self, graph: GraphIndex) -> List[int]:
        assert self._current_doc_id is not None

        if self.edge_mode == "incoming":
            return graph.neighbors_in(self._current_doc_id)
        if self.edge_mode == "outgoing":
            return graph.neighbors_out(self._current_doc_id)

        # edge_mode == "both"
        incoming = graph.neighbors_in(self._current_doc_id)
        outgoing = graph.neighbors_out(self._current_doc_id)

        # If both directions are empty, caller will handle the empty case.
        if not incoming and not outgoing:
            return []

        # If one side is empty, just use the other.
        if not incoming:
            return outgoing
        if not outgoing:
            return incoming

        # Both sides non-empty: defer to caller to choose direction using weights.
        # This helper just returns a concatenation; direction choice uses weights
        # in the main step logic.
        return incoming + outgoing

    def propose_next(
        self,
        graph: GraphIndex,
        rng: random.Random,
        current_pack_ids: List[int],  # noqa: ARG002 - used only as fallback for initial position
    ) -> int:
        """
        Propose the next document id along the random walk.

        The walk can:
            - Teleport to a random node with probability ``restart_prob``, or
            - Follow a randomly chosen neighbor of the current node, using
              incoming/outgoing edges as configured.
        """
        # Initialize current node if needed (e.g., if reset was not called).
        if self._current_doc_id is None:
            if current_pack_ids:
                self._current_doc_id = current_pack_ids[-1]
            else:
                return self._teleport(graph, rng)

        # Restart (teleport) step.
        if rng.random() < self.restart_prob:
            return self._teleport(graph, rng)

        # Edge-based transition.
        if self.edge_mode in {"incoming", "outgoing"}:
            neighbors = self._choose_neighbors(graph)
        else:
            # edge_mode == "both"
            incoming = graph.neighbors_in(self._current_doc_id)
            outgoing = graph.neighbors_out(self._current_doc_id)

            if not incoming and not outgoing:
                neighbors = []
            else:
                # Decide whether to use incoming or outgoing based on weights.
                total_weight = max(self.w_in + self.w_out, 0.0)
                # If weights are non-positive, fall back to unweighted choice.
                if total_weight <= 0.0:
                    neighbors = incoming + outgoing
                else:
                    r = rng.random() * total_weight
                    use_incoming = r < self.w_in
                    if use_incoming and incoming:
                        neighbors = incoming
                    elif (not use_incoming) and outgoing:
                        neighbors = outgoing
                    elif incoming:
                        neighbors = incoming
                    else:
                        neighbors = outgoing

        if neighbors:
            next_doc_id = rng.choice(neighbors)
            self._current_doc_id = next_doc_id
            return next_doc_id

        # No neighbors available: fall back to teleporting to a random node.
        return self._teleport(graph, rng)


class BFSStrategy:
    """
    A breadth-first traversal strategy over the graph.

    For each pack, the strategy maintains a queue-based frontier and a
    per-pack visited set. Nodes are proposed in BFS order relative to the
    starting ``first_doc_id``, with optional restarts into unexplored
    components when the frontier is exhausted.
    """

    def __init__(self, edge_mode: str = "outgoing") -> None:
        """
        Args:
            edge_mode: Which edges to follow when expanding the frontier.
                One of ``{"incoming", "outgoing", "both"}``.
        """
        if edge_mode not in {"incoming", "outgoing", "both"}:
            raise ValueError(
                f"Invalid edge_mode={edge_mode!r}. Must be 'incoming', 'outgoing', or 'both'."
            )
        self.edge_mode = edge_mode
        self._queue: Deque[int] = deque()
        self._visited: Set[int] = set()

    def reset_for_new_pack(
        self,
        graph: GraphIndex,  # noqa: ARG002 - graph is not needed at reset time
        rng: random.Random,  # noqa: ARG002
        first_doc_id: int,
    ) -> None:
        """
        Initialize the BFS frontier for a new pack starting at ``first_doc_id``.

        Args:
            graph: Unused at reset; the topology is accessed during expansion.
            rng: Unused at reset; randomness is used when restarting.
            first_doc_id: The document id chosen as the starting point.
        """
        del graph, rng
        self._queue.clear()
        self._visited.clear()
        self._visited.add(first_doc_id)
        self._queue.append(first_doc_id)

    def _expand_frontier(self, graph: GraphIndex, center_id: int) -> None:
        if self.edge_mode in {"incoming", "both"}:
            for nid in graph.neighbors_in(center_id):
                if nid not in self._visited:
                    self._visited.add(nid)
                    self._queue.append(nid)
        if self.edge_mode in {"outgoing", "both"}:
            for nid in graph.neighbors_out(center_id):
                if nid not in self._visited:
                    self._visited.add(nid)
                    self._queue.append(nid)

    def _restart_if_needed(self, graph: GraphIndex, rng: random.Random) -> None:
        """If the queue is empty, pick a new unseen node (if possible) to seed it."""
        if self._queue:
            return

        num_nodes = len(graph)
        if num_nodes == 0:
            raise RuntimeError("BFSStrategy cannot operate on an empty graph.")

        # Attempt to find an unseen node. We bound the number of retries so that,
        # in the worst case where nearly all nodes are visited, we eventually give
        # up rather than looping indefinitely.
        max_tries = min(num_nodes * 2, 1000)
        for _ in range(max_tries):
            candidate = rng.randrange(num_nodes)
            if candidate not in self._visited:
                self._visited.add(candidate)
                self._queue.append(candidate)
                return

        # If we reach here, we've likely visited almost all nodes; pick any node
        # to keep the process moving, even if it has been seen before.
        fallback = rng.randrange(num_nodes)
        if fallback not in self._visited:
            self._visited.add(fallback)
        self._queue.append(fallback)

    def propose_next(
        self,
        graph: GraphIndex,
        rng: random.Random,
        current_pack_ids: List[int],  # noqa: ARG002 - BFS relies on its own frontier
    ) -> int:
        """
        Propose the next document id in BFS order.

        The method:
            1. Ensures the frontier is non-empty, restarting into a (preferably
               unseen) node if needed.
            2. Pops the next id from the queue.
            3. Expands its neighbors into the frontier (subject to ``edge_mode``).
            4. Returns the popped id.
        """
        del current_pack_ids

        # Ensure we have something in the frontier.
        self._restart_if_needed(graph, rng)

        center_id = self._queue.popleft()
        self._expand_frontier(graph, center_id)
        return center_id


class DFSStrategy:
    """
    A depth-first traversal strategy over the graph.

    For each pack, the strategy maintains a stack-based frontier and a
    per-pack visited set. Nodes are proposed in DFS order relative to the
    starting ``first_doc_id``, with restarts into unexplored components when
    the stack is exhausted.
    """

    def __init__(self, edge_mode: str = "outgoing") -> None:
        """
        Args:
            edge_mode: Which edges to follow when expanding the frontier.
                One of ``{"incoming", "outgoing", "both"}``.
        """
        if edge_mode not in {"incoming", "outgoing", "both"}:
            raise ValueError(
                f"Invalid edge_mode={edge_mode!r}. Must be 'incoming', 'outgoing', or 'both'."
            )
        self.edge_mode = edge_mode
        self._stack: List[int] = []
        self._visited: Set[int] = set()

    def reset_for_new_pack(
        self,
        graph: GraphIndex,  # noqa: ARG002 - graph is not needed at reset time
        rng: random.Random,  # noqa: ARG002
        first_doc_id: int,
    ) -> None:
        """
        Initialize the DFS frontier for a new pack starting at ``first_doc_id``.

        Args:
            graph: Unused at reset; the topology is accessed during expansion.
            rng: Unused at reset; randomness is used when restarting and ordering neighbors.
            first_doc_id: The document id chosen as the starting point.
        """
        del graph, rng
        self._stack.clear()
        self._visited.clear()
        self._visited.add(first_doc_id)
        self._stack.append(first_doc_id)

    def _expand_frontier(self, graph: GraphIndex, rng: random.Random, center_id: int) -> None:
        # Collect neighbors according to edge_mode.
        neighbors: List[int] = []
        if self.edge_mode in {"incoming", "both"}:
            neighbors.extend(graph.neighbors_in(center_id))
        if self.edge_mode in {"outgoing", "both"}:
            neighbors.extend(graph.neighbors_out(center_id))

        # Randomize neighbor order for a more canonical DFS feel.
        rng.shuffle(neighbors)

        for nid in neighbors:
            if nid not in self._visited:
                self._visited.add(nid)
                self._stack.append(nid)

    def _restart_if_needed(self, graph: GraphIndex, rng: random.Random) -> None:
        """If the stack is empty, pick a new unseen node (if possible) to seed it."""
        if self._stack:
            return

        num_nodes = len(graph)
        if num_nodes == 0:
            raise RuntimeError("DFSStrategy cannot operate on an empty graph.")

        max_tries = min(num_nodes * 2, 1000)
        for _ in range(max_tries):
            candidate = rng.randrange(num_nodes)
            if candidate not in self._visited:
                self._visited.add(candidate)
                self._stack.append(candidate)
                return

        # Fallback: if everything looks visited, push any node to keep moving.
        fallback = rng.randrange(num_nodes)
        if fallback not in self._visited:
            self._visited.add(fallback)
        self._stack.append(fallback)

    def propose_next(
        self,
        graph: GraphIndex,
        rng: random.Random,
        current_pack_ids: List[int],  # noqa: ARG002 - DFS relies on its own frontier
    ) -> int:
        """
        Propose the next document id in DFS order.

        The method:
            1. Ensures the frontier (stack) is non-empty, restarting into a
               (preferably unseen) node if needed.
            2. Pops the next id from the top of the stack.
            3. Expands its neighbors into the stack (subject to ``edge_mode``),
               pushing newly discovered nodes.
            4. Returns the popped id.
        """
        del current_pack_ids

        self._restart_if_needed(graph, rng)

        center_id = self._stack.pop()
        self._expand_frontier(graph, rng, center_id)
        return center_id


class CompositeTraversalStrategy:
    """
    A traversal strategy that mixes multiple child strategies within a pack.

    Each pack begins by resetting all child strategies to the same starting
    document id. Subsequent calls to ``propose_next`` delegate to one child at
    a time according to a configurable mixing rule (mode).

    Supported modes:
        - ``"per_step_random"``: on each call to ``propose_next``, select a
          child strategy according to the provided weights and delegate.
        - ``"alternate"``: cycle deterministically through the list of child
          strategies in round-robin order, ignoring weights.
    """

    def __init__(
        self,
        strategies: List[Tuple[TraversalStrategy, float]],
        mode: str = "per_step_random",
    ) -> None:
        """
        Args:
            strategies: A list of ``(strategy_instance, weight)`` pairs. The
                weights are used only in ``"per_step_random"`` mode.
            mode: The mixing rule to use. One of ``{"per_step_random", "alternate"}``.
        """
        if not strategies:
            raise ValueError("CompositeTraversalStrategy requires at least one child strategy.")
        if mode not in {"per_step_random", "alternate"}:
            raise ValueError(
                f"Invalid mode={mode!r}. Must be 'per_step_random' or 'alternate'."
            )

        self._strategies: List[Tuple[TraversalStrategy, float]] = strategies
        self.mode = mode

        # Precompute weight-related helpers for per-step random selection.
        self._weights = [max(float(w), 0.0) for _, w in self._strategies]
        self._weight_sum = sum(self._weights)

        self._alt_index: int = 0

    def reset_for_new_pack(
        self,
        graph: GraphIndex,
        rng: random.Random,
        first_doc_id: int,
    ) -> None:
        """
        Reset all child strategies for the new pack starting at ``first_doc_id``.

        Each child receives the same starting document id so that their internal
        states are aligned at the beginning of the pack.
        """
        for strategy, _ in self._strategies:
            strategy.reset_for_new_pack(graph, rng, first_doc_id)

        self._alt_index = 0

    def _choose_index_per_step_random(self, rng: random.Random) -> int:
        # If all weights are non-positive, fall back to uniform selection.
        if self._weight_sum <= 0.0:
            return rng.randrange(len(self._strategies))

        r = rng.random() * self._weight_sum
        cumulative = 0.0
        for idx, weight in enumerate(self._weights):
            cumulative += weight
            if r <= cumulative:
                return idx
        # Numerical edge case: return last index.
        return len(self._strategies) - 1

    def propose_next(
        self,
        graph: GraphIndex,
        rng: random.Random,
        current_pack_ids: List[int],
    ) -> int:
        """
        Delegate to one of the child strategies according to the configured mode.

        The chosen child strategy receives the same arguments and its proposed
        id is returned verbatim. The composite strategy does not perform any
        additional filtering or book-keeping beyond child selection.
        """
        if self.mode == "per_step_random":
            idx = self._choose_index_per_step_random(rng)
        else:  # "alternate"
            idx = self._alt_index
            self._alt_index = (self._alt_index + 1) % len(self._strategies)

        strategy, _ = self._strategies[idx]
        return strategy.propose_next(graph, rng, current_pack_ids)


