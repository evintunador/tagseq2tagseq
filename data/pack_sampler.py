import logging
import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Set, Dict

from experiments.dagseq2dagseq.data.dataset import GraphIndex
from experiments.dagseq2dagseq.data.traversal import TraversalStrategy


logger = logging.getLogger(__name__)


@dataclass
class DocPlacement:
    """
    Description of how a single document contributes tokens to a pack.

    This structure is produced by ``PackBatchSampler`` and later consumed by
    collate functions to slice pre-tokenized sequences. It captures
    per-document truncation (due to per-doc budgets) as well as pack-level
    truncation decisions.

    Attributes:
        doc_id: Integer id of the document within the ``GraphIndex``.
        effective_len: Number of tokens from this document that will appear in
            the final packed sequence (after any per-doc or pack-level
            truncation).
        truncated: Whether this document is truncated relative to its full
            pre-tokenized length. This flag becomes ``True`` if either a
            per-document budget or the final pack-level adjustment reduces the
            number of tokens used from this document.
        doc_trim_side: Which side of the document should be trimmed when
            applying truncation at the token level, either ``"head"`` or
            ``"tail"``. For causal language modeling we typically trim from the
            ``"tail"``, but this is recorded explicitly so that collate logic
            can slice tokens correctly.
    """

    doc_id: int
    effective_len: int
    truncated: bool
    doc_trim_side: str


class PackBatchSampler:
    """
    Iterable that builds token-limited packs of document ids using a
    configurable graph traversal strategy over the document graph.

    Each iteration of this sampler yields a list of ``DocPlacement`` objects
    describing a single pack. Higher-level dataset / collate code can convert
    these placements into token tensors by:

    1. Looking up each document's token sequence from the pre-tokenized
       backend.
    2. Slicing according to ``effective_len`` and ``doc_trim_side``.
    3. Concatenating slices to form a single packed sequence of length
       approximately equal to ``token_budget``.

    Notes:
        - No document id appears more than once within a single pack.
        - Multiple disconnected components are allowed within each pack.
        - The sampler may temporarily overshoot the global token budget while
          assembling a pack; a final truncation pass on the *first* document in
          the ordered pack enforces the budget exactly (up to optionally
          dropping that document if it would be reduced to zero length).
    """

    def __init__(
        self,
        graph: GraphIndex,
        strategy_factory: Callable[[], TraversalStrategy],
        token_budget: int,
        doc_budget: Optional[int] = None,
        overflow_policy: str = "truncate",
        doc_level_trim_side: str = "tail",
        pack_level_trim_side: str = "head",
        max_candidates_per_component: int = 1000,
        seed: int = 0,
        order_mode: str = "as_traversed",
    ) -> None:
        """
        Args:
            graph: The ``GraphIndex`` defining the universe of documents and
                their token lengths and edges.
            strategy_factory: Callable that returns a fresh ``TraversalStrategy``
                instance for each new graph walk / subgraph started in a pack.
            token_budget: Global maximum number of tokens allowed in a single
                pack (typically ``batch_size * seq_len``).
            doc_budget: Optional maximum number of tokens drawn from any single
                document. ``None`` means "use the full document length".
            overflow_policy: Behavior when ``full_len > doc_budget``:
                - ``"skip"``: the document is never included in any pack.
                - ``"truncate"``: the document may be included, but only up to
                  ``doc_budget`` tokens, marked as truncated.
            doc_level_trim_side: Side from which to trim tokens within each
                document when truncation is applied, either ``"head"`` or
                ``"tail"``. For causal LM workloads this will typically be
                ``"tail"``.
            pack_level_trim_side: Side of the *document list* from which to
                remove tokens when the final pack-level truncation is applied
                to hit ``token_budget`` exactly. ``"head"`` means trim starting
                from the earliest document in the pack; ``"tail"`` means trim
                starting from the last document in the pack.
            max_candidates_per_component: Safety bound on how many candidate
                doc ids are requested from a traversal strategy for a single
                graph walk (subgraph) before it is considered exhausted.
            seed: Base integer seed for the sampler's internal RNG.
            order_mode: Strategy for ordering documents within a pack after all
                components have been collected. Currently supported:
                - ``"as_traversed"``: keep insertion order.
                - ``"prefer_targets_first"``: heuristic that prefers documents
                  that are linked-to to appear earlier than their linkers.
        """
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}.")
        if doc_budget is not None and doc_budget <= 0:
            raise ValueError(f"doc_budget must be positive when set, got {doc_budget}.")
        if overflow_policy not in {"truncate", "skip"}:
            raise ValueError(
                f"overflow_policy must be 'truncate' or 'skip', got {overflow_policy!r}."
            )
        if doc_level_trim_side not in {"head", "tail"}:
            raise ValueError(
                f"doc_level_trim_side must be 'head' or 'tail', got {doc_level_trim_side!r}."
            )
        if pack_level_trim_side not in {"head", "tail"}:
            raise ValueError(
                f"pack_level_trim_side must be 'head' or 'tail', got {pack_level_trim_side!r}."
            )
        if max_candidates_per_component <= 0:
            raise ValueError(
                f"max_candidates_per_component must be positive, got {max_candidates_per_component}."
            )
        if order_mode not in {"as_traversed", "prefer_targets_first"}:
            raise ValueError(
                f"order_mode must be 'as_traversed' or 'prefer_targets_first', got {order_mode!r}."
            )

        self.graph = graph
        self.strategy_factory = strategy_factory
        self.token_budget = token_budget
        self.doc_budget = doc_budget
        self.overflow_policy = overflow_policy
        self.doc_level_trim_side = doc_level_trim_side
        self.pack_level_trim_side = pack_level_trim_side
        self.max_candidates_per_component = max_candidates_per_component
        self.order_mode = order_mode

        self._rng = random.Random(seed)

    def __iter__(self) -> Iterable[List[DocPlacement]]:
        """
        Yield one pack at a time as a list of ``DocPlacement`` objects.

        High-level steps per pack:
            1. Seed one or more graph walks (subgraphs) with starting documents.
            2. Grow each subgraph using a traversal strategy, enforcing
               per-document budgets and "no duplicates within a pack".
            3. Reorder documents according to ``order_mode``.
            4. Apply a final pack-level truncation, potentially trimming from
               multiple documents at the pack ``head`` or ``tail`` to hit
               ``token_budget`` exactly.

        The iterator continues yielding packs until it is no longer possible to
        construct any non-empty pack under the current constraints, at which
        point iteration terminates.
        """
        if len(self.graph) == 0:
            return

        while True:
            placements, total_tokens = self._build_single_pack()
            if not placements:
                # No more valid packs can be constructed.
                return

            # Apply ordering and final pack-level truncation.
            ordered = self._order_placements(placements)
            final = self._apply_pack_truncation(ordered, total_tokens)
            if not final:
                # If truncation dropped all documents, stop iteration.
                return

            yield final

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _compute_budgeted_length(self, full_len: int) -> Optional[tuple[int, bool]]:
        """
        Apply per-document budget and overflow policy to a raw token length.

        Returns:
            (effective_len, truncated) if the document is usable, otherwise
            ``None`` if it should be skipped entirely.
        """
        if full_len <= 0:
            return None
        if self.doc_budget is None:
            return full_len, False
        if full_len <= self.doc_budget:
            return full_len, False
        if self.overflow_policy == "skip":
            return None
        # overflow_policy == "truncate"
        return self.doc_budget, True

    def _build_single_pack(self) -> tuple[List[DocPlacement], int]:
        """
        Construct a single pack worth of placements (unordered).

        Returns:
            placements: List of accepted ``DocPlacement`` objects in insertion
                order, before any global ordering or pack-level truncation.
            total_tokens: Sum of ``effective_len`` over the returned placements.
        """
        placements: List[DocPlacement] = []
        pack_doc_ids: Set[int] = set()
        current_total_tokens = 0

        num_nodes = len(self.graph)
        if num_nodes == 0:
            return placements, current_total_tokens

        # Outer loop: start new subgraphs while we still have budget left and
        # can find seeds.
        while current_total_tokens < self.token_budget:
            updated_total = self._seed_and_grow_subgraph(
                placements, pack_doc_ids, current_total_tokens
            )
            # _seed_and_grow_subgraph mutates placements and pack_doc_ids and
            # returns the new total token count.
            current_total_tokens = updated_total

            if current_total_tokens < 0:
                # Sentinel for "no suitable seed found"; stop building this pack.
                break

            if current_total_tokens >= self.token_budget:
                break

        # Filter out any placements that ended up with non-positive length
        # (should not generally happen, but is safe).
        placements = [p for p in placements if p.effective_len > 0]
        current_total_tokens = sum(p.effective_len for p in placements)
        return placements, current_total_tokens

    def _seed_and_grow_subgraph(
        self,
        placements: List[DocPlacement],
        pack_doc_ids: Set[int],
        current_total_tokens: int,
    ) -> int:
        """
        Seed a new graph subwalk and grow it using a traversal strategy.

        Args:
            placements: Current list of placements (mutated in-place).
            pack_doc_ids: Set of doc ids already included in this pack.
            current_total_tokens: Running total of tokens in the pack so far.

        Returns:
            Updated ``current_total_tokens`` if a subgraph was added, or
            ``-1`` as a sentinel if no valid seed could be found (in which
            case the caller should stop building this pack).
        """
        num_nodes = len(self.graph)
        if num_nodes == 0:
            return -1

        max_seed_tries = min(num_nodes * 2, 1000)
        seed_added = False
        first_doc_id: Optional[int] = None
        per_doc_len = 0
        truncated = False

        for _ in range(max_seed_tries):
            candidate = self._rng.randrange(num_nodes)
            if candidate in pack_doc_ids:
                continue

            full_len = self.graph.get_token_len(candidate)
            budgeted = self._compute_budgeted_length(full_len)
            if budgeted is None:
                continue

            per_doc_len, truncated = budgeted
            if per_doc_len <= 0:
                continue

            # Accept seed.
            first_doc_id = candidate
            placements.append(
                DocPlacement(
                    doc_id=first_doc_id,
                    effective_len=per_doc_len,
                    truncated=truncated,
                    doc_trim_side=self.doc_level_trim_side,
                )
            )
            pack_doc_ids.add(first_doc_id)
            current_total_tokens += per_doc_len
            seed_added = True
            break

        if not seed_added or first_doc_id is None:
            # No valid seed found.
            return -1

        # Grow this subgraph using the configured traversal strategy.
        strategy = self.strategy_factory()
        strategy.reset_for_new_pack(self.graph, self._rng, first_doc_id)

        # Track the ids that belong to this subgraph/component. The sampler
        # still enforces *pack*-level de-duplication via ``pack_doc_ids``, but
        # the traversal strategy only sees the local history for this component.
        component_doc_ids: List[int] = [first_doc_id]
        attempts_without_accept = 0

        while (
            attempts_without_accept < self.max_candidates_per_component
            and current_total_tokens < self.token_budget
        ):
            candidate = strategy.propose_next(self.graph, self._rng, component_doc_ids)

            if candidate in pack_doc_ids:
                attempts_without_accept += 1
                continue

            full_len = self.graph.get_token_len(candidate)
            budgeted = self._compute_budgeted_length(full_len)
            if budgeted is None:
                attempts_without_accept += 1
                continue

            per_doc_len, truncated = budgeted
            if per_doc_len <= 0:
                attempts_without_accept += 1
                continue

            # Accept candidate.
            placements.append(
                DocPlacement(
                    doc_id=candidate,
                    effective_len=per_doc_len,
                    truncated=truncated,
                    doc_trim_side=self.doc_level_trim_side,
                )
            )
            pack_doc_ids.add(candidate)
            component_doc_ids.append(candidate)
            current_total_tokens += per_doc_len

            attempts_without_accept = 0

            # We allow overshoot here; the final truncation step will enforce
            # the exact token budget.
            if current_total_tokens >= self.token_budget:
                break

        return current_total_tokens

    def _order_placements(self, placements: List[DocPlacement]) -> List[DocPlacement]:
        """
        Order placements according to ``order_mode``.

        ``"as_traversed"`` keeps them in insertion order. ``"prefer_targets_first"``
        performs a heuristic Kahn-style topological sort on the induced
        subgraph where edges are added from targets to linkers so that, as much
        as possible, documents that are linked-to appear before documents that
        link to them. Cycles are broken by falling back to insertion order for
        any remaining nodes.
        """
        if self.order_mode == "as_traversed" or len(placements) <= 1:
            return placements

        # Build induced adjacency: if u -> v in the original graph and both are
        # in this pack, we add an edge v -> u so that v is preferred before u.
        doc_ids = [p.doc_id for p in placements]
        doc_set = set(doc_ids)
        insertion_index: Dict[int, int] = {doc_id: i for i, doc_id in enumerate(doc_ids)}

        adjacency: Dict[int, List[int]] = {doc_id: [] for doc_id in doc_ids}
        indegree: Dict[int, int] = {doc_id: 0 for doc_id in doc_ids}

        for u in doc_ids:
            for v in self.graph.neighbors_out(u):
                if v in doc_set:
                    # Edge from v (target) to u (linker).
                    adjacency[v].append(u)
                    indegree[u] += 1

        # Kahn's algorithm with stable tie-breaking based on insertion order.
        ready = [d for d in doc_ids if indegree[d] == 0]
        ready.sort(key=lambda d: insertion_index[d])

        ordered_ids: List[int] = []
        while ready:
            current = ready.pop(0)
            ordered_ids.append(current)
            for nbr in adjacency[current]:
                indegree[nbr] -= 1
                if indegree[nbr] == 0:
                    ready.append(nbr)
            ready.sort(key=lambda d: insertion_index[d])

        if len(ordered_ids) < len(doc_ids):
            # Remaining nodes are part of cycles; append them in insertion order.
            remaining = [d for d in doc_ids if d not in ordered_ids]
            ordered_ids.extend(remaining)

        placement_by_id: Dict[int, DocPlacement] = {p.doc_id: p for p in placements}
        return [placement_by_id[doc_id] for doc_id in ordered_ids]

    def _apply_pack_truncation(
        self,
        placements: List[DocPlacement],
        total_tokens: int,
    ) -> List[DocPlacement]:
        """
        Apply final pack-level truncation so that the sum of ``effective_len``
        across all placements does not exceed ``token_budget``.

        Depending on ``pack_level_trim_side``, truncation walks from the head
        (earliest documents) or tail (latest documents) of the ordered pack,
        decreasing ``effective_len`` and marking documents as truncated. If a
        document's ``effective_len`` reaches zero, it is dropped from the pack.
        """
        if not placements:
            return placements

        if total_tokens <= self.token_budget:
            return placements

        overshoot = total_tokens - self.token_budget

        if self.pack_level_trim_side == "head":
            indices = list(range(len(placements)))
        else:  # "tail"
            indices = list(reversed(range(len(placements))))

        for idx in indices:
            if overshoot <= 0:
                break
            p = placements[idx]
            if p.effective_len <= 0:
                continue

            trim_amount = min(p.effective_len, overshoot)
            if trim_amount <= 0:
                continue

            p.effective_len -= trim_amount
            overshoot -= trim_amount
            p.truncated = True

        # Drop any documents that have been reduced to zero length.
        final_placements = [p for p in placements if p.effective_len > 0]
        return final_placements



