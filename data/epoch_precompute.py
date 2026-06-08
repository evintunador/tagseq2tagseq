"""
Offline epoch pre-computation for density-aware batch scheduling.

Pre-computes all packs for an epoch, assigns kv_block_count density metrics
(proxy for FlexAttention backward cost), and groups packs into density buckets
for load-balanced DDP training.

Supported datasets: TheStack only (repo-partitioned graph with "owner/repo:path"
identifiers).  Wikipedia / SimpleWiki must use the online PackedSequenceDataset.

TODO — Wikipedia / flat-identifier dataset support:
    The TheStack restriction exists because _partition_repos groups documents by
    their "owner/repo" prefix so that all files in a repo land on the same worker.
    This matters because _ShardedEpochView only exposes each worker's own doc IDs;
    BFS traversal stops at the shard boundary, so if linked documents scatter across
    workers the resulting packs are effectively doc_causal (no cross-doc grants fire).

    Wikipedia identifiers have no repo prefix, so _partition_repos degenerates to
    one singleton "repo" per article — linked articles scatter randomly, defeating
    the purpose.

    The fix is a graph-community partitioner: multi-source BFS Voronoi.
    High-level algorithm:
      1. Pick n_workers random seed doc IDs.
      2. Maintain one BFS queue per worker; initialize each with its seed.
      3. Interleave expansion round-robin: dequeue one doc from each worker in
         turn, claim all unclaimed neighbors into that worker's queue.
         Once a worker's territory reaches ~(len(graph) / n_workers * 1.5) docs,
         stop expanding it (size cap prevents hub-heavy nodes from dominating).
         Re-seed any worker that exhausts its queue before the cap from the
         remaining unclaimed docs.
      4. Any unclaimed docs after all queues are empty (isolated nodes, capped
         overflow) are assigned round-robin.
    This produces n_workers connected subgraphs so BFS traversal stays
    intra-shard, yielding the same density-scheduling benefits as TheStack.
    Partitioning is O(n) — same order as the existing _partition_repos scan —
    and runs in the main process before workers start, so it adds negligible
    wall time.
"""

import collections
import json
import logging
import math
import multiprocessing
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from data.collate import build_packed_batch
from data.dataset import GraphIndex, PretokShardedBackend
from data.layout import make_layout_policy
from data.pack_sampler import DocPlacement, PackBatchSampler
from data.traversal import BFSStrategy, DFSStrategy, RandomWalkStrategy, RandomSelectionStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PackRecord
# ---------------------------------------------------------------------------

@dataclass
class PackRecord:
    """Serializable record for a single pre-computed pack."""
    pack_id:             int
    doc_ids:             List[int]
    effective_lens:      List[int]
    truncated_flags:     List[bool]
    trim_sides:          List[str]          # "head" or "tail" per doc
    link_end_positions:  List[int]          # keys of link_to_target
    link_target_doc_ids: List[List[int]]    # values of link_to_target
    kv_block_count:      int = -1           # filled by GPU pass
    bucket_id:           int = -1           # filled by bucketing


def _record_to_placements(record: PackRecord) -> List[DocPlacement]:
    """Reconstruct a DocPlacement list from a PackRecord."""
    return [
        DocPlacement(d, e, t, s)
        for d, e, t, s in zip(
            record.doc_ids,
            record.effective_lens,
            record.truncated_flags,
            record.trim_sides,
        )
    ]


# ---------------------------------------------------------------------------
# TheStack dataset validation
# ---------------------------------------------------------------------------

def _assert_thestack_dataset(graph: GraphIndex) -> None:
    """Raise ValueError if graph is not a TheStack (repo-partitioned) dataset.

    TODO: remove once the graph-community partitioner is implemented (see module
    docstring).  The restriction exists solely because _partition_repos relies on
    the "owner/repo:" prefix; the rest of the pipeline is dataset-agnostic.
    """
    if len(graph) == 0:
        raise ValueError("Graph is empty.")
    first_id = graph.get_normed_identifier(0)
    if ":" not in first_id:
        raise ValueError(
            "Density pre-computation currently requires a TheStack dataset (repo-partitioned "
            "graph with 'owner/repo:path' identifiers). Wikipedia / flat-identifier datasets "
            "need a graph-community partitioner — see the module-level TODO for the design."
        )


def _get_repo_prefix(normed_identifier: str) -> str:
    """Return the 'owner/repo' prefix from a TheStack normed_identifier."""
    return normed_identifier.split(":")[0]


def _partition_repos(
    graph: GraphIndex,
    n_workers: int,
    seed: int,
) -> List[List[int]]:
    """Partition all repos across n_workers shards (round-robin by repo)."""
    repo_to_ids: Dict[str, List[int]] = collections.defaultdict(list)
    for doc_id in range(len(graph)):
        prefix = _get_repo_prefix(graph.get_normed_identifier(doc_id))
        repo_to_ids[prefix].append(doc_id)
    repos = list(repo_to_ids.values())
    random.Random(seed).shuffle(repos)
    shards: List[List[int]] = [[] for _ in range(n_workers)]
    for i, ids in enumerate(repos):
        shards[i % n_workers].extend(ids)
    return shards


# ---------------------------------------------------------------------------
# Shard + epoch view of GraphIndex
# ---------------------------------------------------------------------------

class _ShardedEpochView:
    """Wraps GraphIndex to restrict traversal to one repo shard and exclude
    epoch-visited docs.  Out-of-shard / visited docs appear to have tok_len=0
    (so PackBatchSampler skips them as budgeting rejects zero-length docs) and
    their neighbors are filtered out of BFS/DFS expansions.
    """

    def __init__(
        self,
        graph: GraphIndex,
        shard_doc_ids: Set[int],
        epoch_visited: Set[int],
    ) -> None:
        self._graph = graph
        self._shard = shard_doc_ids
        self._visited = epoch_visited  # live reference — caller updates it

    # ---- GraphIndex interface used by PackBatchSampler / traversal strategies ----

    def __len__(self) -> int:
        return len(self._graph)

    def __contains__(self, normed_identifier: str) -> bool:
        return normed_identifier in self._graph

    def get_token_len(self, doc_id: int) -> int:
        if doc_id not in self._shard or doc_id in self._visited:
            return 0
        return self._graph.get_token_len(doc_id)

    def get_normed_identifier(self, doc_id: int) -> str:
        return self._graph.get_normed_identifier(doc_id)

    def get_raw_identifier(self, normed_identifier: str) -> Optional[str]:
        return self._graph.get_raw_identifier(normed_identifier)

    def get_outgoing_links(self, normed_identifier: str) -> List[str]:
        return self._graph.get_outgoing_links(normed_identifier)

    def get_incoming_links(self, normed_identifier: str) -> List[str]:
        return self._graph.get_incoming_links(normed_identifier)

    def get_id(self, normed_identifier: str) -> int:
        return self._graph.get_id(normed_identifier)

    def get_node(self, normed_identifier: str) -> Optional[Dict[str, Any]]:
        return self._graph.get_node(normed_identifier)

    def neighbors_out(self, doc_id: int) -> List[int]:
        return [
            n for n in self._graph.neighbors_out(doc_id)
            if n in self._shard and n not in self._visited
        ]

    def neighbors_in(self, doc_id: int) -> List[int]:
        return [
            n for n in self._graph.neighbors_in(doc_id)
            if n in self._shard and n not in self._visited
        ]


# ---------------------------------------------------------------------------
# Picklable strategy factories (lambdas are not picklable for spawn)
# ---------------------------------------------------------------------------

class _BFSFactory:
    def __call__(self) -> BFSStrategy:
        return BFSStrategy(edge_mode="outgoing")


class _DFSFactory:
    def __call__(self) -> DFSStrategy:
        return DFSStrategy(edge_mode="outgoing")


class _RandomWalkFactory:
    def __call__(self) -> RandomWalkStrategy:
        return RandomWalkStrategy()


class _RandomFactory:
    def __call__(self) -> RandomSelectionStrategy:
        return RandomSelectionStrategy()


_STRATEGY_FACTORIES = {
    "bfs": _BFSFactory,
    "dfs": _DFSFactory,
    "random_walk": _RandomWalkFactory,
    "random": _RandomFactory,
}


# ---------------------------------------------------------------------------
# Worker config and function (spawned subprocess)
# ---------------------------------------------------------------------------

@dataclass
class _WorkerConfig:
    dataset_dir:      str
    shard_doc_ids:    List[int]
    token_budget:     int
    strategy:         str
    link_detector:    str
    layout_policy:    str
    seed:             int
    order_mode:       str
    worker_idx:       int
    epoch_idx:        int = 0        # propagated to stochastic layout policies
    use_analytical:   bool = True    # compute kv_block_count in worker (CPU, ~1ms/pack)


def _worker_fn(
    config: "_WorkerConfig",
    result_queue: "multiprocessing.Queue",
) -> None:
    """Worker subprocess: generate PackRecords for one repo shard.

    Sends PackRecord objects to result_queue; sends None sentinel when done.
    All file handles are opened fresh inside the worker (no inherited fds).
    """
    import logging
    import tiktoken

    from data.collate import build_packed_batch
    from data.dataset import GraphIndex, PretokShardedBackend
    from data.layout import make_layout_policy
    from data.pack_sampler import PackBatchSampler
    from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
    from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
    from model.graph_traversal.python_import_detector import PythonImportDetector

    logging.basicConfig(level=logging.WARNING)

    graph = GraphIndex(config.dataset_dir)
    backend = PretokShardedBackend(graph)
    enc = tiktoken.get_encoding(graph.metadata.get("tokenizer", "gpt2"))
    layout = make_layout_policy(config.layout_policy, encode_fn=enc.encode_ordinary)
    if hasattr(layout, "set_epoch"):
        layout.set_epoch(config.epoch_idx)

    if config.link_detector == "python":
        detector = PythonImportDetector(decode_fn=enc.decode)
    else:
        detector = MarkdownLinkDetector(decode_fn=enc.decode)

    creator = CrossDocLinkMaskCreator(link_detector=detector)

    strategy_factory = _STRATEGY_FACTORIES[config.strategy]()
    shard_set = set(config.shard_doc_ids)
    epoch_visited: Set[int] = set()
    view = _ShardedEpochView(graph, shard_set, epoch_visited)

    sampler = PackBatchSampler(
        graph=view,
        strategy_factory=strategy_factory,
        token_budget=config.token_budget,
        overflow_policy="truncate",
        doc_level_trim_side="tail",
        pack_level_trim_side="head",
        max_candidates_per_component=1000,
        seed=config.seed,
        order_mode=config.order_mode,
        layout_policy=layout,   # must match the layout used in _materialize at training time
    )

    # pack_id base: unique across workers so records never collide when merged
    pack_id = config.worker_idx * 10_000_000

    for placements in sampler:
        if not placements:
            continue

        batch = build_packed_batch(graph, backend, layout, placements, as_2d=True)
        tokens = batch["tokens"]
        doc_spans = batch["doc_spans"]

        if tokens.shape[-1] != config.token_budget:
            logger.debug(
                "Worker %d: dropping pack with T=%d (expected %d).",
                config.worker_idx, tokens.shape[-1], config.token_budget,
            )
            pack_id += 1
            continue

        for p in placements:
            epoch_visited.add(p.doc_id)

        if hasattr(detector, "detect_links_for_doc"):
            links = creator._collect_links_per_doc(tokens, doc_spans)
        else:
            links = detector.detect_links(tokens[0])
        link_to_target = creator._match_links_to_docs(links, doc_spans)

        link_end_positions = list(link_to_target.keys())
        link_target_doc_ids = [link_to_target[k] for k in link_end_positions]

        kv_bc = -1
        if config.use_analytical:
            from model.graph_traversal.cross_doc_mask import _kv_block_count_analytical
            kv_bc = _kv_block_count_analytical(
                doc_spans, link_to_target, tokens.shape[-1]
            )

        result_queue.put(PackRecord(
            pack_id=pack_id,
            doc_ids=[p.doc_id for p in placements],
            effective_lens=[p.effective_len for p in placements],
            truncated_flags=[p.truncated for p in placements],
            trim_sides=[p.doc_trim_side for p in placements],
            link_end_positions=link_end_positions,
            link_target_doc_ids=link_target_doc_ids,
            kv_block_count=kv_bc,
            bucket_id=-1,
        ))
        pack_id += 1

    result_queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# GPU pass: fill kv_block_count (Method B — BlockMask)
# ---------------------------------------------------------------------------

def _fill_kv_block_counts(
    records: List[PackRecord],
    graph: GraphIndex,
    backend: PretokShardedBackend,
    layout: Any,
    cross_doc_creator: Any,  # CrossDocLinkMaskCreator
    device: torch.device,
) -> None:
    """Fill kv_block_count for all records using BlockMask.kv_num_blocks.sum()."""
    for i, record in enumerate(records):
        if i % 1000 == 0:
            logger.info("kv_block_count GPU pass: %d / %d", i, len(records))
        placements = _record_to_placements(record)
        batch = build_packed_batch(graph, backend, layout, placements, as_2d=True)
        tokens = batch["tokens"].to(device)
        doc_spans = batch["doc_spans"]
        link_to_target = dict(
            zip(record.link_end_positions, record.link_target_doc_ids)
        )
        block_mask = cross_doc_creator(tokens, doc_spans, link_to_target=link_to_target)
        # Total non-empty block pairs = partial blocks + full blocks.
        # FlexAttention stores them separately; backward cost depends on both.
        record.kv_block_count = int(
            (block_mask.kv_num_blocks.sum() + block_mask.full_kv_num_blocks.sum()).item()
        )


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def _assign_buckets(records: List[PackRecord], n_buckets: int) -> None:
    """Assign equal-count quantile buckets by kv_block_count."""
    records.sort(key=lambda r: r.kv_block_count)
    n = len(records)
    for i, r in enumerate(records):
        r.bucket_id = int(i / n * n_buckets)


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def _records_to_table(records: List[PackRecord]) -> pa.Table:
    return pa.table({
        "pack_id":             pa.array([r.pack_id             for r in records], pa.int32()),
        "bucket_id":           pa.array([r.bucket_id           for r in records], pa.int32()),
        "kv_block_count":      pa.array([r.kv_block_count      for r in records], pa.int32()),
        "doc_ids":             pa.array([r.doc_ids             for r in records], pa.list_(pa.int32())),
        "effective_lens":      pa.array([r.effective_lens      for r in records], pa.list_(pa.int32())),
        "truncated_flags":     pa.array([r.truncated_flags     for r in records], pa.list_(pa.bool_())),
        "trim_sides":          pa.array([r.trim_sides          for r in records], pa.list_(pa.string())),
        "link_end_positions":  pa.array([r.link_end_positions  for r in records], pa.list_(pa.int32())),
        "link_target_doc_ids": pa.array(
            [r.link_target_doc_ids for r in records],
            pa.list_(pa.list_(pa.int32())),
        ),
    })


def _table_to_records(table: pa.Table) -> List[PackRecord]:
    cols = table.to_pydict()
    n = len(cols["pack_id"])
    return [
        PackRecord(
            pack_id=int(cols["pack_id"][i]),
            bucket_id=int(cols["bucket_id"][i]),
            kv_block_count=int(cols["kv_block_count"][i]),
            doc_ids=[int(x) for x in cols["doc_ids"][i]],
            effective_lens=[int(x) for x in cols["effective_lens"][i]],
            truncated_flags=[bool(x) for x in cols["truncated_flags"][i]],
            trim_sides=[str(x) for x in cols["trim_sides"][i]],
            link_end_positions=[int(x) for x in cols["link_end_positions"][i]],
            link_target_doc_ids=[[int(y) for y in row]
                                 for row in cols["link_target_doc_ids"][i]],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# EpochPrecomputer
# ---------------------------------------------------------------------------

class EpochPrecomputer:
    """Pre-computes all packs for one epoch and writes packs.parquet + metadata.json.

    Epoch directory layout::

        epoch_dir/
            packs.parquet   # snappy-compressed, sorted by bucket_id then pack_id
            metadata.json   # n_buckets, n_packs, token_budget, strategy, seed, …

    Resume semantics: ``run()`` skips epochs whose directory already contains
    ``packs.parquet`` — re-run with a clean directory to regenerate.
    """

    def __init__(
        self,
        dataset_dir: str,
        token_budget: int,
        n_buckets: int = 32,
        n_workers: int = 8,
        strategy: str = "bfs",
        link_detector: str = "python",
        layout_policy: str = "null",
        max_grants: int = 64,
        order_mode: str = "prefer_targets_first",
        device: Optional[torch.device] = None,
        use_analytical: bool = True,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.token_budget = token_budget
        self.n_buckets = n_buckets
        self.n_workers = n_workers
        self.strategy = strategy
        self.link_detector = link_detector
        self.layout_policy_name = layout_policy
        self.max_grants = max_grants
        self.order_mode = order_mode
        self.use_analytical = use_analytical
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def run(self, epoch_dir: str, epoch_idx: int, seed: int) -> None:
        """Pre-compute one epoch.  No-op if packs.parquet already exists."""
        out_path = Path(epoch_dir)
        parquet_path = out_path / "packs.parquet"
        if parquet_path.exists():
            logger.info("Epoch dir %s already exists, skipping.", epoch_dir)
            return

        out_path.mkdir(parents=True, exist_ok=True)
        logger.info("Pre-computing epoch %d → %s (seed=%d)", epoch_idx, epoch_dir, seed)

        graph = GraphIndex(self.dataset_dir)
        _assert_thestack_dataset(graph)

        # 1. Partition repos across workers
        shards = _partition_repos(graph, self.n_workers, seed)
        logger.info(
            "Partitioned %d docs into %d shards (sizes: %s)",
            len(graph), self.n_workers, [len(s) for s in shards],
        )

        # 2. Spawn workers (spawn context avoids CUDA fork issues)
        ctx = multiprocessing.get_context("spawn")
        result_queue: "multiprocessing.Queue[Optional[PackRecord]]" = ctx.Queue(maxsize=10_000)
        worker_configs = [
            _WorkerConfig(
                dataset_dir=self.dataset_dir,
                shard_doc_ids=shards[i],
                token_budget=self.token_budget,
                strategy=self.strategy,
                link_detector=self.link_detector,
                layout_policy=self.layout_policy_name,
                seed=seed + i,
                order_mode=self.order_mode,
                worker_idx=i,
                epoch_idx=epoch_idx,
                use_analytical=self.use_analytical,
            )
            for i in range(self.n_workers)
        ]
        workers = [
            ctx.Process(target=_worker_fn, args=(cfg, result_queue), daemon=True)
            for cfg in worker_configs
        ]
        for w in workers:
            w.start()

        records: List[PackRecord] = []
        done_count = 0
        while done_count < self.n_workers:
            item = result_queue.get()
            if item is None:
                done_count += 1
            else:
                records.append(item)
        for w in workers:
            w.join()
        logger.info("Workers done. Generated %d packs.", len(records))

        # 3. GPU pass: fill kv_block_count (skip if analytical method used in workers)
        if self.use_analytical:
            logger.info("Skipping GPU pass (kv_block_count already computed analytically).")
        else:
            self._gpu_pass(records, graph, seed)

        # 4. Assign buckets
        _assign_buckets(records, self.n_buckets)

        # 5. Write output (sorted by bucket_id then pack_id)
        records.sort(key=lambda r: (r.bucket_id, r.pack_id))
        table = _records_to_table(records)
        pq.write_table(table, str(parquet_path), compression="snappy")

        metadata = {
            "n_buckets":     self.n_buckets,
            "n_packs":       len(records),
            "token_budget":  self.token_budget,
            "strategy":      self.strategy,
            "seed":          seed,
            "epoch_idx":     epoch_idx,
            "kv_method":     "analytical" if self.use_analytical else "block_mask",
            "link_detector": self.link_detector,
            "layout_policy": self.layout_policy_name,
            "max_grants":    self.max_grants,
        }
        with open(out_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Epoch %d done: %d packs, %d buckets → %s",
            epoch_idx, len(records), self.n_buckets, epoch_dir,
        )

    def _gpu_pass(self, records: List[PackRecord], graph: GraphIndex, seed: int) -> None:
        """GPU pass to fill kv_block_count using BlockMask (Method B)."""
        import tiktoken
        from model.graph_traversal.cross_doc_mask import CrossDocLinkMaskCreator
        from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
        from model.graph_traversal.python_import_detector import PythonImportDetector

        backend = PretokShardedBackend(graph)
        enc = tiktoken.get_encoding(graph.metadata.get("tokenizer", "gpt2"))
        layout = make_layout_policy(self.layout_policy_name, encode_fn=enc.encode_ordinary)

        if self.link_detector == "python":
            detector = PythonImportDetector(decode_fn=enc.decode)
        else:
            detector = MarkdownLinkDetector(decode_fn=enc.decode)

        cross_doc_creator = CrossDocLinkMaskCreator(
            link_detector=detector,
            max_grants=self.max_grants,
        )
        _fill_kv_block_counts(records, graph, backend, layout, cross_doc_creator, self.device)

