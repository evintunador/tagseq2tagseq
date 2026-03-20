# Density-Aware Pre-Computed Batch Scheduling — Implementation Plan

## Overview

This plan covers the full implementation of offline epoch pre-computation and
density-bucketed scheduling. Online training (`PackBatchSampler` + `PackedSequenceDataset`)
is replaced entirely by two new pipeline stages:

1. **Pre-computation** (`precompute_epoch.py`): graph traversal → pack generation →
   grant detection → bucketing → ladder scheduling → Arrow/Parquet files on disk.
2. **Training** (`ScheduledPackDataset`): read schedule from disk → materialize tokens
   → reconstruct bitmasks from stored grants → feed model.

---

## Glossary

| Term | Definition |
|------|-----------|
| `local_seq_len` | Token budget per pack = `token_budget` = e.g. 32768 |
| `G` | `global_batch_size_packs` — packs per optimizer step, fixed at pre-compute time |
| `W` | `world_size` — GPUs in a given training run |
| `A` | `grad_accum` — gradient accumulation steps |
| Constraint | `W × A = G` must hold for any training run using a given schedule |
| `slot` | Index 0..G-1 within a single optimizer step's pack list |
| `global_batch_id` | Monotonically increasing optimizer step index within an epoch |

---

## New Files

```
precompute_epoch.py                  CLI entry point for pre-computation
data/epoch_precompute.py             EpochPrecomputer, PackRecord, worker logic
data/epoch_schedule.py               EpochSchedule loader, ScheduledPackDataset
```

## Modified Files

```
model/graph_traversal/cross_doc_mask.py   Accept pre-computed link_to_target from batch
main.py                                   Wire ScheduledPackDataset when data.schedule_dir set
```

---

## Data Structures

### `PackRecord`

Stored in `packs.parquet`, one row per pack.

```python
@dataclass
class PackRecord:
    pack_id: int
    # Per-document placement (mirrors data.pack_sampler.DocPlacement)
    doc_ids: List[int]           # doc_id for each doc in the pack
    effective_lens: List[int]    # tokens used from each doc (after truncation)
    truncated_flags: List[bool]  # whether each doc was truncated
    trim_sides: List[str]        # "head" or "tail" per doc
    # Grant data (from CrossDocLinkMaskCreator._match_links_to_docs)
    link_end_positions: List[int]         # link_end_pos values (keys of link_to_target)
    link_target_doc_ids: List[List[int]]  # target_doc_ids per link_end_pos (values)
    grant_count: int             # total resolved grants — bucketing key
```

**Why not store bitmask tensors?** At `seq_len=32768`, each pack's bitmasks are ~512 KB.
290k packs × 512 KB = 149 GB. The `link_to_target` mapping (a few dozen ints per pack)
is ~60 MB total uncompressed and allows bitmasks to be reconstructed in microseconds.

### Arrow/Parquet Schema

```
packs.parquet
  pack_id              int32
  doc_ids              list<int32>
  effective_lens       list<int32>
  truncated_flags      list<bool>
  trim_sides           list<string>
  link_end_positions   list<int32>
  link_target_doc_ids  list<list<int32>>   (nested list — pyarrow supports this)
  grant_count          int32

schedule.arrow
  global_batch_id      int32               one row per optimizer step
  pack_ids             list<int32>         length G; slot s = pack_ids[s]
  bucket_id            int32               which density bucket this step drew from

metadata.json
  local_seq_len        int
  global_batch_size_packs (G)   int
  n_buckets            int
  strategy             str                 "dfs" | "bfs" | "random_walk" | "random"
  seed                 int
  epoch_idx            int
  dataset_dir          str
  n_optimizer_steps    int
  n_packs_total        int
```

The `schedule.arrow` + `packs.parquet` together form a **schedule directory** (`schedule_dir`).
One `schedule_dir` per epoch. Multi-epoch training uses `schedule_dirs: [dir0, dir1, ...]`
in config.

---

## Pre-Computation Implementation

### `data/epoch_precompute.py`

#### `EpochPrecomputer`

```python
class EpochPrecomputer:
    def __init__(
        self,
        graph: GraphIndex,
        backend: PretokShardedBackend,
        strategy_factory: Callable[[], TraversalStrategy],
        layout_policy: DocLayoutPolicy,
        link_detector: LinkDetector,
        token_budget: int,
        doc_budget: Optional[int],
        max_grants: int,
        global_batch_size_packs: int,
        n_buckets: int,
        n_workers: int,
        seed: int,
        order_mode: str = "prefer_targets_first",
    ): ...

    def run(self) -> Tuple[List[PackRecord], pd.DataFrame]:
        """
        1. Spawn n_workers processes, each running _worker_generate_packs().
        2. Collect and concatenate PackRecord lists from all workers.
        3. Compute quantile bucket boundaries from grant_count distribution.
        4. Assign bucket_id to each pack.
        5. Build ladder schedule: assign packs to (global_batch_id, slot) pairs.
        6. Return (pack_records, schedule_df).
        """
```

#### Worker function

```python
def _worker_generate_packs(
    worker_id: int,
    n_workers: int,
    graph: GraphIndex,
    backend: PretokShardedBackend,
    strategy_factory: Callable,
    layout_policy: DocLayoutPolicy,
    link_detector: LinkDetector,
    token_budget: int,
    doc_budget: Optional[int],
    max_grants: int,
    seed: int,
    order_mode: str,
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Generates one shard of packs. Each worker uses seed = base_seed + worker_id
    and a PackBatchSampler that exhausts the graph.

    For each pack produced by the sampler:
      1. Call build_packed_batch() to get tokens + doc_spans.
      2. Run link_detector.detect_links(tokens[0]) → links.
      3. Call _match_links_to_docs(links, doc_spans) → link_to_target.
         (Reuses CrossDocLinkMaskCreator._match_links_to_docs as a staticmethod
          or standalone function — does NOT create a BlockMask, only the dict.)
      4. Build PackRecord from placements + link_to_target + grant_count.
      5. Put into result_queue.
    """
```

**Note on worker isolation**: `GraphIndex` and `PretokShardedBackend` are opened inside
each worker process after `fork()` so that memory-mapped file descriptors are not shared.

#### Bucketing

```python
def _assign_buckets(packs: List[PackRecord], n_buckets: int) -> List[PackRecord]:
    """Quantile-based: sort by grant_count, assign bucket_id = int(i/N * n_buckets)."""
    packs_sorted = sorted(packs, key=lambda p: p.grant_count)
    n = len(packs_sorted)
    for i, p in enumerate(packs_sorted):
        p.bucket_id = int(i / n * n_buckets)
    return packs_sorted
```

Quantile bucketing guarantees every bucket has exactly `floor(n/n_buckets)` or
`ceil(n/n_buckets)` packs regardless of density distribution skew.

#### Ladder scheduling

```python
def _build_schedule(
    packs: List[PackRecord],
    n_buckets: int,
    G: int,  # global_batch_size_packs
) -> List[Tuple[int, int, List[int]]]:
    """
    Returns list of (global_batch_id, bucket_id, pack_ids[G]).

    Ladder pattern: 0, 1, 2, ..., N-1, N-2, ..., 1, 0, 1, 2, ...
    Each global batch pulls G packs from the current bucket's queue.
    If current bucket is exhausted, step to nearest non-empty adjacent bucket.
    """
    from collections import deque
    bucket_queues = {
        b: deque(p for p in packs if p.bucket_id == b)
        for b in range(n_buckets)
    }
    schedule = []
    global_batch_id = 0
    period = 2 * (n_buckets - 1)

    while True:
        pos = global_batch_id % period
        target_bucket = pos if pos < n_buckets else period - pos
        # Find nearest non-empty bucket
        bucket = _nearest_nonempty(bucket_queues, target_bucket, n_buckets, G)
        if bucket is None:
            break  # all buckets exhausted
        batch_packs = [bucket_queues[bucket].popleft() for _ in range(G)]
        schedule.append((global_batch_id, bucket, [p.pack_id for p in batch_packs]))
        global_batch_id += 1

    return schedule


def _nearest_nonempty(queues, target, n_buckets, G):
    """Find closest bucket with >= G packs remaining."""
    for delta in range(n_buckets):
        for b in [target - delta, target + delta]:
            if 0 <= b < n_buckets and len(queues[b]) >= G:
                return b
    return None
```

### `precompute_epoch.py` — CLI

```
python precompute_epoch.py \
    --dataset-dir  data/pretokenized_datasets/stack_100m \
    --schedule-dir schedules/stack100m_bfs_epoch0 \
    --strategy     bfs \
    --local-seq-len 32768 \
    --global-batch-size-packs 16 \
    --n-buckets    32 \
    --n-workers    16 \
    --max-grants   64 \
    --epoch-idx    0 \
    --seed         42 \
    --link-detector python \
    --layout-policy null
```

For N epochs, run the script N times with different `--epoch-idx` (which offsets the seed)
and different `--schedule-dir` output paths.

---

## Training-Time Implementation

### `data/epoch_schedule.py`

#### `EpochSchedule`

```python
@dataclass
class EpochSchedule:
    schedule_dir: Path
    metadata: dict                    # from metadata.json
    packs_table: pa.Table             # packs.parquet loaded into memory
    schedule_table: pa.Table          # schedule.arrow loaded into memory

    @classmethod
    def load(cls, schedule_dir: str | Path) -> "EpochSchedule": ...

    def get_pack(self, pack_id: int) -> PackRecord:
        """Row lookup by pack_id. packs_table is indexed by pack_id."""

    def get_global_batch(self, global_batch_id: int) -> List[PackRecord]:
        """Returns the G PackRecords for this optimizer step."""

    @property
    def G(self) -> int:
        return self.metadata["global_batch_size_packs"]

    @property
    def n_optimizer_steps(self) -> int:
        return self.metadata["n_optimizer_steps"]
```

#### `ScheduledPackDataset`

```python
class ScheduledPackDataset(IterableDataset):
    def __init__(
        self,
        schedule_dirs: List[str | Path],   # one per epoch, consumed in order
        backend: PretokShardedBackend,
        graph: GraphIndex,
        layout: DocLayoutPolicy,
        rank: int,
        world_size: int,
        grad_accum: int,
        start_optimizer_step: int = 0,     # for resume — global across all epochs
    ):
        # Validates world_size * grad_accum == G for each schedule
        G = EpochSchedule.load(schedule_dirs[0]).G
        assert world_size * grad_accum == G, (
            f"world_size ({world_size}) × grad_accum ({grad_accum}) = "
            f"{world_size * grad_accum} ≠ G ({G}). "
            "Adjust grad_accum to match the pre-computed global batch size."
        )
        ...

    def __iter__(self) -> Iterator[Dict]:
        global_step = self.start_optimizer_step
        for schedule in self._epoch_schedules_from(global_step):
            epoch_start = self._epoch_start_step(schedule)
            for step_within_epoch in range(
                global_step - epoch_start,
                schedule.n_optimizer_steps
            ):
                batch_packs = schedule.get_global_batch(step_within_epoch)
                for accum_k in range(self.grad_accum):
                    slot = accum_k * self.world_size + self.rank
                    pack = batch_packs[slot]
                    yield self._materialize(pack)
                global_step += 1

    def _materialize(self, pack: PackRecord) -> Dict:
        """
        1. Reconstruct DocPlacement list from pack fields.
        2. Call build_packed_batch(graph, backend, layout, placements) → tokens, doc_spans.
        3. Reconstruct link_to_target dict from pack.link_end_positions + pack.link_target_doc_ids.
        4. Return {"tokens": ..., "doc_spans": ..., "link_to_target": ...}
        """
```

**Key**: `link_to_target` is passed through in the batch dict to the mask creator.

### `CrossDocLinkMaskCreator` modification

In `cross_doc_mask.py`, `__call__` gains a fast path:

```python
def __call__(self, tokens, doc_spans, link_to_target=None, **kwargs):
    device = tokens.device
    seq_len = tokens.shape[-1]

    if link_to_target is None:
        # Online path: run link detection (kept for backwards compatibility / online use)
        links = self.link_detector.detect_links(tokens[0])
        link_to_target = self._match_links_to_docs(links, doc_spans)
    # else: pre-computed path — skip detection entirely

    document_ids = ...  # same as before
    q_bms, kv_bms = self._build_grant_bitmasks(seq_len, doc_spans, link_to_target, device)
    # rest unchanged
```

The batch dict from `ScheduledPackDataset` includes `"link_to_target"`, which flows through
`make_mask_creator_callable_from`'s `**batch` expansion to `CrossDocLinkMaskCreator.__call__`.
No change required to the training module or DDP wrapper.

---

## `main.py` Integration

Add a branch in the data loading section (section 2):

```python
schedule_dirs = cfg.get('data', {}).get('schedule_dirs')  # list of schedule dir paths

if schedule_dirs:
    # Pre-computed mode
    G = EpochSchedule.load(schedule_dirs[0]).G
    accum_steps = cfg.get('train_loop', {}).get('atomic_feature_kwargs', {}).get('accum_steps', 1)
    assert dist.world_size * accum_steps == G, (
        f"world_size ({dist.world_size}) × accum_steps ({accum_steps}) must equal "
        f"global_batch_size_packs ({G}) from the pre-computed schedule."
    )
    dataset = ScheduledPackDataset(
        schedule_dirs=schedule_dirs,
        backend=backend,
        graph=graph_index,
        layout=layout_policy,
        rank=dist.rank,
        world_size=dist.world_size,
        grad_accum=accum_steps,
        start_optimizer_step=resumed_steps,
    )
    train_loader = DataLoader(dataset, batch_size=None, num_workers=0)
    # No LimitedDataLoader needed — ScheduledPackDataset is finite (exhausts schedule)

else:
    # Online mode (kept for local debugging only)
    ...  # existing PackBatchSampler + PackedSequenceDataset code
```

**Config**: add `data.schedule_dirs` as a list of paths. Setting it enables pre-computed mode.

Example config snippet:
```yaml
data:
  dataset_dir: data/pretokenized_datasets/stack_100m
  schedule_dirs:
    - schedules/stack100m_bfs_epoch0
    - schedules/stack100m_bfs_epoch1
    - schedules/stack100m_bfs_epoch2
```

---

## Resume Semantics

The checkpoint already saves `metadata.step` = `optimizer_step`. On resume:

1. `resumed_steps` is read from checkpoint as before (existing logic in `main.py`).
2. `ScheduledPackDataset(start_optimizer_step=resumed_steps)` skips to the right position.
3. The epoch index is inferred from `resumed_steps` and the `n_optimizer_steps` of each
   epoch schedule — `ScheduledPackDataset` iterates epoch schedules in order and skips
   past completed ones.
4. `max_optimizer_steps` adjustment in `main.py` (existing code) still works because the
   dataset is now finite; `LimitedDataLoader` is not needed.

When changing `world_size` during resume:
- Set `grad_accum = G / new_world_size`. If not an integer, choose a different node count
  or recompute the schedule with a different G.
- No other changes needed.

---

## Misc Notes

- **`main.py` argparse `default="random"` for `--strategy`** (line 453): this default is
  vestigial once `ScheduledPackDataset` is the primary path. When online mode is kept for
  debug use, the strategy arg should be changed to `default=None` (no default) to prevent
  accidental random-strategy runs.
- **Validation**: the same pre-computation pipeline can produce a val schedule from a
  separate seed and/or a held-out node subset. For now, the existing stopgap
  (online val sampler with different seed) can be retained until proper held-out splits
  are defined.
- **Parquet row access**: `packs.parquet` will be loaded fully into an Arrow table for
  O(1) row lookup by `pack_id`. At ~60 MB uncompressed, this fits comfortably in memory
  on any training node.
- **Strategy logging**: the schedule's `metadata.json` records the strategy used. The
  training run should log this at startup so the relationship between checkpoint and
  strategy is always auditable.
