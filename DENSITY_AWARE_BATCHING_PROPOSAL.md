# Density-Aware Pre-Computed Batch Scheduling

## Problem

In multi-node DDP training, all ranks synchronize gradients via an NCCL all-reduce at the end
of each backward pass. The all-reduce ring cannot proceed until every rank has finished its
backward. Currently, backward time varies 2–6× across ranks within a single step because
different ranks draw different packs from the graph, and pack density (number of resolved
cross-document link grants) determines FlexAttention backward cost. A rank that lands on a
dense Python repo (many mutual imports) can take 6s backward while a rank with a sparse pack
finishes in 2s. The 4s gap is dead time — the fast rank stalls at the barrier.

NCCL hardware itself is fast (560 MB in 6ms at 176 GB/s with IB/GDR). The stall is purely a
rank-synchronization problem driven by unbalanced batch density.

## Proposed Solution: Offline Pre-Computation + Density-Bucketed Scheduling

Instead of generating batches online during training, do a single offline pre-computation pass
over the dataset that:

1. Generates all packs for one epoch using the existing graph traversal infrastructure
2. Computes each pack's mask density (grant count) as a scheduling proxy
3. Mitigates node repeats across the epoch using repeat-aware sampling
4. Sorts packs into many density buckets and arranges the training sequence so that all ranks
   in a given step draw from the same bucket

Training then becomes a simple sequential read from the pre-computed schedule.

---

## Pre-Computation Phase

### Inputs
- `GraphIndex` + `PretokShardedBackend` (same as training)
- `PackBatchSampler` with any supported `strategy` (`dfs`, `bfs`, `random_walk`, `random`)
  — strategy is a first-class parameter; all strategies are equally supported for pre-computation
- Link detector (`PythonImportDetector`, `MarkdownLinkDetector`, etc.) for grant detection
- Config: `token_budget`, `doc_budget`, `max_grants`, `n_workers`, `n_epochs`, `seed`

### Algorithm

The pre-computation runs the full graph traversal to produce all packs for one epoch,
detecting grants for each pack in parallel. **No epoch termination threshold** — we
generate packs until the traversal naturally exhausts, covering 100% of the data.
Sparse packs (near-zero grants) land in the lowest-density bucket and are handled exactly
like dense packs during scheduling. There is no harm in batches that happen to have
nearly doc-causal attention; they are simply fast steps.

Multi-worker parallelism: N workers each generate a disjoint shard of packs using
different seeds, then the main process merges all shards and proceeds to bucketing.
`GraphIndex` is read-only and fork-safe.

### What to store per pack

For each pack we store the minimal data needed to reconstruct the batch and the
attention mask at training time — no token arrays:

- **DocPlacement list**: `(doc_id, effective_len, truncated, doc_trim_side)` per document.
  Together with the layout policy (which is a fixed training config, not per-pack), this
  fully determines the packed token sequence.
- **`link_to_target`**: the matched grant mapping `{link_end_pos: [target_doc_id, ...]}`.
  This is a few dozen integers per pack. At training time, `_build_grant_bitmasks` is
  called directly on this stored mapping — skipping re-running the link detector entirely.
- **`grant_count`**: total number of matched grants (scalar int). Used only for bucketing.

Bitmask tensors are NOT stored — they are O(n_chunks × seq_len) ≈ 512 KB per pack,
which would be ~149 GB for 290k packs. They are cheap to recompute from `link_to_target`
at training time.

### Output Format

Arrow/Parquet files (via `pyarrow`). One pack record per row:

```
packs.parquet
  pack_id:                int32
  doc_ids:                list<int32>
  effective_lens:         list<int32>
  truncated_flags:        list<bool>
  trim_sides:             list<string>      # "head" or "tail" per doc
  link_end_positions:     list<int32>       # keys of link_to_target
  link_target_doc_ids:    list<list<int32>> # values of link_to_target
  grant_count:            int32

schedule.arrow
  global_batch_id:        int32             # which optimizer step
  pack_ids:               list<int32>       # length = global_batch_size_packs, one per slot
```

`schedule.arrow` contains one row per optimizer step. Each row's `pack_ids` list has
`global_batch_size_packs` entries ordered by slot index (slot 0..G-1). At training time,
with `world_size=W` and `grad_accum=A` (where `W × A = global_batch_size_packs`), rank r
at accumulation step k reads slot `k * W + r`.

Tokens are NOT stored — they are fetched at training time via `PretokShardedBackend`
using `doc_id` + `effective_len` + `trim_side`.

---

## Density Bucketing and Step Scheduling

### Global Batch Size

The pre-computation commits to two fixed quantities:
- `local_seq_len`: tokens per pack (= `token_budget`, e.g. 32768)
- `global_batch_size_packs` (G): number of packs per optimizer step (e.g. 16 for 512K tokens/step at 32K local)

These define the "common currency" for all training runs using this schedule. The
actual number of nodes and gradient accumulation steps are free parameters constrained
by `world_size × grad_accum = G`. If you scale up or down nodes, adjust `grad_accum`
accordingly — no recompute needed. Changing `local_seq_len` or `G` requires recompute.

### Bucketing

Buckets are defined by quantiles of `grant_count` so that each bucket contains
approximately equal total packs. The number of buckets is configurable at pre-compute
time. Quantile-based boundaries handle skewed density distributions (many sparse packs,
few very dense) without any bucket becoming nearly empty.

```python
N_BUCKETS = 32   # configurable at pre-compute time
packs.sort(key=lambda p: p.grant_count)
for i, pack in enumerate(packs):
    pack.bucket_id = int(i / len(packs) * N_BUCKETS)
```

Buckets are thus exactly equal in size (±1 pack). The total number of complete global
batches per bucket is `floor(packs_per_bucket / G)`. Any remainder packs (< G) are
dropped (`drop_last` semantics) — this only affects the last few optimizer steps of
each bucket and is a negligible loss of data.

### Ladder Scheduling

Arrange the training sequence so that:
1. Within each step, all `world_size` ranks draw from the **same bucket** (or adjacent
   buckets) — this is the core constraint that eliminates rank stalls
2. Across steps, the bucket sequence follows a **triangle-wave (ladder) pattern**:
   `0, 1, 2, …, N-1, N-2, …, 1, 0, 1, 2, …`

   This gives each density level proportional representation while keeping adjacent steps
   close in difficulty (smooth loss curve, no sudden spikes from jumping bucket 0→31).

```python
def ladder_sequence(n_steps, n_buckets):
    """Generates a triangle-wave index over [0, n_buckets-1]."""
    period = 2 * (n_buckets - 1)
    for step in range(n_steps):
        pos = step % period
        yield pos if pos < n_buckets else period - pos

# Build step→bucket mapping, then assign packs to (step, rank) slots
bucket_queues = {b: deque(packs_in_bucket(b)) for b in range(N_BUCKETS)}
schedule = []
for step, bucket in enumerate(ladder_sequence(n_steps, N_BUCKETS)):
    for rank in range(world_size):
        pack = bucket_queues[bucket].popleft()   # or nearest non-empty bucket
        schedule.append((step, rank, pack.pack_id))
```

When the ladder visits a bucket that has no remaining complete global batches, it falls
back to the nearest non-empty adjacent bucket. This only happens toward the end of an
epoch and affects a small number of steps — the density balance guarantee is degraded
only there.

---

## Training-Time Dataset Interface

The pre-computed schedule replaces `PackBatchSampler` + `PackedSequenceDataset` with
`ScheduledPackDataset`. At training time there is no graph traversal and no link
detection — both happened offline.

```python
class ScheduledPackDataset(IterableDataset):
    def __init__(self, schedule_dir, backend, graph, layout,
                 rank, world_size, grad_accum, start_optimizer_step=0):
        # Validates that world_size * grad_accum == global_batch_size_packs
        ...

    def __iter__(self):
        # For each optimizer step:
        #   for each accum step k in range(grad_accum):
        #     slot = k * world_size + rank
        #     pack = schedule[optimizer_step].pack_ids[slot]
        #     tokens, doc_spans = materialize(pack, backend, graph, layout)
        #     yield {"tokens": tokens, "doc_spans": doc_spans,
        #            "link_to_target": pack.link_to_target}
```

`CrossDocLinkMaskCreator` is extended to accept a pre-computed `link_to_target` dict
from the batch, bypassing link detection and calling `_build_grant_bitmasks` directly.
When `link_to_target` is absent from the batch (online mode), it falls back to running
the link detector as before — but online mode is no longer used in practice.

The rest of the training loop (model, optimizer, DDP) is unchanged. The dataset is now a
deterministic iterator — no random graph traversal or link detection at training time.

---

## Properties and Trade-offs

| Property | Current (online) | Proposed (pre-computed) |
|---|---|---|
| Rank density balance | Uncontrolled | Guaranteed within bucket |
| Expected NCCL stall | 2–4s/step | Near zero (6ms wire time) |
| Pre-computation cost | 0 | ~1h for Stack-100M (parallelised) |
| Training reproducibility | Seeded but dynamic | Fully deterministic |
| Flexibility (mid-run changes) | Trivial | Requires re-pre-computing |
| Node count flexibility | Fixed at launch | Free — adjust grad_accum to compensate |
| Multi-epoch | Re-traverses graph | Pre-compute N separate epoch schedules |

---

## Implementation Notes

1. **Storage size**: Stack-100M at ~3.5M nodes, ~12 docs/pack, ~290k packs/epoch.
   DocPlacement list + link_to_target per pack: ~200 bytes/pack × 290k = ~60 MB uncompressed.
   Parquet with snappy compression will be well under 20 MB. Trivial.
   Token sequences are re-fetched from the existing memmap shards — do not re-store tokens.

2. **Multi-epoch**: Pre-compute N separate epoch schedules, each with a different seed,
   before training starts. Each epoch is a separate `schedule_dir` containing its own
   `packs.parquet` + `schedule.arrow`.

3. **Validation set**: Apply the same pre-computation with a different seed; the existing
   validation split logic carries over unchanged.

4. **Strategy choice**: All strategies (`dfs`, `bfs`, `random_walk`, `random`) are fully
   supported. `random` is only appropriate as a baseline (near-zero link density → cross-doc
   masks never active). Interesting experiment runs use `dfs`, `bfs`, or `random_walk`.
   Strategy is fixed at pre-computation time.

5. **Parallelising pre-computation**: N workers each generate a disjoint shard using
   different seeds, then merge and bucket globally. `GraphIndex` is read-only and fork-safe.

6. **Resume semantics**: The checkpoint saves `optimizer_step`. `ScheduledPackDataset`
   accepts `start_optimizer_step` and seeks to `start_optimizer_step * G` in the schedule.
   The epoch number is `optimizer_step // steps_per_epoch`; the dataset automatically
   switches schedule directories at epoch boundaries.

7. **Integration point**: `main.py` selects between `PackedSequenceDataset` (online) and
   `ScheduledPackDataset` (pre-computed) based on whether `data.schedule_dir` is set in
   config. No changes to model, optimizer, or training loop.

8. **`CrossDocLinkMaskCreator` change**: Extended to accept `link_to_target` as a batch
   key. When present, skips link detection entirely and calls `_build_grant_bitmasks`
   directly. This is the only required change to existing mask-creation code.
