"""
BucketedPackDataset — loads pre-computed epoch packs and yields them in
density-bucket order for load-balanced DDP training.

Each rank in a DDP run draws packs from the same density bucket at every
accum step, eliminating FlexAttention backward-cost variance across ranks.

Key design decisions
--------------------
* world_size is NOT baked in at pre-compute time.  At step t rank r draws
  pack at index ``bucket_consumed[B] + r`` from bucket B, then
  ``bucket_consumed[B] += world_size``.  Changing world_size or grad_accum
  on resume only requires reloading ``BucketState`` — no data re-generation.
* num_workers=0 (synchronous): sub-ms data load vs several-second GPU step,
  and avoids stale ``get_state()`` from a subprocess.
* Epoch tail uses drop_last semantics: if fewer than world_size packs remain
  in all non-empty buckets, the epoch ends.
* Per-pack layout: a pack may name the DocLayoutPolicy it was budgeted under
  (``PackRecord.layout_name``); ``_materialize`` resolves each name through a
  cache so one dataset can serve a mixed corpus whose sources use different
  layouts (e.g. arxiv's latex-comment card vs wiki's identifier card). An empty
  name falls back to the ``layout`` passed at construction (single-source epochs).
"""

import collections
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from torch.utils.data import IterableDataset

from data.collate import build_packed_batch
from data.dataset import GraphIndex, PretokShardedBackend
from data.epoch_precompute import PackRecord, _record_to_placements, _table_to_records
from data.layout import DocLayoutPolicy, make_layout_policy

logger = logging.getLogger(__name__)


@dataclass
class BucketState:
    """Serializable dataset position for checkpointing and resume."""
    epoch_idx:         int
    global_accum_step: int               # position in bucket selection sequence
    bucket_consumed:   Dict[int, int]    # total individual packs consumed per bucket


def _make_bucket_sequence(n_buckets: int, seed: int, n_repeats: int = 1000) -> List[int]:
    """Repeating shuffled sequence over n_buckets.

    Over any n_repeats×n_buckets consecutive accum steps each bucket appears
    exactly n_repeats times.  Adjacent steps see different buckets (no runs
    of the same bucket).
    """
    rng = random.Random(seed)
    buckets = list(range(n_buckets))
    seq: List[int] = []
    for _ in range(n_repeats):
        rng.shuffle(buckets)
        seq.extend(list(buckets))
    return seq


class BucketedPackDataset(IterableDataset):
    """Iterable dataset yielding pre-computed packs in density-bucket order.

    Args:
        epoch_dirs:   Ordered list of epoch directories (each with packs.parquet).
        graph:        GraphIndex for reconstructing packed batches.
        backend:      PretokShardedBackend for token data.
        layout:       DocLayoutPolicy for prefix/suffix decoration.
        rank:         This rank's index in the DDP group (0-based).
        world_size:   Total number of ranks.
        start_state:  Optional BucketState to resume from a checkpoint.
    """

    def __init__(
        self,
        epoch_dirs: List[str],
        graph: GraphIndex,
        backend: PretokShardedBackend,
        layout: DocLayoutPolicy,
        rank: int,
        world_size: int,
        start_state: Optional[BucketState] = None,
        encode_fn=None,
    ) -> None:
        super().__init__()
        self.epoch_dirs = epoch_dirs
        self.graph = graph
        self.backend = backend
        self.layout = layout
        self.rank = rank
        self.world_size = world_size

        # Per-pack layout support (mixed-source corpora): packs may carry a
        # layout_name naming the DocLayoutPolicy they were budgeted under.
        # _materialize resolves each name to a policy through this cache so one
        # dataset can serve packs decorated with different layouts (e.g. arxiv's
        # latex_comment card vs wiki's identifier card).  encode_fn is needed to
        # build prefix layouts on demand; if not passed, derive it from the
        # default layout (all prefix policies stash their tokeniser as ._encode).
        self._encode_fn = encode_fn or getattr(layout, "_encode", None)
        self._layout_cache: Dict[str, DocLayoutPolicy] = {"": layout}

        # State tracks position across calls to __iter__
        if start_state is not None:
            self._epoch_idx = start_state.epoch_idx
            self._global_accum_step = start_state.global_accum_step
            self._bucket_consumed = dict(start_state.bucket_consumed)
        else:
            self._epoch_idx = 0
            self._global_accum_step = 0
            self._bucket_consumed = {}

        # Signal initial epoch to every cached stochastic layout policy.
        self._set_epoch_all(self._epoch_idx)

    def _set_epoch_all(self, epoch_idx: int) -> None:
        """Fan the epoch counter out to every cached layout policy.

        The stochastic prefix layouts flip a per-(doc, epoch) coin, so each
        cached policy — not just the default — must see the current epoch or
        its coin would freeze at epoch 0 while others advance.
        """
        for pol in self._layout_cache.values():
            if hasattr(pol, "set_epoch"):
                pol.set_epoch(epoch_idx)

    def _get_layout(self, layout_name: str) -> DocLayoutPolicy:
        """Resolve a pack's layout_name to a policy, building+caching on first use.

        Empty name → the default layout (single-source epochs). A newly built
        policy is immediately advanced to the current epoch so its stochastic
        coin is in sync with the rest.
        """
        pol = self._layout_cache.get(layout_name)
        if pol is None:
            pol = make_layout_policy(layout_name, encode_fn=self._encode_fn)
            if hasattr(pol, "set_epoch"):
                pol.set_epoch(self._epoch_idx)
            self._layout_cache[layout_name] = pol
        return pol

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while self._epoch_idx < len(self.epoch_dirs):
            epoch_dir = self.epoch_dirs[self._epoch_idx]
            parquet_path = os.path.join(epoch_dir, "packs.parquet")
            meta_path = os.path.join(epoch_dir, "metadata.json")

            if not os.path.exists(parquet_path):
                raise FileNotFoundError(f"packs.parquet not found in {epoch_dir}")

            with open(meta_path) as f:
                meta = json.load(f)
            n_buckets: int = meta["n_buckets"]
            self._token_budget: Optional[int] = meta.get("token_budget")

            # Warn if max_grants warmup is active (bucketing is approximate during warmup)
            max_grants_start = meta.get("max_grants_start")
            max_grants = meta.get("max_grants", 64)
            max_grants_warmup = meta.get("max_grants_warmup_steps", 0)
            if max_grants_start is not None and max_grants_start < max_grants:
                logger.warning(
                    "max_grants warmup active: kv_block_count bucketing reflects final "
                    "max_grants (%d); density balance is approximate during warmup "
                    "(%d→%d over %d steps).",
                    max_grants, max_grants_start, max_grants, max_grants_warmup,
                )

            # Load and group packs by bucket
            import pyarrow.parquet as pq
            table = pq.read_table(parquet_path)
            all_records = _table_to_records(table)
            bucket_lists: Dict[int, List[PackRecord]] = collections.defaultdict(list)
            for r in all_records:
                bucket_lists[r.bucket_id].append(r)
            for b in bucket_lists:
                bucket_lists[b].sort(key=lambda r: r.pack_id)

            bucket_seq = _make_bucket_sequence(n_buckets, seed=self._epoch_idx)

            while True:
                chosen = bucket_seq[self._global_accum_step % len(bucket_seq)]

                # Fallback: scan outward by |b - chosen| to find a non-empty bucket
                actual_bucket: Optional[int] = None
                for cand in sorted(range(n_buckets), key=lambda b: abs(b - chosen)):
                    consumed = self._bucket_consumed.get(cand, 0)
                    available = len(bucket_lists.get(cand, []))
                    if available - consumed >= self.world_size:
                        actual_bucket = cand
                        break

                if actual_bucket is None:
                    break  # epoch exhausted (drop_last)

                consumed = self._bucket_consumed.get(actual_bucket, 0)
                pack_idx = consumed + self.rank
                pack = bucket_lists[actual_bucket][pack_idx]

                # Update state BEFORE yield so get_state() always reflects
                # the position after the most recent item.
                self._bucket_consumed[actual_bucket] = consumed + self.world_size
                self._global_accum_step += 1

                yield self._materialize(pack)

            # Epoch exhausted — advance to next
            self._epoch_idx += 1
            self._set_epoch_all(self._epoch_idx)
            self._global_accum_step = 0
            self._bucket_consumed = {}

        raise RuntimeError(
            f"All pre-computed epoch dirs exhausted after {len(self.epoch_dirs)} epochs. "
            "Re-run precompute_epochs.py to generate more."
        )

    def get_state(self) -> BucketState:
        """Return the current dataset position for checkpointing."""
        return BucketState(
            epoch_idx=self._epoch_idx,
            global_accum_step=self._global_accum_step,
            bucket_consumed=dict(self._bucket_consumed),
        )

    def set_state(self, state: BucketState) -> None:
        """Restore the schedule position captured by get_state().

        Used to make a throw-away read (e.g. a pre-training compile warmup that
        fetches a batch before the real loop) side-effect-free: snapshot with
        get_state(), consume the batch, then set_state() back so the warmup pack
        is re-yielded by the real training loop and the per-epoch dedup accounting
        is unperturbed.
        """
        self._epoch_idx = state.epoch_idx
        self._global_accum_step = state.global_accum_step
        self._bucket_consumed = dict(state.bucket_consumed)

    def _materialize(self, pack: PackRecord) -> Dict[str, Any]:
        """Reconstruct a full batch dict from a PackRecord."""
        placements = _record_to_placements(pack)
        layout = self._get_layout(getattr(pack, "layout_name", ""))
        batch = build_packed_batch(self.graph, self.backend, layout, placements)
        T = batch["tokens"].shape[-1]
        budget = getattr(self, '_token_budget', None)
        if budget is not None and T != budget:
            import logging as _logging
            _logging.getLogger(__name__).error(
                "Materialized pack has T=%d, expected token_budget=%d (layout=%r). "
                "Re-run precompute_epochs.py to regenerate packs at the correct length.",
                T, budget, getattr(pack, "layout_name", ""),
            )
            raise AssertionError(
                f"Materialized pack has T={T}, expected token_budget={budget} "
                f"(layout={getattr(pack, 'layout_name', '')!r}). "
                "Re-run precompute_epochs.py to regenerate packs at the correct length."
            )
        batch["link_to_target"] = dict(
            zip(pack.link_end_positions, pack.link_target_doc_ids)
        )
        return batch
