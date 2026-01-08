from typing import Any, Dict, Iterator

from torch.utils.data import IterableDataset

from .collate import build_packed_batch
from .dataset import GraphIndex, PretokShardedBackend
from .layout import DocLayoutPolicy
from .pack_sampler import PackBatchSampler


class PackedSequenceDataset(IterableDataset):
    """
    Iterable dataset that streams fully materialised packed batches.

    Each iteration yields a dictionary suitable as model input, containing:

        - ``tokens``: torch.LongTensor of shape [1, T] (or [T] if as_2d=False)
        - ``doc_spans``: List[DocSpan] describing per-doc spans in the sequence
        - ``doc_ids``: List[int] of document ids in order
        - ``titles``: List[str] of document titles in order

    The underlying ``PackBatchSampler`` is responsible for graph traversal and
    budgeting; this dataset only performs token materialisation and collation.
    """

    def __init__(
        self,
        graph: GraphIndex,
        backend: PretokShardedBackend,
        pack_sampler: PackBatchSampler,
        layout_policy: DocLayoutPolicy,
        as_2d: bool = True,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.backend = backend
        self.pack_sampler = pack_sampler
        self.layout = layout_policy
        self.as_2d = as_2d

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # In an IterableDataset, this method may be invoked in worker processes
        # when used with a DataLoader and num_workers > 0.
        for placements in self.pack_sampler:
            if not placements:
                continue

            batch = build_packed_batch(
                graph=self.graph,
                backend=self.backend,
                layout=self.layout,
                placements=placements,
                as_2d=self.as_2d,
            )
            yield batch



