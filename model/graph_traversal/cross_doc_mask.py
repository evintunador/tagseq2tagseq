"""
CrossDocLinkMaskCreator — FlexAttention mask for cross-document link attention.

Combines a doc_causal base mask with cross-document attention grants derived from
in-text links detected by a pluggable LinkDetector.

See also:
    link_detector.py         — LinkInfo, LinkDetector protocol
    markdown_link_detector.py — MarkdownLinkDetector (Wikipedia / Markdown)
    python_import_detector.py — PythonImportDetector (Python / TheStack)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .link_detector import LinkDetector, LinkInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BlockInteractionMask — precomputed block-level interaction table
# ---------------------------------------------------------------------------

@dataclass
class BlockInteractionMask:
    """Block-level interaction table for Triton cross-doc attention kernels.

    Precomputed once per batch from the grant bitmasks.  Reused across all
    heads and all transformer layers — the mask is the same everywhere, so
    the O((T/bs)²) Python precomputation is amortised over every layer.

    The table encodes, for each block pair (Q-block, KV-block), whether any
    position in Q-block can attend to any position in KV-block (accounting
    for causality, same-doc identity, and cross-doc grant bitmasks).

    Two CSR (Compressed Sparse Row) lists:
      q_kv_*  — for each Q-block, which KV-blocks to process
                (used by the forward pass + backward STAGE 2 dLdQ)
      kv_q_*  — for each KV-block, which Q-blocks to process
                (used by backward STAGE 1 dLdK / dLdV)

    Analogous to FlexAttention's BlockMask but without the B/H dimensions
    (the cross-doc mask is identical for every head and every layer).
    """

    seq_len:      int
    block_size:   int

    # Forward + backward-STAGE-2: for each Q-block, sorted KV-block indices
    q_kv_counts:  torch.Tensor   # [n_blocks]   int32
    q_kv_ptrs:    torch.Tensor   # [n_blocks+1] int32  (exclusive prefix sums)
    q_kv_indices: torch.Tensor   # [nnz_fwd]    int32

    # Backward STAGE 1: for each KV-block, sorted Q-block indices
    kv_q_counts:  torch.Tensor   # [n_blocks]   int32
    kv_q_ptrs:    torch.Tensor   # [n_blocks+1] int32
    kv_q_indices: torch.Tensor   # [nnz_bwd]    int32

    @property
    def n_blocks(self) -> int:
        return int(self.q_kv_counts.shape[0])

    @property
    def sparsity(self) -> float:
        """Fraction of causal block pairs that are entirely empty (skipped)."""
        n = self.n_blocks
        total = n * (n + 1) // 2
        nnz = int(self.q_kv_counts.sum().item())
        return 1.0 - nnz / max(total, 1)


# ---------------------------------------------------------------------------
# kv_block_count helpers (Methods A, B, C for Step 1 / test_kv_block_count)
# ---------------------------------------------------------------------------

def _kv_block_count_from_dense(mask: torch.Tensor, block_size: int = 128) -> int:
    """Method A: count non-empty (q_block, kv_block) pairs in a full bool mask [T, T].

    Equivalent to (block_mask.kv_num_blocks + block_mask.full_kv_num_blocks).sum()
    for the corresponding BlockMask (partial + full blocks combined).
    """
    T = mask.shape[0]
    n = math.ceil(T / block_size)
    count = 0
    for qi in range(n):
        for ki in range(n):
            q_sl = slice(qi * block_size, min((qi + 1) * block_size, T))
            k_sl = slice(ki * block_size, min((ki + 1) * block_size, T))
            if mask[q_sl, k_sl].any():
                count += 1
    return count


def _kv_block_count_analytical(
    doc_spans: List[Any],
    link_to_target: Dict[int, List[int]],
    seq_len: int,
    block_size: int = 128,
) -> int:
    """Method C: set-based analytical count of non-empty (q_block, kv_block) pairs.

    Exact when every target doc ends before the corresponding link_end_pos
    (which holds for standard packing order).  Counts both causal+same_doc
    blocks and cross-doc grant blocks without double-counting.
    """
    non_empty: Set[Tuple[int, int]] = set()

    # causal + same_doc: each doc contributes a lower-triangular block region
    for span in doc_spans:
        if span.end <= span.start:
            continue
        first_blk = span.start // block_size
        last_blk = (span.end - 1) // block_size
        for q_blk in range(first_blk, last_blk + 1):
            for kv_blk in range(first_blk, q_blk + 1):
                non_empty.add((q_blk, kv_blk))

    # cross-doc grants
    span_by_doc_id = {s.doc_id: s for s in doc_spans}
    for link_pos, target_doc_ids in link_to_target.items():
        q_span = None
        for span in doc_spans:
            if span.start < link_pos <= span.end:
                q_span = span
                break
        if q_span is None:
            continue
        grant_end = min(seq_len, q_span.end)
        if link_pos >= grant_end:
            continue

        q_first_blk = link_pos // block_size
        q_last_blk = (grant_end - 1) // block_size

        for target_doc_id in target_doc_ids:
            kv_span = span_by_doc_id.get(target_doc_id)
            if kv_span is None:
                continue
            target_start = max(0, kv_span.start)
            target_end = min(seq_len, kv_span.end)
            if target_start >= target_end:
                continue
            kv_first_blk = target_start // block_size
            kv_last_blk = (target_end - 1) // block_size
            for q_blk in range(q_first_blk, q_last_blk + 1):
                for kv_blk in range(kv_first_blk, kv_last_blk + 1):
                    non_empty.add((q_blk, kv_blk))

    return len(non_empty)


class CrossDocLinkMaskCreator:
    """
    Creates attention masks that grant cross-document attention based on in-text links.

    This is a callable class that can be passed to TS2TSTrainingModule as the
    block_mask_creator parameter.

    Args:
        link_detector: A LinkDetector implementation appropriate for the dataset
                       (e.g. MarkdownLinkDetector for Wikipedia).

    Example:
        >>> import tiktoken
        >>> enc = tiktoken.get_encoding('gpt2')
        >>> from model.graph_traversal.markdown_link_detector import MarkdownLinkDetector
        >>> detector = MarkdownLinkDetector(decode_fn=enc.decode)
        >>> mask_creator = CrossDocLinkMaskCreator(link_detector=detector)
        >>> model = TS2TSTrainingModule(
        ...     block_mask_creator=mask_creator,
        ...     vocab_size=50257,
        ...     ...
        ... )
    """

    def __init__(
        self,
        link_detector: LinkDetector,
        max_grants: int = 64,
        max_grants_start: Optional[int] = None,
        max_grants_warmup_steps: int = 0,
        backend: str = "flex",
        triton_block_size: int = 64,
    ):
        """
        Args:
            link_detector:          Dataset-specific link extractor.
            max_grants:             Maximum cross-doc grants per batch.
            max_grants_start:       Cosine-warmup starting value (None = no warmup).
            max_grants_warmup_steps: Steps over which to ramp max_grants.
            backend:                ``"flex"``   — return FlexAttention ``BlockMask``
                                                   (default, used during training
                                                   with torch.compile + DDP).
                                    ``"triton"`` — return ``BlockInteractionMask``
                                                   (precomputed block index lists for
                                                   our custom Triton kernels; avoids
                                                   per-block OR-reductions at runtime).
            triton_block_size:      Block granularity for ``BlockInteractionMask``.
                                    Should match the BLOCK_SIZE used by the Triton
                                    kernel (typically the autotune winner, default 64).
        """
        assert backend in ("flex", "triton"), \
            f"backend must be 'flex' or 'triton', got {backend!r}"
        self.link_detector = link_detector
        self.max_grants = max_grants
        self.backend = backend
        self.triton_block_size = triton_block_size
        # Schedule: cosine ascent from max_grants_start → max_grants over
        # max_grants_warmup_steps forward passes.  Disabled when start is None
        # or warmup_steps is 0 (max_grants is used from step 0).
        self._max_grants_start = max_grants_start if max_grants_start is not None else max_grants
        self._max_grants_warmup_steps = max_grants_warmup_steps
        self._step = 0
        # _n_chunks is always sized for the *final* max_grants so the triton
        # kernel shape is stable throughout training.  Early in warmup the
        # extra chunks are all-zero and contribute no grants.
        self._n_chunks = max(1, (max_grants + 63) // 64)
        logger.info(
            f"Initialized CrossDocLinkMaskCreator with detector: "
            f"{type(link_detector).__name__}, max_grants={max_grants} "
            f"({self._n_chunks} int64 chunk(s)), "
            f"warmup: {self._max_grants_start}→{max_grants} over "
            f"{max_grants_warmup_steps} steps, "
            f"backend={backend}"
            + (f" (block_size={triton_block_size})" if backend == "triton" else "")
        )

    def _current_max_grants(self) -> int:
        """Cosine ascent schedule: ramps from max_grants_start to max_grants."""
        if self._max_grants_warmup_steps <= 0 or self._step >= self._max_grants_warmup_steps:
            return self.max_grants
        frac = self._step / self._max_grants_warmup_steps
        cosine_factor = (1.0 - math.cos(math.pi * frac)) / 2.0
        return round(
            self._max_grants_start
            + (self.max_grants - self._max_grants_start) * cosine_factor
        )

    def _match_links_to_docs(
        self,
        links: List[LinkInfo],
        doc_spans: List[Any],
    ) -> Dict[int, List[int]]:
        """
        Match detected links to documents in the batch.

        Uses ``self.link_detector.index_doc_span(span)`` to build the lookup
        key for each span.  This lets dataset-specific detectors (e.g.
        ``PythonImportDetector``) match on a sub-component of ``raw_identifier``
        (e.g. the bare file path) rather than the full identifier.

        Multiple ``LinkInfo`` objects can share the same ``link_end_pos`` when
        a single import generates several candidate file paths; all matches are
        collected in a list so every valid target receives cross-doc attention.

        Args:
            links:     List of LinkInfo objects from the detector.
            doc_spans: List of DocSpan objects for the current batch.

        Returns:
            Mapping ``link_end_pos -> [target_doc_id, ...]`` for valid links.
            Only includes links where the target document appears earlier in
            the batch (DAG property).
        """
        # Build detector-key -> (doc_id, start_pos) mapping
        index_to_doc: Dict[str, tuple] = {}
        for span in doc_spans:
            key = self.link_detector.index_doc_span(span)
            index_to_doc[key] = (span.doc_id, span.start)

        link_to_target: Dict[int, List[int]] = {}
        matched = 0

        for link in links:
            if link.target_str not in index_to_doc:
                logger.debug(
                    f"Link target '{link.target_str}' not found in batch, skipping"
                )
                continue

            target_doc_id, target_start_pos = index_to_doc[link.target_str]

            # Enforce DAG property: target must start before the link position
            if target_start_pos >= link.link_end_pos:
                logger.debug(
                    f"Link at {link.link_end_pos} to '{link.target_str}' violates DAG "
                    f"(target starts at {target_start_pos}), skipping"
                )
                continue

            link_to_target.setdefault(link.link_end_pos, []).append(target_doc_id)
            matched += 1
            logger.debug(
                f"Matched link at {link.link_end_pos} -> "
                f"doc {target_doc_id} ('{link.target_str}')"
            )

        logger.info(
            f"Matched {matched}/{len(links)} links to documents in batch"
        )
        return link_to_target

    def _build_cross_doc_mask(
        self,
        seq_len: int,
        doc_spans: List[Any],
        link_to_target: Dict[int, List[int]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a 2D cross-document attention mask for link-based attention.

        Returns:
            cross_doc_mask: Tensor of shape [seq_len, seq_len] where
                            cross_doc_mask[q, kv] = True if position q can attend
                            to position kv via a link (not including same-doc attention).
        """
        cross_doc_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

        for link_pos, target_doc_ids in sorted(link_to_target.items()):
            for target_doc_id in target_doc_ids:
                # Find the source document containing this link
                link_doc_span = None
                for span in doc_spans:
                    if span.start < link_pos <= span.end:
                        link_doc_span = span
                        break

                # link_end_pos is exclusive (one past the closing token), so the
                # correct containment check is span.start < link_pos <= span.end.
                if link_doc_span is None:
                    logger.warning(f"Link at position {link_pos} not in any document span")
                    continue

                # Find the target document span
                target_doc_span = None
                for span in doc_spans:
                    if span.doc_id == target_doc_id:
                        target_doc_span = span
                        break

                if target_doc_span is None:
                    logger.warning(f"Target doc {target_doc_id} not found in doc_spans")
                    continue

                # Grant access: positions from link_pos onward (within source doc)
                # can attend to all positions in the target doc
                grant_start = link_pos
                grant_end = min(seq_len, link_doc_span.end)
                target_start = max(0, target_doc_span.start)
                target_end = min(seq_len, target_doc_span.end)

                if grant_start < grant_end and target_start < target_end:
                    cross_doc_mask[grant_start:grant_end, target_start:target_end] = True
                    logger.debug(
                        f"Link at {link_pos}: positions [{grant_start}, {grant_end}) "
                        f"can attend to doc {target_doc_id} at [{target_start}, {target_end})"
                    )

        return cross_doc_mask

    def _build_grant_bitmasks(
        self,
        seq_len: int,
        doc_spans: List[Any],
        link_to_target: Dict[int, List[int]],
        device: torch.device,
        max_grants_cap: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Build lists of int64 bitmask tensors encoding cross-doc grants.

        Returns ``(q_bitmasks, kv_bitmasks)``, each a list of ``n_chunks``
        tensors of shape [seq_len], where ``n_chunks = ceil(max_grants / 64)``.

        Grant k uses chunk ``k // 64``, bit position ``k % 64``.  Bit 63 of
        each chunk is stored as INT64_MIN (the signed representation of 2^63),
        which has the same bit pattern and works correctly with ``!= 0``.

        The mask_mod ORs across all chunks::

            in_grant = any((q_bitmasks[c][q_idx] & kv_bitmasks[c][kv_idx]) != 0
                           for c in range(n_chunks))

        Fully pointwise — no reductions over the sequence dimension.

        Memory: 2 × n_chunks × seq_len × 8 bytes.
        At seq_len=32768: 512 KB (n_chunks=1) to 2 MB (n_chunks=4),
        vs the original O(seq_len²) dense bool mask (1 GB).
        """
        cap = max_grants_cap if max_grants_cap is not None else self.max_grants
        q_bitmasks = [torch.zeros(seq_len, dtype=torch.int64, device=device)
                      for _ in range(self._n_chunks)]
        kv_bitmasks = [torch.zeros(seq_len, dtype=torch.int64, device=device)
                       for _ in range(self._n_chunks)]

        grant_idx = 0
        truncated = False
        for link_pos, target_doc_ids in sorted(link_to_target.items()):
            if truncated:
                break
            for target_doc_id in target_doc_ids:
                if grant_idx >= cap:
                    logger.warning(
                        f"Batch has >{cap} grants; ignoring remainder. "
                        f"(max_grants={self.max_grants}, cap={cap})"
                    )
                    truncated = True
                    break

                link_doc_span = None
                for span in doc_spans:
                    if span.start < link_pos <= span.end:
                        link_doc_span = span
                        break
                if link_doc_span is None:
                    logger.warning(f"Link at position {link_pos} not in any document span")
                    continue

                target_doc_span = None
                for span in doc_spans:
                    if span.doc_id == target_doc_id:
                        target_doc_span = span
                        break
                if target_doc_span is None:
                    logger.warning(f"Target doc {target_doc_id} not found in doc_spans")
                    continue

                grant_start = link_pos
                grant_end = min(seq_len, link_doc_span.end)
                target_start = max(0, target_doc_span.start)
                target_end = min(seq_len, target_doc_span.end)

                if grant_start < grant_end and target_start < target_end:
                    chunk = grant_idx // 64
                    bit_pos = grant_idx % 64
                    # bit_pos 63 would be 2^63, which overflows signed int64;
                    # use INT64_MIN (same bit pattern, valid as signed int64).
                    bit = (1 << bit_pos) if bit_pos < 63 else -(1 << 63)
                    q_bitmasks[chunk][grant_start:grant_end] |= bit
                    kv_bitmasks[chunk][target_start:target_end] |= bit
                    logger.debug(
                        f"Grant {grant_idx} (chunk {chunk}, bit {bit_pos}): "
                        f"q[{grant_start},{grant_end}) → kv[{target_start},{target_end})"
                    )
                    grant_idx += 1

        logger.info(
            f"Built {grant_idx} cross-doc attention grants "
            f"across {self._n_chunks} chunk(s)"
        )
        return q_bitmasks, kv_bitmasks

    def _build_block_interaction_mask(
        self,
        seq_len: int,
        document_ids: torch.Tensor,
        q_bms: List[torch.Tensor],
        kv_bms: List[torch.Tensor],
        device: torch.device,
    ) -> "BlockInteractionMask":
        """Precompute block-level interaction table for the Triton backend.

        Runs entirely on CPU with simple Python loops — fast because it executes
        once per batch (not per layer/head).  At T=32768 / block_size=64 / 64
        grants: ~2 ms on a modern CPU.

        For each (Q-block, KV-block) pair that satisfies causality, records
        whether any position pair can interact via same-doc or cross-doc grant.
        Same-doc check: do the two blocks share any document ID?
        Grant check:    is any bit set in both q_union[c] and kv_union[c]?
        """
        bs = self.triton_block_size
        n_blocks = (seq_len + bs - 1) // bs
        n_chunks = len(q_bms)

        # All computation on CPU with numpy — fully vectorised, no Python loops
        # over block pairs.  For T=32768, block_size=64: ~1 ms total.

        # 1. Block-level OR-union of bitmasks via reshape + vectorised reduce.
        #    Pad each bitmask to n_blocks * bs so it reshapes cleanly.
        pad_len = n_blocks * bs - seq_len
        q_union_np  = np.zeros((n_chunks, n_blocks), dtype=np.int64)
        kv_union_np = np.zeros((n_chunks, n_blocks), dtype=np.int64)
        for c in range(n_chunks):
            q_pad  = np.zeros(n_blocks * bs, dtype=np.int64)
            kv_pad = np.zeros(n_blocks * bs, dtype=np.int64)
            q_pad[:seq_len]  = q_bms[c].cpu().numpy()
            kv_pad[:seq_len] = kv_bms[c].cpu().numpy()
            q_union_np[c]  = np.bitwise_or.reduce(q_pad.reshape(n_blocks, bs), axis=1)
            kv_union_np[c] = np.bitwise_or.reduce(kv_pad.reshape(n_blocks, bs), axis=1)

        # 2. Same-doc pairs: block q and block kv share a doc iff their doc_id
        #    ranges overlap.  doc_ids are non-decreasing in the packed sequence
        #    (each doc is contiguous and packed in order), so the range of doc IDs
        #    present in a block is exactly [block_start_doc, block_end_doc].
        doc_ids_np = document_ids.cpu().numpy()
        block_starts = np.arange(n_blocks) * bs
        block_ends   = np.minimum(block_starts + bs - 1, seq_len - 1)
        blk_min_doc  = doc_ids_np[block_starts]           # [n_blocks]
        blk_max_doc  = doc_ids_np[block_ends]              # [n_blocks]
        # Ranges overlap: max(min_q, min_kv) <= min(max_q, max_kv)
        same_doc = (
            np.maximum(blk_min_doc[:, None], blk_min_doc[None, :])
            <= np.minimum(blk_max_doc[:, None], blk_max_doc[None, :])
        )  # [n_blocks, n_blocks] bool

        # 3. Grant overlap pairs: any chunk has (q_union & kv_union) != 0
        grant = np.zeros((n_blocks, n_blocks), dtype=np.bool_)
        for c in range(n_chunks):
            grant |= (
                np.bitwise_and(q_union_np[c][:, None], kv_union_np[c][None, :]) != 0
            )

        # 4. Causal mask: kv_block <= q_block
        causal = np.arange(n_blocks)[:, None] >= np.arange(n_blocks)[None, :]

        # 5. Final interaction matrix and CSR construction via np.where
        interact = causal & (same_doc | grant)  # [n_blocks, n_blocks] bool

        # q_kv: for each Q-block, sorted list of KV-blocks (forward + bwd STAGE 2)
        q_idxs, kv_idxs = np.where(interact)   # row-major → already sorted by q_b
        kv_idxs_T, q_idxs_T = np.where(interact.T)  # kv→q (backward STAGE 1)

        def _pack_csr(
            row_idxs: np.ndarray,
            col_idxs: np.ndarray,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            counts = np.bincount(row_idxs, minlength=n_blocks).astype(np.int32)
            ptrs   = np.zeros(n_blocks + 1, dtype=np.int32)
            ptrs[1:] = np.cumsum(counts)
            indices = col_idxs.astype(np.int32)
            return (
                torch.from_numpy(counts).to(device),
                torch.from_numpy(ptrs).to(device),
                torch.from_numpy(indices).to(device),
            )

        q_kv_counts, q_kv_ptrs, q_kv_indices   = _pack_csr(q_idxs,   kv_idxs)
        kv_q_counts, kv_q_ptrs, kv_q_indices   = _pack_csr(kv_idxs_T, q_idxs_T)

        bim = BlockInteractionMask(
            seq_len=seq_len,
            block_size=bs,
            q_kv_counts=q_kv_counts,
            q_kv_ptrs=q_kv_ptrs,
            q_kv_indices=q_kv_indices,
            kv_q_counts=kv_q_counts,
            kv_q_ptrs=kv_q_ptrs,
            kv_q_indices=kv_q_indices,
        )
        logger.info(
            f"Built BlockInteractionMask: {n_blocks} blocks, "
            f"{int(q_kv_counts.sum())} non-empty fwd pairs, "
            f"sparsity={bim.sparsity:.1%} (block_size={bs})"
        )
        return bim

    def __call__(
        self,
        tokens: torch.Tensor,
        doc_spans: List[Any],
        link_to_target: Optional[Dict[int, List[int]]] = None,
        **kwargs,
    ):
        """
        Create a cross-document link-aware attention mask.

        Args:
            tokens:         Tensor of shape [B, T] with token IDs (input sequence only)
            doc_spans:      List of DocSpan objects with start, end, doc_id, raw_identifier
            link_to_target: Pre-computed link mapping (link_end_pos → [target_doc_id, ...]).
                            When provided the online link detection step is skipped entirely
                            (fast path for pre-computed epochs).  When None (default) links
                            are detected from ``tokens`` as usual.
            **kwargs:       Additional batch information (unused)

        Returns:
            ``BlockMask``             if ``self.backend == "flex"``  (default)
            ``BlockInteractionMask``  if ``self.backend == "triton"``
        """
        device = tokens.device
        seq_len = tokens.shape[-1]

        if link_to_target is None:
            # Online path: detect links from tokens
            input_ids = tokens[0]  # [seq_len]
            links = self.link_detector.detect_links(input_ids)
            logger.info(f"Found {len(links)} links in batch")
            link_to_target = self._match_links_to_docs(links, doc_spans)
        else:
            logger.info(
                f"Using pre-computed link_to_target with {len(link_to_target)} entries"
            )

        # Build document_ids tensor for base doc_causal mask
        document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for span in doc_spans:
            start = max(0, span.start)
            end = min(seq_len, span.end)
            if start < end:
                document_ids[start:end] = span.doc_id

        # Cosine warmup schedule: ramp max_grants from start → final over
        # max_grants_warmup_steps forward passes.  _n_chunks is always sized
        # for the final value so the triton kernel shape is stable.
        current_max_grants = self._current_max_grants()
        self._step += 1
        if current_max_grants < self.max_grants:
            logger.debug(
                f"max_grants schedule: step={self._step - 1}, "
                f"current_max_grants={current_max_grants}/{self.max_grants}"
            )

        q_bms, kv_bms = self._build_grant_bitmasks(
            seq_len, doc_spans, link_to_target, device,
            max_grants_cap=current_max_grants,
        )

        if self.backend == "triton":
            # Precompute block interaction table — no FlexAttention BlockMask built.
            return self._build_block_interaction_mask(
                seq_len, document_ids, q_bms, kv_bms, device
            )

        # backend == "flex": build FlexAttention BlockMask (default, DDP training path)
        def cross_doc_link_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            same_doc = document_ids[q_idx] == document_ids[kv_idx]
            # OR across chunks: bit k in chunk c is set in both iff grant
            # (64*c + k) covers this (q, kv) pair. The Python loop over a
            # fixed-length list unrolls at trace time.
            in_grant = (q_bms[0][q_idx] & kv_bms[0][kv_idx]) != 0
            for i in range(1, len(q_bms)):
                in_grant = in_grant | ((q_bms[i][q_idx] & kv_bms[i][kv_idx]) != 0)
            return causal & (same_doc | in_grant)

        block_mask = create_block_mask(
            cross_doc_link_mod,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device
        )

        logger.info(f"Created cross-document link mask for sequence length {seq_len}")
        return block_mask

    def build_dense_mask_for_visualization(
        self,
        tokens: torch.Tensor,
        doc_spans: List[Any],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Build a dense 2D boolean mask for visualization.

        Uses the exact same logic as __call__ but returns a dense tensor instead
        of a FlexAttention BlockMask.
        """
        if device is None:
            device = tokens.device

        seq_len = tokens.shape[-1]
        input_ids = tokens[0]

        links = self.link_detector.detect_links(input_ids)
        link_to_target = self._match_links_to_docs(links, doc_spans)
        cross_doc_mask = self._build_cross_doc_mask(
            seq_len, doc_spans, link_to_target, device
        )

        document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for span in doc_spans:
            start = max(0, span.start)
            end = min(seq_len, span.end)
            if start < end:
                document_ids[start:end] = span.doc_id

        q_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1]
        k_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]

        causal_mask = q_indices >= k_indices
        same_doc_mask = document_ids.unsqueeze(1) == document_ids.unsqueeze(0)

        dense_mask = causal_mask & (same_doc_mask | cross_doc_mask)
        return dense_mask
