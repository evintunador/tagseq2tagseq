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
from typing import Any, Dict, List, Tuple

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .link_detector import LinkDetector, LinkInfo

logger = logging.getLogger(__name__)


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

    def __init__(self, link_detector: LinkDetector, max_grants: int = 64):
        self.link_detector = link_detector
        self.max_grants = max_grants
        self._n_chunks = max(1, (max_grants + 63) // 64)
        logger.info(
            f"Initialized CrossDocLinkMaskCreator with detector: "
            f"{type(link_detector).__name__}, max_grants={max_grants} "
            f"({self._n_chunks} int64 chunk(s))"
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
                if grant_idx >= self.max_grants:
                    logger.warning(
                        f"Batch has >{self.max_grants} grants; ignoring remainder. "
                        f"Consider increasing max_grants (currently {self.max_grants})."
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

    def __call__(self, tokens: torch.Tensor, doc_spans: List[Any], **kwargs) -> BlockMask:
        """
        Create a cross-document link-aware attention mask.

        Args:
            tokens:    Tensor of shape [B, T] with token IDs (input sequence only)
            doc_spans: List of DocSpan objects with start, end, doc_id, raw_identifier
            **kwargs:  Additional batch information (unused)

        Returns:
            BlockMask for FlexAttention
        """
        device = tokens.device
        seq_len = tokens.shape[-1]

        input_ids = tokens[0]  # [seq_len]

        links = self.link_detector.detect_links(input_ids)
        logger.info(f"Found {len(links)} links in batch")

        link_to_target = self._match_links_to_docs(links, doc_spans)

        # Build document_ids tensor for base doc_causal mask
        document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for span in doc_spans:
            start = max(0, span.start)
            end = min(seq_len, span.end)
            if start < end:
                document_ids[start:end] = span.doc_id

        # Lists of n_chunks int64 bitmask tensors: O(n_chunks × seq_len).
        # At seq_len=32768: 512 KB (1 chunk) to 2 MB (4 chunks) vs 1 GB for
        # the O(seq_len²) dense bool mask.
        # mask_mod ORs across chunks (all pointwise, no sequence-dimension
        # reductions), satisfying FlexAttention's mask_mod constraint.
        q_bms, kv_bms = self._build_grant_bitmasks(
            seq_len, doc_spans, link_to_target, device
        )

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
