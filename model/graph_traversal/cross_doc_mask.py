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
from typing import Any, Dict, List

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

    def __init__(self, link_detector: LinkDetector):
        self.link_detector = link_detector
        logger.info(
            f"Initialized CrossDocLinkMaskCreator with detector: "
            f"{type(link_detector).__name__}"
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
                    if span.start <= link_pos < span.end:
                        link_doc_span = span
                        break

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

        cross_doc_mask = self._build_cross_doc_mask(
            seq_len, doc_spans, link_to_target, device
        )

        # Build document_ids tensor for base doc_causal mask
        document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for span in doc_spans:
            start = max(0, span.start)
            end = min(seq_len, span.end)
            if start < end:
                document_ids[start:end] = span.doc_id

        def cross_doc_link_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            same_doc = document_ids[q_idx] == document_ids[kv_idx]
            cross_doc_link = cross_doc_mask[q_idx, kv_idx]
            return causal & (same_doc | cross_doc_link)

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
