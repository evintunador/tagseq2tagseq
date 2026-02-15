"""
Cross-Document Link Attention Mask

This module implements a custom attention mask that allows tokens to attend to
previously-linked documents in the batch. When a document contains a markdown link
[text](target_title), all tokens appearing AFTER that link can attend to the
target document if it appears earlier in the batch.

Key Features:
- Link-aware attention: tokens after links gain access to linked documents
- Cumulative access: multiple links accumulate, granting access to multiple docs
- Causal by default: maintains causality within and across documents
- DAG-aware: only works for forward references (earlier docs in batch)
"""

import torch
from torch.nn.attention.flex_attention import create_block_mask, BlockMask
from typing import List, Any, Callable, Dict
import logging

from .link_detector_protocol import LinkInfo, TokenizedLinkDetector

logger = logging.getLogger(__name__)


class CrossDocLinkMaskCreator:
    """
    Creates attention masks that grant cross-document attention based on in-text links.

    This is a callable class that can be passed to DS2DSTrainingModule as the
    block_mask_creator parameter.

    Args:
        tokenizer_decode_fn: Function that takes List[int] and returns str (e.g., tokenizer.decode)
        link_detector: TokenizedLinkDetector implementation (e.g., MarkdownLinkDetector, PythonImportDetector)
        bos_token_id: Token ID for beginning of sequence (optional)
        eos_token_id: Token ID for end of sequence (optional)

    Example:
        >>> import tiktoken
        >>> from model.tokenizer_config import TokenizerConfig
        >>> from model.graph_traversal.link_detectors import MarkdownLinkDetector
        >>> 
        >>> enc = tiktoken.get_encoding('gpt2')
        >>> tokenizer_config = TokenizerConfig.from_tokenizer(enc, name='gpt2')
        >>> link_detector = MarkdownLinkDetector(tokenizer_config)
        >>> 
        >>> mask_creator = CrossDocLinkMaskCreator(
        ...     tokenizer_decode_fn=enc.decode,
        ...     link_detector=link_detector
        ... )
        >>> model = DS2DSTrainingModule(
        ...     block_mask_creator=mask_creator,
        ...     vocab_size=50257,
        ...     ...
        ... )
    """

    def __init__(
        self,
        tokenizer_decode_fn: Callable[[List[int]], str],
        link_detector: TokenizedLinkDetector = None,
        bos_token_id: int = None,
        eos_token_id: int = None,
    ):
        """
        Args:
            tokenizer_decode_fn: Function to decode token IDs to text
            link_detector: Implementation for detecting links (default: MarkdownLinkDetector)
            bos_token_id: Token ID for beginning of sequence (optional)
            eos_token_id: Token ID for end of sequence (optional)
        """
        self.tokenizer_decode_fn = tokenizer_decode_fn
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Default to markdown links if not specified
        if link_detector is None:
            from .link_detectors import MarkdownLinkDetector
            from model.tokenizer_config import TokenizerConfig
            # Use GPT-2 defaults if no detector provided
            link_detector = MarkdownLinkDetector(TokenizerConfig.gpt2_defaults())
        
        self.link_detector = link_detector
        logger.info(f"Initialized CrossDocLinkMaskCreator with detector: {type(link_detector).__name__}")

    def _match_links_to_docs(
        self,
        links: List[LinkInfo],
        input_ids: torch.Tensor,
        doc_spans: List[Any]
    ) -> Dict[int, int]:
        """
        Match detected links to documents in the batch.

        Two modes:
        1. Target-based (markdown): Decode target from tokens, match to clean_title
        2. Outgoing-based (Python): Use doc_spans.outgoing_titles from graph

        Args:
            links: List of detected LinkInfo objects
            input_ids: 1D tensor of token IDs
            doc_spans: List of DocSpan objects

        Returns:
            Dictionary mapping link_end_pos -> target_doc_id for valid links.
            Only includes links where the target document appears earlier in the batch.
        """
        # Build a mapping from clean_title to (doc_id, start_pos)
        title_to_doc = {}
        for span in doc_spans:
            title_to_doc[span.clean_title] = (span.doc_id, span.start)

        link_to_target = {}

        # MODE 1: Use outgoing_titles from graph (Python imports, etc.)
        if self.link_detector.uses_outgoing_titles:
            # Build position -> doc_span mapping
            pos_to_span = {}
            for span in doc_spans:
                for pos in range(span.start, span.end):
                    pos_to_span[pos] = span
            
            # For each link position, find its document and use outgoing_titles
            for link in links:
                source_span = pos_to_span.get(link.link_start_pos)
                if not source_span or not source_span.outgoing_titles:
                    continue
                
                # Grant access to ALL outgoing docs from this source doc
                for target_title in source_span.outgoing_titles:
                    if target_title not in title_to_doc:
                        continue
                    
                    target_doc_id, target_start_pos = title_to_doc[target_title]
                    
                    # Ensure DAG property
                    if target_start_pos >= link.link_end_pos:
                        continue
                    
                    # Map this import position to the target doc
                    link_to_target[link.link_end_pos] = target_doc_id
                    logger.debug(
                        f"Matched import at {link.link_end_pos} -> doc {target_doc_id} ('{target_title}' from outgoing_titles)"
                    )
                    break  # Only match to first valid target per import
        
        # MODE 2: Decode target from tokens (markdown links, etc.)
        else:
            for link in links:
                # Extract and decode target title tokens
                target_tokens = input_ids[link.target_start:link.target_end].tolist()
                try:
                    target_title = self.tokenizer_decode_fn(target_tokens).strip()
                except Exception as e:
                    logger.warning(f"Failed to decode link target tokens: {e}")
                    continue

                # Check if this title exists in the batch
                if target_title not in title_to_doc:
                    logger.debug(f"Link target '{target_title}' not found in batch, skipping")
                    continue

                target_doc_id, target_start_pos = title_to_doc[target_title]

                # Ensure DAG property: target must appear EARLIER in the sequence
                if target_start_pos >= link.link_end_pos:
                    logger.debug(
                        f"Link at {link.link_end_pos} to '{target_title}' violates DAG "
                        f"(target starts at {target_start_pos}), skipping"
                    )
                    continue

                # Valid link! Map the link end position to the target doc
                link_to_target[link.link_end_pos] = target_doc_id
                logger.debug(
                    f"Matched link at {link.link_end_pos} -> doc {target_doc_id} ('{target_title}')"
                )

        logger.info(f"Matched {len(link_to_target)}/{len(links)} links to documents in batch")
        return link_to_target

    def _build_cross_doc_mask(
        self,
        seq_len: int,
        doc_spans: List[Any],
        link_to_target: Dict[int, int],
        device: torch.device
    ) -> torch.Tensor:
        """
        Build a 2D cross-document attention mask for link-based attention.

        Args:
            seq_len: Length of the input sequence
            doc_spans: List of DocSpan objects
            link_to_target: Mapping from link_end_pos to target_doc_id
            device: Device to create tensors on

        Returns:
            cross_doc_mask: Tensor of shape [seq_len, seq_len] where
                           cross_doc_mask[q, kv] = True if position q can attend
                           to position kv via a link (not including same-doc attention).
        """
        cross_doc_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

        # For each link, grant cross-document access
        for link_pos, target_doc_id in sorted(link_to_target.items()):
            # Find the document containing this link
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

            # Grant access: positions after the link in the source doc
            # can attend to all positions in the target doc
            grant_start = link_pos + 1
            grant_end = min(seq_len, link_doc_span.end)
            target_start = max(0, target_doc_span.start)
            target_end = min(seq_len, target_doc_span.end)

            if grant_start < grant_end and target_start < target_end:
                # Set cross_doc_mask[q, kv] = True for all q in [grant_start, grant_end)
                # and kv in [target_start, target_end)
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
            tokens: Tensor of shape [B, T] with token IDs (full sequence including target)
            doc_spans: List of DocSpan objects with start, end, doc_id, clean_title
            **kwargs: Additional batch information

        Returns:
            BlockMask for FlexAttention
        """
        device = tokens.device
        # Input sequence length (model sees tokens[:-1])
        seq_len = tokens.shape[-1] - 1

        # Get input_ids as 1D tensor for link detection
        input_ids = tokens[0, :-1]  # [seq_len]

        # Step 2: Detect all links in the sequence using pluggable detector
        links = self.link_detector.detect_links(input_ids, self.tokenizer_decode_fn)

        # Step 3: Match links to documents in the batch
        link_to_target = self._match_links_to_docs(links, input_ids, doc_spans)

        # Step 4: Build cross-doc attention mask (2D)
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

        # Create the combined attention mask function
        def cross_doc_link_mod(b, h, q_idx, kv_idx):
            # Base causal constraint (always required)
            causal = q_idx >= kv_idx

            # Same document (always allowed if causal)
            same_doc = document_ids[q_idx] == document_ids[kv_idx]

            # Cross-document via link (precomputed 2D mask)
            cross_doc_link = cross_doc_mask[q_idx, kv_idx]

            # Allow if: causal AND (same_doc OR cross_doc_link)
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
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Build a dense 2D boolean mask for visualization.

        This uses the EXACT same logic as __call__ but returns a dense tensor
        instead of a FlexAttention BlockMask. Use this for visualization to ensure
        the logic is identical.

        Args:
            tokens: Tensor of shape [B, T] with token IDs
            doc_spans: List of DocSpan objects
            device: Device for tensors (defaults to tokens.device)

        Returns:
            Dense boolean mask of shape [seq_len, seq_len]
        """
        if device is None:
            device = tokens.device

        seq_len = tokens.shape[-1] - 1
        input_ids = tokens[0, :-1]

        # Step 1: Detect links using pluggable detector
        links = self.link_detector.detect_links(input_ids, self.tokenizer_decode_fn)

        # Step 2: Match links to docs
        link_to_target = self._match_links_to_docs(links, input_ids, doc_spans)

        # Step 3: Build cross-doc mask
        cross_doc_mask = self._build_cross_doc_mask(
            seq_len, doc_spans, link_to_target, device
        )

        # Step 4: Build document IDs tensor
        document_ids = torch.full((seq_len,), -1, dtype=torch.int32, device=device)
        for span in doc_spans:
            start = max(0, span.start)
            end = min(seq_len, span.end)
            if start < end:
                document_ids[start:end] = span.doc_id

        # Step 5: Build dense mask with SAME logic as mask_mod in __call__
        q_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1]
        k_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]

        # Causal constraint
        causal_mask = q_indices >= k_indices

        # Same document
        same_doc_mask = document_ids.unsqueeze(1) == document_ids.unsqueeze(0)

        # Final combination: causal AND (same_doc OR cross_doc_link)
        # This mirrors the logic in the mask_mod function
        dense_mask = causal_mask & (same_doc_mask | cross_doc_mask)

        return dense_mask
