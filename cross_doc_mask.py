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
from typing import List, Any, Optional, NamedTuple, Callable, Dict
import logging

logger = logging.getLogger(__name__)


class LinkInfo(NamedTuple):
    """Metadata about a detected link in the token sequence."""
    link_start_pos: int  # Position of '[' token
    link_mid_pos: int    # Position of '](' token
    link_end_pos: int    # Position of ')' token
    target_start: int    # First token of target title (after '](' )
    target_end: int      # Last token of target title (before ')')


class CrossDocLinkMaskCreator:
    """
    Creates attention masks that grant cross-document attention based on in-text links.

    This is a callable class that can be passed to DS2DSTrainingModule as the
    block_mask_creator parameter.

    Args:
        tokenizer_decode_fn: Function that takes List[int] and returns str (e.g., tokenizer.decode)
        link_start_token_ids: Token IDs for '[' variants (default: [58, 685] for GPT-2)
        link_mid_token_id: Token ID for '](' (default: 16151 for GPT-2)
        link_end_token_id: Token ID for ')' (default: 8 for GPT-2)
        bos_token_id: Token ID for beginning of sequence (optional)
        eos_token_id: Token ID for end of sequence (optional)

    Example:
        >>> import tiktoken
        >>> enc = tiktoken.get_encoding('gpt2')
        >>> mask_creator = CrossDocLinkMaskCreator(
        ...     tokenizer_decode_fn=enc.decode,
        ...     link_start_token_ids=[58, 685],
        ...     link_mid_token_id=16151,
        ...     link_end_token_id=8
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
        link_start_token_ids: Optional[List[int]] = None,  # Possible '[' tokens
        link_mid_token_id: int = 16151,   # '](' in GPT-2
        link_end_token_id: int = 8,       # ')' in GPT-2
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        # Default to common '[' token variants in GPT-2
        if link_start_token_ids is None:
            link_start_token_ids = [58, 685]  # '[' and ' ['

        self.tokenizer_decode_fn = tokenizer_decode_fn
        self.link_start_token_ids = set(link_start_token_ids)
        self.link_mid_token_id = link_mid_token_id
        self.link_end_token_id = link_end_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        logger.info(
            f"Initialized CrossDocLinkMaskCreator with token IDs: "
            f"'[' variants = {link_start_token_ids}, '](' = {link_mid_token_id}, ')' = {link_end_token_id}"
        )

    def _detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """
        Detect all markdown link patterns [text](target) in the token sequence.

        Args:
            input_ids: 1D tensor of token IDs (shape [seq_len])

        Returns:
            List of LinkInfo objects describing each detected link
        """
        links = []
        seq_len = input_ids.shape[0]

        # Find all positions of '](' token
        link_mid_positions = (input_ids == self.link_mid_token_id).nonzero(as_tuple=True)[0]

        for mid_pos in link_mid_positions:
            mid_pos = mid_pos.item()

            # Search backwards for '[' token (check multiple possible token IDs)
            link_start_pos = None
            for i in range(mid_pos - 1, -1, -1):
                if input_ids[i].item() in self.link_start_token_ids:
                    link_start_pos = i
                    break
                # Stop searching if we've gone too far (e.g., 100 tokens)
                if mid_pos - i > 100:
                    break

            if link_start_pos is None:
                # No matching '[' found, skip this ']('
                continue

            # Search forwards for ')' token
            link_end_pos = None
            for i in range(mid_pos + 1, min(mid_pos + 101, seq_len)):
                if input_ids[i] == self.link_end_token_id:
                    link_end_pos = i
                    break

            if link_end_pos is None:
                # No matching ')' found, skip this link
                continue

            # Calculate target title span (tokens between '](' and ')')
            target_start = mid_pos + 1
            target_end = link_end_pos  # exclusive

            if target_start >= target_end:
                # Empty target, skip
                continue

            links.append(LinkInfo(
                link_start_pos=link_start_pos,
                link_mid_pos=mid_pos,
                link_end_pos=link_end_pos,
                target_start=target_start,
                target_end=target_end
            ))

        logger.debug(f"Detected {len(links)} links in sequence of length {seq_len}")
        return links

    def _match_links_to_docs(
        self,
        links: List[LinkInfo],
        input_ids: torch.Tensor,
        doc_spans: List[Any]
    ) -> Dict[int, int]:
        """
        Match detected links to documents in the batch.

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

        for link in links:
            # Extract and decode target title tokens
            target_tokens = input_ids[link.target_start:link.target_end].tolist()
            try:
                target_title = self.tokenizer_decode_fn(target_tokens)
            except Exception as e:
                logger.warning(f"Failed to decode link target tokens {target_tokens}: {e}")
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

        # Step 2: Detect all links in the sequence
        links = self._detect_links(input_ids)
        logger.info(f"Found {len(links)} links in batch")

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

        # Step 1: Detect links
        links = self._detect_links(input_ids)

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
