import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

import numpy as np
import torch

from .dataset import GraphIndex, PretokShardedBackend
from .layout import DocLayoutPolicy
from .pack_sampler import DocPlacement


logger = logging.getLogger(__name__)


# Cache for formatter instances per dataset directory
_formatter_cache: Dict[str, Any] = {}


@dataclass
class DocSpan:
    """
    Span metadata describing where a single document appears in a packed sequence.

    Indices are expressed in terms of the final 1D token sequence emitted by
    ``build_packed_batch``. The span covers the concatenation of:

        [prefix(doc_id)] + [body_slice(doc_id)] + [suffix(doc_id)]
    """

    doc_id: int
    title: str
    start: int  # inclusive
    end: int  # exclusive
    truncated: bool
    outgoing_titles: List[str]
    clean_title: str


def _slice_body_tokens(
    full_body: np.ndarray,
    effective_len: int,
    trim_side: str,
    doc_id: int,
) -> np.ndarray:
    """
    Slice the body tokens according to the placement's ``effective_len`` and
    ``doc_trim_side``.

    The sampler is expected to ensure ``0 < effective_len <= len(full_body)``.
    This function defensively clips ``effective_len`` to the available length
    and logs a warning if it needs to do so.
    """
    if effective_len <= 0:
        return full_body[:0]

    L = int(full_body.shape[0])
    k = int(effective_len)

    if L == 0:
        return full_body[:0]

    if k > L:
        logger.warning(
            "Doc %s has effective_len=%d > body_len=%d; clipping to body_len.",
            doc_id,
            k,
            L,
        )
        k = L

    if trim_side == "tail":
        return full_body[:k]
    if trim_side == "head":
        return full_body[L - k :]

    logger.warning(
        "Doc %s has unknown trim_side=%r; defaulting to 'tail'.", doc_id, trim_side
    )
    return full_body[:k]


def get_title_formatter(run_directory: Path):
    """
    Get the appropriate title formatter for a dataset.
    
    Loads the dataset config and returns the corresponding formatter.
    Caches formatters to avoid repeated file reads.
    
    Args:
        run_directory: Path to pretokenized dataset directory
        
    Returns:
        TitleFormatter instance for this dataset
    """
    from .dataset_config import load_config_from_pretokenized_dir
    
    cache_key = str(run_directory.resolve())
    
    if cache_key not in _formatter_cache:
        try:
            config = load_config_from_pretokenized_dir(run_directory)
            formatter = config.get_formatter()
            _formatter_cache[cache_key] = formatter
            logger.info(
                f"Loaded title formatter for dataset '{config.name}': "
                f"{formatter.__class__.__name__}"
            )
        except FileNotFoundError:
            # Fall back to flat formatter if no config
            from model.title_formats import FlatTitleFormatter
            logger.warning(
                f"No dataset_config.json found in {run_directory}, "
                f"using default FlatTitleFormatter"
            )
            formatter = FlatTitleFormatter()
            _formatter_cache[cache_key] = formatter
    
    return _formatter_cache[cache_key]


def build_packed_batch(
    graph: GraphIndex,
    backend: PretokShardedBackend,
    layout: DocLayoutPolicy,
    placements: List[DocPlacement],
    as_2d: bool = True,
    formatter: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Materialise a single packed batch from an ordered list of ``DocPlacement``.

    The sampler and layout policy are responsible for all length decisions,
    including prefix / body / suffix budgeting and truncation. This function
    simply:

        1. Fetches prefix / suffix decoration tokens via the layout policy.
        2. Fetches full body tokens from the backend.
        3. Slices the body according to ``DocPlacement.effective_len`` and
           ``DocPlacement.doc_trim_side``.
        4. Concatenates prefix + body_slice + suffix for each doc.
        5. Concatenates all docs into a single packed sequence.
        6. Returns the packed tokens and per-doc ``DocSpan`` metadata.
    """
    segments: List[np.ndarray] = []
    spans: List[DocSpan] = []
    offset = 0

    for p in placements:
        prefix_ids = layout.prefix_tokens(p.doc_id)
        suffix_ids = layout.suffix_tokens(p.doc_id)

        body = backend.get_tokens_by_id(p.doc_id)
        if body is None:
            logger.warning("No tokens found for doc_id=%s; skipping.", p.doc_id)
            continue

        body = np.asarray(body)
        body_slice = _slice_body_tokens(
            full_body=body,
            effective_len=p.effective_len,
            trim_side=p.doc_trim_side,
            doc_id=p.doc_id,
        )

        if body_slice.size == 0 and not prefix_ids and not suffix_ids:
            # Entirely empty contribution; skip to avoid degenerate spans.
            continue

        prefix_arr = (
            np.asarray(prefix_ids, dtype=np.int64)
            if prefix_ids
            else np.empty(0, dtype=np.int64)
        )
        body_arr = np.asarray(body_slice, dtype=np.int64)
        suffix_arr = (
            np.asarray(suffix_ids, dtype=np.int64)
            if suffix_ids
            else np.empty(0, dtype=np.int64)
        )

        doc_tokens = np.concatenate([prefix_arr, body_arr, suffix_arr], axis=0)
        doc_len = int(doc_tokens.shape[0])

        title = graph.get_title(p.doc_id)
        outgoing_titles = graph.get_outgoing_links(title)
        
        # Strip hash to get clean title using formatter if available
        if formatter is not None:
            clean_title = formatter.strip_hash(title)
        else:
            # Fall back to default pattern (6-char hex hash)
            clean_title = re.sub(r'_[0-9a-f]{6}$', '', title)
        
        span = DocSpan(
            doc_id=p.doc_id,
            title=title,
            start=offset,
            end=offset + doc_len,
            truncated=p.truncated,
            outgoing_titles=outgoing_titles,
            clean_title=clean_title,
        )

        segments.append(doc_tokens)
        spans.append(span)
        offset += doc_len

    if segments:
        tokens = np.concatenate(segments, axis=0).astype(np.int64, copy=False)
    else:
        tokens = np.empty(0, dtype=np.int64)

    tokens = torch.from_numpy(tokens)
    if as_2d:
        # Shape as [1, T] for convenience in typical LM workloads.
        tokens = tokens.view(1, -1)

    return {
        "tokens": tokens,
        "doc_spans": spans,
        "doc_ids": [span.doc_id for span in spans],
        "titles": [span.title for span in spans],
    }


