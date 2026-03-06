"""
LinkDetector protocol and shared LinkInfo type.

All dataset-specific link detectors implement this protocol:
    MarkdownLinkDetector  (markdown_link_detector.py)  — Wikipedia / Markdown
    PythonImportDetector  (python_import_detector.py)  — Python / TheStack
"""
from __future__ import annotations

from typing import Any, List, NamedTuple, Protocol, runtime_checkable

import torch


class LinkInfo(NamedTuple):
    """Metadata about a detected link in the token sequence."""
    link_end_pos: int   # Token position just after the link's closing delimiter;
                        # attention to the target is granted from this position onward.
    target_str: str     # Decoded target identifier string (matched against DocSpan.raw_identifier)


@runtime_checkable
class LinkDetector(Protocol):
    """
    Protocol for dataset-specific link detection in packed token sequences.

    Implementations scan a 1D input_ids tensor and return structured information
    about each link they find. Different datasets need different implementations:
    - Wikipedia / Markdown:  [text](target) syntax          → MarkdownLinkDetector
    - Python / TheStack:     import statements              → PythonImportDetector
    - LaTeX / ArXiv:         \\cite{key} or \\ref{label}   → not yet implemented (see below)
    - Other languages:       Ruby require, JS/TS import,
                             Rust use, etc.                 → not yet implemented (see below)

    # TODO(@jamesljr): a LaTeX detector will be needed for the ArXiv dataset.
    # The exact abstraction is still TBD — this is a rough sketch, not a
    # confident design.  LaTeX cross-references can take many forms
    # (\\cite, \\citep, \\citet, \\ref, \\hyperref, \\input, \\include, and
    # others depending on the document class and packages used), and it is not
    # yet clear which of these should create attention links, what target_str
    # should look like, or how index_doc_span should map ArXiv identifiers.
    # Requires deeper understanding of the ArXiv graph structure before
    # committing to an implementation.

    # TODO(@jamesljr): additional programming languages (Ruby require, JS/TS
    # import, Rust use, Go import, etc.) will each need their own detector if
    # those datasets are added.  The right abstraction here is uncertain — the
    # examples below are illustrative guesses, not a confident design.  Open
    # questions include: one detector per language vs. a dispatch wrapper,
    # whether to identify language from the file extension in raw_identifier or
    # store it as metadata, and how import semantics differ enough across
    # languages to require fundamentally different logic vs. just different
    # regex patterns.  Revisit once a second code dataset is being added.

    The detector is responsible for all decoding; it returns already-decoded
    target strings rather than token spans, so CrossDocLinkMaskCreator has no
    dependency on a tokenizer.
    """

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """
        Detect links in a 1D token sequence.

        Args:
            input_ids: 1D tensor of token IDs, shape [seq_len]

        Returns:
            List of LinkInfo objects. Each describes the position in the sequence
            where attention to the target begins (link_end_pos) and the already-
            decoded target identifier string (target_str).
        """
        ...

    def index_doc_span(self, span: Any) -> str:
        """
        Return the lookup key for a DocSpan when building the target-matching index.

        Defaults to ``span.raw_identifier`` (exact match).  Detectors for datasets
        whose ``target_str`` is only a sub-component of ``raw_identifier`` (e.g.
        ``PythonImportDetector`` returns a bare file path while ``raw_identifier``
        includes a repo prefix) should override this to return the matching
        sub-component.
        """
        return span.raw_identifier
