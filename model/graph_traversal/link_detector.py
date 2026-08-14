"""
LinkDetector protocol and shared LinkInfo type.

All dataset-specific link detectors implement this protocol:
    MarkdownLinkDetector  (markdown_link_detector.py)  — Wikipedia / Markdown
    PythonImportDetector  (python_import_detector.py)  — Python / TheStack
    ArxivCiteDetector     (arxiv_cite_detector.py)     — ArXiv / unarXive

Use ``make_link_detector(name, decode_fn)`` to construct one by config name
rather than dispatching on the name at each call site.
"""
from __future__ import annotations

from typing import Any, Callable, List, NamedTuple, Protocol, runtime_checkable

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
    about each link they find. Different datasets have different implementations:
    - Wikipedia / Markdown:  [text](target) syntax     → MarkdownLinkDetector
    - Python / TheStack:     import statements         → PythonImportDetector
    - LaTeX / ArXiv:         \\cite{Title} citations   → ArxivCiteDetector

    # TODO: additional programming languages (Ruby require, JS/TS import,
    # Rust use, etc.) will each need their own detector if those datasets are
    # added. Revisit once a second code dataset is being added.

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


# Valid names accepted by ``make_link_detector`` (and the ``model.link_detector``
# config key), kept here as the single source of truth for error messages.
LINK_DETECTOR_NAMES = ("markdown", "python", "go", "java", "typescript", "javascript", "kotlin", "rust", "zig", "dart", "arxiv", "null", "composite")


def make_link_detector(name: str, decode_fn: Callable[[List[int]], str]) -> "LinkDetector":
    """
    Construct a dataset-specific LinkDetector by config name.

    Single dispatch point for ``model.link_detector`` so adding a dataset means
    one entry here instead of editing every training/inference/profiling script.

    Args:
        name: ``'markdown'`` (Wikipedia), ``'python'`` (TheStack),
              ``'go'`` (TheStack Go), or ``'arxiv'`` (unarXive
              ``\\cite{Title}`` citations).
        decode_fn: Token-ids → str callable (typically ``tiktoken_enc.decode``).

    Raises:
        ValueError: If ``name`` is unknown.
    """
    # Imported lazily: the concrete detector modules import LinkInfo from this
    # module, so importing them at top level would create a circular import.
    if name == "markdown":
        from .markdown_link_detector import MarkdownLinkDetector
        return MarkdownLinkDetector(decode_fn=decode_fn)
    if name == "python":
        from .python_import_detector import PythonImportDetector
        return PythonImportDetector(decode_fn=decode_fn)
    if name == "go":
        from .go_import_detector import GoImportDetector
        return GoImportDetector(decode_fn=decode_fn)
    if name == "java":
        from .java_import_detector import JavaImportDetector
        return JavaImportDetector(decode_fn=decode_fn)
    if name == "typescript":
        from .typescript_import_detector import TypeScriptImportDetector
        return TypeScriptImportDetector(decode_fn=decode_fn)
    if name == "javascript":
        from .javascript_import_detector import JavaScriptImportDetector
        return JavaScriptImportDetector(decode_fn=decode_fn)
    if name == "kotlin":
        from .kotlin_import_detector import KotlinImportDetector
        return KotlinImportDetector(decode_fn=decode_fn)
    if name == "rust":
        from .rust_import_detector import RustImportDetector
        return RustImportDetector(decode_fn=decode_fn)
    if name == "zig":
        from .zig_import_detector import ZigImportDetector
        return ZigImportDetector(decode_fn=decode_fn)
    if name == "dart":
        from .dart_import_detector import DartImportDetector
        return DartImportDetector(decode_fn=decode_fn)
    if name == "arxiv":
        from .arxiv_cite_detector import ArxivCiteDetector
        return ArxivCiteDetector(decode_fn=decode_fn)
    if name == "null":
        from .null_link_detector import NullLinkDetector
        return NullLinkDetector(decode_fn=decode_fn)
    if name == "composite":
        # Per-document dispatch across all merged-corpus sub-detectors, for
        # inference/generation on a merged model where links must be detected
        # from raw multi-source text (no single syntax fires everywhere).
        from .composite_link_detector import CompositeLinkDetector
        return CompositeLinkDetector(decode_fn=decode_fn)
    raise ValueError(
        f"Unknown model.link_detector '{name}'. "
        f"Valid options: {', '.join(LINK_DETECTOR_NAMES)} "
        "('markdown'=Wikipedia, 'python'=TheStack, 'go'=TheStack Go, "
        "'arxiv'=unarXive, 'null'=edgeless/FineWeb, "
        "'composite'=merged multi-source model)."
    )
