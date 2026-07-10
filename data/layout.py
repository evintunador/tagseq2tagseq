import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol


@dataclass
class DocLayoutInfo:
    """
    All known information about a document at layout time.

    Passed to every ``DocLayoutPolicy`` method so that policies can use
    whatever subset of information they need without the call sites growing
    new positional arguments when new fields are added.

    Fields that are genuinely unavailable at a given call site are left at
    their defaults (empty lists / None) rather than being omitted.  Policies
    must handle these gracefully.

    Attributes:
        raw_identifier: Human-readable document identifier (e.g. "Python
            (programming language)").  Empty string for the generation root,
            which has no corpus identity.
        normed_identifier: Normalised + hashed form used as the stable corpus
            key.  Empty string when ``raw_identifier`` is empty.
        outgoing_identifiers: Normed identifiers of documents this document
            links to.  Available in training (GraphIndex) and in generation
            after link detection has run.  Empty list when unknown.
        incoming_identifiers: Normed identifiers of documents that link to
            this document.  Available in training (GraphIndex).  Empty list
            in generation — structurally unavailable without a full reverse
            index of the corpus.
        body_tokens: The document body token ids (body only, excluding prefix
            and suffix decoration).  None when the body has not yet been
            fetched or generated (e.g. during pack-sampler length budgeting).
        categories: Space-separated subject categories for the document (e.g.
            ArXiv's ``"cs.CV eess.IV"``).  Empty string for datasets without
            categories (Wikipedia, TheStack) and for the generation root.  Used
            by the LaTeX-comment prefix-card policies.
    """

    raw_identifier: str
    normed_identifier: str
    outgoing_identifiers: List[str] = field(default_factory=list)
    incoming_identifiers: List[str] = field(default_factory=list)
    body_tokens: Optional[List[int]] = None
    categories: str = ""


class DocLayoutPolicy(Protocol):
    """
    Policy describing how each document is laid out in the packed sequence.

    Conceptually, every document in a batch contributes three segments:

        [prefix(doc_id)] + [body(doc_id)] + [suffix(doc_id)]

    The pack sampler is responsible for budgeting the *total* number of tokens
    per batch (including prefix and suffix), but it only ever truncates the
    body segment. Implementations of this protocol may use graph metadata,
    tokenizers, and caching internally, but expose only simple length and
    token-accessors here.
    """

    def prefix_length(self, info: DocLayoutInfo) -> int:
        """Number of prefix tokens emitted before the body for this doc."""

    def suffix_length(self, info: DocLayoutInfo) -> int:
        """Number of suffix tokens emitted after the body for this doc."""

    def prefix_tokens(self, info: DocLayoutInfo) -> List[int]:
        """
        Token ids for the prefix; to be consumed later in the collate layer.

        The pack sampler does not inspect these tokens; it only reasons about
        lengths. The collate function will use these tokens when materialising
        the final packed tensor.
        """

    def suffix_tokens(self, info: DocLayoutInfo) -> List[int]:
        """
        Token ids for the suffix; to be consumed later in the collate layer.

        As with ``prefix_tokens``, the sampler only needs lengths; callers in
        the collate layer will use these tokens when building the batch.
        """


class NullLayoutPolicy(DocLayoutPolicy):
    """
    Trivial layout policy that adds no decoration around document bodies.

    Under this policy, each document contributes exactly its body tokens to the
    batch and no additional prefix or suffix tokens. This preserves the current
    semantics of ``PackBatchSampler`` where ``effective_len`` equals the total
    number of tokens contributed by the document.
    """

    def prefix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 0

    def suffix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 0

    def prefix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return []

    def suffix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return []


class EOSLayoutPolicy(DocLayoutPolicy):
    """
    Layout policy that adds an end-of-sequence (EOS) token as the suffix for
    each document and no prefix.

    The EOS suffix gives the model a clean stop signal and makes the document
    boundary explicit in the token stream without wasting a context position
    on a redundant BOS prefix (doc_causal attention already provides a hard
    boundary at the start of each document).
    """

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id

    def prefix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 0

    def suffix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 1

    def prefix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return []

    def suffix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return [self.eos_token_id]


# ---------------------------------------------------------------------------
# Prefix-card formatters
# ---------------------------------------------------------------------------
# A "format function" maps a DocLayoutInfo to the prefix string for that document.
# It is the ONLY dataset-specific part of a prefix policy; everything else
# (caching, the stochastic coin-flip, length/token agreement, the EOS suffix) is
# shared by PrefixLayoutPolicy below.

def _markdown_heading_format(info: DocLayoutInfo) -> str:
    """Wikipedia / TheStack style: ``# {raw_identifier}\\n\\n``."""
    return f"# {info.raw_identifier}\n\n"


def _latex_comment_card(info: DocLayoutInfo) -> str:
    """ArXiv style LaTeX-comment card: ``% Title: ...``.

    Title-only by design: the citation link ``\\cite{Title}`` carries only the
    title, so seeding an aux doc at inference can only ever reconstruct the title
    line.  Emitting ``% Categories: ...`` at training time (where categories are
    known from the graph) but never at inference (where the cite has no
    categories) would create a train/inference mismatch — the model would learn
    to expect a categories line after the title that the inference seed never
    provides.  Keeping the card title-only matches the identifier-prefix policies
    used by the other datasets.  The card is valid LaTeX (``%`` begins a comment),
    keeping the token stream in-distribution for an otherwise-LaTeX ArXiv body.
    """
    return f"% Title: {info.raw_identifier}\n\n"


class PrefixLayoutPolicy(DocLayoutPolicy):
    """
    Unified prefix-card layout policy parameterised over three orthogonal axes:

    * ``format_fn``    — maps a ``DocLayoutInfo`` to the prefix string (the only
                         dataset-specific bit; e.g. ``_markdown_heading_format``
                         or ``_latex_comment_card``).
    * ``stochastic``   — when True, the prefix is included on a per-(doc, epoch)
                         50-50 coin flip (so the model is not OOD on benchmark
                         prompts that lack a prefix); when False the prefix is
                         always included and tokens are cached per format string.
    * ``eos_token_id`` — when not None, appended as a 1-token suffix.

    The coin flip is ``md5(normed_identifier + ":" + epoch) % 2 == 0`` —
    deterministic across ranks, restarts, and subprocesses (unlike Python's
    salted ``hash()``), and identical between ``prefix_length`` and
    ``prefix_tokens`` for a given doc so the two never disagree.  Stochastic mode
    intentionally does not cache (the decision is cheap and per-epoch); the
    deterministic mode caches tokens keyed by the formatted string.

    The named subclasses below pin these axes for the config-name factory and
    preserve the historical class names that callers import directly.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        format_fn: Callable[[DocLayoutInfo], str],
        *,
        stochastic: bool = False,
        eos_token_id: Optional[int] = None,
    ):
        self._encode = encode_fn
        self._format = format_fn
        self._stochastic = stochastic
        self.eos_token_id = eos_token_id
        self._epoch: int = 0
        self._cache: Dict[str, List[int]] = {}

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch counter (drives the stochastic coin flip).

        Defined unconditionally; harmless for deterministic policies and lets the
        training loop call it via a single ``hasattr(layout, 'set_epoch')`` check.
        """
        self._epoch = epoch

    def _include_prefix(self, info: DocLayoutInfo) -> bool:
        if not self._stochastic:
            return True
        key = f"{info.normed_identifier}:{self._epoch}".encode()
        return int(hashlib.md5(key).hexdigest(), 16) % 2 == 0

    def _prefix_tokens(self, info: DocLayoutInfo) -> List[int]:
        text = self._format(info)
        if self._stochastic:
            return self._encode(text)  # no cache: decision varies per epoch
        if text not in self._cache:
            self._cache[text] = self._encode(text)
        return self._cache[text]

    def prefix_length(self, info: DocLayoutInfo) -> int:
        if not self._include_prefix(info):
            return 0
        return len(self._prefix_tokens(info))

    def prefix_tokens(self, info: DocLayoutInfo) -> List[int]:
        if not self._include_prefix(info):
            return []
        return list(self._prefix_tokens(info))

    def suffix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 0 if self.eos_token_id is None else 1

    def suffix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return [] if self.eos_token_id is None else [self.eos_token_id]


# Named policies: thin wrappers pinning PrefixLayoutPolicy's axes. They preserve
# the class names that callers (generate.py, demo_traversal.py, eval) import and
# isinstance-check, while sharing all behaviour with the unified base.

class IdentifierPrefixLayoutPolicy(PrefixLayoutPolicy):
    """``# {raw_identifier}\\n\\n`` prefix, no suffix (Wikipedia / TheStack)."""

    def __init__(self, encode_fn: Callable[[str], List[int]]):
        super().__init__(encode_fn, _markdown_heading_format)


class IdentifierPrefixEOSLayoutPolicy(PrefixLayoutPolicy):
    """``# {raw_identifier}\\n\\n`` prefix + EOS suffix."""

    def __init__(self, encode_fn: Callable[[str], List[int]], eos_token_id: int):
        super().__init__(encode_fn, _markdown_heading_format, eos_token_id=eos_token_id)


class StochasticIdentifierPrefixLayoutPolicy(PrefixLayoutPolicy):
    """50-50 per-(doc, epoch) ``# {raw_identifier}\\n\\n`` prefix + EOS suffix.

    Train with 50-50 prefix/no-prefix so the model is not OOD on external
    benchmarks that lack a prefix, while still being able to start aux docs during
    generation.  Wire a deterministic ``identifier_prefix_eos`` policy for
    inference via ``data.inference_layout_policy``.
    """

    def __init__(self, encode_fn: Callable[[str], List[int]], eos_token_id: int):
        super().__init__(encode_fn, _markdown_heading_format,
                         stochastic=True, eos_token_id=eos_token_id)


class LatexCommentPrefixLayoutPolicy(PrefixLayoutPolicy):
    """Deterministic LaTeX-comment card (title + categories) + EOS suffix.

    The inference counterpart to ``StochasticLatexCommentPrefixLayoutPolicy``
    (wire via ``data.inference_layout_policy``): generation needs a deterministic
    starting card for aux docs, whereas training randomises card inclusion.
    """

    def __init__(self, encode_fn: Callable[[str], List[int]], eos_token_id: int):
        super().__init__(encode_fn, _latex_comment_card, eos_token_id=eos_token_id)


class StochasticLatexCommentPrefixLayoutPolicy(PrefixLayoutPolicy):
    """50-50 per-(doc, epoch) LaTeX-comment card (title + categories) + EOS suffix.

    Mirrors ``StochasticIdentifierPrefixLayoutPolicy`` but emits a LaTeX-comment
    card and surfaces ``categories`` (a conceptual, learnable signal that survives
    both inference-time corpus-fetch and document-generation).  Train with
    50-50 card/no-card; wire ``latex_comment_prefix`` for inference.
    """

    def __init__(self, encode_fn: Callable[[str], List[int]], eos_token_id: int):
        super().__init__(encode_fn, _latex_comment_card,
                         stochastic=True, eos_token_id=eos_token_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_GPT2_EOS = 50256  # <|endoftext|>


def make_layout_policy(
    name: str,
    encode_fn: Optional[Callable[[str], List[int]]] = None,
    eos_token_id: int = _GPT2_EOS,
) -> DocLayoutPolicy:
    """
    Construct a layout policy by name.

    Args:
        name: One of ``"null"``, ``"eos"``, ``"identifier_prefix"``,
              ``"identifier_prefix_eos"``, ``"stochastic_identifier_prefix"``,
              ``"latex_comment_prefix"``, ``"stochastic_latex_comment_prefix"``.
        encode_fn: Required for policies that tokenise the identifier
            (``"identifier_prefix"``, ``"identifier_prefix_eos"``,
            ``"stochastic_identifier_prefix"``, ``"latex_comment_prefix"``,
            ``"stochastic_latex_comment_prefix"``).
        eos_token_id: EOS token id (default: GPT-2 ``<|endoftext|>`` = 50256).

    Returns:
        A ``DocLayoutPolicy`` instance.

    Raises:
        ValueError: If ``name`` is unknown or ``encode_fn`` is missing where required.
    """
    if name == "null":
        return NullLayoutPolicy()
    if name == "eos":
        return EOSLayoutPolicy(eos_token_id=eos_token_id)
    if name in ("identifier_prefix", "identifier_prefix_eos"):
        if encode_fn is None:
            raise ValueError(
                f"layout_policy='{name}' requires encode_fn (a tokeniser callable)."
            )
        if name == "identifier_prefix":
            return IdentifierPrefixLayoutPolicy(encode_fn)
        return IdentifierPrefixEOSLayoutPolicy(
            encode_fn=encode_fn,
            eos_token_id=eos_token_id,
        )
    if name == "stochastic_identifier_prefix":
        if encode_fn is None:
            raise ValueError(
                "layout_policy='stochastic_identifier_prefix' requires encode_fn "
                "(a tokeniser callable)."
            )
        return StochasticIdentifierPrefixLayoutPolicy(encode_fn, eos_token_id=eos_token_id)
    if name == "latex_comment_prefix":
        if encode_fn is None:
            raise ValueError(
                "layout_policy='latex_comment_prefix' requires encode_fn "
                "(a tokeniser callable)."
            )
        return LatexCommentPrefixLayoutPolicy(encode_fn, eos_token_id=eos_token_id)
    if name == "stochastic_latex_comment_prefix":
        if encode_fn is None:
            raise ValueError(
                "layout_policy='stochastic_latex_comment_prefix' requires encode_fn "
                "(a tokeniser callable)."
            )
        return StochasticLatexCommentPrefixLayoutPolicy(encode_fn, eos_token_id=eos_token_id)
    raise ValueError(
        f"Unknown layout_policy '{name}'. "
        "Valid options: 'null', 'eos', 'identifier_prefix', "
        "'identifier_prefix_eos', 'stochastic_identifier_prefix', "
        "'latex_comment_prefix', 'stochastic_latex_comment_prefix'."
    )


# ---------------------------------------------------------------------------
# Detector -> inference layout mapping
# ---------------------------------------------------------------------------
#
# For a mixed-corpus model there is no single inference layout: a \cite prompt
# must use arxiv's latex-comment card, a markdown prompt the identifier card.
# The DETERMINISTIC inference-time layout is a clean function of the link
# detector, so eval can pick the correct layout per benchmark from the detector
# the benchmark implies (Tier-1 per-benchmark inference).  These pin the
# DETERMINISTIC variants (the stochastic coin is a training-only device); they
# mirror the inference_layout_policy fields in the per-source configs.
_DETECTOR_INFERENCE_LAYOUT = {
    "python":   "identifier_prefix_eos",
    "markdown": "identifier_prefix_eos",
    "arxiv":    "latex_comment_prefix",
    "null":     "eos",
}


def inference_layout_for_detector(detector_name: Optional[str]) -> str:
    """Return the deterministic inference layout name for a link detector.

    ``None`` (no detector, e.g. doc_causal / edgeless) maps to ``"eos"``.
    Raises on an unknown detector so a new detector can't silently fall back to
    a wrong layout.
    """
    if detector_name is None:
        return "eos"
    try:
        return _DETECTOR_INFERENCE_LAYOUT[detector_name]
    except KeyError as exc:
        raise ValueError(
            f"No inference layout mapping for link detector {detector_name!r}. "
            f"Known: {sorted(_DETECTOR_INFERENCE_LAYOUT)}."
        ) from exc
