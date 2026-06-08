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
    """

    raw_identifier: str
    normed_identifier: str
    outgoing_identifiers: List[str] = field(default_factory=list)
    incoming_identifiers: List[str] = field(default_factory=list)
    body_tokens: Optional[List[int]] = None


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


class IdentifierPrefixLayoutPolicy(DocLayoutPolicy):
    """
    Layout policy that prepends "# {raw_identifier}\\n\\n" as a prefix for each document.

    Tokens are produced by an external encode function (e.g. a tiktoken encoding's
    encode_ordinary method) and cached per identifier to avoid repeated tokenization.
    No suffix is added.
    """

    def __init__(self, encode_fn: Callable[[str], List[int]]):
        self._encode = encode_fn
        self._cache: Dict[str, List[int]] = {}

    def _get_prefix_tokens(self, raw_identifier: str) -> List[int]:
        if raw_identifier not in self._cache:
            self._cache[raw_identifier] = self._encode(f"# {raw_identifier}\n\n")
        return self._cache[raw_identifier]

    def prefix_length(self, info: DocLayoutInfo) -> int:
        return len(self._get_prefix_tokens(info.raw_identifier))

    def suffix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 0

    def prefix_tokens(self, info: DocLayoutInfo) -> List[int]:
        return list(self._get_prefix_tokens(info.raw_identifier))

    def suffix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return []


class IdentifierPrefixEOSLayoutPolicy(DocLayoutPolicy):
    """
    Layout policy that combines a title prefix with an EOS suffix.

    Each document is laid out as:

        encode("# {raw_identifier}\\n\\n") + [body] + [EOS]

    This gives the model a human-readable identifier before its content
    (useful for code filenames and article titles) plus a clean EOS stop
    signal, without a redundant BOS prefix.

    Token counts: prefix = len(encode("# {title}\\n\\n")), suffix = 1.
    """

    def __init__(
        self,
        encode_fn: Callable[[str], List[int]],
        eos_token_id: int,
    ):
        self._encode = encode_fn
        self.eos_token_id = eos_token_id
        self._cache: Dict[str, List[int]] = {}

    def _get_title_tokens(self, raw_identifier: str) -> List[int]:
        if raw_identifier not in self._cache:
            self._cache[raw_identifier] = self._encode(f"# {raw_identifier}\n\n")
        return self._cache[raw_identifier]

    def prefix_length(self, info: DocLayoutInfo) -> int:
        return len(self._get_title_tokens(info.raw_identifier))

    def suffix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 1

    def prefix_tokens(self, info: DocLayoutInfo) -> List[int]:
        return list(self._get_title_tokens(info.raw_identifier))

    def suffix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return [self.eos_token_id]


class StochasticIdentifierPrefixLayoutPolicy(DocLayoutPolicy):
    """
    Layout policy that randomly includes or omits the ``"# {raw_identifier}\\n\\n"``
    identifier prefix on a per-document, per-epoch basis.

    Inclusion is decided by::

        hash(normed_identifier + ":" + str(epoch)) % 2 == 0

    This is deterministic: the same ``(doc, epoch)`` pair always produces the
    same decision across ranks, restarts, and between the ``prefix_length()``
    and ``prefix_tokens()`` calls for the same document, with no shared state
    or cache required.

    An EOS token is always appended as suffix so that aux docs can end cleanly
    during generation regardless of whether the prefix was included.

    Use case: train with 50-50 prefix/no-prefix so the model is not OOD on
    external benchmarks that have no identifier prefix, while still being able
    to generate aux docs (which need a starting string).  During inference,
    wire a separate deterministic policy (e.g. ``identifier_prefix_eos``)
    via the ``data.inference_layout_policy`` config key.

    Call ``set_epoch(n)`` at the start of each epoch.  The training loop does
    this automatically via ``hasattr(layout, 'set_epoch')`` check.
    """

    def __init__(self, encode_fn: Callable[[str], List[int]], eos_token_id: int):
        self._encode = encode_fn
        self.eos_token_id = eos_token_id
        self._epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch counter. Called by BucketedPackDataset on epoch advance."""
        self._epoch = epoch

    def _include_prefix(self, normed_identifier: str) -> bool:
        """Deterministic per-(doc, epoch) coin flip; no cache, no shared state.

        Uses hashlib.md5 rather than Python's built-in hash() so the result is
        stable across processes, interpreter restarts, and subprocesses (Python's
        hash() is randomized per-process by default via PYTHONHASHSEED).
        """
        key = f"{normed_identifier}:{self._epoch}".encode()
        return int(hashlib.md5(key).hexdigest(), 16) % 2 == 0

    def _get_prefix_tokens(self, raw_identifier: str) -> List[int]:
        return self._encode(f"# {raw_identifier}\n\n")

    def prefix_length(self, info: DocLayoutInfo) -> int:
        if not self._include_prefix(info.normed_identifier):
            return 0
        return len(self._get_prefix_tokens(info.raw_identifier))

    def suffix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 1

    def prefix_tokens(self, info: DocLayoutInfo) -> List[int]:
        if not self._include_prefix(info.normed_identifier):
            return []
        return self._get_prefix_tokens(info.raw_identifier)

    def suffix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return [self.eos_token_id]


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
              ``"identifier_prefix_eos"``, ``"stochastic_identifier_prefix"``.
        encode_fn: Required for policies that tokenise the identifier
            (``"identifier_prefix"``, ``"identifier_prefix_eos"``, and
            ``"stochastic_identifier_prefix"``).
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
    raise ValueError(
        f"Unknown layout_policy '{name}'. "
        "Valid options: 'null', 'eos', 'identifier_prefix', "
        "'identifier_prefix_eos', 'stochastic_identifier_prefix'."
    )
