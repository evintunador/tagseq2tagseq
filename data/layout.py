import abc
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


class BOSEOSLayoutPolicy(DocLayoutPolicy):
    """
    Layout policy that adds a beginning-of-sequence (BOS) token as the prefix
    and an end-of-sequence (EOS) token as the suffix for each document.
    """

    def __init__(self, bos_token_id: int, eos_token_id: int):
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def prefix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 1

    def suffix_length(self, info: DocLayoutInfo) -> int:  # noqa: ARG002
        return 1

    def prefix_tokens(self, info: DocLayoutInfo) -> List[int]:  # noqa: ARG002
        return [self.bos_token_id]

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
