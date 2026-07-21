"""
Python Import Statement Link Detector

Detects Python ``import`` and ``from ... import`` statements in tokenized source
code sequences and converts the module paths to candidate file paths for matching
against document identifiers in the packed batch.

The *build-time* import extractor (``data/github_graph_extractor/extract.py``)
uses a tree-sitter grammar to enumerate import statements when constructing the
shipped graph. This runtime detector re-detects links from *tokens* at
train/inference time and — like the Go/Rust/TypeScript runtime detectors — works
in token space with a comment/string-blanking pass plus compiled regexes (a
tree-sitter parse per packed sequence would be far too slow in the training hot
path). The two are graded for agreement against a THIRD, independent tree-sitter
oracle (``data/graph_harness``).

Design
------
Detection works in four stages:

1. **Decode once**: the full token sequence is decoded to a string in a single
   ``decode_fn`` call.
2. **Blank comments/strings**: ``_blank_comments_and_strings`` overwrites the
   bodies of ``#`` comments and string literals (including triple-quoted
   docstrings) with equal-length spaces, preserving every character offset. This
   mirrors the Go/TS detectors and matches the tree-sitter oracle, which never
   treats an ``import`` written inside a docstring or string as a real import.
3. **Regex parse**: ``_parse_imports`` finds all import statements via two
   compiled patterns (one for plain ``import``, one for ``from ... import``
   including multi-line parenthesised forms). ``as`` aliases are stripped from
   both the module (``import x as y`` -> ``x``) and the imported names
   (``from x import y as z`` -> name ``y``).
4. **Char -> token mapping**: a cumulative character-length index (built with a
   single batch ``decode_tokens_bytes`` call when available, else per-token) maps
   regex character offsets back to token positions.  For Python source (nearly
   all ASCII) this is exact; for the rare UTF-8 edge case the position may be off
   by one token, which is acceptable.

Relative imports
----------------
``detect_links`` operates on a flat packed sequence with no knowledge of which
document each token belongs to, so relative imports are skipped there.

``detect_links_for_doc`` is the per-document variant: it receives a single
document's token span plus its ``raw_identifier`` (which encodes the file path),
enabling full relative-import resolution via ``_parse_relative_imports`` and
``_resolve_relative_import``.  ``CrossDocLinkMaskCreator`` uses this method when
available (detected via ``hasattr``), looping over each ``DocSpan`` and offsetting
the returned local positions back to global packed-sequence coordinates.

Limitations
-----------
- **Dynamic / conditional imports** (``__import__``, ``importlib.import_module``,
  imports inside ``if TYPE_CHECKING:`` blocks, etc.) are not detected.
- The module-to-file mapping assumes the repo root equals the Python path root.
  Packages installed into site-packages or manipulated via ``sys.path`` will not
  resolve correctly, but those files are unlikely to be co-located in the same
  batch anyway.

A ``from foo.bar import baz`` statement generates candidate paths for *baz* as
both a submodule (``foo/bar/baz.py``) and a symbol in the parent module
(``foo/bar.py``), because we cannot tell at parse time which case applies.
"""

import bisect
import logging
import re
from typing import Any, Callable, List, Tuple

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "import foo.bar.baz [as alias]" — optionally comma-separated.
# Captures the entire import list as group 1; individual names are split later.
_SIMPLE_IMPORT_RE = re.compile(
    r"^[ \t]*import\s+"
    r"([\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*)[ \t]*(?:#.*)?$",
    re.MULTILINE,
)

# "from foo.bar import name1, name2 [as alias]" — single-line, no parentheses.
# Group 1: module path.  Group 2: imported names (each optionally ``as alias``)
# or "*".  Aliases are stripped from the names later by ``_strip_alias``.
_FROM_IMPORT_INLINE_RE = re.compile(
    r"^[ \t]*from\s+([\w.]+)\s+import\s+"
    r"(\*|\w+(?:\s+as\s+\w+)?(?:\s*,\s*\w+(?:\s+as\s+\w+)?)*)[ \t]*(?:#.*)?$",
    re.MULTILINE,
)

# "from foo.bar import (\n    name1,\n    name2 as alias,\n)" — parenthesised,
# may span lines.  Group 1: module path.  Group 2: contents of the parentheses
# (names + optional ``as`` aliases + whitespace).
_FROM_IMPORT_PAREN_RE = re.compile(
    r"^[ \t]*from\s+([\w.]+)\s+import\s+\(([^)]*)\)",
    re.MULTILINE | re.DOTALL,
)


def _strip_alias(name: str) -> str:
    """Return the imported name with any ``as <alias>`` suffix removed.

    ``'y as z'`` -> ``'y'``; ``'y'`` -> ``'y'``.  Splitting on whitespace and
    taking the first token is robust to arbitrary spacing around ``as``.  Returns
    ``''`` for an all-whitespace input (e.g. a trailing-comma artifact).
    """
    parts = name.split()
    return parts[0] if parts else ""


# ---------------------------------------------------------------------------
# Comment / string blanking
# ---------------------------------------------------------------------------


def _blank_comments_and_strings(text: str) -> str:
    """Blank ``#`` comments and string-literal bodies to equal-length spaces.

    Newlines are preserved (kept as ``\\n``) so line structure — and therefore
    every character offset used for ``link_end_pos`` — is unchanged, and the
    ``re.MULTILINE`` ``^`` anchors still see the true line boundaries.

    This mirrors the Go/Rust/TypeScript runtime detectors and the tree-sitter
    oracle: an ``import`` statement written inside a docstring, a comment, or any
    string literal is NOT real code and must not be detected.  Unlike Go/TS —
    where the import *specifier* is itself a quoted string — a Python import
    target is always a bare identifier, so every string body can be blanked
    without erasing anything the parser needs.

    A hand-written scanner (not a regex) is used so that ``#`` inside a string
    and quotes inside a comment are handled correctly, and so triple-quoted
    strings (docstrings) are treated as a single span.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # --- line comment ---
        if c == "#":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        # --- string literal (single/double, optionally triple-quoted) ---
        if c == '"' or c == "'":
            quote = c
            # triple-quoted?
            if i + 2 < n and text[i + 1] == quote and text[i + 2] == quote:
                out[i] = out[i + 1] = out[i + 2] = " "
                i += 3
                while i < n:
                    if text[i] == "\\":  # escape — blank the pair
                        out[i] = " "
                        if i + 1 < n and text[i + 1] != "\n":
                            out[i + 1] = " "
                        i += 2
                        continue
                    if (i + 2 < n and text[i] == quote
                            and text[i + 1] == quote and text[i + 2] == quote):
                        out[i] = out[i + 1] = out[i + 2] = " "
                        i += 3
                        break
                    if text[i] != "\n":
                        out[i] = " "
                    i += 1
                continue
            # single-quoted (ends at matching quote or newline)
            out[i] = " "
            i += 1
            while i < n:
                if text[i] == "\\":
                    out[i] = " "
                    if i + 1 < n and text[i + 1] != "\n":
                        out[i + 1] = " "
                    i += 2
                    continue
                if text[i] == quote:
                    out[i] = " "
                    i += 1
                    break
                if text[i] == "\n":
                    break
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def module_path_to_file_paths(module_path: str, from_name: str = "") -> List[str]:
    """
    Convert a dotted Python module path to candidate relative file paths.

    Returns paths in order of specificity (most specific first).  All paths
    assume the repository root as the filesystem root.

    Args:
        module_path: Dotted module path, e.g. ``'foo.bar.baz'``.
        from_name:   The imported name from a ``from X import Y`` statement.
                     Pass ``''`` or ``'*'`` for plain ``import X`` or star
                     imports; in both cases only the module itself is a
                     candidate.

    Returns:
        List of candidate file paths.

    Examples::

        >>> module_path_to_file_paths('foo.bar.baz')
        ['foo/bar/baz.py', 'foo/bar/baz/__init__.py']

        >>> module_path_to_file_paths('foo.bar', 'baz')
        ['foo/bar/baz.py', 'foo/bar/baz/__init__.py', 'foo/bar.py', 'foo/bar/__init__.py']

        >>> module_path_to_file_paths('foo.bar', '*')
        ['foo/bar.py', 'foo/bar/__init__.py']
    """
    base = module_path.replace(".", "/")
    module_candidates = [f"{base}.py", f"{base}/__init__.py"]

    if from_name and from_name != "*":
        # ``from foo.bar import baz``: baz may be a submodule of foo.bar
        sub = f"{base}/{from_name}"
        return [f"{sub}.py", f"{sub}/__init__.py"] + module_candidates

    return module_candidates


def _parse_relative_imports(text: str) -> List[Tuple[str, str, int, int]]:
    """
    Find all *relative* Python import statements in *text*.

    Same return format as ``_parse_imports`` — ``(module_path, from_name,
    char_start, char_end)`` — but only entries whose ``module_path`` starts
    with ``'.'``.  Plain ``import .foo`` is a syntax error in Python so only
    the ``from … import`` patterns are checked.  ``as`` aliases on the imported
    names are stripped; comments/strings are blanked first.
    """
    text = _blank_comments_and_strings(text)
    results: List[Tuple[str, str, int, int]] = []

    # --- "from .foo import name1, name2 [as alias]" (single-line) ---
    for m in _FROM_IMPORT_INLINE_RE.finditer(text):
        module_path = m.group(1)
        if not module_path.startswith("."):
            continue
        names_str = m.group(2).strip()
        if names_str == "*":
            results.append((module_path, "*", m.start(), m.end()))
        else:
            for name in names_str.split(","):
                name = _strip_alias(name)
                if name:
                    results.append((module_path, name, m.start(), m.end()))

    # --- "from .foo import (\n    name1,\n    name2 as alias\n)" ---
    for m in _FROM_IMPORT_PAREN_RE.finditer(text):
        module_path = m.group(1)
        if not module_path.startswith("."):
            continue
        clean_lines = [
            line.split("#")[0].rstrip("\\")
            for line in m.group(2).split("\n")
        ]
        for name in " ".join(clean_lines).split(","):
            name = _strip_alias(name)
            if name:
                results.append((module_path, name, m.start(), m.end()))

    return results


def _resolve_relative_import(
    module_path: str,
    from_name: str,
    source_file_path: str,
) -> List[str]:
    """
    Resolve a single relative import to candidate absolute file paths.

    Args:
        module_path:      Dotted module path starting with one or more ``'.'``
                          characters, e.g. ``'.'``, ``'.utils'``, ``'..models'``.
        from_name:        Imported name (``'*'`` or ``''`` → no submodule candidates).
        source_file_path: Repo-relative path of the importing file, e.g.
                          ``'pkg/sub/mod.py'``.  Both ``/`` and ``\\`` separators
                          are accepted.

    Returns:
        Candidate file paths (most-specific first), or ``[]`` when the import
        walks above the repo root or ``from_name`` is ``'*'`` / ``''``.

    Examples::

        >>> _resolve_relative_import('.', 'utils', 'pkg/sub/mod.py')
        ['pkg/sub/utils.py', 'pkg/sub/utils/__init__.py']

        >>> _resolve_relative_import('..', 'models', 'pkg/sub/mod.py')
        ['pkg/models.py', 'pkg/models/__init__.py']

        >>> _resolve_relative_import('.schema', 'User', 'pkg/sub/mod.py')
        ['pkg/sub/schema/User.py', 'pkg/sub/schema/User/__init__.py',
         'pkg/sub/schema.py', 'pkg/sub/schema/__init__.py']
    """
    dot_count = len(module_path) - len(module_path.lstrip("."))
    base_module = module_path[dot_count:]  # '' for '.', 'utils' for '.utils'

    # Start from the directory containing the source file.
    dir_parts = source_file_path.replace("\\", "/").split("/")[:-1]

    # Walk up (dot_count - 1) levels.  A single dot means "current package"
    # (the directory itself), so dot_count=1 requires no upward traversal.
    for _ in range(dot_count - 1):
        if not dir_parts:
            return []  # import walks above repo root — unresolvable
        dir_parts.pop()

    if base_module:
        resolved_parts = dir_parts + base_module.split(".")
    else:
        # "from . import X" — X lives directly in the current package directory.
        resolved_parts = dir_parts

    if not resolved_parts:
        # Root-level file with "from . import X": X is at the repo root.
        if from_name and from_name != "*":
            return [f"{from_name}.py", f"{from_name}/__init__.py"]
        return []

    return module_path_to_file_paths(".".join(resolved_parts), from_name)


def _parse_imports(text: str) -> List[Tuple[str, str, int, int]]:
    """
    Find all Python import statements in *text*.

    Returns a list of ``(module_path, from_name, char_start, char_end)`` tuples:

    - ``module_path``: dotted module path string (e.g. ``'os.path'``).
    - ``from_name``:   imported name for ``from X import Y``, ``'*'`` for
                       star imports, ``''`` for plain ``import X``.
    - ``char_start``:  character offset of the start of the statement.
    - ``char_end``:    character offset just past the end of the statement.

    Relative imports (``module_path`` starting with ``'.'``) are skipped.
    Multiple entries with the same ``(char_start, char_end)`` are emitted when
    a single statement imports several names
    (e.g. ``from foo import bar, baz``).

    Comments and string literals are blanked before matching, so ``import``
    statements appearing inside docstrings/strings/comments are NOT detected.
    Character offsets in the returned tuples index into the ORIGINAL ``text``
    (blanking preserves every offset).
    """
    text = _blank_comments_and_strings(text)
    results: List[Tuple[str, str, int, int]] = []

    # --- "import foo.bar [as x], baz.qux [as y]" ---
    for m in _SIMPLE_IMPORT_RE.finditer(text):
        for item in m.group(1).split(","):
            item = item.strip()
            module_path = item.split(" as ")[0].strip()
            if module_path and not module_path.startswith("."):
                results.append((module_path, "", m.start(), m.end()))

    # --- "from foo.bar import name1, name2 [as alias]" (single-line) ---
    for m in _FROM_IMPORT_INLINE_RE.finditer(text):
        module_path = m.group(1)
        if module_path.startswith("."):
            continue
        names_str = m.group(2).strip()
        if names_str == "*":
            results.append((module_path, "*", m.start(), m.end()))
        else:
            for name in names_str.split(","):
                name = _strip_alias(name)
                if name:
                    results.append((module_path, name, m.start(), m.end()))

    # --- "from foo.bar import (\n    name1,\n    name2 as alias\n)" ---
    for m in _FROM_IMPORT_PAREN_RE.finditer(text):
        module_path = m.group(1)
        if module_path.startswith("."):
            continue
        # Comments are already blanked upstream, but keep the ``#`` / line-cont
        # strip as belt-and-braces before comma-splitting.
        clean_lines = [
            line.split("#")[0].rstrip("\\")
            for line in m.group(2).split("\n")
        ]
        for name in " ".join(clean_lines).split(","):
            name = _strip_alias(name)
            if name:
                results.append((module_path, name, m.start(), m.end()))

    return results


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class PythonImportDetector:
    """
    Detects Python import statements in tokenized source code sequences.

    Implements the ``LinkDetector`` protocol from ``cross_doc_mask``.

    Converts detected imports to lists of candidate file paths.  Multiple
    ``LinkInfo`` objects may share the same ``link_end_pos`` when a single
    import yields several candidates (e.g. ``foo/bar.py`` and
    ``foo/bar/__init__.py`` for ``import foo.bar``).

    Matching against ``DocSpan`` identifiers uses the path component only
    (everything after ``':'`` in the ``raw_identifier``), so the repo-prefix part
    of the identifier is ignored during lookup.

    Args:
        decode_fn: Callable mapping ``List[int]`` → ``str``.
                   Typically ``tiktoken_enc.decode``.
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn
        # Fast path for the char->token index: tiktoken exposes
        # ``decode_tokens_bytes`` on the bound method's ``__self__`` encoder,
        # which returns per-token byte strings in ONE call instead of one
        # ``decode_fn`` call per token (O(tokens) -> O(1) Python round-trips,
        # ~2x faster on large files). Detected once here; falls back cleanly.
        enc = getattr(decode_fn, "__self__", None)
        batch = getattr(enc, "decode_tokens_bytes", None)
        self._decode_tokens_bytes = batch if callable(batch) else None

    # ------------------------------------------------------------------
    # LinkDetector protocol
    # ------------------------------------------------------------------

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """
        Detect Python import statements in *input_ids* and emit candidate
        file-path links.

        The full sequence is decoded once; a per-token character-length index
        is built to map regex character offsets back to token positions.

        Args:
            input_ids: 1-D token-ID tensor of shape ``[seq_len]``.

        Returns:
            List of ``LinkInfo`` objects.  Multiple entries may share the same
            ``link_end_pos`` when one import yields several candidate paths.
        """
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        raw_imports = _parse_imports(full_text)
        logger.debug(
            f"PythonImportDetector: parsed {len(raw_imports)} raw import entries "
            f"from sequence of length {len(tokens)}"
        )

        links: List[LinkInfo] = []
        for module_path, from_name, _char_start, char_end in raw_imports:
            link_end_pos = self._char_pos_to_token_pos(cumulative, char_end)
            for file_path in module_path_to_file_paths(module_path, from_name):
                links.append(LinkInfo(link_end_pos=link_end_pos, target_str=file_path))

        logger.debug(
            f"PythonImportDetector: produced {len(links)} LinkInfos "
            f"({len(raw_imports)} import entries × avg candidates)"
        )
        return links

    def detect_links_for_doc(
        self,
        span_tokens: torch.Tensor,
        raw_identifier: str,
    ) -> List[LinkInfo]:
        """
        Detect import links for a *single* document span.

        Unlike ``detect_links``, which operates on a full packed sequence with
        no per-document context, this method also resolves relative imports
        (``from . import foo``, ``from ..models import User``, etc.) by
        extracting the source file path from ``raw_identifier``.

        Args:
            span_tokens:    1-D token-ID tensor for this document only
                            (``tokens[span.start:span.end]``).
            raw_identifier: The span's ``raw_identifier``, e.g.
                            ``'owner/repo:pkg/sub/mod.py'``.

        Returns:
            List of ``LinkInfo`` objects with positions **local to the span**
            (i.e. offset 0 = first token of this span).  The caller is
            responsible for adding ``span.start`` to convert to global
            packed-sequence coordinates.
        """
        tokens = span_tokens.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        links: List[LinkInfo] = []

        # Absolute imports — same as detect_links.
        for module_path, from_name, _char_start, char_end in _parse_imports(full_text):
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            for fp in module_path_to_file_paths(module_path, from_name):
                links.append(LinkInfo(link_end_pos=pos, target_str=fp))

        # Relative imports — resolved using the source file path.
        source_file_path = (
            raw_identifier.split(":", 1)[1] if ":" in raw_identifier else raw_identifier
        )
        for module_path, from_name, _char_start, char_end in _parse_relative_imports(full_text):
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            for fp in _resolve_relative_import(module_path, from_name, source_file_path):
                links.append(LinkInfo(link_end_pos=pos, target_str=fp))

        logger.debug(
            "PythonImportDetector.detect_links_for_doc: %d LinkInfos for %r",
            len(links),
            raw_identifier,
        )
        return links

    def index_doc_span(self, span: Any) -> str:
        """
        Return the path component of a span's ``raw_identifier`` for matching.

        For Stack identifiers of the form ``'repo/name:path/to/file.py'``
        this returns ``'path/to/file.py'``.  For titles without ``':'``
        the full ``raw_identifier`` is returned as a fallback.
        """
        parts = span.raw_identifier.split(":", 1)
        return parts[1] if len(parts) > 1 else span.raw_identifier

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_char_to_token_index(self, tokens: List[int]) -> List[int]:
        """
        Build a cumulative character-count index for O(log N) char→token lookup.

        ``cumulative[i]`` = total number of characters in
        ``decode_fn(tokens[:i])``, with ``cumulative[0] = 0``.

        Each token's character contribution is summed.  When the encoder exposes
        ``decode_tokens_bytes`` (tiktoken) all per-token byte strings are fetched
        in a single call and decoded individually to chars — ~2x faster than one
        ``decode_fn`` call per token on large files, with identical results.
        For pure-ASCII source (the common case for Python code) this is exact.
        For the rare multi-byte UTF-8 token the count may be off by a character
        or two, which is acceptable for ``link_end_pos`` precision.
        """
        cumulative = [0] * (len(tokens) + 1)
        if self._decode_tokens_bytes is not None:
            try:
                per_token = self._decode_tokens_bytes(tokens)
                for i, b in enumerate(per_token):
                    cumulative[i + 1] = cumulative[i] + len(b.decode("utf-8", "replace"))
                return cumulative
            except Exception:
                # Fall through to the safe per-token path on any surprise.
                cumulative = [0] * (len(tokens) + 1)
        for i, tok in enumerate(tokens):
            try:
                char_len = len(self.decode_fn([tok]))
            except Exception:
                char_len = 1  # safe fallback: assume 1 char
            cumulative[i + 1] = cumulative[i] + char_len
        return cumulative

    def _char_pos_to_token_pos(self, cumulative: List[int], char_pos: int) -> int:
        """
        Return the smallest token index *t* such that
        ``cumulative[t] >= char_pos``.

        Uses ``bisect_left`` for O(log N) lookup.  The result is clamped to
        ``[0, len(tokens)]``.
        """
        idx = bisect.bisect_left(cumulative, char_pos)
        return max(0, min(idx, len(cumulative) - 1))
