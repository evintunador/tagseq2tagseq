"""
Python Import Statement Link Detector

Detects Python ``import`` and ``from ... import`` statements in tokenized source
code sequences and converts the module paths to candidate file paths for matching
against document identifiers in the packed batch.

Design
------
Detection works in three stages:

1. **Decode once**: the full token sequence is decoded to a string in a single
   ``decode_fn`` call.
2. **Regex parse**: ``_parse_imports`` finds all import statements via two
   compiled patterns (one for plain ``import``, one for ``from ... import``
   including multi-line parenthesised forms).
3. **Char → token mapping**: a cumulative character-length index built by
   decoding each token individually maps regex character offsets back to token
   positions.  For Python source (nearly all ASCII) this is exact; for the rare
   UTF-8 edge case the position may be off by one token, which is acceptable.

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

# "from foo.bar import name1, name2" — single-line, no parentheses.
# Group 1: module path.  Group 2: imported names or "*".
_FROM_IMPORT_INLINE_RE = re.compile(
    r"^[ \t]*from\s+([\w.]+)\s+import\s+"
    r"(\*|\w+(?:\s*,\s*\w+)*)[ \t]*(?:#.*)?$",
    re.MULTILINE,
)

# "from foo.bar import (\n    name1,\n    name2,\n)" — parenthesised, may span lines.
# Group 1: module path.  Group 2: contents of the parentheses (names + whitespace).
_FROM_IMPORT_PAREN_RE = re.compile(
    r"^[ \t]*from\s+([\w.]+)\s+import\s+\(([^)]*)\)",
    re.MULTILINE | re.DOTALL,
)


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
    the ``from … import`` patterns are checked.
    """
    results: List[Tuple[str, str, int, int]] = []

    # --- "from .foo import name1, name2" (single-line) ---
    for m in _FROM_IMPORT_INLINE_RE.finditer(text):
        module_path = m.group(1)
        if not module_path.startswith("."):
            continue
        names_str = m.group(2).strip()
        if names_str == "*":
            results.append((module_path, "*", m.start(), m.end()))
        else:
            for name in names_str.split(","):
                name = name.strip()
                if name:
                    results.append((module_path, name, m.start(), m.end()))

    # --- "from .foo import (\n    name1,\n    name2\n)" ---
    for m in _FROM_IMPORT_PAREN_RE.finditer(text):
        module_path = m.group(1)
        if not module_path.startswith("."):
            continue
        clean_lines = [
            line.split("#")[0].rstrip("\\")
            for line in m.group(2).split("\n")
        ]
        for name in " ".join(clean_lines).split(","):
            name = name.strip()
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
    """
    results: List[Tuple[str, str, int, int]] = []

    # --- "import foo.bar [as x], baz.qux [as y]" ---
    for m in _SIMPLE_IMPORT_RE.finditer(text):
        for item in m.group(1).split(","):
            item = item.strip()
            module_path = item.split(" as ")[0].strip()
            if module_path and not module_path.startswith("."):
                results.append((module_path, "", m.start(), m.end()))

    # --- "from foo.bar import name1, name2" (single-line) ---
    for m in _FROM_IMPORT_INLINE_RE.finditer(text):
        module_path = m.group(1)
        if module_path.startswith("."):
            continue
        names_str = m.group(2).strip()
        if names_str == "*":
            results.append((module_path, "*", m.start(), m.end()))
        else:
            for name in names_str.split(","):
                name = name.strip()
                if name:
                    results.append((module_path, name, m.start(), m.end()))

    # --- "from foo.bar import (\n    name1,\n    name2\n)" ---
    for m in _FROM_IMPORT_PAREN_RE.finditer(text):
        module_path = m.group(1)
        if module_path.startswith("."):
            continue
        # Strip inline comments line-by-line BEFORE comma-splitting so that
        # a comment like ``bar,  # the bar module\n    baz`` doesn't absorb
        # names on the lines that follow.
        clean_lines = [
            line.split("#")[0].rstrip("\\")
            for line in m.group(2).split("\n")
        ]
        for name in " ".join(clean_lines).split(","):
            name = name.strip()
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

        Each token is decoded individually to get its character contribution.
        For pure-ASCII source (the common case for Python code) this is exact.
        For the rare multi-byte UTF-8 token the count may be off by a character
        or two, which is acceptable for ``link_end_pos`` precision.
        """
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
