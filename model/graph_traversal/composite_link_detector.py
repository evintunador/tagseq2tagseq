"""
CompositeLinkDetector — per-document dispatch across all source-specific detectors.

A model trained on the merged corpus (``merged_all_v2``) has learned 11 different
link syntaxes at once: markdown hyperlinks (wiki), ``\\cite{}`` (arxiv), and the
import/use syntax of nine programming languages.  At training and at graph-edge
eval time links are known ahead of time (baked ``link_to_target`` / Option B), so
no text detection runs.  But **generation** has no graph-edge shortcut: the
generation loop (``model/generation_loop.py``) and the inference mask both detect
links from raw text.  A single configured detector (e.g. ``markdown``) fires on
only one source's syntax and finds nothing in the other ten.

``CompositeLinkDetector`` fixes this by dispatching, per document, to the correct
sub-detector.  It satisfies the full ``LinkDetector`` surface used across the
codebase:

* ``index_doc_span(span)`` — dispatch by ``span.raw_identifier`` (identifier
  sniff).  This is the *target* side of matching; the transforming detectors
  (python/java/rust/ts/js/kotlin/dart/zig) all key off a ``:``/``::``/extension
  that the identifier sniff detects reliably, so the right key transform is
  applied.  When the identifier is ambiguous (a bare wiki/arxiv title), every
  candidate detector returns the identifier unchanged, so the default identity
  key is correct.

* ``detect_links_for_doc(span_tokens, raw_identifier)`` — the per-doc path the
  mask creator prefers (selected via ``hasattr``).  Dispatch by identifier first
  (reliable), falling back to a content sniff of the span's tokens.  Routes to the
  sub-detector's own ``detect_links_for_doc`` when it has one (so relative imports
  still resolve), else its ``detect_links``.  Positions stay local to the span;
  the caller offsets by ``span.start``.

* ``detect_links(input_ids)`` — the whole-sequence / generation-loop path, where
  **no identifier is available**.  Dispatch by content sniff only.

Dispatch philosophy — pick exactly ONE sub-detector per document rather than
running all of them and merging.  Running all invites cross-firing: the TS/JS
``from "..."`` / ``require(...)`` regexes are unanchored and match arbitrary text,
and the markdown detector keys on the ``](`` token which appears in code.  Picking
one avoids that, and is cheap in the autoregressive hot loop (one decode + one
sub-detector instead of eleven).  A mis-sniff degrades gracefully: a wrong
detector's ``target_str`` almost never matches a real co-packed span key (Option-B
key mismatch across sources) nor a real corpus document, so a spurious link simply
no-ops downstream (``_match_links_to_docs`` / ``corpus.has_document``).

Not covered (known limitation): a *single generated document* that mixes link
syntaxes (e.g. a markdown doc embedding a ``\\cite{}``) is classified by its
dominant language and detected with that one sub-detector.  Per-token detector
routing within one document ("Tier 2") was deliberately deferred — no current
use case needs it, and the dominant-language assumption holds for qualitative
single-root generation.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

import torch

from .link_detector import LinkInfo, make_link_detector

logger = logging.getLogger(__name__)

# The 11 linked sources of the merged corpus.  ``null`` (edgeless) and
# ``composite`` itself are intentionally excluded.
COMPOSITE_MEMBERS = (
    "markdown", "python", "arxiv", "go", "java",
    "typescript", "javascript", "kotlin", "rust", "zig", "dart",
)

# Recognised source-code file extensions → sub-detector name.  Used by the
# identifier sniff.  Wiki/arxiv identifiers carry no extension and are handled
# by the fall-through (identity key) / content sniff instead.
_EXT_TO_NAME = {
    ".py": "python",
    ".java": "java",
    ".kt": "kotlin", ".kts": "kotlin",
    ".dart": "dart",
    ".zig": "zig",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".go": "go",
    ".rs": "rust",
}

# ---------------------------------------------------------------------------
# Content-sniff signatures.  Each entry: (compiled regex, weight).  The score
# for a language is the sum over its patterns of weight × match-count.  The
# highest-scoring language wins; ties broken by _CONTENT_PRIORITY order.
# Patterns are chosen to be as source-distinctive as possible to minimise
# cross-firing (see module docstring).
# ---------------------------------------------------------------------------
_CONTENT_SIGNATURES: Dict[str, List[tuple]] = {
    # \cite{...}, \citep{...}, \citeauthor[..]{...} — LaTeX-only, very distinctive.
    "arxiv": [(re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\{"), 3.0)],
    # @import("...") — Zig-unique sigil.
    "zig": [(re.compile(r"@import\s*\(\s*[\"']"), 3.0)],
    # import "pkg.dart"; / export 'x.dart'; — Dart directives name a .dart uri.
    "dart": [
        (re.compile(r"(?m)^\s*(?:import|export)\s+[\"'][^\"']*\.dart[\"']"), 3.0),
        (re.compile(r"(?m)^\s*library\s+\w"), 1.0),
    ],
    # use foo::bar;  +  fn — Rust module paths use ::.
    "rust": [
        (re.compile(r"(?m)^\s*(?:pub\s+)?use\s+[\w:]+::"), 2.0),
        (re.compile(r"(?m)^\s*(?:pub\s+)?fn\s+\w"), 0.5),
        (re.compile(r"(?m)^\s*mod\s+\w+\s*;"), 1.0),
    ],
    # import x.y.Z;  — Java imports terminate with a semicolon; + class/interface.
    "java": [
        (re.compile(r"(?m)^\s*import\s+(?:static\s+)?[\w.]+\s*;"), 2.0),
        (re.compile(r"\b(?:public|private|protected)\s+(?:final\s+|abstract\s+)?(?:class|interface|enum)\b"), 1.0),
    ],
    # import foo.bar (no semicolon)  +  fun — Kotlin.
    "kotlin": [
        (re.compile(r"(?m)^\s*import\s+[\w.]+\s*(?:as\s+\w+\s*)?$"), 1.5),
        (re.compile(r"(?m)^\s*(?:public\s+|private\s+|internal\s+)?fun\s+\w"), 0.6),
    ],
    # import ( ... )  / import "pkg"  +  func / package — Go.
    "go": [
        (re.compile(r"(?m)^\s*import\s+(?:\(|[\"'])"), 2.0),
        (re.compile(r"(?m)^\s*func\s+\w"), 0.5),
        (re.compile(r"(?m)^\s*package\s+\w"), 1.0),
    ],
    # from x.y import z / import a.b  +  def — Python (dotted, bare names).
    "python": [
        (re.compile(r"(?m)^\s*from\s+[.\w]+\s+import\s"), 2.0),
        (re.compile(r"(?m)^\s*import\s+[\w.]+(?:\s*,\s*[\w.]+)*\s*(?:#.*)?$"), 1.0),
        (re.compile(r"(?m)^\s*def\s+\w+\s*\("), 0.5),
    ],
    # ES-module imports: `... from "x"`, `require("x")`, `import(...)`.  Shared by
    # TypeScript and JavaScript; the TS/JS split is decided separately below.
    "_esmodule": [
        (re.compile(r"\b(?:import|export)\b[^;\n]*\bfrom\s*[\"']"), 2.0),
        (re.compile(r"\brequire\s*\(\s*[\"']"), 2.0),
        (re.compile(r"(?m)^\s*import\s*[\"']"), 1.5),
        (re.compile(r"\bimport\s*\("), 1.0),
    ],
    # [text](target) — markdown hyperlink.  Low weight per link so a stray ](
    # in code never outvotes real import structure.
    "markdown": [
        (re.compile(r"\[[^\]\n]+\]\([^)\n]+\)"), 1.0),
        (re.compile(r"(?m)^#{1,6}\s+\S"), 0.3),
    ],
}

# TypeScript-only markers — presence tips an _esmodule win toward TS over JS.
# (TS is a superset of JS and the two detectors are near-identical, so the split
# rarely changes the resolved links; this just picks the closer-fit detector.)
_TS_MARKERS = re.compile(
    r"\binterface\s+\w+|\btype\s+\w+\s*=|\benum\s+\w+|:\s*(?:string|number|boolean)\b|\bimplements\s+\w"
)

# Tie-break order when scores are equal: most-distinctive syntaxes first, prose
# (markdown) last.  ``_esmodule`` resolves to typescript/javascript before this.
_CONTENT_PRIORITY = (
    "arxiv", "zig", "dart", "rust", "go", "java", "kotlin",
    "python", "typescript", "javascript", "markdown",
)


def _strip_repo_prefix(raw_identifier: str) -> str:
    """Return the path part of ``repo:path`` / ``owner/repo@modpath`` identifiers.

    Python/Java/TS/JS/Kotlin/Dart/Zig store ``<repo>:<path>``; Rust stores
    ``<owner/repo>@<crate::mod::path>``.  Wiki/arxiv/go carry no such separator
    and are returned unchanged.
    """
    if "@" in raw_identifier and "::" in raw_identifier:
        # Rust: repo tag is before the first '@', module path (with ::) after.
        return raw_identifier.split("@", 1)[1]
    if ":" in raw_identifier:
        return raw_identifier.split(":", 1)[1]
    return raw_identifier


def _sniff_by_identifier(raw_identifier: str) -> Optional[str]:
    """Pick a sub-detector name from a document identifier, or None if ambiguous.

    Reliable for exactly the detectors whose ``index_doc_span`` transforms the
    key (they key off ``:``/``::``/extension).  Returns None for bare titles
    (wiki/arxiv) and other ambiguous shapes, leaving the caller to fall back to
    a content sniff or the identity key.
    """
    if not raw_identifier:
        return None
    # Rust module paths use '::' (with an optional 'owner/repo@' prefix).
    if "::" in raw_identifier:
        return "rust"
    path = _strip_repo_prefix(raw_identifier)
    # Extension match (case-insensitive) — the strongest identifier signal.
    dot = path.rfind(".")
    if dot != -1:
        ext = path[dot:].lower()
        name = _EXT_TO_NAME.get(ext)
        if name is not None:
            return name
    # Go import paths look like a hostname-rooted path: 'github.com/owner/repo/pkg'
    # (no ':' prefix survived, no recognised extension, contains a dotted host).
    if ":" not in raw_identifier and "/" in path:
        head = path.split("/", 1)[0]
        if "." in head and " " not in raw_identifier:  # 'github.com', 'gopkg.in', ...
            return "go"
    return None  # ambiguous → content sniff / identity fallback


def _sniff_by_content(text: str) -> Optional[str]:
    """Pick a sub-detector name from document text, or None if nothing matches.

    Scores each language's distinctive syntax markers and returns the argmax
    (ties broken by ``_CONTENT_PRIORITY``).  Returns None when no marker fires
    at all — the correct outcome for plain prose with no links.
    """
    if not text:
        return None
    scores: Dict[str, float] = {}
    for name, patterns in _CONTENT_SIGNATURES.items():
        s = 0.0
        for rx, weight in patterns:
            n = len(rx.findall(text))
            if n:
                s += weight * n
        if s > 0.0:
            scores[name] = s
    if not scores:
        return None

    # Resolve the shared ES-module score to a concrete TS/JS detector.
    if "_esmodule" in scores:
        scores["typescript" if _TS_MARKERS.search(text) else "javascript"] = scores.pop("_esmodule")

    best = max(scores.values())
    winners = [n for n, v in scores.items() if v == best]
    if len(winners) == 1:
        return winners[0]
    for name in _CONTENT_PRIORITY:
        if name in winners:
            return name
    return winners[0]


class CompositeLinkDetector:
    """Dispatches link detection to a per-document source-specific sub-detector.

    Implements the ``LinkDetector`` protocol plus the optional
    ``detect_links_for_doc`` per-doc path.  Constructs one instance of each of
    the 11 merged-corpus sub-detectors (via ``make_link_detector``) up front;
    all are stateless regex/token matchers, so this is cheap.

    Args:
        decode_fn: Token-ids → str callable (typically ``tiktoken_enc.decode``),
            forwarded to every sub-detector.
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn
        self._subs: Dict[str, Any] = {
            name: make_link_detector(name, decode_fn) for name in COMPOSITE_MEMBERS
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_for_sniff(self, tokens_list: List[int]) -> str:
        """Decode tokens to text for the content sniff, swallowing decode errors."""
        try:
            return self.decode_fn(tokens_list)
        except Exception:  # pragma: no cover — decode should not fail in practice
            return ""

    def _resolve_for_doc(self, span_tokens: torch.Tensor, raw_identifier: str) -> Optional[str]:
        """Choose a sub-detector for a document: identifier first, then content."""
        name = _sniff_by_identifier(raw_identifier)
        if name is not None:
            return name
        return _sniff_by_content(self._decode_for_sniff(span_tokens.tolist()))

    # ------------------------------------------------------------------
    # LinkDetector protocol
    # ------------------------------------------------------------------

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """Detect links in a 1-D token sequence with NO identifier context.

        This is the generation-loop / whole-sequence path.  The language is
        inferred from content alone; the chosen sub-detector's ``detect_links``
        runs on the full sequence.  Returns ``[]`` when no source syntax is
        detected (correct for link-free prose).
        """
        name = _sniff_by_content(self._decode_for_sniff(input_ids.tolist()))
        if name is None:
            return []
        return self._subs[name].detect_links(input_ids)

    def detect_links_for_doc(
        self,
        span_tokens: torch.Tensor,
        raw_identifier: str,
    ) -> List[LinkInfo]:
        """Detect links for a single document span (mask creator's per-doc path).

        Dispatch by ``raw_identifier`` first, then a content sniff of the span.
        Routes to the sub-detector's own ``detect_links_for_doc`` when present
        (preserving relative-import resolution), else its ``detect_links``.
        Positions are local to the span; the caller offsets by ``span.start``.
        """
        name = self._resolve_for_doc(span_tokens, raw_identifier)
        if name is None:
            return []
        sub = self._subs[name]
        if hasattr(sub, "detect_links_for_doc"):
            return sub.detect_links_for_doc(span_tokens, raw_identifier)
        return sub.detect_links(span_tokens)

    def index_doc_span(self, span: Any) -> str:
        """Return the target-matching key for a span, via its source's detector.

        Dispatch by ``span.raw_identifier``.  When the identifier is ambiguous
        (a bare wiki/arxiv title), fall back to the identity key
        (``raw_identifier``) — which is exactly what markdown/arxiv/go and a
        prefix-less kotlin title all return anyway, so the key is correct.
        """
        name = _sniff_by_identifier(span.raw_identifier)
        if name is None:
            return span.raw_identifier
        return self._subs[name].index_doc_span(span)
