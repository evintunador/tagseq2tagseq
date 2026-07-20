"""
Kotlin Import Statement Link Detector

Detects Kotlin ``import`` declarations in tokenized source and emits the imported
symbol's fully-qualified name (FQN) as the link target.

Node model — CRITICAL difference from Java
-------------------------------------------
Kotlin is FQN/JVM-family like Java (import strings are globally unique, so the
corpus is multi-repo capable), BUT:

  * a Kotlin filename does NOT determine the class/symbol name, and
  * ONE ``.kt`` file can declare MANY top-level symbols (classes, objects,
    interfaces, top-level funcs/vals/vars, typealiases), each with FQN
    ``<package>.<SymbolName>``.

So ``import com.ex.util.Helper`` names a DECLARATION, not a file — you cannot map
filename -> FQN like Java. Resolution uses a SYMBOL -> FILE index built by the
extractor (data/kotlin_graph_extractor/build_kotlin_graph.py) and the fixture node
builder (data/graph_harness/kotlin_nodes.py): the node unit is ONE NODE PER
DECLARED FQN (a multi-symbol file contributes several nodes sharing the file
content). That keeps the frozen resolver's one-key-per-node contract intact — an
imported FQN resolves by EXACT string match to the node keyed by that FQN.

Forms handled:
    import com.ex.util.Helper        -> com.ex.util.Helper
    import com.ex.util.helperFn      -> com.ex.util.helperFn  (top-level func/prop)
    import com.ex.foo.bar as Baz     -> com.ex.foo.bar        (alias STRIPPED)
    import com.ex.*                  -> (wildcard/on-demand: DROPPED, no target)
    import kotlin.math.max           -> kotlin.math.max       (stdlib; won't resolve)

Kotlin has no submodule-vs-symbol ambiguity (an import names exactly one symbol),
so one non-wildcard import emits exactly one dotted-FQN candidate. Wildcard imports
have no single target and are dropped (the oracle agrees — see kotlin_spec.py).

``index_doc_span`` returns ``raw_identifier`` unchanged: a Kotlin node's
``raw_identifier`` IS the FQN it declares (build_kotlin_file_nodes /
build_kotlin_graph key nodes by FQN), so matching an emitted target FQN is exact
string equality (like Go).

Detection mechanics mirror GoImportDetector: decode once, blank comments/strings
(so imports inside comments aren't matched), regex the imports, map char offsets to
token positions via a cumulative per-token char index.
"""
from __future__ import annotations

import bisect
import logging
import re
from typing import Any, Callable, List, Tuple

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)

# import <dotted.fqn>[.*] [as Alias]
#   fqn:   dotted identifier path (whitespace around dots tolerated defensively)
#   star:  optional trailing ".*" -> wildcard/on-demand (dropped)
#   alias: optional "as Alias" (stripped; we keep only the fqn)
_IMPORT_RE = re.compile(
    r"^[ \t]*import[ \t]+"
    r"(?P<fqn>[A-Za-z_`][\w`]*(?:\s*\.\s*[A-Za-z_`][\w`]*)*)"
    r"(?P<star>\s*\.\s*\*)?"
    r"(?:[ \t]+as[ \t]+[A-Za-z_`][\w`]*)?"
    r"[ \t]*(?:;|$)",
    re.MULTILINE,
)


def _blank_comments(text: str) -> str:
    """Replace Kotlin comments with equal-length spaces (preserving char offsets).

    ``import`` lines that appear inside ``//`` line comments or ``/* */`` block
    comments (e.g. usage examples in KDoc) are not real code and must not be
    matched — tree-sitter ignores them, so the token-space detector must too.
    Comments are blanked to spaces (not deleted) so downstream char offsets — and
    thus ``link_end_pos`` — are unchanged.

    A hand-written scanner (not regex) is used so comment markers INSIDE string
    literals are not mistaken for comment starts, and quotes inside comments are
    not mistaken for strings. Kotlin block comments nest, so a depth counter is
    kept. Kotlin has raw triple-quoted strings (\"\"\" ... \"\"\") handled first.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # raw triple-quoted string: """ ... """ (no escapes)
        if c == '"' and text[i + 1:i + 3] == '""':
            i += 3
            while i < n and text[i:i + 3] != '"""':
                i += 1
            i += 3
            continue
        # regular string literal
        if c == '"':
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        # char literal 'x'
        if c == "'":
            i += 1
            while i < n and text[i] != "'":
                if text[i] == "\\":
                    i += 1
                i += 1
            i += 1
            continue
        # line comment
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        # block comment (nesting)
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            depth = 1
            out[i] = out[i + 1] = " "
            i += 2
            while i < n and depth > 0:
                if text[i] == "/" and i + 1 < n and text[i + 1] == "*":
                    depth += 1
                    out[i] = out[i + 1] = " "
                    i += 2
                    continue
                if text[i] == "*" and i + 1 < n and text[i + 1] == "/":
                    depth -= 1
                    out[i] = out[i + 1] = " "
                    i += 2
                    continue
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def _parse_imports(text: str) -> List[Tuple[str, int]]:
    """Return ``(fqn, char_end)`` for every non-wildcard import in *text*.

    ``char_end`` is the offset just past the FQN (before any ``as Alias``), so
    ``link_end_pos`` lands right after the imported symbol. Wildcard imports
    (``a.b.*``) are dropped (no single target). Comments/strings are blanked first.
    """
    text = _blank_comments(text)
    results: List[Tuple[str, int]] = []
    for m in _IMPORT_RE.finditer(text):
        if m.group("star") is not None:
            continue  # wildcard/on-demand import: no single target -> drop
        fqn = re.sub(r"\s+", "", m.group("fqn")).replace("`", "")
        if fqn:
            results.append((fqn, m.end("fqn")))
    return results


class KotlinImportDetector:
    """Detects Kotlin import declarations in tokenized source sequences.

    Implements the ``LinkDetector`` protocol. Emits one ``LinkInfo`` per
    non-wildcard import, with the dotted FQN as ``target_str`` (alias stripped).
    Matching against a corpus node is exact string equality on the FQN (see
    ``index_doc_span``). Args: ``decode_fn`` (List[int] -> str).
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)
        links: List[LinkInfo] = []
        for fqn, char_end in _parse_imports(full_text):
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=pos, target_str=fqn))
        logger.debug("KotlinImportDetector: %d links from %d tokens",
                     len(links), len(tokens))
        return links

    def index_doc_span(self, span: Any) -> str:
        """Return the node's FQN for matching.

        A Kotlin node's ``raw_identifier`` IS the fully-qualified name of the
        symbol it declares (``com.ex.util.Helper``) — exactly what
        ``detect_links`` emits as ``target_str`` — so matching is exact string
        equality and this returns ``raw_identifier`` unchanged (like Go). If a
        stored ``raw_identifier`` carries a ``<repo>:`` prefix, strip it and turn
        a path tail into a dotted name defensively.
        """
        raw = span.raw_identifier
        if ":" in raw:
            raw = raw.split(":", 1)[1]
        if raw.endswith(".kt"):
            raw = raw[: -len(".kt")].replace("/", ".")
        return raw

    def _build_char_to_token_index(self, tokens: List[int]) -> List[int]:
        cumulative = [0] * (len(tokens) + 1)
        for i, tok in enumerate(tokens):
            try:
                char_len = len(self.decode_fn([tok]))
            except Exception:
                char_len = 1
            cumulative[i + 1] = cumulative[i] + char_len
        return cumulative

    def _char_pos_to_token_pos(self, cumulative: List[int], char_pos: int) -> int:
        idx = bisect.bisect_left(cumulative, char_pos)
        return max(0, min(idx, len(cumulative) - 1))
