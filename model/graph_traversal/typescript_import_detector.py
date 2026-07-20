"""
TypeScript Import Statement Link Detector

Detects TypeScript ``import`` / ``export ... from`` / CommonJS ``require`` /
dynamic ``import()`` statements in tokenized source and emits candidate file-path
links for the imported modules.

Node unit
---------
A TypeScript NODE is a source FILE (like Python/Java), keyed by its repo-relative
path WITHOUT extension (``src/util/helper``). Only RELATIVE specifiers (``./foo``,
``../x/y``) are intra-repo and resolvable; bare specifiers (``react``, ``lodash``)
are external node_modules deps that legitimately do NOT resolve to a corpus node.

Relative resolution (the Python-like part)
------------------------------------------
Because a relative specifier is resolved against the importing file's directory,
resolution needs per-document context (which file are we in?). So — exactly like
``PythonImportDetector`` — this detector exposes TWO methods:

* ``detect_links`` operates on a flat packed sequence with NO per-doc context, so
  it cannot resolve ``../``. It emits the normalized SPECIFIER-space detection keys
  (``./sub/b.ts`` -> ``sub/b`` and ``sub/b/index``), used by the DETECTION-axis
  grader (run_detection) against the tree-sitter oracle.
* ``detect_links_for_doc`` receives a single document's tokens + its
  ``raw_identifier`` (encoding the file path), so it resolves ``./``/``../``
  against the importing file's dir and emits repo-relative path candidates — the
  RESOLUTION-axis keys training/generation match via ``index_doc_span``.
  ``CrossDocLinkMaskCreator`` prefers this method (detected via ``hasattr``).

TS resolution rules implemented (deterministic, no filesystem):
    ``./foo``  -> ``<dir>/foo``            (foo.ts | foo.tsx | foo/index.ts ...)
    ``../x/y`` -> resolve ``..`` up a dir
    ``./foo.js`` -> ``<dir>/foo``          (.js->.ts remap: strip the .js and match
                                            the sibling foo.ts node key)
Extension is usually omitted in the import and inferred; a directory import
resolves to its ``index`` file. Both are emitted as candidates; whichever matches a
real node resolves (like Python's ``foo/bar.py`` + ``foo/bar/__init__.py``).

Forms handled: default / named ``{a,b}`` / namespace ``* as ns`` / side-effect
``import "./x"`` / type-only ``import type {...} from "./x"`` (still a file edge) /
re-export ``export {q} from "./q"`` and ``export * from "./r"`` / CommonJS
``require("./z")`` / dynamic ``import("./x")`` with a literal string. Non-literal
dynamic imports (``import(varName)``) are dropped.

Detection mechanics mirror the other detectors: decode once, blank comments +
strings (so an import-looking string inside a comment or a template literal is not
matched), regex the specifiers, map char offsets to token positions via a
cumulative per-token char index.
"""
from __future__ import annotations

import bisect
import logging
import re
from typing import Any, Callable, List, Optional, Set, Tuple

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)

# Recognized module extensions, longest-first so ``.d.ts`` strips before ``.ts``.
_EXTS = (".d.ts", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs")

# --- specifier extractors (run on comment/string-blanked text) ---------------
# `import ... from "spec"` and `export ... from "spec"` (list may span lines).
_FROM_RE = re.compile(
    r"\bfrom\s*(?P<q>[\"'])(?P<spec>[^\"'\n]*)(?P=q)",
)
# side-effect `import "spec"` / `import 'spec'` (a quote right after import).
_SIDE_EFFECT_RE = re.compile(
    r"\bimport\s*(?P<q>[\"'])(?P<spec>[^\"'\n]*)(?P=q)",
)
# dynamic `import("spec")`.
_DYNAMIC_RE = re.compile(
    r"\bimport\s*\(\s*(?P<q>[\"'])(?P<spec>[^\"'\n]*)(?P=q)",
)
# CommonJS `require("spec")` — bare identifier only, NOT `foo.require(...)`.
_REQUIRE_RE = re.compile(
    r"(?<![.\w])require\s*\(\s*(?P<q>[\"'])(?P<spec>[^\"'\n]*)(?P=q)",
)


def _strip_ext(spec: str) -> str:
    for ext in _EXTS:
        if spec.endswith(ext):
            return spec[: -len(ext)]
    return spec


def _spec_to_detection_keys(spec: str) -> Set[str]:
    """Relative specifier -> normalized detection key set (specifier space).

    Bare/external specifiers return {} (not in-corpus). Mirrors the frozen oracle
    (typescript_spec._spec_to_detection_keys) — they MUST agree on the same key
    set, but neither imports the other.
    """
    if not (spec.startswith("./") or spec.startswith("../")):
        return set()
    s = _strip_ext(spec)
    out = []
    for seg in s.split("/"):
        if seg in ("", "."):
            continue
        out.append(seg)
    key = "/".join(out)
    if key == "":
        return {"index"}
    keys = {key}
    if not key.endswith("/index"):
        keys.add(f"{key}/index")
    return keys


def _resolve_relative_spec(spec: str, source_file_path: str) -> List[str]:
    """Resolve a relative specifier against the importing file's dir.

    Returns repo-relative path candidates WITHOUT extension (``<resolved>`` and
    ``<resolved>/index``), most-specific first. Returns [] for a bare specifier or
    one that escapes the repo root.
    """
    if not (spec.startswith("./") or spec.startswith("../")):
        return []
    stripped = _strip_ext(spec)
    base_dir = source_file_path.replace("\\", "/").split("/")[:-1]
    cur = list(base_dir)
    for seg in stripped.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if cur:
                cur.pop()
            else:
                return []  # escapes repo root — unresolvable
        else:
            cur.append(seg)
    resolved = "/".join(cur)
    if not resolved:
        return []
    cands = [resolved]
    if not resolved.endswith("/index"):
        cands.append(f"{resolved}/index")
    return cands


def _blank_comments_and_strings(text: str) -> str:
    """Blank JS/TS comments to equal-length spaces; leave string bodies intact.

    Preserves char offsets so ``link_end_pos`` is unchanged. A hand scanner (not
    regex) so comment markers inside strings/templates and quotes inside comments
    are not misread. Comments are blanked so an import statement written inside a
    doc comment is NOT detected (tree-sitter ignores it, so we must too). Quoted-
    string bodies are NOT blanked — the import specifier is itself a ``"..."`` /
    ``'...'`` string, so erasing them would erase the very specifier we detect.
    Template-literal (backtick) bodies ARE blanked: a module specifier is never a
    template, but codegen templates commonly embed an ``import ... from "./x"``
    STRING that is not a real import (tree-sitter treats it as template text), so
    blanking prevents that false positive.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        # quoted string: skip to matching close (honor escapes), body left INTACT
        # (the import specifier is itself a "..." / '...' string).
        if c in ("\"", "'"):
            quote = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        # template literal `...`: BLANK the body. Module specifiers are never
        # backtick templates, but a code-generator's template can contain an
        # `import ... from "./x"` STRING that is not a real import — tree-sitter
        # sees it as template text, so we must blank it to avoid a false positive.
        if c == "`":
            out[i] = " "
            i += 1
            while i < n:
                if text[i] == "\\":
                    if text[i] != "\n":
                        out[i] = " "
                    i += 1
                    if i < n and text[i] != "\n":
                        out[i] = " "
                    i += 1
                    continue
                if text[i] == "`":
                    out[i] = " "
                    i += 1
                    break
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            continue
        # line comment //
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
            continue
        # block comment /* */
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if text[i] != "\n":
                    out[i] = " "
                i += 1
            if i + 1 < n:
                out[i] = " "
                out[i + 1] = " "
                i += 2
            continue
        i += 1
    return "".join(out)


def _parse_import_specs(text: str) -> List[Tuple[str, int]]:
    """Find all import/require/dynamic-import specifiers in *text*.

    Returns ``(specifier, char_end)`` where ``char_end`` is just past the closing
    quote (used for ``link_end_pos``). Comments are blanked first. Dedup by
    (specifier, char_end) so the ``from`` and side-effect regexes don't double-emit.
    """
    text = _blank_comments_and_strings(text)
    seen: Set[Tuple[str, int]] = set()
    results: List[Tuple[str, int]] = []

    def add(spec: str, end: int):
        key = (spec, end)
        if spec and key not in seen:
            seen.add(key)
            results.append((spec, end))

    for m in _FROM_RE.finditer(text):
        add(m.group("spec"), m.end())
    for m in _DYNAMIC_RE.finditer(text):
        add(m.group("spec"), m.end())
    for m in _REQUIRE_RE.finditer(text):
        add(m.group("spec"), m.end())
    # side-effect `import "x"`: must not double-count `import ... from "x"` (that
    # has no quote right after `import`, so _SIDE_EFFECT_RE won't match it anyway).
    for m in _SIDE_EFFECT_RE.finditer(text):
        add(m.group("spec"), m.end())

    return results


class TypeScriptImportDetector:
    """Detects TypeScript import statements in tokenized source sequences.

    Implements the ``LinkDetector`` protocol (+ optional ``detect_links_for_doc``
    for relative-import resolution). Args: ``decode_fn`` (List[int] -> str).
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn

    # ------------------------------------------------------------------
    # LinkDetector protocol
    # ------------------------------------------------------------------

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        """Flat-sequence detection (no per-doc context).

        Emits SPECIFIER-space detection keys for relative specifiers only (bare
        specifiers emit nothing — they never resolve). Used by the detection
        grader. Multiple LinkInfos may share a ``link_end_pos`` (the bare-key and
        the ``/index`` candidate).
        """
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        links: List[LinkInfo] = []
        for spec, char_end in _parse_import_specs(full_text):
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            for key in sorted(_spec_to_detection_keys(spec)):
                links.append(LinkInfo(link_end_pos=pos, target_str=key))
        logger.debug("TypeScriptImportDetector: %d links from %d tokens",
                     len(links), len(tokens))
        return links

    def detect_links_for_doc(
        self,
        span_tokens: torch.Tensor,
        raw_identifier: str,
    ) -> List[LinkInfo]:
        """Per-document detection that RESOLVES relative specifiers.

        The importing file's path is read from ``raw_identifier`` (post-``:``), so
        ``./``/``../`` resolve against its directory. Emits repo-relative path
        candidates (``<resolved>`` + ``<resolved>/index``) with positions LOCAL to
        the span (caller offsets by ``span.start``).
        """
        tokens = span_tokens.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)

        source_file_path = (
            raw_identifier.split(":", 1)[1] if ":" in raw_identifier else raw_identifier
        )
        # node keys are extension-less; strip one defensively if present.
        source_file_path = _strip_ext(source_file_path)

        links: List[LinkInfo] = []
        for spec, char_end in _parse_import_specs(full_text):
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            for cand in _resolve_relative_spec(spec, source_file_path):
                links.append(LinkInfo(link_end_pos=pos, target_str=cand))
        logger.debug("TypeScriptImportDetector.detect_links_for_doc: %d links for %r",
                     len(links), raw_identifier)
        return links

    def index_doc_span(self, span: Any) -> str:
        """Repo-relative path (extension-less) of a node's ``raw_identifier``.

        ``"repo:src/util/helper"`` -> ``"src/util/helper"``. Matches the resolved
        candidates ``detect_links_for_doc`` emits. Strips a recognized extension
        defensively (node keys are stored extension-less by the builder).
        """
        parts = span.raw_identifier.split(":", 1)
        path = parts[1] if len(parts) > 1 else span.raw_identifier
        return _strip_ext(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
