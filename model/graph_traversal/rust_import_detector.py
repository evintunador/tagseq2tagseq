"""
Rust ``use`` / ``mod`` Link Detector

Detects Rust ``use`` declarations (and file-backed ``mod`` declarations) in
tokenized source and emits crate-relative module-path candidates as link targets.

Node unit
---------
A Rust corpus NODE is a source FILE, keyed by its MODULE PATH (``crate::net::tcp``)
-- the chain of ``mod`` declarations from the crate root (``src/lib.rs`` /
``src/main.rs`` / ``src/bin/*.rs``) down to the file. Resolution is SINGLE-REPO
(really single-crate): ``crate::`` is a fixed keyword root that every crate uses to
refer to itself, so module paths are unique *within* a crate but collide across
crates (like Python's bare relative paths). See docs/multilang_code_datasets_DESIGN
§4 (Rust) and this project's memory.

Why ``crate::`` as the fixed root
---------------------------------
The Stack (dedup) has NO ``Cargo.toml`` (filtered to source extensions), so we
cannot read a crate's *published* name. But in the 2018+ edition every crate refers
to its own root as ``crate::``. So an intra-crate ``use crate::a::b::C`` names the
file whose module path is ``crate::a::b`` (``C`` is a symbol) OR ``crate::a::b::C``
(if that is itself a submodule). We emit BOTH candidates and let resolution pick the
one that is a real node (exactly Python's submodule-vs-symbol candidate expansion).
``use std::...`` / ``use somecrate::...`` (a bare, non ``crate``/``self``/``super``
root) are EXTERNAL -- emitted as candidates but they never resolve to an intra-crate
node, so the recall denominator excludes them (design §6.2).

Forms handled (all after comment/string blanking):

    use crate::net::tcp::Conn;          -> crate::net::tcp , crate::net::tcp::Conn
    use self::x;                        -> self , self::x        (self=current module)
    use super::y::Z;                    -> super::y , super::y::Z (super=parent module)
    use std::collections::HashMap;      -> std::collections , std::collections::HashMap  (external)
    use crate::a::{b::C, d::E};         -> crate::a::b , crate::a::b::C , crate::a::d , crate::a::d::E
    use crate::a::{self, b};            -> crate::a , crate::a::b
    use crate::foo::*;                  -> crate::foo             (glob names a MODULE)
    pub use crate::a::B;                -> crate::a , crate::a::B (re-export is a real edge)
    use crate::a as foo;               -> crate::a               (alias dropped)
    mod net;                            -> net (bare) / crate::...::net (with module context)

Design decisions (documented for the human gate):
  * GLOB ``use crate::foo::*``: has no single item, but names a target MODULE
    (``crate::foo``). We emit that module path -- a legitimate edge, unlike a
    wildcard with no target. The oracle agrees.
  * RE-EXPORT ``pub use crate::a::B``: still an edge to ``crate::a``. Included.
  * GROUPED/NESTED ``use a::{b::{c,d}, e}``: expanded leaf-by-leaf; each leaf
    licenses its own candidate keys (the RICH path -- one statement, several keys).
  * INLINE ``mod foo { ... }`` (a body, no file): NOT a file-backed edge -- skipped.
    Only ``mod foo;`` (semicolon, no body) declares a separate file.

Two positional modes, like PythonImportDetector:
  * ``detect_links`` (packed sequence, no per-doc context): ``self``/``super``/mod
    stay relative -- used by the DETECTION gate + generation fallback.
  * ``detect_links_for_doc(span_tokens, raw_identifier)``: ``raw_identifier`` is the
    span's MODULE PATH, so ``self::``/``super::``/``mod`` are rewritten to absolute
    ``crate::`` paths. This is what CrossDocLinkMaskCreator uses at train time.

Detection mechanics mirror the Go/Python detectors: decode once, blank comments +
strings with a hand-scanner (so ``use`` inside a string/comment isn't matched, and
lifetimes ``'a`` aren't mistaken for char literals), scan ``use``/``mod``, then map
char offsets to token positions via a cumulative per-token char-length index.
"""
from __future__ import annotations

import bisect
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

import torch

from .link_detector import LinkInfo

logger = logging.getLogger(__name__)


# Repo separator for multi-repo node ids: "owner/repo@crate::a::b". Chosen over
# ':' because Rust module paths already contain '::'.
_REPO_SEP = "@"


def _module_path_of(raw_identifier: str) -> str:
    """Strip an optional ``owner/repo@`` repo prefix, returning the module path."""
    if _REPO_SEP in raw_identifier:
        return raw_identifier.split(_REPO_SEP, 1)[1]
    return raw_identifier


# ---------------------------------------------------------------------------
# Comment / string blanking (Rust-aware hand scanner)
# ---------------------------------------------------------------------------

def _blank_comments_and_strings(text: str) -> str:
    """Replace Rust comments and string literals with equal-length spaces.

    A ``use`` / ``mod`` that appears inside a ``//`` line comment, a (nestable)
    ``/* */`` block comment, or a string literal is NOT code and must not be
    matched -- tree-sitter ignores them, so the token-space detector must too. We
    blank to spaces (not delete) so every char offset (and thus ``link_end_pos``)
    is preserved.

    Handles: line comments, NESTED block comments (Rust allows nesting), normal
    ``"..."`` strings (with backslash escapes), raw strings ``r"..."`` /
    ``r#"..."#`` / ``br#"..."#`` (any number of ``#``), and byte strings ``b"..."``.
    Char literals ``'x'`` / ``'\\n'`` are skipped WITHOUT confusing them with
    lifetimes ``'a`` (a ``'`` followed by an identifier and no closing quote).
    """
    out = list(text)
    i, n = 0, len(text)

    def blank(a: int, b: int):
        for j in range(a, min(b, n)):
            if out[j] != "\n":
                out[j] = " "

    while i < n:
        c = text[i]

        # raw string: (b?)r #* " ... " #*   (matching count of '#')
        if (c == "r" or c == "b") and i + 1 < n:
            j = i
            if text[j] == "b" and j + 1 < n and text[j + 1] == "r":
                j += 1  # br"..."
            if text[j] == "r":
                k = j + 1
                hashes = 0
                while k < n and text[k] == "#":
                    hashes += 1
                    k += 1
                if k < n and text[k] == '"':
                    # raw string body until '"' followed by `hashes` '#'
                    close = '"' + "#" * hashes
                    end = text.find(close, k + 1)
                    end = (end + len(close)) if end != -1 else n
                    blank(i, end)
                    i = end
                    continue

        # normal / byte string literal
        if c == '"' or (c == "b" and i + 1 < n and text[i + 1] == '"'):
            start = i
            i += 1 if c == '"' else 2
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == '"':
                    i += 1
                    break
                i += 1
            blank(start, i)
            continue

        # char literal vs lifetime: '  -> if it looks like a char literal skip it,
        # otherwise (lifetime 'a) just advance past the quote and leave text alone.
        if c == "'":
            if i + 1 < n and text[i + 1] == "\\":
                # escape -> char literal; skip to closing '
                j = i + 2
                while j < n and text[j] != "'":
                    j += 1
                i = j + 1
                continue
            if i + 2 < n and text[i + 2] == "'":
                # 'x' char literal
                i += 3
                continue
            # lifetime ('a, 'static, ...) -- not a string; just move on
            i += 1
            continue

        # line comment
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = i
            while j < n and text[j] != "\n":
                j += 1
            blank(i, j)
            i = j
            continue

        # block comment (nested)
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            depth = 1
            j = i + 2
            while j < n and depth > 0:
                if text[j] == "/" and j + 1 < n and text[j + 1] == "*":
                    depth += 1
                    j += 2
                    continue
                if text[j] == "*" and j + 1 < n and text[j + 1] == "/":
                    depth -= 1
                    j += 2
                    continue
                j += 1
            blank(i, j)
            i = j
            continue

        i += 1

    return "".join(out)


# ---------------------------------------------------------------------------
# use-tree parsing (string space) + shared candidate rule
# ---------------------------------------------------------------------------

_WS_COLON_RE = re.compile(r"\s*::\s*")
_SPACE_RE = re.compile(r"\s+")


def _norm_path(path: str) -> str:
    """Normalize a ``::``-joined path: collapse whitespace, strip leading ``::``."""
    path = _WS_COLON_RE.sub("::", path.strip())
    path = _SPACE_RE.sub("", path)
    while path.startswith("::"):
        path = path[2:]
    return path


def _split_top_level_commas(s: str) -> List[str]:
    """Split ``s`` on commas that are NOT inside nested ``{ }`` braces."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "{":
            depth += 1
            cur.append(ch)
        elif ch == "}":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def expand_use_tree_string(tree: str) -> List[str]:
    """Expand a Rust use-tree STRING (no leading ``use``, no trailing ``;``) to its
    leaf paths (still possibly relative: ``self``/``super`` kept, ``as`` dropped).

    ``crate::a::{b::C, d::E}`` -> [``crate::a::b::C``, ``crate::a::d::E``]
    ``crate::a::{self, b}``    -> [``crate::a``, ``crate::a::b``]
    ``crate::foo::*``          -> [``crate::foo::*``]
    ``crate::a as foo``        -> [``crate::a``]
    """
    tree = tree.strip()
    if not tree:
        return []

    # find a TOP-LEVEL '{' (a grouped use-list). Prefix is everything before it.
    depth = 0
    brace_open = -1
    for idx, ch in enumerate(tree):
        if ch == "{":
            if depth == 0:
                brace_open = idx
                break
            depth += 1
        elif ch == "}":
            depth -= 1

    if brace_open == -1:
        # simple path: drop `as ALIAS`, keep trailing `*`
        leaf = re.split(r"\s+as\s+", tree, maxsplit=1)[0].strip()
        if leaf == "*":
            return ["*"]
        if leaf.endswith("*"):
            base = _norm_path(leaf[:-1].rstrip(":"))
            return [f"{base}::*"] if base else ["*"]
        leaf = _norm_path(leaf)
        return [leaf] if leaf else []

    prefix = tree[:brace_open].strip()
    prefix = _norm_path(prefix.rstrip().rstrip(":"))
    # matching close brace
    depth = 0
    close = -1
    for idx in range(brace_open, len(tree)):
        if tree[idx] == "{":
            depth += 1
        elif tree[idx] == "}":
            depth -= 1
            if depth == 0:
                close = idx
                break
    inner = tree[brace_open + 1: close if close != -1 else len(tree)]

    leaves: List[str] = []
    for sub in _split_top_level_commas(inner):
        for sub_leaf in expand_use_tree_string(sub):
            if sub_leaf in ("self", "self::*"):
                combined = prefix
            elif prefix:
                combined = f"{prefix}::{sub_leaf}"
            else:
                combined = sub_leaf
            if combined:
                leaves.append(combined)
    return leaves


def leaf_to_candidates(leaf: str) -> List[str]:
    """Project ONE leaf path into the shared candidate key space.

    Rule (mirrors Python's submodule-vs-symbol expansion):
      * glob ``a::b::*``       -> {``a::b``}                (the target module)
      * ``a::b::C`` (>=2 segs) -> {``a::b``, ``a::b::C``}   (parent module + full)
      * ``a`` (1 seg)          -> {``a``}
    ``self`` alone / bare ``*`` -> {} (no target).
    """
    leaf = _norm_path(leaf)
    if not leaf or leaf == "*":
        return []
    # macro metavariables ($crate, $x) are not module paths — drop.
    if any(seg.startswith("$") for seg in leaf.split("::")):
        return []
    if leaf.endswith("::*"):
        mod = leaf[:-3]
        return [mod] if mod else []
    if leaf == "self":
        return []
    segs = leaf.split("::")
    if len(segs) >= 2:
        return ["::".join(segs[:-1]), leaf]
    return [leaf]


def leaf_paths_to_candidates(leaves: List[str]) -> List[str]:
    """Flatten many leaf paths into a deduped candidate list (order-stable)."""
    seen, out = set(), []
    for leaf in leaves:
        for c in leaf_to_candidates(leaf):
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


# ---------------------------------------------------------------------------
# self / super rewriting against a module context
# ---------------------------------------------------------------------------

def rewrite_relative(leaf: str, module_path: str) -> str:
    """Rewrite a ``self::``/``super::`` leaf to an absolute ``crate::`` path.

    ``module_path`` is the CURRENT file's module path (``crate::net::tcp``).
      self::x            (M=crate::net::tcp) -> crate::net::tcp::x
      super::y::Z        (M=crate::net::tcp) -> crate::net::y::Z
      super::super::z    (M=crate::net::tcp) -> crate::z
    Absolute (``crate::``) and external (``std::``) leaves are returned unchanged.
    """
    leaf = _norm_path(leaf)
    glob = leaf.endswith("::*")
    core = leaf[:-3] if glob else leaf
    segs = core.split("::") if core else []
    if not segs:
        return leaf
    mod_segs = module_path.split("::") if module_path else []
    if segs[0] == "self":
        combined = mod_segs + segs[1:]
    elif segs[0] == "super":
        base = list(mod_segs)
        rest = segs
        while rest and rest[0] == "super":
            if base:
                base.pop()
            rest = rest[1:]
        combined = base + rest
    else:
        return leaf
    result = "::".join(combined)
    return f"{result}::*" if glob else result


# ---------------------------------------------------------------------------
# statement scanning (char positions)
# ---------------------------------------------------------------------------

# `use` keyword at a statement boundary (optionally `pub`/`pub(...)`), then the
# tree up to the terminating `;` at brace-depth 0.
_USE_RE = re.compile(
    r"(?:^|[;{}\s])(?:pub(?:\s*\([^)]*\))?\s+)?use\s",
    re.MULTILINE,
)
# file-backed `mod ident;` (semicolon, no body). Inline `mod ident { }` won't match.
_MOD_RE = re.compile(
    r"^[ \t]*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(?P<name>[A-Za-z_][\w]*)\s*;",
    re.MULTILINE,
)


def _scan_use_statements(text: str) -> List[Tuple[str, int]]:
    """Return (use_tree_string, char_end) for each ``use`` decl in *text*.

    ``char_end`` is just past the terminating ``;`` -- where the grant begins.
    """
    results: List[Tuple[str, int]] = []
    for m in _USE_RE.finditer(text):
        # tree starts right after the matched `use ` keyword
        tree_start = m.end()
        i, n = tree_start, len(text)
        depth = 0
        while i < n:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == ";" and depth == 0:
                break
            i += 1
        tree = text[tree_start:i]
        results.append((tree, i + 1))
    return results


def _parse_targets(text: str, module_path: Optional[str] = None) -> List[Tuple[str, int]]:
    """Parse *text* into (target_candidate, char_end) pairs.

    When ``module_path`` is given, ``self``/``super`` leaves and ``mod`` names are
    rewritten to absolute ``crate::`` paths; otherwise they stay relative.
    """
    text = _blank_comments_and_strings(text)
    out: List[Tuple[str, int]] = []

    for tree, char_end in _scan_use_statements(text):
        leaves = expand_use_tree_string(tree)
        if module_path is not None:
            leaves = [rewrite_relative(lf, module_path) for lf in leaves]
        for cand in leaf_paths_to_candidates(leaves):
            out.append((cand, char_end))

    for m in _MOD_RE.finditer(text):
        name = m.group("name")
        if module_path is not None:
            target = f"{module_path}::{name}" if module_path else name
        else:
            target = name
        out.append((target, m.end()))

    return out


class RustImportDetector:
    """Detects Rust ``use`` / file-backed ``mod`` declarations in tokenized source.

    Implements the ``LinkDetector`` protocol. Emits crate-relative module-path
    candidates (parent module + full path per leaf; the module for a glob). Args:
    ``decode_fn`` (List[int] -> str).
    """

    def __init__(self, decode_fn: Callable[[List[int]], str]) -> None:
        self.decode_fn = decode_fn

    def detect_links(self, input_ids: torch.Tensor) -> List[LinkInfo]:
        tokens = input_ids.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)
        links: List[LinkInfo] = []
        for target, char_end in _parse_targets(full_text, module_path=None):
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=pos, target_str=target))
        logger.debug("RustImportDetector: %d links from %d tokens", len(links), len(tokens))
        return links

    def detect_links_for_doc(
        self,
        span_tokens: torch.Tensor,
        raw_identifier: str,
    ) -> List[LinkInfo]:
        """Detect links for a single doc span, resolving ``self``/``super``/``mod``.

        ``raw_identifier`` encodes the span's MODULE PATH (``crate::net::tcp``),
        optionally prefixed by a repo tag (``owner/repo@crate::net::tcp``) to keep
        node ids unique across a multi-repo build; the ``@`` prefix is stripped to
        recover the module path used to rewrite ``self``/``super``. Returns
        span-local positions (the caller offsets by ``span.start``), mirroring
        PythonImportDetector.
        """
        module_path = _module_path_of(raw_identifier)
        tokens = span_tokens.tolist()
        full_text = self.decode_fn(tokens)
        cumulative = self._build_char_to_token_index(tokens)
        links: List[LinkInfo] = []
        for target, char_end in _parse_targets(full_text, module_path=module_path):
            pos = self._char_pos_to_token_pos(cumulative, char_end)
            links.append(LinkInfo(link_end_pos=pos, target_str=target))
        return links

    def index_doc_span(self, span: Any) -> str:
        """Return the node's MODULE PATH for matching.

        A Rust node's ``raw_identifier`` is its module path (``crate::net::tcp``),
        optionally prefixed by a ``owner/repo@`` repo tag (added at build time to
        keep node ids unique across repos, since module paths collide across
        crates). We strip that prefix so the returned key is exactly what
        ``detect_links_for_doc`` emits -- matching is then exact string equality.
        Uses ``@`` (not ``:``) as the separator because Rust paths contain ``::``.
        """
        return _module_path_of(span.raw_identifier)

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
