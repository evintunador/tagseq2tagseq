"""Internal cross-doc benchmark — self-built from held-out ``test_community`` splits.

For the code languages that have NO external RepoBench-analogous cross-file
benchmark (go, rust, javascript, dart, zig) — and, for calibration, the four
that DO (python, java, kotlin, typescript) — this port reconstructs cross-doc
examples from our OWN held-out ``test_community`` split of each pretokenized
dataset. The point of running it on the four externally-benchmarked languages is
calibration: scored through the SAME harness (same scopes, placebo control,
bootstrap CIs), the internal numbers can be compared against the external ports,
so the internal-only languages inherit that credibility band.

Provenance / no-leakage
-----------------------
Examples are drawn ONLY from ``splits/test_community`` — the held-out subgraph
that ``data/split_graph.py`` excludes from ``train`` (see its module docstring:
``test_community`` is "held back for paper"). Each split subdir is written
self-contained with ``outgoing``/``incoming`` edges filtered to same-split nodes
only, and ``test_community`` is carved as whole BFS subgraphs, so an importing
doc and the docs it imports land in the same held-out community together. NO
training content enters the benchmark; the only thing read from the graph is the
edge structure OF THE HELD-OUT SPLIT (which held-out doc imports which held-out
doc). This is a weaker independence guarantee than the external ports (a
different dataset entirely) — the split shares the training pipeline/distribution
— so the placebo control (right-vs-wrong aux) remains the load-bearing legitimacy
test, exactly as for the external ports.

How an example is built
-----------------------
A NODE is one source unit (file, or Go package). For each source node with ≥1
in-split ``outgoing`` edge:
  * ``aux``      = the decoded content of each outgoing TARGET node (these ARE the
                   import-resolved cross-file docs — the graph already resolved
                   the edge, so unlike the ASE-Kotlin port there is no
                   import-mining/resolution step here).
  * ``context``  = the source text up to and including the line of the LAST import
                   the language's own ``LinkDetector`` finds (so the import block
                   is present for Tier 0, and the boundary is defined by the exact
                   detector the scorer uses — uniform across all 9 languages).
  * ``target``   = the first non-empty body line after that boundary (the port's
                   NATIVE target; arbitrary like ASE-Kotlin's, NOT the headline).
  * ``full_file``= the whole decoded source text, so ``scopes.py`` can re-anchor
                   scoring at genuine use sites (``use_line``/``use_block``/
                   ``rest_of_doc`` — the headline scopes rebuild context from
                   ``full_file`` and ignore the native ``context``).

Identifier shaping (the audited component)
------------------------------------------
``identifier_fn`` must produce a ``raw_identifier`` whose ``index_doc_span`` key
equals what the language's detector emits as ``target_str`` for the import.
Two regimes, verified empirically against each detector:
  * ABSOLUTE-import langs (python/go/java/kotlin/rust): the node's
    ``normed_identifier`` IS the resolution key — ``index_doc_span`` maps it back
    into the detector's emission space (python ``repo:path.py``→``path.py``; java/
    kotlin dotted FQN verbatim; go full import path; rust strips ``owner/repo@``→
    ``crate::…`` module path). So ``identifier_fn`` is IDENTITY (aux path = target
    ``normed_identifier``) and links fire at production fidelity.
  * RELATIVE-import langs (typescript/javascript/dart/zig): the detector emits
    dir-relative SPECIFIER keys (``./foo``→``foo``), and ``identifier_fn`` cannot
    see the importing file's directory, so — exactly as the reviewed
    ``crosscodeeval_ts`` port documents — the maximal single-key projection is the
    extension-shaped basename, which resolves the dominant same-directory import
    and under-fires on subdir/parent (``../x/y``) imports. Tier 1 fire-rate is
    therefore ADVISORY for these four (Tier 2 is the arbiter), and a v2 that
    computes the exact source→target specifier from the known edge is filed in
    TODOS.

Determinism: source nodes are visited in sorted ``normed_identifier`` order, aux
docs are sorted by path, and no randomness / wall-clock is used, so
``examples_fn`` is reproducible (Tier 0 checks it).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..schema import AuxDoc, CrossDocExample, PortAdapter

# Bulk data on /fss-data (never /fss). Dataset dir name differs from the spec
# language name only for python (its Stack dataset is "thestack").
_DATASET_ROOT = Path(
    "/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets"
)
_DATASET_DIR: Dict[str, str] = {
    "python": "thestack",
    "java": "java",
    "kotlin": "kotlin",
    "go": "go",
    "rust": "rust",
    "typescript": "typescript",
    "javascript": "javascript",
    "dart": "dart",
    "zig": "zig",
}

# Languages whose in-corpus imports are dir-relative specifiers (detector emits
# basename-space keys, not the full node path). identifier_fn uses basename
# shaping for these; identity for the rest. See module docstring.
_RELATIVE_LANGS = frozenset({"typescript", "javascript", "dart", "zig"})

# Recognized module extensions per relative-import language, longest-first,
# mirroring each detector/spec. The basename key is extension-SHAPED to match
# what the detector emits (ts/js/zig strip the extension in their specifier
# space; dart keeps ``.dart``).
_RELATIVE_EXTS: Dict[str, Tuple[str, ...]] = {
    "typescript": (".d.ts", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs"),
    "javascript": (".jsx", ".mjs", ".cjs", ".js"),
    "dart": (".dart",),
    "zig": (".zig",),
}
# Whether the language's detector KEEPS the extension in its emitted key
# (verified against each detector's emission): dart does (``../x/y.dart``);
# ts/js/zig strip it (``../theme``, ``src/build/fs``).
_RELATIVE_KEEP_EXT: Dict[str, bool] = {
    "typescript": False, "javascript": False, "dart": True, "zig": False,
}

# Bound the aux-doc count per example so packs stay well under the RoPE cap;
# code out-degree is low (≈1–2.5) so this rarely bites. Sorted by path first, so
# the cap is deterministic. Reported by the harness via meta if it fires.
_MAX_AUX = 12


def _split_dir(language: str) -> Path:
    return _DATASET_ROOT / _DATASET_DIR[language] / "splits" / "test_community"


@lru_cache(maxsize=None)
def _load_backend(language: str):
    """(GraphIndex, PretokShardedBackend, decode_fn) for one language's split."""
    import tiktoken
    from data.dataset import GraphIndex, PretokShardedBackend

    idx = GraphIndex(_split_dir(language))
    backend = PretokShardedBackend(idx)
    decode = tiktoken.get_encoding("gpt2").decode
    return idx, backend, decode


@lru_cache(maxsize=8192)
def _node_text(language: str, normed_id: str) -> Optional[str]:
    idx, backend, decode = _load_backend(language)
    toks = backend.get_tokens(normed_id)
    if toks is None or len(toks) == 0:
        return None
    return decode(toks.tolist())


def _import_block_end_line(language: str, text: str, decode: Callable) -> int:
    """0-based line index such that ``text``'s lines[:end+1] is a context that
    the language's own detector re-parses with ≥1 import firing.

    Uses the SAME LinkDetector the scorer uses (the scorer re-detects links on
    the primary doc = context + completion), so the boundary is defined
    identically across all 9 languages and needs no per-language import grammar.

    The detector's ``link_end_pos`` marks the token after an import REFERENCE,
    which for bracketed blocks (Go ``import ( … )``, Rust nested ``use { … }``)
    can precede the block's syntactic close — truncating there yields an
    unparseable context that fires nothing. So we take the detector's last-import
    line as a floor and extend FORWARD a bounded number of lines until the
    truncated prefix itself fires ≥1 import (recovering the closing token).
    Returns -1 if the detector finds no import in the full source, or if no
    bounded prefix fires (degenerate).
    """
    import torch
    import tiktoken

    from model.graph_traversal.link_detector import make_link_detector

    enc = tiktoken.get_encoding("gpt2")
    detector = make_link_detector(language, decode)
    ids = enc.encode(text, disallowed_special=())
    if not ids:
        return -1
    links = detector.detect_links(torch.tensor(ids))
    if not links:
        return -1
    last_tok = max(0, min(max(lk.link_end_pos for lk in links), len(ids)))
    floor_line = decode(ids[:last_tok]).count("\n")

    lines = text.split("\n")
    n_lines = len(lines)
    # Extend forward until the prefix context re-fires (closing bracket / newline
    # recovered). Bounded so a pathological file can't scan the whole body.
    _MAX_EXTEND = 6
    for end in range(floor_line, min(floor_line + _MAX_EXTEND + 1, n_lines)):
        prefix = "\n".join(lines[: end + 1])
        pids = enc.encode(prefix, disallowed_special=())
        if pids and detector.detect_links(torch.tensor(pids)):
            return end
    return -1


def _basename_identifier(language: str, repo: str, path: str) -> str:
    """Extension-shaped basename key for a relative-import aux path.

    Mirrors ``crosscodeeval_ts._ts_basename_identifier``: strip the module
    extension per segment, keep the parent segment for directory-index files
    (``foo/index``), and re-attach the extension only for languages whose
    detector keeps it (dart). Returns ``<repo>:<base>`` so ``index_doc_span``
    yields ``<base>`` — the key a same-directory relative import emits.
    """
    exts = _RELATIVE_EXTS[language]

    def strip_ext(seg: str) -> str:
        for e in exts:
            if seg.endswith(e):
                return seg[: -len(e)]
        return seg

    p = path.replace("\\", "/").strip().strip("/")
    # Drop a repo prefix if the node id carried one (``repo:src/x`` style).
    if ":" in p and "/" not in p.split(":", 1)[0]:
        p = p.split(":", 1)[1]
    segs = p.split("/")
    stripped = [strip_ext(s) for s in segs]
    if stripped and stripped[-1] == "index" and len(stripped) >= 2:
        base = f"{stripped[-2]}/index"
    else:
        base = stripped[-1] if stripped else p
    if _RELATIVE_KEEP_EXT.get(language) and exts:
        base = base + exts[-1]
    return f"{repo}:{base}"


def _repo_of(language: str, normed_id: str) -> str:
    """Best-effort repo id for grouping/aux-path display.

    rust uses ``owner/repo@module``; python/ts/js/dart/zig use ``repo:path``;
    go/java/kotlin have no delimiter (import-path / FQN) — the whole id stands in
    (edges are guaranteed same-repo by the extractors + same-split by the split
    writer, so a strict repo check is not needed here).
    """
    if language == "rust" and "@" in normed_id:
        return normed_id.split("@", 1)[0]
    if ":" in normed_id and language in _RELATIVE_LANGS.union({"python"}):
        return normed_id.split(":", 1)[0]
    return normed_id


def _aux_path(language: str, target_nid: str) -> str:
    """Repo-relative resolution path for an aux doc — the string BOTH the
    detector's ``index_doc_span`` and Tier 1's oracle suffix-projection agree on.

    The Tier-1 oracle projects the raw aux path by stripping leading ``/``-
    components; it cannot strip a repo prefix glued to the first component
    (``repo:virtool/foo.py`` → ``foo.py``, never ``virtool/foo.py``). So we strip
    the repo prefix HERE, uniformly, leaving the bare import-space path:
      * rust  — drop ``owner/repo@`` (module path ``crate::…`` remains).
      * python/typescript/javascript/dart/zig — drop ``repo:`` (repo tokens
        contain no ``/``, so a ``:`` whose left side is ``/``-free is the repo
        delimiter; a ``::`` inside a path is left intact).
      * go/java/kotlin — no prefix; the id already IS the key.
    ``index_doc_span`` tolerates the stripped form (it splits on ``:``/``@`` and
    returns the tail unchanged when the prefix is already absent), so identity
    ``identifier_fn`` still fires at production fidelity. Result is Tier-0-clean
    (no leading ``/``, no ``..``).
    """
    if language == "rust":
        return target_nid.split("@", 1)[1] if "@" in target_nid else target_nid
    # python/typescript/javascript/dart/zig node ids are ``<repo>:<path>`` where
    # <repo> is ``owner/name`` (may contain '/'); the path after the FIRST ':' is
    # the bare import-space path. go/java/kotlin ids carry no ':' delimiter (Go
    # import paths and Java/Kotlin FQNs use '/'/'.' only), so they pass through.
    if language in _RELATIVE_LANGS or language == "python":
        return target_nid.split(":", 1)[1] if ":" in target_nid else target_nid
    return target_nid


def _build_examples(language: str, max_examples: Optional[int]) -> List[CrossDocExample]:
    try:
        idx, backend, decode = _load_backend(language)
    except FileNotFoundError:
        return []

    out: List[CrossDocExample] = []
    for src_nid in sorted(idx.get_all_normed_identifiers()):
        if max_examples is not None and len(out) >= max_examples:
            break
        targets = [t for t in idx.get_outgoing_links(src_nid) if t in idx]
        if not targets:
            continue

        src_text = _node_text(language, src_nid)
        if not src_text or not src_text.strip():
            continue

        end_line = _import_block_end_line(language, src_text, decode)
        if end_line < 0:
            continue  # no detectable import -> not a cross-doc example

        lines = src_text.split("\n")
        context = "\n".join(lines[: end_line + 1])
        if not context.strip():
            continue
        # Native target = first non-empty body line after the import block.
        native_target = None
        for ln in lines[end_line + 1:]:
            if ln.strip():
                native_target = ln
                break
        if native_target is None:
            continue  # no body after imports -> nothing to score / scope

        aux: List[AuxDoc] = []
        for tgt_nid in sorted(targets):
            tgt_text = _node_text(language, tgt_nid)
            if not tgt_text or not tgt_text.strip():
                continue
            aux.append(AuxDoc(path=_aux_path(language, tgt_nid), content=tgt_text))
        if not aux:
            continue
        capped = len(aux) > _MAX_AUX
        if capped:
            aux = aux[:_MAX_AUX]

        out.append(CrossDocExample(
            repo=_repo_of(language, src_nid),
            file_path=_aux_path(language, src_nid),
            context=context,
            target="\n" + native_target,
            aux=tuple(aux),
            meta={"source_id": src_nid,
                  "n_outgoing": len(targets),
                  "aux_capped": capped,
                  "split": "test_community"},
            full_file=src_text,
        ))
    return out


def _identity_identifier(repo: str, path: str, content: str) -> str:
    return path


def _make_relative_identifier(language: str):
    def _fn(repo: str, path: str, content: str) -> str:
        return _basename_identifier(language, repo, path)
    return _fn


def _make_detector_factory(language: str):
    def _factory(decode_fn):
        from model.graph_traversal.link_detector import make_link_detector
        return make_link_detector(language, decode_fn)
    return _factory


def _make_port(language: str) -> PortAdapter:
    identifier_fn = (
        _make_relative_identifier(language)
        if language in _RELATIVE_LANGS
        else _identity_identifier
    )
    return PortAdapter(
        name=f"internal_{language}",
        language=language,
        examples_fn=lambda n, _lang=language: _build_examples(_lang, n),
        identifier_fn=identifier_fn,
        detector_factory=_make_detector_factory(language),
    )


INTERNAL_PORTS: Dict[str, PortAdapter] = {
    f"internal_{lang}": _make_port(lang) for lang in _DATASET_DIR
}
