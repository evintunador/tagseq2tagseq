"""CrossCodeEval TypeScript port — cross-file line-completion benchmark.

Maps the TypeScript split of CrossCodeEval (AWS, github.com/amazon-science/cceval,
Apache-2.0) into the canonical CrossDocExample schema, mirroring the RepoBench
reference adapter (ports/repobench.py). Convenience source is the HF parquet
mirror ``ZHENGRAN/cross_code_eval_typescript`` (3,356 rows — the paper's TS split),
cached under /fss-data (NOT /fss).

Field mapping
-------------
  * context   = ``prompt`` — the in-file left context INCLUDING the import block
    (verified: every row has ``metadata.context_start_lineno == 0``, so no import
    block is cropped; rows with a non-zero start are DROPPED defensively).
  * target    = ``groundtruth`` — the statement-level completion (avg ~1-1.7 lines).
  * file_path = ``metadata.file`` (repo-relative path of the primary file).
  * repo      = ``metadata.repository``.
  * aux       = ``crossfile_context_retrieval`` list → AuxDoc(path=filename,
    content=retrieved_chunk). These are 10-line RETRIEVAL CHUNKS keyed by the
    source file's repo-relative path (a known v1 limitation: a chunk's path can
    be import-licensed while the chunk TEXT lacks the imported symbol). Sorted by
    (path, retrieval-rank) for determinism, then de-duplicated on the exact
    (path, content) pair (Tier 0 forbids fully-identical aux pairs).

identifier shaping (the audited component)
------------------------------------------
A TypeScript NODE is a source file keyed by its repo-relative path WITHOUT
extension (``src/util/helper``); ``index_doc_span`` returns ``raw_identifier``
after the ``:`` with a recognized extension stripped.

TS in-corpus imports are all RELATIVE (``./foo``, ``../x/y``). The eval pipeline
(Tier 1 AND ``score_completion_with_context_docs``) matches via the FLAT
``detect_links``, which has no per-doc context and therefore emits SPECIFIER-space
keys — the import path as written, relative to the importing file's directory
(``./apiDebug`` → ``apiDebug``; ``./sub/b`` → ``sub/b``; ``../x/y`` → ``../x/y``).
It matches those against ``index_doc_span(raw_identifier)`` per aux span.

``identifier_fn`` receives only (repo, aux_path, content) — NOT the importing
file's directory — so it cannot reconstruct the dir-relative specifier for
subdir/parent imports. The maximal single-key choice that a specifier can match
is therefore the extension-stripped BASENAME of the aux path (index files keep
their parent segment; see ``_ts_basename_identifier``), which resolves the
DOMINANT same-directory ``./foo`` import case (foo.ts imported as ``./foo`` from a
sibling file → key ``foo`` == basename). Subdir (``./sub/foo`` → ``sub/foo``) and
parent (``../x/y``) specifiers carry directory context that only the runtime
``source_file_path`` relative resolution can recover; the static basename
projection cannot, and those links are lost on the port side exactly as the
oracle's suffix projection loses ``..``-prefixed keys.

Measured on the first 500 examples (2026-07-25): oracle-reachable-rate 0.480,
Tier-1 basename fire-rate 0.400 at precision 1.000 — just under the
0.9×reachable = 0.432 parity gate. The ~0.03 shortfall is the ~5.6% of examples
reachable ONLY through a subdir/parent specifier (``./sub/x`` / ``../x/y``), which
NO stateless single-key projection can reach because ``identifier_fn`` lacks the
importing file's dir. The runtime scorer DOES recover these:
``score_completion_with_context_docs`` passes ``source_file_path`` and resolves
``./``/``../`` against it (full-path key match), lifting fire-rate to 0.674 on the
same 500 — so the benchmark is genuinely cross-doc-discriminating at Tier 2 even
though Tier 1's static path-suffix projection under-fires by design. This is an
honest, expected gap for a relative-import language, NOT an identifier-shaping
bug. ``detect_links_for_doc`` (full-path resolution, what the TRAINING pipeline
uses via ``CrossDocLinkMaskCreator``) is deliberately NOT used here: neither Tier
1 nor the runtime scorer invokes it.
"""
from __future__ import annotations

import os
from typing import List, Optional

from ..schema import AuxDoc, CrossDocExample, PortAdapter

# HF parquet mirror of the paper's TS split, cached on /fss-data (NOT /fss).
_PARQUET = (
    "/fss-data/evin_t/tagseq2tagseq_artifacts/raw/crosscodeeval_ts/"
    "data/train-00000-of-00001.parquet"
)

# Recognized TS module extensions, longest-first (mirrors the detector/spec).
_EXTS = (".d.ts", ".tsx", ".ts", ".jsx", ".js", ".mjs", ".cjs")


def _strip_ext(spec: str) -> str:
    for ext in _EXTS:
        if spec.endswith(ext):
            return spec[: -len(ext)]
    return spec


def _ts_basename_identifier(repo: str, path: str, content: str) -> str:
    """Shape an aux file path into the specifier-space key ``detect_links`` emits.

    ``src/apiDebug.ts`` -> ``<repo>:apiDebug``. ``index_doc_span`` returns the
    post-``:`` component with the extension stripped, giving ``apiDebug`` — the
    key a same-directory ``import ... from './apiDebug'`` produces under the flat
    detector. See the module docstring for why the basename (not the full
    repo-relative path) is the correct single-key projection for TS.

    Directory-index files are special-cased: ``src/foo/index.ts`` keeps its
    parent segment (``foo/index``), because a same-directory ``import from
    './foo'`` emits BOTH ``foo`` and ``foo/index`` under the flat detector, and
    ``foo/index`` is the one this aux resolves to. A bare basename ``index``
    would collide across every ``index.ts`` in the repo and never match.
    """
    p = path.replace("\\", "/").strip().strip("/")
    parts = [_strip_ext(seg) for seg in p.split("/")]
    if parts[-1] == "index" and len(parts) >= 2:
        base = f"{parts[-2]}/index"
    else:
        base = parts[-1]
    return f"{repo}:{base}"


def _load_crosscodeeval_ts(max_examples: Optional[int]) -> List[CrossDocExample]:
    if not os.path.exists(_PARQUET):
        return []
    import pandas as pd

    df = pd.read_parquet(_PARQUET)
    out: List[CrossDocExample] = []
    for _, row in df.iterrows():
        meta = row["metadata"]
        # Defensive: a non-zero context_start_lineno means the import block was
        # cropped upstream — drop it (the whole split is 0 as of 2026-07-24).
        if int(meta.get("context_start_lineno", 0)) != 0:
            continue

        target = row["groundtruth"]
        context = row["prompt"]
        if not (target and target.strip()) or not (context and context.strip()):
            continue

        # crossfile_context_retrieval is a {'list': ndarray-of-dicts} column.
        cf = row["crossfile_context_retrieval"]
        items = cf["list"] if isinstance(cf, dict) else cf
        # Deterministic order: retrieval rank is meaningful (score-desc), so keep
        # the shipped order but make it reproducible by de-duplicating exact
        # (path, content) pairs (Tier 0 forbids fully-identical aux) while
        # preserving first-seen order.
        seen = set()
        aux_list: List[AuxDoc] = []
        for it in items:
            fn = it.get("filename", "")
            chunk = it.get("retrieved_chunk", "")
            if not (chunk and chunk.strip()):
                continue
            key = (fn, chunk)
            if key in seen:
                continue
            seen.add(key)
            aux_list.append(AuxDoc(path=fn, content=chunk))

        out.append(CrossDocExample(
            repo=meta.get("repository", "repo"),
            file_path=meta.get("file", ""),
            context=context,
            target=target,
            aux=tuple(aux_list),
            meta={
                "task_id": meta.get("task_id"),
                "groundtruth_start_lineno": meta.get("groundtruth_start_lineno"),
            },
        ))
        if max_examples is not None and len(out) >= max_examples:
            break
    return out


def _ts_detector(decode_fn):
    from model.graph_traversal.typescript_import_detector import TypeScriptImportDetector
    return TypeScriptImportDetector(decode_fn)


CROSSCODEEVAL_TS = PortAdapter(
    name="crosscodeeval_ts",
    language="typescript",
    examples_fn=lambda n: _load_crosscodeeval_ts(n),
    identifier_fn=_ts_basename_identifier,
    detector_factory=_ts_detector,
)
