"""ASE-2025 Kotlin port — self-mined cross-doc benchmark.

Source: JetBrains/Mistral ASE 2025 Context Collection Challenge (Zenodo record
16964765, CC-BY-4.0). Each datapoint is a FIM point (``prefix``/``middle``/
``suffix``) drawn from a repo file, plus a FULL repo-snapshot zip that PRE-DATES
the ground-truth commit (no leakage). Unlike RepoBench there is no pre-baked
cross-file context: we mine it ourselves by resolving the primary file's import
FQNs against sibling ``.kt`` files inside the snapshot.

Field mapping (mirrors run_repobench_cross_doc's next-line scope):
  * context   = ``prefix`` (begins at file top, so the ``package``/``import``
                block is visible to the detector).
  * target    = the FIRST LINE of ``middle`` (RepoBench next-line scope). The
                prefix cuts mid-line, so — unlike RepoBench, whose ``next_line``
                is a fresh line prefixed with ``\\n`` — no newline is prepended;
                the target continues the prefix's current line directly.
  * file_path = ``path`` (repo-relative primary file path).
  * aux       = each snapshot ``.kt`` file whose source-root-stripped path
                dotifies to an import FQN present in the prefix, EXCLUDING files
                listed in ``modified`` (self-referential / same-commit edits).

Identifier shaping reuses the Java FQN machinery: Kotlin imports are dotted FQNs
(``com.foo.Bar``) and Kotlin files live under Java-family build source roots
(``src/main/kotlin/`` AND, in practice, ``src/main/java/``). Stripping the source
root leaves ``com/foo/Bar.kt``, which ``KotlinImportDetector.index_doc_span``
dotifies (dropping ``.kt``) back to the emitted ``target_str`` ``com.foo.Bar``.

Resolution is FILE-PATH matching, exactly like the Java port: an import FQN
resolves to the snapshot file whose source-root-stripped path dotifies to that
FQN. Kotlin's filename-does-not-determine-symbol-name case (``import
com.foo.util.Helper`` where ``Helper`` lives in ``Utils.kt``) is deliberately NOT
mined: such a file's identifier dotifies to ``com.foo.util.Utils`` which never
equals the emitted ``com.foo.util.Helper``, so it could neither fire nor be
oracle-reachable — it would be dead context at inference (the precise-mode link
detector attends only to matched files). Keeping resolution to file-path matches
mirrors the calibration reference and keeps every aux genuinely link-reachable.

Determinism: datapoints are read in file order (practice then public), the
per-example aux list is sorted by (path), and no randomness or wall-clock is
used, so ``examples_fn`` is reproducible (Tier 0 checks it).

Data layout (materialized 2026-07-24 under /fss-data, NOT /fss):
    raw/ase2025_kotlin/kotlin-practice.jsonl   (30 points)
    raw/ase2025_kotlin/kotlin-public.jsonl     (400 points)
    raw/ase2025_kotlin/kotlin-practice.zip     (zip-of-zips: 30 snapshots)
    raw/ase2025_kotlin/kotlin-public.zip       (zip-of-zips: 400 snapshots)
Each outer zip contains one inner snapshot zip per datapoint, named by the
datapoint's ``archive`` field; inner entries are repo-relative paths.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..schema import AuxDoc, CrossDocExample, PortAdapter
from eval.nlp_benchmarks import _strip_java_source_root

# Bulk data lives on /fss-data (never /fss). Practice + public splits only
# (private split has no ground-truth middle for the public challenge).
_DATA_DIR = Path(
    "/fss-data/evin_t/tagseq2tagseq_artifacts/raw/ase2025_kotlin"
)
_SPLITS: Tuple[Tuple[str, str], ...] = (
    ("kotlin-practice.jsonl", "kotlin-practice.zip"),
    ("kotlin-public.jsonl", "kotlin-public.zip"),
)

# Prefix import scan is only used to know WHICH sibling files to resolve into
# aux (the frozen KotlinImportDetector / oracle do the graded matching later).
# Mirror the detector: non-wildcard `import <dotted.fqn>` at line start, alias
# tolerated. Wildcards are ignored (no single target file).
_IMPORT_RE = re.compile(
    r"^[ \t]*import[ \t]+([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)[ \t]*(?:;|$)",
    re.MULTILINE,
)


def _path_to_fqn(path: str) -> Optional[str]:
    """Source-root-stripped .kt path -> dotted FQN of the file's path.

    ``app/src/main/java/fe/linksheet/Foo.kt`` -> ``fe.linksheet.Foo`` — the same
    key ``KotlinImportDetector.index_doc_span`` produces for this file's shaped
    identifier, so an import of ``fe.linksheet.Foo`` resolves here by exact
    string equality (Java-port file-path matching).
    """
    s = _strip_java_source_root(path).replace("\\", "/")
    if not s.endswith(".kt"):
        return None
    return s[: -len(".kt")].replace("/", ".")


def _build_snapshot_index(namelist: List[str]) -> Dict[str, str]:
    """Map file-path FQN -> repo-relative .kt path for one snapshot.

    Deterministic: files are processed in sorted path order, and a key already
    claimed by an earlier (lexicographically smaller) path is not overwritten,
    so index construction is order-independent.
    """
    index: Dict[str, str] = {}
    for p in sorted(n for n in namelist if n.endswith(".kt")):
        fqn = _path_to_fqn(p)
        if fqn and fqn not in index:
            index[fqn] = p
    return index


def _resolve_aux(
    prefix: str,
    inner_zip: zipfile.ZipFile,
    file_index: Dict[str, str],
    modified: set,
) -> List[AuxDoc]:
    """Resolve prefix imports to snapshot sibling files -> sorted AuxDocs.

    For each non-wildcard import FQN in the prefix, resolve to the snapshot .kt
    file whose source-root-stripped path dotifies to that FQN (exact match).
    Files listed in ``modified`` (the ground-truth commit's changed files) are
    excluded to keep aux non-self-referential and leakage-free. Returns AuxDocs
    sorted by path, deduped by path.
    """
    resolved_paths: Dict[str, str] = {}  # path -> import (dedupe by path)
    seen_imp = set()
    for m in _IMPORT_RE.finditer(prefix):
        imp = m.group(1)
        if imp in seen_imp:
            continue
        seen_imp.add(imp)
        path = file_index.get(imp)
        if path is None or path in modified:
            continue
        resolved_paths.setdefault(path, imp)

    aux: List[AuxDoc] = []
    for path in sorted(resolved_paths):
        try:
            content = inner_zip.read(path).decode("utf-8", "replace")
        except Exception:
            continue
        if content.strip():
            aux.append(AuxDoc(path=path, content=content))
    return aux


@lru_cache(maxsize=None)
def _load_outer_zip(zip_name: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(_DATA_DIR / zip_name)


def _load_examples(max_examples: Optional[int]) -> List[CrossDocExample]:
    out: List[CrossDocExample] = []
    for jsonl_name, zip_name in _SPLITS:
        jsonl_path = _DATA_DIR / jsonl_name
        outer_path = _DATA_DIR / zip_name
        if not jsonl_path.exists() or not outer_path.exists():
            continue
        outer = _load_outer_zip(zip_name)
        outer_members = set(outer.namelist())
        with open(jsonl_path) as fh:
            for line in fh:
                if max_examples is not None and len(out) >= max_examples:
                    return out
                rec = json.loads(line)
                prefix = rec.get("prefix", "")
                middle = rec.get("middle", "")
                suffix = rec.get("suffix", "")
                path = rec.get("path", "")
                archive = rec.get("archive", "")
                if not prefix.strip() or archive not in outer_members:
                    continue
                target = middle.split("\n", 1)[0]
                if not target.strip():
                    continue
                # FIM prefix+middle+suffix reconstructs the full primary file,
                # which scopes.py needs to re-anchor scoring at use sites (the
                # native scope uses only prefix→first-line-of-middle).
                full_file = prefix + middle + suffix
                inner_bytes = outer.read(archive)
                inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
                file_index = _build_snapshot_index(inner.namelist())
                modified = set(rec.get("modified", []) or [])
                aux = _resolve_aux(prefix, inner, file_index, modified)
                if not aux:
                    continue  # no cross-file context -> not a cross-doc example
                out.append(CrossDocExample(
                    repo=rec.get("repo", "repo"),
                    file_path=path,
                    context=prefix,
                    target=target,
                    aux=tuple(aux),
                    meta={"id": rec.get("id"),
                          "revision": rec.get("revision"),
                          "archive": archive,
                          "n_modified": len(modified)},
                    full_file=full_file,
                ))
    return out


def _kotlin_aux_identifier(repo: str, path: str, content: str) -> str:
    """Aux DocSpan raw_identifier so index_doc_span matches the detector FQN.

    Reuse the Java source-root strip: drop the build root from the repo-relative
    path, leaving ``com/foo/Bar.kt``. KotlinImportDetector.index_doc_span strips
    the ``<repo>:`` prefix and dotifies the ``.kt`` tail to ``com.foo.Bar`` — the
    exact dotted FQN the detector emits from an ``import com.foo.Bar``.
    """
    return f"{repo}:{_strip_java_source_root(path)}"


def _kotlin_detector(decode_fn):
    from model.graph_traversal.kotlin_import_detector import KotlinImportDetector
    return KotlinImportDetector(decode_fn)


ASE_KOTLIN = PortAdapter(
    name="ase_kotlin",
    language="kotlin",
    examples_fn=_load_examples,
    identifier_fn=_kotlin_aux_identifier,
    detector_factory=_kotlin_detector,
)
