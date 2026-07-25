"""CoLT-132K Go port — cross-file API-invocation scenario (`go_api`).

Maps the Go test split of CoLT-132K (aiXcoder-7B-v2, Zenodo 15019938,
CC-BY-4.0) into the canonical CrossDocExample schema, mirroring the
RepoBench reference adapter (ports/repobench.py):

  * context = `prefix` (starts at the file's `package` line, so the
    `import (...)` block is visible — required for the Go import detector).
  * target  = first line of `middle` (next-line scope, like RepoBench's
    next_line), prefixed with a newline so the flat/packed completion ids match.
  * file_path = `code_file_path`; repo = `project_dir`.
  * aux = each `cross_file_dependency` entry as
    AuxDoc(path=code_file_path, content=abstraction). `abstraction` is a
    declaration-only signature skeleton (expected — weaker than a full file).

identifier shaping
------------------
A Go corpus NODE is a PACKAGE (a directory), and GoImportDetector emits the
bare package import path as target_str, returning it unchanged from
index_doc_span. A CoLT aux `code_file_path` is a repo-relative *file* path
(`<module>/pkg/sub/file.go`) whose first component is the module root
(== project_dir). The package import path the detector matches is therefore
the aux file path's DIRECTORY (drop the `file.go` component). So
identifier_fn returns dirname(path) — no `repo:` prefix, because the path
already carries the module-qualified prefix and index_doc_span does an exact
string compare against the emitted import path.

DATASET BLOCKER (verified 2026-07-25)
-------------------------------------
Every Go test example in the release (all 3,000 across go_api / go_line /
go_structured_span, and also po_data/sft_data) ships an EMPTY
`cross_file_dependency` list — the dependency records live in external
`/data/godata/.../*.json` files (see `dependency_file_path`) that are NOT
included in CoLT-132K.zip. Python (6,796 dep entries / 1,000 examples) and
Java (128,039 / 1,000) are fully populated; only Go's dependency aux is
missing. Consequently this port produces examples with NO aux docs and fails
Tier 0's no-aux gate. The mapping code below is correct and will yield real
cross-doc examples if/when the dependency JSONs are recovered; until then Go
via CoLT-132K is blocked. `similar_functions` (retrieval, populated) is
deliberately NOT used as primary aux — those are Jaccard-similar code blocks,
not import edges, so they carry no import-licensed cross-doc signal.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from ..schema import AuxDoc, CrossDocExample, PortAdapter

# The go_api split = cross-file API-invocation scenario (the dependency-based
# one), per the design doc's "port only the cross-file API-invocation scenario
# first" note.
_GO_API_JSONL = (
    "/fss-data/evin_t/tagseq2tagseq_artifacts/raw/colt132k/"
    "test_data/go_api.jsonl"
)


def _first_line(text: str) -> str:
    """First non-empty line of `middle` (next-line completion scope)."""
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def _go_package_identifier(repo: str, path: str, content: str) -> str:
    """Shape an aux file path into the Go package import path the detector emits.

    `<module>/pkg/sub/file.go` -> `<module>/pkg/sub`. GoImportDetector emits the
    bare import path (a package directory) and index_doc_span returns it
    unchanged, so the raw_identifier must be exactly that directory. Drops a
    trailing `.go` file component; a path that is already a directory is
    returned unchanged.
    """
    p = path.replace("\\", "/").strip().strip("/")
    if p.endswith(".go"):
        p = os.path.dirname(p)
    return p


def _load_colt_go(max_examples: Optional[int]) -> List[CrossDocExample]:
    if not os.path.exists(_GO_API_JSONL):
        return []
    out: List[CrossDocExample] = []
    with open(_GO_API_JSONL) as f:
        for line in f:
            rec = json.loads(line)
            target = _first_line(rec.get("middle", ""))
            if not target.strip():
                continue
            context = rec.get("prefix", "")
            # Dependency edges (SCIP import graph) are the real cross-doc aux.
            # Sorted by (path, content) for determinism.
            deps = rec.get("cross_file_dependency", []) or []
            aux_items = sorted(
                (
                    (d.get("code_file_path", ""), d.get("abstraction", ""))
                    for d in deps
                    if (d.get("abstraction", "") or "").strip()
                ),
                key=lambda t: (t[0], t[1]),
            )
            aux = tuple(AuxDoc(path=p, content=c) for p, c in aux_items)
            out.append(CrossDocExample(
                repo=rec.get("project_dir", "repo"),
                file_path=rec.get("code_file_path", ""),
                context=context,
                target="\n" + target,
                aux=aux,
                meta={
                    "namespace": rec.get("namespace"),
                    "function_name": rec.get("function_name"),
                    "invoked_item": rec.get("invoked_item"),
                    "dependency_file_path": rec.get("dependency_file_path"),
                },
            ))
            if max_examples is not None and len(out) >= max_examples:
                break
    return out


def _go_detector(decode_fn):
    from model.graph_traversal.go_import_detector import GoImportDetector
    return GoImportDetector(decode_fn)


COLT_GO = PortAdapter(
    name="colt_go",
    language="go",
    examples_fn=lambda n: _load_colt_go(n),
    identifier_fn=_go_package_identifier,
    detector_factory=_go_detector,
)
