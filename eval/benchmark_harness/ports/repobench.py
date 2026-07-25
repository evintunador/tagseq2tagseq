"""RepoBench python/java ports — the CALIBRATION REFERENCE.

Maps tianyang/repobench_{python,java}_v1.1 cross_file_first into the canonical
CrossDocExample schema using exactly the field mapping and identifier shaping
of eval/nlp_benchmarks.py::run_repobench_cross_doc (incl. the java
source-root strip). Running the harness on these two ports defines the
legitimacy band new ports are judged against.
"""
from __future__ import annotations

from typing import List, Optional

from ..schema import AuxDoc, CrossDocExample, PortAdapter
from eval.nlp_benchmarks import _repobench_aux_identifier

_CACHE_DIR = "data/.cache/repobench"


def _load_repobench(language: str, max_examples: Optional[int]) -> List[CrossDocExample]:
    from datasets import load_dataset
    raw = load_dataset(
        f"tianyang/repobench_{language}_v1.1",
        split="cross_file_first",
        cache_dir=_CACHE_DIR,
        verification_mode="no_checks",
    )
    out: List[CrossDocExample] = []
    limit = max_examples if max_examples is not None else len(raw)
    for ex in raw.select(range(min(limit, len(raw)))):
        next_line = ex.get("next_line", "")
        if not next_line.strip():
            continue
        aux = tuple(
            AuxDoc(path=item.get("path", ""), content=item.get("snippet", ""))
            for item in ex.get("context", [])
            if item.get("snippet", "").strip()
        )
        out.append(CrossDocExample(
            repo=ex.get("repo_name", "repo"),
            file_path=ex.get("file_path", ""),
            context=ex.get("import_statement", "") + "\n" + ex.get("cropped_code", ""),
            target="\n" + next_line,
            aux=aux,
            meta={"gold_snippet_index": ex.get("gold_snippet_index"),
                  "created_at": ex.get("created_at"),
                  "level": ex.get("level")},
        ))
        if max_examples is not None and len(out) >= max_examples:
            break
    return out


def _python_detector(decode_fn):
    from model.graph_traversal.python_import_detector import PythonImportDetector
    return PythonImportDetector(decode_fn)


def _java_detector(decode_fn):
    from model.graph_traversal.java_import_detector import JavaImportDetector
    return JavaImportDetector(decode_fn)


REPOBENCH_PYTHON = PortAdapter(
    name="repobench_python",
    language="python",
    examples_fn=lambda n: _load_repobench("python", n),
    identifier_fn=lambda repo, path, content: _repobench_aux_identifier(
        "python", repo, path, content),
    detector_factory=_python_detector,
)

REPOBENCH_JAVA = PortAdapter(
    name="repobench_java",
    language="java",
    examples_fn=lambda n: _load_repobench("java", n),
    identifier_fn=lambda repo, path, content: _repobench_aux_identifier(
        "java", repo, path, content),
    detector_factory=_java_detector,
)
