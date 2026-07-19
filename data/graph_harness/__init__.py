"""
graph_harness — language-conformance harness for code-dataset link detection.

Grades a language's link-detection + graph-extraction implementation against an
INDEPENDENT tree-sitter oracle (detection axis) plus structural invariants and
resolvability (resolution axis). See docs/multilang_code_datasets_DESIGN.md.

The harness is FROZEN: language implementers add a `LanguageSpec` (specs/) and
fixtures; they must NOT edit the scoring code here. That separation is what makes
the gate un-gameable — the oracle query and the scoring logic are authored once,
independently of any single language implementation.

Public API:
    LanguageSpec        — per-language adapter (grammar, oracle query, normalizers)
    TreeSitterOracle    — independent ground-truth import extractor
    score_detection     — precision/recall of an import set vs. the oracle
    DetectionScore      — result dataclass with pass/fail + example mismatches
"""
from .spec import LanguageSpec
from .oracle import TreeSitterOracle, OracleImport
from .scoring import DetectionScore, score_detection, PrecisionRecall

__all__ = [
    "LanguageSpec",
    "TreeSitterOracle",
    "OracleImport",
    "DetectionScore",
    "score_detection",
    "PrecisionRecall",
]
