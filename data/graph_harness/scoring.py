"""
Frozen detection scorer.

Compares the set of canonical import keys an implementation emits against the set
the tree-sitter oracle finds, and reports precision / recall plus the concrete
false-positive and false-negative keys (so a failing implementation gets
actionable examples, not just a number).

Scoring is done over SETS of canonical keys per file, then micro-averaged across
files (sum TP/FP/FN, then divide) so that files with many imports weigh
proportionally — a detector can't inflate its score by nailing a few
import-light files while failing import-heavy ones.

DESIGN NOTE — why this closes the reward hacks (see design doc §2):
  * recall < 1 is the ONLY thing that catches "emit nothing" / "emit only the
    easy imports" — a resolution-rate metric rewards exactly those hacks, so
    recall against an independent oracle is load-bearing.
  * precision < 1 catches hallucinated / spurious links.
  * because the oracle key set comes from tree-sitter on files the implementer
    does not curate, neither number can be gamed by teaching to a fixed test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


@dataclass(frozen=True)
class PrecisionRecall:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return 1.0 if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return 1.0 if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

    def __add__(self, other: "PrecisionRecall") -> "PrecisionRecall":
        return PrecisionRecall(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


@dataclass
class DetectionScore:
    """Micro-averaged detection score across a set of files, with examples."""
    counts: PrecisionRecall
    n_files: int
    # Up to `max_examples` concrete mismatches, each (file_label, key).
    false_positive_examples: List[tuple] = field(default_factory=list)
    false_negative_examples: List[tuple] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.counts.precision

    @property
    def recall(self) -> float:
        return self.counts.recall

    @property
    def f1(self) -> float:
        return self.counts.f1

    def passes(self, min_precision: float, min_recall: float) -> bool:
        return self.precision >= min_precision and self.recall >= min_recall

    def summary(self) -> str:
        return (
            f"files={self.n_files} "
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
            f"(tp={self.counts.tp} fp={self.counts.fp} fn={self.counts.fn})"
        )


def _project_targets(
    targets: Iterable[str],
    canonical_target,
) -> Set[str]:
    """Project detector/extractor target strings into the canonical key space.

    Drops any target the spec maps to None. The MANY-to-one collapse here is what
    lets a detector legitimately emit `foo/bar.py` AND `foo/bar/__init__.py` for
    one import without being penalized: both project to the same canonical key.
    """
    keys: Set[str] = set()
    for t in targets:
        k = canonical_target(t)
        if k is not None:
            keys.add(k)
    return keys


def score_detection(
    per_file: Dict[str, tuple],
    canonical_target,
    max_examples: int = 20,
) -> DetectionScore:
    """Score an implementation's detected imports against oracle imports.

    Args:
        per_file: mapping file_label -> (oracle_keys: set[str],
                  detected_targets: iterable[str]). oracle_keys are already
                  canonical (from TreeSitterOracle.import_keys); detected_targets
                  are raw strings the implementation emitted, projected here via
                  canonical_target.
        canonical_target: spec.canonical_target — projects a raw emitted target
                  into the canonical key space (or None to ignore it).
        max_examples: cap on stored mismatch examples per category.

    Returns:
        DetectionScore (micro-averaged P/R/F1 + example FPs/FNs).
    """
    total = PrecisionRecall(0, 0, 0)
    fp_examples: List[tuple] = []
    fn_examples: List[tuple] = []
    n_files = 0

    for label, (oracle_keys, detected_targets) in per_file.items():
        n_files += 1
        oracle_set: Set[str] = set(oracle_keys)
        detected_set = _project_targets(detected_targets, canonical_target)

        tp_keys = detected_set & oracle_set
        fp_keys = detected_set - oracle_set
        fn_keys = oracle_set - detected_set

        total = total + PrecisionRecall(len(tp_keys), len(fp_keys), len(fn_keys))

        for k in sorted(fp_keys):
            if len(fp_examples) < max_examples:
                fp_examples.append((label, k))
        for k in sorted(fn_keys):
            if len(fn_examples) < max_examples:
                fn_examples.append((label, k))

    return DetectionScore(
        counts=total,
        n_files=n_files,
        false_positive_examples=fp_examples,
        false_negative_examples=fn_examples,
    )
