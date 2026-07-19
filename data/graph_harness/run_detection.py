"""
CLI: grade a language's DETECTION implementation against the tree-sitter oracle.

Usage:
    python -m data.graph_harness.run_detection <lang> --files <dir_or_glob> \
        [--min-precision 0.95] [--min-recall 0.90] [--max-files N]

`<lang>` selects a registered LanguageSpec (python, go, ...). The runner:
  1. reads source files (by extension) from the given path(s),
  2. computes oracle import keys per file (tree-sitter, independent),
  3. runs the language's registered LinkDetector via `make_link_detector` on the
     tokenized file (GPT-2, matching training), collecting emitted target_str,
  4. scores micro-averaged precision/recall and prints concrete FP/FN examples,
  5. exits non-zero if thresholds are not met.

This is the DETECTION axis only (design doc §2). Resolution (does a target_str
hit a real corpus node?) is graded separately against fixtures + toolchain +
invariants — a detector passing here has NOT been shown to resolve correctly.

The detector name passed to `make_link_detector` defaults to `<lang>` but can be
overridden with --detector for languages whose registered name differs.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

import torch

from data.graph_harness import TreeSitterOracle, score_detection
from data.graph_harness.specs import get_spec


def _iter_source_files(paths: List[str], extensions: frozenset, max_files: int):
    exts = {e.lstrip(".") for e in extensions}
    seen = 0
    for path in paths:
        candidates = []
        if os.path.isdir(path):
            for root, _dirs, files in os.walk(path):
                for f in files:
                    candidates.append(os.path.join(root, f))
        else:
            candidates = glob.glob(path, recursive=True)
        for fp in sorted(candidates):
            if fp.rsplit(".", 1)[-1] not in exts:
                continue
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                    yield fp, fh.read()
            except (OSError, UnicodeError):
                continue
            seen += 1
            if max_files and seen >= max_files:
                return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("lang", help="registered LanguageSpec name (python, go, ...)")
    ap.add_argument("--files", nargs="+", required=True,
                    help="directories or globs of source files to grade on")
    ap.add_argument("--detector", default=None,
                    help="link_detector name for make_link_detector "
                         "(default: same as <lang>)")
    ap.add_argument("--min-precision", type=float, default=0.95)
    ap.add_argument("--min-recall", type=float, default=0.90)
    ap.add_argument("--max-files", type=int, default=0, help="0 = no limit")
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args(argv)

    spec = get_spec(args.lang)
    oracle = TreeSitterOracle(spec)

    import tiktoken
    from model.graph_traversal.link_detector import make_link_detector
    enc = tiktoken.get_encoding(args.tokenizer)
    detector = make_link_detector(args.detector or args.lang, enc.decode)

    per_file = {}
    for fp, source in _iter_source_files(args.files, spec.extensions, args.max_files):
        oracle_keys = oracle.import_keys(source)
        ids = torch.tensor(enc.encode(source), dtype=torch.long)
        detected = [li.target_str for li in detector.detect_links(ids)]
        per_file[fp] = (oracle_keys, detected)

    if not per_file:
        print("No source files found for extensions "
              f"{sorted(spec.extensions)} under {args.files}", file=sys.stderr)
        return 2

    score = score_detection(per_file, spec.canonical_target)
    print(f"[{args.lang}] detection score: {score.summary()}")
    if score.false_negative_examples:
        print(f"  MISSED (recall gaps), up to {len(score.false_negative_examples)}:")
        for label, key in score.false_negative_examples:
            print(f"    - {label}: {key}")
    if score.false_positive_examples:
        print(f"  SPURIOUS (precision gaps), up to {len(score.false_positive_examples)}:")
        for label, key in score.false_positive_examples:
            print(f"    + {label}: {key}")

    ok = score.passes(args.min_precision, args.min_recall)
    print(f"  gate: precision>={args.min_precision} recall>={args.min_recall} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
