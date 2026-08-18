#!/usr/bin/env python3
"""Verify the paper's numbers are grounded in run provenance; fail on drift.

Intended as a pre-push / CI gate (mirrors the code_briefs drift protocol). Checks:

  (a) Ledger resolves      -- every entry resolves against a real run record, all its
                              run_ids agree, and any `expected` still matches (catches an
                              eval_results.json re-merge that silently changed a number).
  (b) values.tex fresh     -- regenerate in memory and diff the committed
                              paper/generated/values.tex; stale = fail.
  (c) \\val usage           -- every \\val{key} cited in paper/sections/**/*.tex exists in
                              the ledger (fail); every ledger key is cited (warn; fail under
                              --strict).
  (d) Run-dir liveness     -- warn on records whose source_run_dirs vanished but aren't
                              marked run_dir_exists=false (run distill_runs.py --prune-missing).
  (e) Patch integrity      -- every dirty record's referenced patch file exists and its
                              sha256 matches the recorded hash.

`kind: literal` entries are reported in a separate "declared but ungrounded" bucket
(log-derived numbers to reverify at camera-ready). Exit code is nonzero if any hard
check fails.

Examples:
  python scripts/check_grounding.py
  python scripts/check_grounding.py --strict     # orphan ledger keys also fail
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance_lib as pl  # noqa: E402
import gen_values_tex  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VAL_RE = re.compile(r"\\val\{([^}]*)\}")


def check(args):
    failures = []
    warnings = []
    literals = []

    entries = pl.load_ledger(args.ledger)
    records = pl.load_records(args.runs)

    # (a) resolve every entry
    try:
        content = gen_values_tex.build_values_tex(entries, records)
    except pl.ResolveError as e:
        failures.append(f"(a) ledger resolution: {e}")
        content = None

    for key, entry in entries.items():
        if (entry.get("source") or {}).get("kind") == "literal":
            literals.append((key, entry.get("note", "")))

    # (b) values.tex freshness
    values_path = Path(args.values)
    if content is not None:
        on_disk = values_path.read_text() if values_path.exists() else None
        if on_disk is None:
            failures.append(f"(b) {values_path} missing -- run gen_values_tex.py")
        elif on_disk != content:
            failures.append(f"(b) {values_path} is stale -- run gen_values_tex.py")

    # (c) \val usage cross-check
    used = set()
    for tex in Path(args.paper_sections).rglob("*.tex"):
        for m in VAL_RE.finditer(tex.read_text()):
            used.add(m.group(1))
    for key in sorted(used - set(entries)):
        failures.append(f"(c) paper cites \\val{{{key}}} but ledger has no such key")
    orphans = sorted(set(entries) - used)
    if orphans:
        msg = f"(c) {len(orphans)} ledger keys never cited by \\val: {', '.join(orphans)}"
        (failures if args.strict else warnings).append(msg)

    # (d) run-dir liveness
    for run_id, rec in records.items():
        if rec.get("run_dir_exists") is False:
            continue
        dirs = rec.get("source_run_dirs") or []
        if dirs and not any(Path(d).is_dir() for d in dirs):
            warnings.append(f"(d) record {run_id} run dirs gone but not archived "
                            f"(run distill_runs.py --prune-missing)")

    # (e) patch integrity
    for run_id, rec in records.items():
        patch = (rec.get("reproduce") or {}).get("patch")
        if not patch:
            continue
        pf = REPO_ROOT / patch["path"]
        if not pf.is_file():
            failures.append(f"(e) record {run_id} references missing patch {patch['path']}")
        elif hashlib.sha256(pf.read_bytes()).hexdigest() != patch["hash"]:
            failures.append(f"(e) patch {patch['path']} sha256 != recorded hash")

    # report
    print(f"check_grounding: {len(entries)} ledger entries, {len(records)} run records")
    for w in warnings:
        print(f"  WARN  {w}")
    if literals:
        print(f"  {len(literals)} literal (log-derived, ungrounded) entries to reverify:")
        for key, note in literals:
            print(f"    - {key}: {note.strip().splitlines()[0] if note else ''}")
    for f in failures:
        print(f"  FAIL  {f}")
    if failures:
        print(f"\nFAILED: {len(failures)} grounding check(s) failed")
        return 1
    print(f"\nOK: grounding checks passed"
          + (f" ({len(warnings)} warning(s))" if warnings else ""))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default=str(REPO_ROOT / "provenance" / "ledger.yaml"))
    ap.add_argument("--runs", default=str(REPO_ROOT / "provenance" / "runs"))
    ap.add_argument("--values", default=str(REPO_ROOT / "paper" / "generated" / "values.tex"))
    ap.add_argument("--paper-sections", default=str(REPO_ROOT / "paper" / "sections"))
    ap.add_argument("--strict", action="store_true", help="Orphan ledger keys also fail")
    args = ap.parse_args()
    return check(args)


if __name__ == "__main__":
    raise SystemExit(main())
