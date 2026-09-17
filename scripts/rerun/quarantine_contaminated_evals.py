#!/usr/bin/env python
"""Quarantine eval output that the old eval_checkpoints.py wrote INTO training run dirs.

Before the standalone-eval-dir fix, every eval run wrote `<run>/eval/<ts>/` (its
ReproducibilityManager capture) and merged its numbers into the training run's own
`<run>/eval_results.json`. Those files are not trustworthy as training provenance, so
this script MOVES them (never deletes) into a quarantine root, mirroring the run layout,
and writes a manifest so the move is reversible.

Default is a dry run. Run only AFTER the distiller that re-attaches evals from
$TS2TS_EVALS_ROOT is on the branch you distill from — before that, moving these files
drops the affected metrics from provenance/runs/ records.

    python scripts/rerun/quarantine_contaminated_evals.py            # list only
    python scripts/rerun/quarantine_contaminated_evals.py --execute # move + manifest
    python scripts/rerun/quarantine_contaminated_evals.py --undo <manifest.json>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULT_ROOTS = [
    "/fss/evin_t/tagseq2tagseq/runs",
    "/fss-data/evin_t/tagseq2tagseq_artifacts/runs",
]
DEFAULT_QUARANTINE = "/fss-data/evin_t/tagseq2tagseq_artifacts/quarantine_contaminated_evals"
# Anything touched more recently than this is treated as possibly live and skipped.
DEFAULT_MIN_AGE_DAYS = 3


def _training_last_mtime(run: Path):
    cands = list((run / "checkpoints").glob("*")) + list(run.glob("*.log")) + list(run.glob("train*"))
    return max((c.stat().st_mtime for c in cands), default=None)


def collect(roots, min_age_days, include_results_json):
    now = time.time()
    out = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for run in sorted(p for p in root.iterdir() if p.is_dir()):
            tdone = _training_last_mtime(run)
            targets = [run / "eval"]
            if include_results_json:
                targets.append(run / "eval_results.json")
            for t in targets:
                if not t.exists():
                    continue
                mtime = t.stat().st_mtime
                age_days = (now - mtime) / 86400
                after_training = None if tdone is None else (mtime > tdone + 60)
                out.append({
                    "path": str(t),
                    "kind": "eval_dir" if t.is_dir() else "eval_results",
                    "mtime": time.strftime("%Y-%m-%d", time.gmtime(mtime)),
                    "written_after_training": after_training,
                    "skip_reason": ("recent" if age_days < min_age_days else None),
                })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--quarantine-root", default=DEFAULT_QUARANTINE)
    ap.add_argument("--min-age-days", type=float, default=DEFAULT_MIN_AGE_DAYS)
    ap.add_argument("--eval-dirs-only", action="store_true",
                    help="Move only <run>/eval/ (pure eval-run output); leave eval_results.json, "
                         "which may also hold the training run's own end-of-training eval.")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--undo", metavar="MANIFEST", help="Move everything in MANIFEST back.")
    args = ap.parse_args(argv)

    if args.undo:
        man = json.load(open(args.undo))
        for e in man["moved"]:
            Path(e["src"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(e["dst"], e["src"])
        print(f"restored {len(man['moved'])} paths")
        return 0

    entries = collect(args.roots, args.min_age_days, include_results_json=not args.eval_dirs_only)
    todo = [e for e in entries if e["skip_reason"] is None]
    skipped = [e for e in entries if e["skip_reason"]]
    for e in entries:
        flag = f"SKIP({e['skip_reason']})" if e["skip_reason"] else "MOVE"
        print(f"{flag:14s} {e['kind']:13s} {e['mtime']}  after_training={e['written_after_training']}  {e['path']}")
    print(f"\n{len(todo)} to move, {len(skipped)} skipped as recent (< {args.min_age_days} days)")
    if not args.execute:
        print("dry run — pass --execute to move")
        return 0

    qroot = Path(args.quarantine_root)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "moved": []}
    for e in todo:
        src = Path(e["path"])
        # mirror <root-basename>/<run_id>/<eval|eval_results.json>
        dst = qroot / src.parent.parent.name / src.parent.name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        manifest["moved"].append({"src": str(src), "dst": str(dst), **e})
    qroot.mkdir(parents=True, exist_ok=True)
    mpath = qroot / f"manifest_{manifest['created'].replace(':', '')}.json"
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"moved {len(manifest['moved'])} paths; manifest → {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
