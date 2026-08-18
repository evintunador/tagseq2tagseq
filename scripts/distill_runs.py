#!/usr/bin/env python3
"""Distill minimal, committable provenance records from run directories.

Training/eval runs record rich provenance (git SHA + dirty patch, resolved config,
invocation, environment, eval metrics) under `runs/`, but `runs/` is gitignored and
ephemeral -- the provenance dies with the run dir. This harvester reads each run dir
and emits a small, self-contained JSON record into the tracked `provenance/` tree, so
the code + config state behind every result survives run-dir deletion and can ground
the paper.

For each run it reads (via the `reproducibility/main` symlink -> rank-0 node dir):
git_info.json, run_invocation.json, software_environment.json, runtime_environment.json,
plus run-dir-level hyperparameters.json and eval_results.json. Git-dirty runs get their
uncommitted patch copied into a content-addressed store (`provenance/patches/<hash>.patch`)
so a dirty run reproduces from `git checkout <commit> && git apply <patch>`.

Idempotent: re-running only rewrites a record when its content actually changed (the
`distilled_at` stamp alone never triggers a rewrite), so re-distilling is a clean no-op.
Records whose run dirs have been deleted are kept; `--prune-missing` marks them archived.
Reads run dirs only -- never modifies them or the external tunalab package.

Examples:
  # See what would change without writing anything:
  python scripts/distill_runs.py --dry-run
  # Harvest every run in both default roots:
  python scripts/distill_runs.py
  # Re-distill a single run:
  python scripts/distill_runs.py --run-id run_20260814_154619_610214
  # Flag records whose run dirs no longer exist:
  python scripts/distill_runs.py --prune-missing
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = [
    REPO_ROOT / "runs",
    Path("/fss-data/evin_t/tagseq2tagseq_artifacts/runs"),
]

# run_YYYYMMDD_HHMMSS_ffffff (submitit) and bare YYYYMMDD_HHMMSS (direct main.py)
RE_SUBMITIT = re.compile(r"^run_(\d{8})_(\d{6})_(\d{6})$")
RE_BARE = re.compile(r"^(\d{8})_(\d{6})$")


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(obj) -> str:
    """Hash a JSON-able object independent of key ordering / whitespace."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def read_json(path: Path):
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def classify_run_id(run_id: str):
    """Return (naming_convention, started_iso) or (None, None) if not a run dir."""
    m = RE_SUBMITIT.match(run_id)
    if m:
        return "submitit", _iso_from_parts(m.group(1), m.group(2))
    m = RE_BARE.match(run_id)
    if m:
        return "bare", _iso_from_parts(m.group(1), m.group(2))
    return None, None


def _iso_from_parts(ymd: str, hms: str):
    try:
        dt = datetime.datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


def parse_invocation_argv(argv):
    """Split a run_invocation argv list into (config_path, {dotted_key: value})."""
    config_path = None
    overrides = {}
    # argv[0] is the entrypoint token ("main"); skip it.
    tokens = list(argv[1:]) if argv else []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not isinstance(tok, str) or not tok.startswith("--"):
            i += 1
            continue
        if "=" in tok:  # --key=value form
            key, val = tok[2:].split("=", 1)
        else:
            key = tok[2:]
            if i + 1 < len(tokens) and not str(tokens[i + 1]).startswith("--"):
                val = tokens[i + 1]
                i += 1
            else:
                val = True  # bare flag
        if key == "config":
            config_path = val
        else:
            overrides[key] = val
        i += 1
    return config_path, overrides


def find_run_dirs(roots):
    """Map run_id -> list of existing run dirs (across roots) that look like runs."""
    found = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            conv, _ = classify_run_id(child.name)
            if conv is None:
                continue
            found.setdefault(child.name, []).append(child)
    return found


def build_record(run_id, run_dirs, patches_dir, dry_run=False):
    """Build a provenance record dict from the run dirs for one run_id.

    Copies the uncommitted patch into `patches_dir` as a side effect for dirty runs
    (unless `dry_run`). Returns the record dict (without a committed `distilled_at` stamp).
    """
    conv, started_iso = classify_run_id(run_id)
    existing_dirs = [d for d in run_dirs if d.is_dir()]
    source_run_dirs = [str(d) for d in existing_dirs]

    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "naming_convention": conv,
        "source_run_dirs": source_run_dirs,
        "run_dir_exists": bool(existing_dirs),
        "warnings": [],
        "reproduce": None,
        "environment": None,
        "hyperparameters": None,
        "hyperparameters_sha256": None,
        "timestamps": {"run_started": started_iso},
        "eval": {"present": False, "eval_results_sha256": None, "metrics": None},
    }
    warnings = record["warnings"]

    # Pick a canonical dir: the first with a readable reproducibility/main/git_info.json.
    canonical = None
    git_info = None
    for d in existing_dirs:
        gi = read_json(d / "reproducibility" / "main" / "git_info.json")
        if gi is not None:
            canonical, git_info = d, gi
            break

    # Collision guard: differing commits across dirs sharing a basename = real conflict.
    commits = set()
    for d in existing_dirs:
        gi = read_json(d / "reproducibility" / "main" / "git_info.json")
        if gi and gi.get("commit_hash"):
            commits.add(gi["commit_hash"])
    if len(commits) > 1:
        raise RuntimeError(
            f"run_id {run_id} maps to dirs with differing commit_hash {commits}; "
            f"disambiguate with --run-id and inspect {source_run_dirs}"
        )

    if canonical is None:
        warnings.append("no reproducibility dir")
        # Still capture hyperparameters if a run dir has them.
        for d in existing_dirs:
            hp = read_json(d / "hyperparameters.json")
            if hp is not None:
                record["hyperparameters"] = hp
                record["hyperparameters_sha256"] = canonical_sha256(hp)
                break
        _attach_eval(record, existing_dirs)
        return record

    main = canonical / "reproducibility" / "main"
    inv = read_json(main / "run_invocation.json") or {}
    sw = read_json(main / "software_environment.json") or {}
    rt = read_json(main / "runtime_environment.json") or {}

    config_path, overrides = parse_invocation_argv(inv.get("argv") or [])
    env = inv.get("env") or {}
    pkgs = sw.get("package_versions") or {}
    cuda_rt = rt.get("cuda_runtime") or {}
    dist = rt.get("distributed") or {}
    devices = rt.get("device_properties") or []

    # Copy the uncommitted patch into the content-addressed store when dirty.
    patch_field = None
    if git_info.get("git_is_dirty"):
        patch_src = main / "uncommitted_changes.patch"
        recorded_hash = git_info.get("patch_file_hash")
        if patch_src.is_file():
            data = patch_src.read_bytes()
            actual_hash = sha256_bytes(data)
            if recorded_hash and actual_hash != recorded_hash:
                warnings.append(
                    f"patch hash mismatch (recorded {recorded_hash[:12]}, "
                    f"actual {actual_hash[:12]})"
                )
            phash = recorded_hash or actual_hash
            dest = patches_dir / f"{phash}.patch"
            if not dry_run and not dest.exists():
                patches_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(patch_src, dest)
            patch_field = {
                "hash": phash,
                "path": f"provenance/patches/{phash}.patch",
                "bytes": len(data),
            }
        else:
            warnings.append("git dirty but no uncommitted_changes.patch found")

    record["reproduce"] = {
        "commit_hash": git_info.get("commit_hash"),
        "branch": git_info.get("branch"),
        "remote_url": git_info.get("remote_url"),
        "github_url": git_info.get("github_url"),
        "git_is_dirty": git_info.get("git_is_dirty"),
        "patch": patch_field,
        "config_path": config_path,
        "cli_overrides": overrides,
        "python_version": (sw.get("python_version") or "").split(" ")[0] or None,
        "torch_version": pkgs.get("torch"),
        "torch_cuda_version": cuda_rt.get("torch_cuda_version"),
        "cudnn_version": cuda_rt.get("cudnn_version"),
        "nvidia_driver_version": cuda_rt.get("nvidia_driver_version"),
        "slurm_job_id": env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID"),
    }

    os_info = rt.get("os") or {}
    record["environment"] = {
        "gpu_model": devices[0].get("name") if devices else None,
        "gpu_count": rt.get("device_count"),
        "world_size": dist.get("world_size"),
        "os_release": os_info.get("release"),
    }

    record["timestamps"] = {
        "run_started": started_iso,
        "slurm_start_time": env.get("SLURM_JOB_START_TIME"),
        "slurm_end_time": env.get("SLURM_JOB_END_TIME"),
    }

    hp = read_json(canonical / "hyperparameters.json")
    if hp is not None:
        record["hyperparameters"] = hp
        record["hyperparameters_sha256"] = canonical_sha256(hp)
    else:
        warnings.append("no hyperparameters.json")

    _attach_eval(record, [canonical] + [d for d in existing_dirs if d != canonical])
    return record


def _attach_eval(record, dirs):
    for d in dirs:
        ev = read_json(d / "eval_results.json")
        if ev is not None:
            record["eval"] = {
                "present": True,
                "eval_results_sha256": canonical_sha256(ev),
                "metrics": ev,
            }
            return


def record_body(rec):
    """A record minus the volatile stamp, for change detection."""
    return {k: v for k, v in rec.items() if k != "distilled_at"}


def write_if_changed(path, record, dry_run):
    """Write record only if its body differs from the on-disk file. Returns action."""
    existing = read_json(path) if path.exists() else None
    if existing is not None and record_body(existing) == record_body(record):
        return "unchanged"
    action = "update" if existing is not None else "create"
    if not dry_run:
        record["distilled_at"] = utcnow()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return action


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=[str(r) for r in DEFAULT_ROOTS],
                    help="Run-dir roots to scan (default: in-repo runs/ + fss-data runs/)")
    ap.add_argument("--out", default=str(REPO_ROOT / "provenance"),
                    help="Provenance output dir (default: provenance/)")
    ap.add_argument("--run-id", nargs="+", default=None,
                    help="Distill only these run_ids (default: all discovered)")
    ap.add_argument("--dry-run", action="store_true", help="Report changes, write nothing")
    ap.add_argument("--prune-missing", action="store_true",
                    help="Set run_dir_exists=false on records whose run dirs are gone")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    runs_out = out / "runs"
    patches_dir = out / "patches"

    discovered = find_run_dirs(args.roots)
    if args.run_id:
        want = set(args.run_id)
        discovered = {k: v for k, v in discovered.items() if k in want}
        for rid in sorted(want - set(discovered)):
            print(f"  requested run_id not found in any root: {rid}", file=sys.stderr)

    counts = {"create": 0, "update": 0, "unchanged": 0, "partial": 0, "error": 0}
    patch_hashes = set()
    eval_count = 0
    warnings_all = []

    for run_id in sorted(discovered):
        run_dirs = discovered[run_id]
        try:
            rec = build_record(run_id, run_dirs, patches_dir, dry_run=args.dry_run)
        except RuntimeError as e:
            counts["error"] += 1
            print(f"  ERROR {run_id}: {e}", file=sys.stderr)
            continue
        if rec["reproduce"] is None:
            counts["partial"] += 1
        if rec["eval"]["present"]:
            eval_count += 1
        if rec["reproduce"] and rec["reproduce"].get("patch"):
            patch_hashes.add(rec["reproduce"]["patch"]["hash"])
        action = write_if_changed(runs_out / f"{run_id}.json", rec, args.dry_run)
        counts[action] += 1
        if rec["warnings"]:
            warnings_all.append((run_id, rec["warnings"]))
        if args.verbose:
            print(f"  {action:9s} {run_id}"
                  + (f"  [{', '.join(rec['warnings'])}]" if rec["warnings"] else ""))

    # Optionally flag records whose run dirs vanished (never delete the record).
    pruned = 0
    if args.prune_missing and runs_out.is_dir():
        live_ids = set(discovered)
        for rec_path in sorted(runs_out.glob("*.json")):
            rid = rec_path.stem
            if rid in live_ids:
                continue
            rec = read_json(rec_path)
            if rec and rec.get("run_dir_exists") is not False:
                rec["run_dir_exists"] = False
                if not args.dry_run:
                    rec["distilled_at"] = utcnow()
                    with rec_path.open("w") as f:
                        json.dump(rec, f, indent=2, ensure_ascii=False)
                        f.write("\n")
                pruned += 1

    verb = "would " if args.dry_run else ""
    print(f"\ndistill_runs summary{' (dry run)' if args.dry_run else ''}:")
    print(f"  discovered run dirs : {len(discovered)}")
    print(f"  {verb}create        : {counts['create']}")
    print(f"  {verb}update        : {counts['update']}")
    print(f"  unchanged          : {counts['unchanged']}")
    print(f"  partial (no repro) : {counts['partial']}")
    print(f"  with eval metrics  : {eval_count}")
    print(f"  unique patches     : {len(patch_hashes)}")
    if args.prune_missing:
        print(f"  {verb}prune-missing : {pruned}")
    if counts["error"]:
        print(f"  ERRORS             : {counts['error']}")
    if warnings_all and not args.verbose:
        print(f"  runs with warnings : {len(warnings_all)} (re-run with --verbose to list)")

    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
