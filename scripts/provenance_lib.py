"""Shared resolution logic for the provenance ledger.

Both `gen_values_tex.py` (writes paper/generated/values.tex) and `check_grounding.py`
(verifies it) import this so they can never disagree -- if the generator and checker
resolved values differently, the freshness check would be meaningless.

A "resolution" turns a ledger entry into a rendered LaTeX string by reading the field it
points at out of the distilled run record(s) under provenance/runs/. Resolution failures
raise ResolveError with a message naming the claim-key, run_id and the missing path.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

KEY_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
# Relative tolerance for expected-vs-resolved and multi-run agreement checks.
DEFAULT_TOLERANCE = 1e-6


class ResolveError(Exception):
    pass


def load_ledger(path):
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    entries = doc.get("entries") or {}
    for key in entries:
        if not KEY_RE.match(key):
            raise ResolveError(f"ledger key {key!r} has characters outside [A-Za-z0-9._/-]")
    return entries


def load_records(runs_dir):
    """run_id -> record dict, from provenance/runs/*.json."""
    records = {}
    for p in sorted(Path(runs_dir).glob("*.json")):
        with p.open() as f:
            records[p.stem] = json.load(f)
    return records


def _dotted_get(obj, dotted, key, run_id, what):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ResolveError(f"{key}: run {run_id} {what} path {dotted!r} missing at {part!r}")
        cur = cur[part]
    return cur


def _resolve_ci(group, field, key, run_id):
    """Return [lo, hi] from either an array field or sibling *_ci_low/_ci_high scalars."""
    if field in group and isinstance(group[field], (list, tuple)) and len(group[field]) == 2:
        return list(group[field])
    base = field[:-3] if field.endswith("_ci") else field
    lo, hi = f"{base}_ci_low", f"{base}_ci_high"
    if lo in group and hi in group:
        return [group[lo], group[hi]]
    raise ResolveError(
        f"{key}: run {run_id} has no CI for field {field!r} "
        f"(need a 2-list, or {lo}/{hi} scalars)"
    )


def resolve_raw(key, entry, records):
    """Resolve a ledger entry to a raw value (float, int, or [lo, hi] list).

    Verifies all run_ids agree within tolerance and, if `expected` is set, that the
    resolved value matches it.
    """
    src = entry.get("source") or {}
    kind = src.get("kind")
    run_ids = entry.get("run_ids") or []
    is_interval = (entry.get("format") or {}).get("kind") == "interval"

    if kind == "literal":
        if not entry.get("note"):
            raise ResolveError(f"{key}: literal source requires a non-empty `note`")
        return src["value"]

    if not run_ids:
        raise ResolveError(f"{key}: source kind {kind!r} needs at least one run_id")

    values = []
    for run_id in run_ids:
        rec = records.get(run_id)
        if rec is None:
            raise ResolveError(f"{key}: no distilled record for run_id {run_id!r} "
                               f"(run scripts/distill_runs.py)")
        if kind == "eval":
            if not (rec.get("eval") or {}).get("present"):
                raise ResolveError(f"{key}: run {run_id} has no eval metrics")
            metrics = rec["eval"]["metrics"]
            mp = src["metric_path"]
            if mp not in metrics:
                raise ResolveError(f"{key}: run {run_id} eval has no metric_path {mp!r}")
            group = metrics[mp]
            field = src["field"]
            if is_interval:
                values.append(_resolve_ci(group, field, key, run_id))
            else:
                if field not in group:
                    raise ResolveError(f"{key}: run {run_id} metric {mp!r} has no field {field!r}")
                values.append(group[field])
        elif kind == "hyperparam":
            hp = rec.get("hyperparameters")
            if hp is None:
                raise ResolveError(f"{key}: run {run_id} has no hyperparameters")
            values.append(_dotted_get(hp, src["path"], key, run_id, "hyperparameter"))
        else:
            raise ResolveError(f"{key}: unknown source kind {kind!r}")

    _assert_agreement(key, values)
    value = values[0]

    expected = entry.get("expected")
    if expected is not None and not _close(value, expected):
        raise ResolveError(f"{key}: resolved value {value} != expected {expected} "
                           f"(ledger `expected` is stale, or the metric changed)")
    return value


def _assert_agreement(key, values, tol=DEFAULT_TOLERANCE):
    first = values[0]
    for v in values[1:]:
        if not _close(v, first):
            raise ResolveError(f"{key}: run values disagree beyond tolerance: {values}")


def _close(a, b, tol=DEFAULT_TOLERANCE):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        denom = max(abs(a), abs(b), 1e-30)
        return abs(a - b) / denom <= tol
    return a == b


def _fmt_num(x, sig_figs):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int) or (isinstance(x, float) and x == int(x) and sig_figs == 0):
        return str(int(x))
    if sig_figs == 0:
        return str(int(round(x)))
    return f"{x:.{sig_figs}g}"


def render(key, entry, records):
    """Resolve and format a ledger entry into the LaTeX string emitted by \\valdef.

    Emits only the number (or interval). `units` is ledger metadata; prose supplies units.
    """
    raw = resolve_raw(key, entry, records)
    fmt = entry.get("format") or {}
    sig = fmt.get("sig_figs", 3)
    if fmt.get("kind") == "interval":
        lo, hi = raw
        return f"[{_fmt_num(lo, sig)}, {_fmt_num(hi, sig)}]"
    return _fmt_num(raw, sig)


def render_all(entries, records):
    """key -> rendered string for every entry, in sorted key order. Raises on first failure."""
    return {key: render(key, entries[key], records) for key in sorted(entries)}
