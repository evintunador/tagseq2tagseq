#!/usr/bin/env python3
"""Launch driver for the epochs-to-degradation / memorization experiment.

Builds the arm matrix (corpus × mask × mode × n_epochs), resolves each arm's
`epoch_dirs` (fresh = distinct-seed epoch_0..N-1; repeat = epoch_0 listed N
times), reads `n_packs` from each epoch's metadata.json to PIN
`max_optimizer_steps = total_packs // (world_size * accum)` (so LR cooldown +
untie fire and fresh/repeat arms at the same N are schedule-matched), and emits
one `launch_slurm.py` command per arm.

Default is DRY RUN: prints the plan table and writes the per-arm commands to a
file. Pass --launch to actually submit, staggered one-at-a-time past the first
training step (per CLAUDE.md launch discipline).

Design ref: memory [[project_epochs_to_degradation_memorization]].

Examples:
  # Dry-run the full matrix:
  python scripts/launch_mem_epochs.py
  # Dry-run just the launchable wiki Stage-1 wave (dc+cdl, fresh+repeat, N<=6):
  python scripts/launch_mem_epochs.py --corpus wiki_merged --masks dc cdl --epochs 2 4 6
  # Actually launch that wave (staggered):
  python scripts/launch_mem_epochs.py --corpus wiki_merged --masks dc cdl --epochs 2 4 6 --launch
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]  # self-locating: this repo (isolated worktree or shared)
ART = Path("/fss-data/evin_t/tagseq2tagseq_artifacts")
SCHED = ART / "schedules"

# mask key -> base config (relative to REPO). All tuned recipe; WD differs by mask.
CONFIGS = {
    "wiki_merged": {
        "dc":     "configs/wiki_merged_doc_causal_best.yaml",   # wd 0.1
        "cdl":    "configs/wiki_crossdoc_best.yaml",            # wd 0.3
        "dcl":    "configs/wiki_docconcatlink_best.yaml",       # wd 0.3
        "concat": "configs/wiki_docconcat_best.yaml",           # wd 0.3
    },
    "go": {
        "dc":     "configs/go_sweep/go_veoff_dc.yaml",
        "cdl":    "configs/go_sweep/go_veoff_cdl.yaml",
        "dcl":    "configs/go_sweep/go_veoff_concatlink.yaml",
        "concat": "configs/go_sweep/go_veoff_concat.yaml",
    },
    "simplewiki": {
        "dc":     "configs/wiki_merged_doc_causal_best.yaml",   # base recipe; dirs overridden below
        "cdl":    "configs/wiki_crossdoc_best.yaml",
        "dcl":    "configs/wiki_docconcatlink_best.yaml",
        "concat": "configs/wiki_docconcat_best.yaml",
    },
}
MASK_TYPE = {"dc": "doc_causal", "cdl": "cross_doc_link",
             "dcl": "doc_concat_link", "concat": "doc_concatenated"}


def epoch_dir(corpus: str, i: int) -> Path:
    return SCHED / f"{corpus}_bfs" / f"epoch_{i}"


def _epoch_meta(corpus: str, i: int) -> dict:
    return json.loads((epoch_dir(corpus, i) / "metadata.json").read_text())


def n_packs(corpus: str, i: int) -> int:
    return _epoch_meta(corpus, i)["n_packs"]


def _achievable_steps(corpus: str, i: int, world_size: int, accum: int) -> int:
    """Optimizer steps actually yielded by one epoch dir under BucketedPackDataset.

    The loader consumes each of the n_buckets quantile buckets in world_size*accum
    chunks and DROPS each bucket's partial tail every epoch, so usable steps =
    n_buckets * floor(n_packs / (n_buckets * world_size * accum)), NOT
    n_packs // (world_size*accum). Pinning to the latter overshoots and the run
    dies "epoch dirs exhausted" a few %% early (benign for long runs, but exit-1
    -> the yield-watcher may resume into a re-exhaustion loop). Pinning to the
    achievable count -> clean exit at the last real step. Quantile buckets are
    equal-count so floor-per-bucket is exact-to-slightly-under = safe.
    """
    m = _epoch_meta(corpus, i)
    nb = int(m.get("n_buckets", 1)) or 1
    per_batch = nb * world_size * accum
    return nb * (m["n_packs"] // per_batch)


def resolve_arm(corpus, mask, mode, n, world_size, accum):
    """Return (epoch_dirs:list[Path], total_packs, max_steps) or raise if a
    required epoch dir / packs.parquet is missing."""
    if mode == "fresh":
        idxs = list(range(n))
    elif mode == "repeat":
        idxs = [0] * n
    else:
        raise ValueError(mode)
    dirs = [epoch_dir(corpus, i) for i in idxs]
    for i, d in zip(idxs, dirs):
        if not (d / "packs.parquet").exists():
            raise FileNotFoundError(f"missing {d}/packs.parquet (corpus={corpus} epoch_{i})")
    total = sum(n_packs(corpus, i) for i in idxs)
    max_steps = sum(_achievable_steps(corpus, i, world_size, accum) for i in idxs)
    return dirs, total, max_steps


def arm_label(corpus, mask, mode, n):
    return f"{corpus}.{mask}.{mode}.e{n}"


def build_command(corpus, mask, mode, n, dirs, max_steps, nodes, gpus, time_limit, no_eval=False, accum=1):
    cfg = CONFIGS[corpus][mask]
    epoch_csv = ",".join(str(d) for d in dirs)
    cmd = [
        str(REPO / ".venv/bin/python"), str(REPO / "launch_slurm.py"),
        "--nodes", str(nodes), "--gpus-per-node", str(gpus),
        "--time", time_limit,
        "--config", cfg,
        "--data.epoch_dirs", epoch_csv,
        "--train_loop.max_optimizer_steps", str(max_steps),
        "--model.mask_type", MASK_TYPE[mask],
    ]
    if accum and int(accum) != 1:
        # Partial-node arm: fewer GPUs (world_size) but grad-accum to keep the
        # effective batch = world_size*accum*seq_len identical to the world=8 arms.
        # max_optimizer_steps is already pinned on world_size*accum, so the WSD
        # schedule matches. Lets us pack onto nodes with only 4 free GPUs.
        cmd += ["--train_loop.atomic_feature_kwargs.accum_steps", str(int(accum))]
    if no_eval:
        # Skip the config's run_on_completion benchmark suite (e.g. tiny pilot
        # arms, or a config whose annotator_corpus doesn't match this corpus).
        cmd += ["--eval.run_on_completion", "false"]
    # Memorization experiment uses a COMMON muon_wd across masks so the cross-mask
    # onset comparison isn't confounded by regularization strength. The wiki
    # cross-doc base configs are tuned to wd 0.3 (val-loss optimum); pin them to
    # 0.1 to match doc_causal. go configs are already all 0.1 (no-op there, but
    # set explicitly for a self-documenting run record).
    cmd += ["--optimizer.muon_wd", "0.1"]
    # simplewiki reuses wiki base configs but must point at simplewiki splits.
    if corpus == "simplewiki":
        base = str(ART / "pretokenized_datasets/simplewiki")
        cmd += [
            "--data.dataset_dir", base,
            "--data.train_dir", f"{base}/splits/train",
            "--data.val_dirs.val_random", f"{base}/splits/val_random",
            "--data.val_dirs.val_community", f"{base}/splits/val_community",
        ]
    return cmd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", default=["wiki_merged", "go"],
                    choices=["wiki_merged", "go", "simplewiki"])
    ap.add_argument("--masks", nargs="+", default=["dc", "cdl", "dcl", "concat"],
                    choices=["dc", "cdl", "dcl", "concat"])
    ap.add_argument("--modes", nargs="+", default=["fresh", "repeat"],
                    choices=["fresh", "repeat"])
    ap.add_argument("--epochs", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--world-size", type=int, default=8, help="nodes*gpus; pins max_optimizer_steps")
    ap.add_argument("--nodes", type=int, default=1)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--time", default="24:00:00")
    ap.add_argument("--no-eval", action="store_true", help="append --eval.run_on_completion false to each arm")
    ap.add_argument("--launch", action="store_true", help="actually submit (staggered); default dry-run")
    ap.add_argument("--first-step-timeout", type=int, default=1800,
                    help="seconds to wait for an arm to reach step 1 before launching the next")
    ap.add_argument("--out", default=str(REPO / "runs" / "MEM_EPOCHS_LAUNCH_CMDS.sh"))
    args = ap.parse_args()

    if args.world_size != args.nodes * args.gpus:
        print(f"NOTE: --world-size {args.world_size} != nodes*gpus {args.nodes*args.gpus}; "
              f"using world_size for step pinning.", file=sys.stderr)

    arms, skipped = [], []
    for corpus in args.corpus:
        for mask in args.masks:
            for mode in args.modes:
                # doc_causal is re-pack-invariant: fresh==repeat. Keep fresh; flag repeat.
                for n in sorted(args.epochs):
                    try:
                        dirs, total, steps = resolve_arm(corpus, mask, mode, n,
                                                         args.world_size, args.accum)
                    except FileNotFoundError as e:
                        skipped.append((arm_label(corpus, mask, mode, n), str(e)))
                        continue
                    arms.append(dict(corpus=corpus, mask=mask, mode=mode, n=n,
                                     dirs=dirs, total=total, steps=steps,
                                     label=arm_label(corpus, mask, mode, n)))

    # ---- report ----
    print(f"{'ARM':<34} {'config':<40} {'packs':>8} {'steps':>7} {'~Btok':>6}  note")
    cmds = ["#!/bin/bash", "# Auto-generated by launch_mem_epochs.py — staggered per CLAUDE.md.", "set -e", ""]
    for a in arms:
        cfg = CONFIGS[a["corpus"]][a["mask"]]
        btok = a["steps"] * args.world_size * args.accum * 32768 / 1e9
        note = "dc: fresh≡repeat (repeat=sanity only)" if (a["mask"] == "dc" and a["mode"] == "repeat") else ""
        print(f"{a['label']:<34} {cfg:<40} {a['total']:>8} {a['steps']:>7} {btok:>6.2f}  {note}")
        cmd = build_command(a["corpus"], a["mask"], a["mode"], a["n"], a["dirs"],
                            a["steps"], args.nodes, args.gpus, args.time, args.no_eval, args.accum)
        cmds.append(f"# {a['label']}  ({a['total']} packs, {a['steps']} steps)")
        cmds.append(" ".join(cmd))
        cmds.append("")

    if skipped:
        print("\nSKIPPED (missing precompute):")
        for lbl, why in skipped:
            print(f"  {lbl}: {why}")

    Path(args.out).write_text("\n".join(cmds) + "\n")
    print(f"\n{len(arms)} arms planned, {len(skipped)} skipped. Commands written to {args.out}")

    if not args.launch:
        print("\nDRY RUN — nothing submitted. Re-run with --launch to submit (staggered).")
        return

    # ---- staggered launch ----
    print(f"\nLAUNCHING {len(arms)} arms, staggered (wait for step 1, timeout "
          f"{args.first_step_timeout}s each)...")
    for a in arms:
        cmd = build_command(a["corpus"], a["mask"], a["mode"], a["n"], a["dirs"],
                            a["steps"], args.nodes, args.gpus, args.time, args.no_eval, args.accum) + ["--no-tail"]
        print(f"\n>>> submitting {a['label']}")
        res = subprocess.run(cmd, cwd=str(REPO), check=True, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout, end="")
        run_dir = _parse_run_dir(res.stdout)
        _wait_for_first_step(a["label"], run_dir, args.first_step_timeout)
    print("\nAll arms submitted.")


def _parse_run_dir(stdout):
    """Extract this arm's run dir from launch_slurm.py's banner ('Run dir : ...')."""
    for line in (stdout or "").splitlines():
        if "Run dir" in line and "runs/" in line:
            return Path(line.split(":", 1)[1].strip())
    return None


def _wait_for_first_step(label, run_dir, timeout):
    """Poll THIS arm's run dir for a training-step marker before returning.

    Polls run_dir/logs/stderr.txt (+ .slurm/*_log.err) for this arm ONLY, so a
    concurrently-running earlier arm can't trigger a false positive (the old
    global-newest glob did, firing the next arm while this one was still PD /
    cold-compiling — a concurrent-compile hazard on cold caches). Falls back to a
    fixed timeout so the sweep never stalls; the yield-watcher resumes preempted
    arms independently. If run_dir is unknown, degrades to a short fixed wait.
    """
    markers = ("Training step", "Epoch 1/1:", "Starting Training")
    if run_dir is None:
        print(f"    {label}: run dir not parsed; waiting {min(180, timeout)}s before next.")
        time.sleep(min(180, timeout))
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(20)
        for lp in list(run_dir.glob("logs/stderr.txt")) + list(run_dir.glob(".slurm/*_log.err")):
            try:
                txt = lp.read_text(errors="ignore")
            except OSError:
                continue
            if any(m in txt for m in markers):
                print(f"    {label}: reached first step; launching next.")
                return
    print(f"    {label}: first-step marker not seen within timeout; proceeding "
          f"(check the run manually — yield-watcher will resume if preempted).")


if __name__ == "__main__":
    main()
