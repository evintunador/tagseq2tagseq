#!/usr/bin/env bash
# Relaunch the 15 code-dataset graph-sparsity interior keep-runs CLEANLY, using the
# baked per-(ds,keep) configs (configs/sparsity/{ds}_{keep}_cdl.yaml) whose epoch_dirs
# point at the subsampled schedules — so they are RESUME-SAFE (no CLI override for the
# watcher to drop; see [[sparsity-watcher-resume-bug]]). Where a last-valid keep-density
# checkpoint exists, resume from it (exact Muon+AdamW); else start fresh.
# Staggered ~12s (avoid run-dir-collision). Launches only into free nodes.
#
# DRYRUN=1 bash scripts/relaunch_sparsity_clean.sh    # print plan
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null

# ds keep resume_run_dir(""=fresh)   — resume dirs from relaunch_plan.py (2026-08-05)
RUNS=(
 "javascript 0p25 run_20260804_171138_972353"
 "javascript 0p5  run_20260804_211844_277741"
 "javascript 0p75 run_20260804_211858_627575"
 "typescript 0p25 "
 "typescript 0p5  run_20260804_211926_167053"
 "typescript 0p75 run_20260804_211939_908957"
 "thestack   0p25 run_20260804_211953_676337"
 "thestack   0p5  run_20260804_212007_463712"
 "thestack   0p75 run_20260804_212021_085438"
 "rust       0p25 run_20260804_212034_839586"
 "rust       0p5  run_20260804_212049_691789"
 "rust       0p75 "
 "dart       0p25 "
 "dart       0p5  "
 "dart       0p75 "
)
n=0
for spec in "${RUNS[@]}"; do
  read -r ds keep resume <<< "$spec"
  cfg="configs/sparsity/${ds}_keep${keep}_cdl.yaml"
  [ -f "$cfg" ] || { echo "MISSING CONFIG $cfg — skip"; continue; }
  resume_args=""
  if [ -n "$resume" ]; then
    ck="$REPO/runs/$resume/checkpoints/latest.pt"
    [ -f "$ck" ] && resume_args="--resume-from $ck" || echo "  warn: $ds $keep resume ckpt missing, going fresh"
  fi
  if [ "${DRYRUN:-0}" = "1" ]; then
    echo "DRYRUN: $ds keep=$keep cfg=$cfg ${resume_args:-<fresh>}"; n=$((n+1)); continue
  fi
  idle=$(sinfo -p compute -h -t idle -o "%D" 2>/dev/null | head -1); idle=${idle:-0}
  if [ "$idle" -lt 1 ]; then echo "no free nodes — stopping at $n launched"; break; fi
  echo "=== launch $ds keep=$keep (idle=$idle) ${resume_args:+[resume]} ==="
  python launch_slurm.py --nodes 1 --gpus-per-node 8 --time 6:00:00 \
    --config "$cfg" --no-tail $resume_args 2>&1 | grep -E "Job ID|Run dir"
  n=$((n+1)); sleep 12
done
echo "=== relaunched $n runs ==="
