#!/bin/bash
# Stagger-launch a list of configs: launch one, wait until it prints a training step
# (past compile) or errors, then launch the next. Respects CLAUDE.md "never launch
# simultaneously". Appends run mapping to runs/CODE_SWEEP_RUNMAP.txt.
# Usage: stagger_launch.sh <label_prefix> <config1> [config2 ...]
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
EXCLUDE="GPU-954,GPU-749"
PREFIX="$1"; shift
for cfg in "$@"; do
  echo "[stagger] launching $cfg"
  out=$(python launch_slurm.py --nodes 1 --gpus-per-node 8 --config "$cfg" \
        --time 96:00:00 --no-tail --exclude "$EXCLUDE" 2>&1)
  jid=$(echo "$out" | grep -oE 'Job ID *: [0-9]+' | grep -oE '[0-9]+')
  rundir=$(echo "$out" | grep -oE 'runs/run_[0-9_]+' | head -1)
  echo "[stagger] $cfg -> job $jid $rundir"
  echo "$jid  $PREFIX  $rundir  $cfg" >> runs/CODE_SWEEP_RUNMAP.txt
  # wait up to ~15 min for training step or error
  for i in $(seq 1 60); do
    if grep -q "smart_compiled_loop.*Training step" "$rundir/logs/stderr.txt" 2>/dev/null; then
      echo "[stagger] $jid REACHED TRAINING"; break; fi
    if grep -qiE "Traceback|out of range|Error:|CUDA out of memory" "$rundir/logs/stderr.txt" "$rundir"/.slurm/*_0_log.err 2>/dev/null; then
      echo "[stagger] $jid ERROR — aborting stagger"; grep -iE "Error|out of range" "$rundir"/.slurm/${jid}_0_log.err 2>/dev/null | tail -3; exit 1; fi
    sleep 15
  done
done
echo "[stagger] all launched."
