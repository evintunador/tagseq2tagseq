#!/usr/bin/env bash
# Launch the 19 graph-sparsity keep-runs FRESH on the SCHEDULE-CORRECTED configs
# (max_optimizer_steps pinned: code 15000, wiki 14507 — matching each endpoint).
# The prior runs were abandoned: they trained on an auto-derived ~37-45k LR schedule
# (wrong warmup/cooldown/untie vs endpoints) — a scientific-validity break. This is
# the first clean, schedule-matched run. See [[sparsity-watcher-resume-bug]].
#
# Idempotent + node-polite: skips any (ds,keep) already RUNNING, launches at most
# (idle nodes) each invocation (re-run as capacity frees), staggered ~12s. No
# --resume-from (fresh). No --exclude (bad nodes GPU-495/954/943 are drained).
# 96h wall-time so runs don't TIMEOUT mid-training.
#
# DRYRUN=1 to preview. MAXLAUNCH=N to cap this invocation.
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true

ALL=(
 javascript_keep0p25 javascript_keep0p5 javascript_keep0p75
 typescript_keep0p25 typescript_keep0p5 typescript_keep0p75
 thestack_keep0p25 thestack_keep0p5 thestack_keep0p75
 rust_keep0p25 rust_keep0p5 rust_keep0p75
 dart_keep0p25 dart_keep0p5 dart_keep0p75
 wiki_merged_keep0p25 wiki_merged_keep0p5 wiki_merged_keep0p75 wiki_merged_keep1p0
)

# what (ds,keep) tags are already running? (skip them)
running_tags="$(python3 /fss/evin_t/.claude/jobs/465d9eaf/tmp/live.py 2>/dev/null \
  | grep -oE '[a-z_]+ keep[0-9p]+' | sed 's/ /_/' | sort -u)"

launched=0
MAXLAUNCH="${MAXLAUNCH:-999}"
for name in "${ALL[@]}"; do
  # normalize tag: config name uses {ds}_keep{K}; running tag uses {ds}_keep{K} too
  if grep -qx "$name" <<< "$running_tags"; then echo "skip $name (already running)"; continue; fi
  [ "$launched" -ge "$MAXLAUNCH" ] && { echo "hit MAXLAUNCH=$MAXLAUNCH"; break; }
  idle=$(sinfo -p compute -h -t idle -o "%D" 2>/dev/null | head -1); idle=${idle:-0}
  if [ "$idle" -lt 1 ]; then echo "no idle nodes — stopping (re-run later for the rest)"; break; fi
  cfg="configs/sparsity/${name}_cdl.yaml"
  [ -f "$cfg" ] || { echo "MISSING $cfg"; continue; }
  if [ "${DRYRUN:-0}" = "1" ]; then echo "DRYRUN: $name (idle=$idle)"; launched=$((launched+1)); continue; fi
  echo "=== launch $name (idle=$idle) ==="
  python launch_slurm.py --nodes 1 --gpus-per-node 8 --time 96:00:00 \
    --config "$cfg" --no-tail 2>&1 | grep -E "Job ID|Run dir"
  launched=$((launched+1)); sleep 12
done
echo "=== launched $launched this invocation ==="
