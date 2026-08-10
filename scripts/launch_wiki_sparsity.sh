#!/usr/bin/env bash
# Launch the 4 wiki graph-sparsity runs (interior keep{0.25,0.5,0.75} + keep=1.0
# retrained on CORRECTED-packing epochs). Baked resume-safe configs; all FRESH (no
# prior wiki keep ckpts). wiki recipe = wiki_crossdoc_best.yaml (lr0.003/VE-off/4ep).
# Staggered ~12s; launches only into free nodes (stops when idle=0).
# DRYRUN=1 to preview.
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null
KEEPS=(keep0p25 keep0p5 keep0p75 keep1p0)
n=0
for k in "${KEEPS[@]}"; do
  cfg="configs/sparsity/wiki_merged_${k}_cdl.yaml"
  [ -f "$cfg" ] || { echo "MISSING $cfg"; continue; }
  if [ "${DRYRUN:-0}" = "1" ]; then echo "DRYRUN: wiki $k cfg=$cfg <fresh>"; n=$((n+1)); continue; fi
  idle=$(sinfo -p compute -h -t idle -o "%D" 2>/dev/null | head -1); idle=${idle:-0}
  if [ "$idle" -lt 1 ]; then echo "no free nodes — stopping at $n (watcher will resume the rest as capacity frees)"; break; fi
  echo "=== launch wiki $k (idle=$idle) ==="
  python launch_slurm.py --nodes 1 --gpus-per-node 8 --time 6:00:00 \
    --config "$cfg" --no-tail 2>&1 | grep -E "Job ID|Run dir"
  n=$((n+1)); sleep 12
done
echo "=== launched $n wiki runs ==="
