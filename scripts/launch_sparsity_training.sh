#!/usr/bin/env bash
# Launch Phase-2 (train-time) graph-sparsity interior arms: for each dataset, a
# cross_doc run at keep ∈ {0.25,0.5,0.75} on the SUBSAMPLED epoch dirs. Endpoints
# (keep=0 ≡ existing doc_causal, keep=1 ≡ existing cross_doc) are reused, NOT retrained
# (except wiki keep=1.0, launched separately on corrected epochs).
#
# Each run reuses its dataset's EXACT endpoint recipe (the _veoff_cdl.yaml sweep
# config / wiki_crossdoc_best.yaml — all VE-off, muon_lr=0.003) so the keep line is
# internally consistent; only --data.epoch_dirs (→ subsampled) changes, plus
# --train_loop.max_optimizer_steps 15000.
#
# LAUNCH DISCIPLINE (CLAUDE.md): 1 node × 8 GPUs per run; warm ONE shared compile
# cache at world=8 first; then launch one-at-a-time, waiting for each to reach its
# first training step ("Training: Nit") before the next, to avoid concurrent
# cold-compile thrash + run-dir-collision. Launches only as many as there are FREE
# nodes (don't queue in front of others).
#
# Usage:
#   DRYRUN=1 bash scripts/launch_sparsity_training.sh          # print the matrix
#   MAXLAUNCH=5 bash scripts/launch_sparsity_training.sh       # launch up to 5 (into free nodes)
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq                     # launch from MAIN repo (has runs/, launch_slurm)
WT=/fss/evin_t/tagseq2tagseq-sparsity              # worktree has the subsampled schedules + code knobs
cd "$REPO"
SCH=/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/schedules
export TS2TS_SHARED_COMPILE_CACHE=/fss-data/evin_t/tagseq2tagseq_artifacts/compile_cache/sparsity_cdl_ws8

# dataset : endpoint-config : n_epochs   (code datasets only; wiki launched separately)
MATRIX=(
  "javascript:configs/javascript_sweep/javascript_veoff_cdl.yaml:1"
  "typescript:configs/typescript_sweep/typescript_veoff_cdl.yaml:1"
  "thestack:configs/thestack_sweep/thestack_veoff_cdl.yaml:1"
  "rust:configs/rust_sweep/rust_veoff_cdl.yaml:4"
  "dart:configs/dart_sweep/dart_veoff_cdl.yaml:9"
)
KEEPS=(0.25 0.5 0.75)
STEPS=15000

# Build the full run list: (dataset, cfg, keep, epoch_dirs_csv)
RUNS=()
for spec in "${MATRIX[@]}"; do
  IFS=: read -r ds cfg nep <<< "$spec"
  for keep in "${KEEPS[@]}"; do
    ktag=keep$(echo "$keep" | sed 's/\./p/')
    dir="$SCH/${ds}_bfs_${ktag}"
    eps=""; for i in $(seq 0 $((nep-1))); do eps="${eps}${eps:+,}$dir/epoch_$i"; done
    # sanity: all epoch parquets present
    ok=1; for i in $(seq 0 $((nep-1))); do [ -f "$dir/epoch_$i/packs.parquet" ] || ok=0; done
    [ "$ok" = 1 ] || { echo "SKIP $ds $keep — missing epoch parquet(s) in $dir"; continue; }
    RUNS+=("$ds|$cfg|$keep|$eps")
  done
done

echo "=== Phase-2 training matrix: ${#RUNS[@]} runs (code datasets, interior keeps) ==="
for r in "${RUNS[@]}"; do IFS='|' read -r ds cfg keep eps <<< "$r"; echo "  $ds keep=$keep  cfg=$cfg  ($(echo "$eps" | tr ',' '\n' | wc -l) epochs)"; done

[ "${DRYRUN:-0}" = "1" ] && { echo "(dryrun) not launching."; exit 0; }

freenodes() { sinfo -p compute -h -t idle -o "%D" 2>/dev/null | head -1; }
launched=0
MAXLAUNCH=${MAXLAUNCH:-99}
for r in "${RUNS[@]}"; do
  [ "$launched" -ge "$MAXLAUNCH" ] && { echo "hit MAXLAUNCH=$MAXLAUNCH"; break; }
  fn=$(freenodes); fn=${fn:-0}
  if [ "$fn" -lt 1 ]; then echo "no free nodes (idle=$fn) — stopping so we don't queue ahead of others"; break; fi
  IFS='|' read -r ds cfg keep eps <<< "$r"
  echo "=== [$((launched+1))] launching $ds keep=$keep (idle nodes=$fn) ==="
  python launch_slurm.py --nodes 1 --gpus-per-node 8 --time 6:00:00 \
    --config "$cfg" \
    --data.epoch_dirs "$eps" \
    --train_loop.max_optimizer_steps "$STEPS" \
    --no-tail &
  lpid=$!
  echo "  launch_slurm pid=$lpid; waiting for first training step before next (CLAUDE.md stagger)..."
  # wait for THIS run to reach 'Training: N' in the newest run log, up to 15 min
  launched=$((launched+1))
  sleep 5
  wait_ok=0
  for _ in $(seq 1 90); do
    newlog=$(ls -t "$REPO"/runs/*/logs/*.log 2>/dev/null | head -1)
    if [ -n "$newlog" ] && grep -qE "Training: [0-9]|Training: Nit|step 1\b" "$newlog" 2>/dev/null; then wait_ok=1; break; fi
    sleep 10
  done
  [ "$wait_ok" = 1 ] && echo "  → reached first step; proceeding." || echo "  ! did not confirm first step in 15min; PAUSING launches for safety"; [ "$wait_ok" = 1 ] || break
done
echo "=== launched $launched run(s) ==="
