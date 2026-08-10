#!/usr/bin/env bash
# Phase-1 eval-time sweep for the graph-sparsity scaling law.
# For each of the 11 solo datasets, load its cross_doc best_model.pt ONCE and
# sweep keep_frac over the community split (mask-time edge dropout), writing one
# JSON per dataset. edge mode = the density line; node mode (interior only) =
# structure-sensitivity robustness check. See memory [[graph-sparsity-scaling-law]].
#
# Runs one dataset per GPU, staggered (CLAUDE.md: never launch simultaneously —
# each first-pack flex-compiles ~3min; a cold-compile storm thrashes inductor).
# GPU 0 is skipped (co-tenant); uses GPUs 1..NGPU.
#
# Usage: sparsity_phase1_sweep.sh
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq-sparsity; cd "$REPO"
source .venv/bin/activate 2>/dev/null || true

MANIFEST="${MANIFEST:-$CLAUDE_JOB_DIR/tmp/phase1_manifest.json}"
OUT="${OUT:-/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/phase1_eval}"
LOGDIR="$OUT/logs"; mkdir -p "$OUT" "$LOGDIR"

KEEP="${KEEP:-0,0.25,0.5,0.75,1.0}"
MODES="${MODES:-edge,node}"
SEEDS="${SEEDS:-0}"
MAXPACKS="${MAXPACKS:-500}"
GPUS=(${GPUS:-1 2 3 4 5 6 7})   # skip GPU 0 (co-tenant)
STAGGER="${STAGGER:-45}"        # seconds between launches on the same wave

# Read the manifest into parallel arrays.
mapfile -t DATASETS < <(python -c "import json,sys;[print(r['dataset']) for r in json.load(open('$MANIFEST'))]")
mapfile -t CKPTS    < <(python -c "import json,sys;[print(r['ckpt']) for r in json.load(open('$MANIFEST'))]")
mapfile -t DDIRS    < <(python -c "import json,sys;[print(r['dataset_dir']) for r in json.load(open('$MANIFEST'))]")

N=${#DATASETS[@]}
echo "=== Phase-1 sparsity sweep: $N datasets, keep=[$KEEP] modes=[$MODES] seeds=[$SEEDS] maxpacks=$MAXPACKS ==="
echo "=== output → $OUT ==="

NG=${#GPUS[@]}
i=0
while [ $i -lt $N ]; do
  pids=()
  for g in "${GPUS[@]}"; do
    [ $i -lt $N ] || break
    ds="${DATASETS[$i]}"; ckpt="${CKPTS[$i]}"; dd="${DDIRS[$i]}"
    outjson="$OUT/${ds}.json"
    log="$LOGDIR/${ds}.log"
    echo "  → [$((i+1))/$N] $ds on cuda:$g  (ckpt=$ckpt)"
    CUDA_VISIBLE_DEVICES=$g nohup python -m eval.sparsity_sweep \
      --checkpoint "$ckpt" \
      --dataset "$dd" \
      --split val_community \
      --max-packs "$MAXPACKS" \
      --keep-fracs "$KEEP" \
      --modes "$MODES" \
      --seeds "$SEEDS" \
      --dataset-tag "$ds" \
      --output "$outjson" > "$log" 2>&1 &
    pids+=($!)
    i=$((i+1))
    sleep "$STAGGER"   # stagger cold compiles within the wave
  done
  echo "  --- wave launched (${#pids[@]} jobs), waiting for completion ---"
  for p in "${pids[@]}"; do wait "$p"; done
  echo "  --- wave done ---"
done
echo "=== ALL Phase-1 sweeps done → $OUT ==="
