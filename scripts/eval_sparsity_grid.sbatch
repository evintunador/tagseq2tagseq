#!/bin/bash
# 2D grid eval: for each checkpoint in LIST (ds|trainkeep|ckpt), eval across the
# FULL eval-keep axis (KEEPS) -> Δnll(train_keep, eval_keep) surface. One 8-GPU
# node, 8 checkpoints concurrent (one per GPU), each doing all eval-keeps in one
# checkpoint-load. Writes {OUT}/{ds}_train{trainkeep}.json (rows = eval-keep points).
#
# Submit: LIST=grid_list_0.txt sbatch scripts/eval_sparsity_grid.sbatch
#SBATCH --job-name=sparsegrid
#SBATCH --partition=compute
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/grid2d/logs/sbatch_%j.out
set -uo pipefail
: "${LIST:?eval list}"; : "${KEEPS:=0,0.25,0.5,0.75,1.0}"; : "${MAXP:=500}"
REPO=/fss/evin_t/tagseq2tagseq-sparsity; cd "$REPO"
source /fss/evin_t/tagseq2tagseq/.venv/bin/activate 2>/dev/null || true
ART=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets
OUT=/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/grid2d
mkdir -p "$OUT/logs"
mapfile -t SPECS < "$LIST"
NG=8; i=0
while [ $i -lt ${#SPECS[@]} ]; do
  pids=()
  for ((g=0; g<NG && i<${#SPECS[@]}; g++, i++)); do
    IFS='|' read -r ds tk ckpt <<< "${SPECS[$i]}"
    out="$OUT/${ds}_train${tk}.json"
    [ -f "$out" ] && { echo "skip $ds train$tk (done)"; continue; }
    echo "[$(date +%H:%M:%S)] $ds train$tk -> gpu $g (eval-keeps=$KEEPS, packs=$MAXP)"
    CUDA_VISIBLE_DEVICES=$g python -m eval.sparsity_sweep \
      --checkpoint "$ckpt" --dataset "$ART/$ds" --split val_community \
      --max-packs "$MAXP" --keep-fracs "$KEEPS" --modes edge --seeds 0 \
      --dataset-tag "$ds" --output "$out" > "$OUT/logs/${ds}_train${tk}.log" 2>&1 &
    pids+=($!); sleep 5
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
echo "[$(date +%H:%M:%S)] GRID SLICE DONE ($LIST)"
