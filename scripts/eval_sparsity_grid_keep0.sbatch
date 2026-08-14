#!/bin/bash
# TRUE train-keep=0 row of the 2D grid: eval each DOC_CAUSAL-trained endpoint
# under a REAL cross-doc mask across the eval-keep axis. Forces the cross arm to
# cross_doc_link (mask_type_override) with the dataset's link detector, so a model
# that never trained with cross-doc links is measured USING them at inference.
# Writes {OUT}/{ds}_train0p0.json (rows = eval-keep points) -> fills the bottom
# row of fig_grid2d + the keep=0 point of the train-time line.
#
# LIST lines: ds|detector|ckpt   (one per doc_causal endpoint)
# Submit: LIST=grid_keep0_list.txt sbatch scripts/eval_sparsity_grid_keep0.sbatch
#SBATCH --job-name=sparsegrid0
#SBATCH --partition=compute
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/grid2d/logs/sbatch0_%j.out
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
    IFS='|' read -r ds det ckpt <<< "${SPECS[$i]}"
    out="$OUT/${ds}_train0p0.json"
    [ -f "$out" ] && { echo "skip $ds train0p0 (done)"; continue; }
    echo "[$(date +%H:%M:%S)] $ds train0p0 (det=$det) -> gpu $g (eval-keeps=$KEEPS, packs=$MAXP)"
    CUDA_VISIBLE_DEVICES=$g python -m eval.sparsity_sweep \
      --checkpoint "$ckpt" --dataset "$ART/$ds" --split val_community \
      --max-packs "$MAXP" --keep-fracs "$KEEPS" --modes edge --seeds 0 \
      --cross-mask-type cross_doc_link --link-detector "$det" \
      --dataset-tag "$ds" --output "$out" > "$OUT/logs/${ds}_train0p0.log" 2>&1 &
    pids+=($!); sleep 5
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
echo "[$(date +%H:%M:%S)] KEEP0 GRID ROW DONE ($LIST)"
