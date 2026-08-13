#!/bin/bash
# Full-fidelity (max-packs=500) community_pack eval for graph-sparsity keep-arms,
# on ONE 8-GPU SLURM node. Reads an eval-list file (lines: ds|keeptag|ckpt_path),
# runs 8 evals concurrently (one per local GPU), waves until the list is done.
# Each eval = eval/sparsity_sweep at keep=1.0 (full-density test) -> {ds}_keep{K}.json.
#
# Submit: LIST=/path/eval_list.txt sbatch scripts/eval_sparsity_500.sbatch
#SBATCH --job-name=sparseval
#SBATCH --partition=compute
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=8:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/phase2_traintime/logs/sbatch_%j.out
set -uo pipefail
: "${LIST:?eval list file}"; : "${MAXP:=500}"
REPO=/fss/evin_t/tagseq2tagseq-sparsity; cd "$REPO"
source /fss/evin_t/tagseq2tagseq/.venv/bin/activate 2>/dev/null || true
ART=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets
OUT=/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/phase2_traintime
mkdir -p "$OUT/logs"
mapfile -t SPECS < "$LIST"
NG=8; i=0
while [ $i -lt ${#SPECS[@]} ]; do
  pids=()
  for ((g=0; g<NG && i<${#SPECS[@]}; g++, i++)); do
    IFS='|' read -r ds keep ckpt <<< "${SPECS[$i]}"
    out="$OUT/${ds}_keep${keep}.json"
    echo "[$(date +%H:%M:%S)] $ds keep$keep -> local gpu $g (maxpacks=$MAXP)"
    CUDA_VISIBLE_DEVICES=$g python -m eval.sparsity_sweep \
      --checkpoint "$ckpt" --dataset "$ART/$ds" --split val_community \
      --max-packs "$MAXP" --keep-fracs 1.0 --modes edge --seeds 0 \
      --dataset-tag "$ds" --output "$out" > "$OUT/logs/${ds}_keep${keep}_500.log" 2>&1 &
    pids+=($!); sleep 5
  done
  for p in "${pids[@]}"; do wait "$p"; done
done
echo "[$(date +%H:%M:%S)] ALL EVALS DONE"
