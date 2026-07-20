#!/bin/bash
# Precompute thestack dfs/random_walk/random BFS schedules on a dedicated compute
# node (login node is CPU-starved). Runs the 3 strategies sequentially, each with
# many workers, against splits/train to match thestack_bfs scope. CPU-only (no GPU
# needed: analytical kv_block_count).
set -euo pipefail
REPO=/fss/evin_t/tagseq2tagseq
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
cd "$REPO"
source .venv/bin/activate 2>/dev/null || true

for strat in dfs random_walk random; do
  echo "=== [$(date)] precomputing thestack_${strat} ==="
  python precompute_epochs.py \
    --dataset-dir "$ART/pretokenized_datasets/thestack/splits/train" \
    --output-dir  "$ART/schedules/thestack_${strat}" \
    --n-epochs 1 --strategy "$strat" --local-seq-len 32768 --n-buckets 32 --n-workers 32 \
    --link-detector python --layout-policy stochastic_identifier_prefix \
    --max-grants 256 --order-mode prefer_targets_first --device cpu
done
echo "=== [$(date)] ALL THREE ABLATION SCHEDULES DONE ==="
