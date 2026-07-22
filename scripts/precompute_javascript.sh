#!/bin/bash
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
python -c "import data.layout, precompute_epochs; from model.graph_traversal.link_detector import LINK_DETECTOR_NAMES; assert 'javascript' in LINK_DETECTOR_NAMES" || { echo "[js] import/detector check FAILED"; exit 1; }
echo "[js] import check OK"
for s in bfs dfs random_walk random; do
  echo "=== [$(date)] javascript/$s → 1 epoch ==="
  python precompute_epochs.py \
    --dataset-dir /fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/javascript/splits/train \
    --output-dir  /fss-data/evin_t/tagseq2tagseq_artifacts/schedules/javascript_${s} \
    --n-epochs 1 --strategy "$s" --local-seq-len 32768 --n-buckets 32 --n-workers 32 \
    --link-detector javascript --layout-policy stochastic_slash_comment_prefix \
    --max-grants 256 --order-mode prefer_targets_first --device cpu \
    || echo "[js] WARN javascript/$s nonzero"
done
echo "=== [$(date)] JAVASCRIPT PRECOMPUTE DONE ==="
