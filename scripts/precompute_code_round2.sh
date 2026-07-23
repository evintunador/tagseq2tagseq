#!/bin/bash
# Round-2 precompute: finish rust/kotlin (interrupted when the dart merge broke
# data/layout.py mid-run) and add dart/zig. Resume-safe (skips existing epochs).
# All C-family langs → stochastic_slash_comment_prefix. Detector = lang name.
# Epoch targets (need 120k packs for 15k steps @ world=8):
#   rust ~37k→4  kotlin ~84k→2  dart ~13.9k→9  zig ~921→16 (zig is far below chinchilla;
#   16 ep ≈ 1840 steps — a pragmatic floor, NOT 15k; zig config caps steps accordingly).
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true

# GUARD: abort immediately if the code doesn't import (catches merge-conflict breakage
# like the one that silently burned the round-1 allocation for 7h).
python -c "import data.layout, precompute_epochs; from model.graph_traversal.link_detector import LINK_DETECTOR_NAMES" || {
  echo "[round2] IMPORT CHECK FAILED — aborting (repo likely mid-merge/broken)"; exit 1; }
echo "[round2] import check OK"

gen () { # lang n_epochs strat
  local lang=$1 n=$2 strat=$3
  echo "=== [$(date)] $lang/$strat → $n epoch(s) ==="
  python precompute_epochs.py \
    --dataset-dir "/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/$lang/splits/train" \
    --output-dir  "/fss-data/evin_t/tagseq2tagseq_artifacts/schedules/${lang}_${strat}" \
    --n-epochs "$n" --strategy "$strat" --local-seq-len 32768 --n-buckets 32 --n-workers 32 \
    --link-detector "$lang" --layout-policy stochastic_slash_comment_prefix \
    --max-grants 256 --order-mode prefer_targets_first --device cpu \
    || echo "[round2] WARN: $lang/$strat exited nonzero"
}

for s in bfs dfs random_walk random; do gen rust   4  "$s"; done
for s in bfs dfs random_walk random; do gen kotlin 2  "$s"; done
for s in bfs dfs random_walk random; do gen dart   9  "$s"; done
for s in bfs dfs random_walk random; do gen zig    16 "$s"; done
echo "=== [$(date)] ROUND-2 PRECOMPUTE DONE ==="
