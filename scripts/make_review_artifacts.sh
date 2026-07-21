#!/bin/bash
# Generate the human/agent review-artifact bundle for a built language dataset.
#
# Produces, under review_artifacts/<lang>/:
#   01_audit_full.txt         - run_audit on the full graph (structural stats)
#   02_audit_train.txt        - run_audit on the train split
#   03_sample_dump.txt        - run_sample_dump: (source -> detected link -> resolved
#                               target -> snippet) tuples. THE main text artifact.
#   04_packed_batches.txt     - visualize_llm_input: real packed batches with
#                               per-doc layout + detected cross-doc links (no-color).
#   graph_stats.json          - the builder's own stats (copied)
#
# Usage:  bash scripts/make_review_artifacts.sh <lang> <link_detector>
#   e.g.  bash scripts/make_review_artifacts.sh rust rust
set -uo pipefail

LANG="${1:?usage: make_review_artifacts.sh <lang> <detector>}"
DET="${2:?usage: make_review_artifacts.sh <lang> <detector>}"

REPO=/fss/evin_t/tagseq2tagseq
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
DS=$ART/pretokenized_datasets/${LANG}
OUT=$REPO/review_artifacts/${LANG}
cd "$REPO"
mkdir -p "$OUT"

echo "[artifacts] $LANG -> $OUT"

echo "  01 audit (full)"
python -m data.graph_harness.run_audit "$DS" > "$OUT/01_audit_full.txt" 2>&1

echo "  02 audit (train split)"
python -m data.graph_harness.run_audit "$DS/splits/train" > "$OUT/02_audit_train.txt" 2>&1

# Sample-dump on val_community (a smaller split): loading a multi-million-node
# GraphIndex + scanning for linked docs is slow, and val_community is dense
# (BFS communities) so links are found fast. Same resolution logic as train.
echo "  03 sample dump (detector=$DET, on val_community)"
python -m data.graph_harness.run_sample_dump "$DS/splits/val_community" \
    --detector "$DET" --n 40 --seed 0 --snippet-chars 400 --max-scan 5000 \
    > "$OUT/03_sample_dump.txt" 2>&1

echo "  04 packed batches (visualize_llm_input)"
python visualize_llm_input.py \
    --dataset-dir "$DS/splits/val_community" \
    --link-detector "$DET" \
    --layout-policy identifier_prefix_eos \
    --num-packs 5 --token-budget 16384 --seed 33 --no-color \
    > "$OUT/04_packed_batches.txt" 2>&1

cp -f "$ART/graphs/${LANG}/graph_stats.json" "$OUT/graph_stats.json" 2>/dev/null || true

echo "[artifacts] $LANG done:"
ls -la "$OUT"
