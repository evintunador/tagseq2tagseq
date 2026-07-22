#!/bin/bash
# Generate the human/agent review-artifact bundle for a built language dataset.
#
# Produces, under review_artifacts/<lang>/:
#   01_audit_full.txt         - run_audit on the full graph (structural stats)
#   02_audit_train.txt        - run_audit on the train split
#   03_sample_dump.txt        - run_sample_dump: (source -> detected link -> resolved
#                               target -> snippet) tuples. THE main text artifact.
#   04_packed_batches_train.txt      - visualize_llm_input under the TRAINING
#                               (stochastic 50-50 card) layout: real packed batches,
#                               per-doc layout + in-pack cross-doc grants.
#   05_packed_batches_inference.txt  - same, under the deterministic inference layout.
#   06_attention_mask.png     - block_mask_creator: the actual FlexAttention
#                               cross_doc_link mask for a real packed batch.
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
# Review bundles live on /fss-data (bulk artifacts off /fss; referenced across runs).
OUT=$ART/review_artifacts/${LANG}
cd "$REPO"
mkdir -p "$OUT"

# Per-language seed so concurrent jobs don't collide on the mask PNG filename
# (block_mask_creator names it mask_viz_<mask>_seed<SEED>.png — seed only, no lang).
case "$LANG" in
  rust) SEED=33 ;; kotlin) SEED=34 ;; typescript) SEED=35 ;;
  go) SEED=36 ;; java) SEED=37 ;; *) SEED=33 ;;
esac

echo "[artifacts] $LANG -> $OUT (seed=$SEED)"

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

# Derive the layouts from the detector so the artifact matches TRAINING exactly.
# Training uses the STOCHASTIC (50-50 per-doc coin-flip) layout; inference uses the
# deterministic variant. We render BOTH so the human sees what the model actually
# trains on (some docs with a card, some without) AND the inference card.
TRAIN_LAYOUT=$(python -c "
from model.graph_traversal.link_detector import make_link_detector  # noqa
from data.layout import inference_layout_for_detector
inf = inference_layout_for_detector('$DET')
# training = stochastic sibling of the deterministic inference layout
train = 'stochastic_slash_comment_prefix' if inf.startswith('slash') else (
        'stochastic_identifier_prefix' if inf.startswith('identifier') else
        'stochastic_latex_comment_prefix' if inf.startswith('latex') else inf)
print(train, inf)
")
TRAIN_LP=$(echo "$TRAIN_LAYOUT" | cut -d' ' -f1)
INF_LP=$(echo "$TRAIN_LAYOUT" | cut -d' ' -f2)
echo "  layouts: train=$TRAIN_LP  inference=$INF_LP"

echo "  04 packed batches — TRAINING layout (stochastic card, epoch 0)"
python visualize_llm_input.py \
    --dataset-dir "$DS/splits/val_community" \
    --link-detector "$DET" \
    --layout-policy "$TRAIN_LP" --epoch 0 \
    --num-packs 5 --token-budget 16384 --seed $SEED --no-color \
    > "$OUT/04_packed_batches_train.txt" 2>&1

echo "  05 packed batches — INFERENCE layout (deterministic card)"
python visualize_llm_input.py \
    --dataset-dir "$DS/splits/val_community" \
    --link-detector "$DET" \
    --layout-policy "$INF_LP" \
    --num-packs 3 --token-budget 16384 --seed $SEED --no-color \
    > "$OUT/05_packed_batches_inference.txt" 2>&1

echo "  06 attention-mask PNG (cross_doc_link, real batch)"
python -m model.graph_traversal.block_mask_creator "$DS/splits/val_community" \
    --mask-type cross_doc_link --link-detector "$DET" \
    --layout-policy "$INF_LP" --token-budget 16384 --seed $SEED \
    > "$OUT/06_mask_build.log" 2>&1 || true
# block_mask_creator writes the PNG under model/graph_traversal/artifacts/
cp -f "$REPO/model/graph_traversal/artifacts/mask_viz_cross_doc_link_seed$SEED.png" \
      "$OUT/06_attention_mask.png" 2>/dev/null || true

cp -f "$ART/graphs/${LANG}/graph_stats.json" "$OUT/graph_stats.json" 2>/dev/null || true

echo "[artifacts] $LANG done:"
ls -la "$OUT"
