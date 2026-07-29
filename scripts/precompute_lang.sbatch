#!/bin/bash
# Precompute a packed-epoch schedule for one (lang, strategy). CPU job (analytical
# kv method — no GPU). Uses the TRAINING layout so pack boundaries match training.
#
# Submit:  LANG=rust DET=rust STRAT=bfs LAYOUT=stochastic_slash_comment_prefix \
#            sbatch scripts/precompute_lang.sbatch
#SBATCH --job-name=precompute
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/schedules/precompute_%x_%j.out

set -euo pipefail
: "${LANG:?}"; : "${DET:?}"; : "${STRAT:?}"; : "${LAYOUT:?}"

REPO=/fss/evin_t/tagseq2tagseq
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
TRAIN=$ART/pretokenized_datasets/${LANG}/splits/train
OUT=$ART/schedules/${LANG}_${STRAT}
cd "$REPO"
source "$REPO/.venv/bin/activate"

echo "[$(date)] precompute $LANG/$STRAT (detector=$DET, layout=$LAYOUT) -> $OUT"
python precompute_epochs.py \
    --dataset-dir "$TRAIN" \
    --output-dir  "$OUT" \
    --n-epochs 1 \
    --strategy "$STRAT" \
    --local-seq-len 32768 \
    --n-buckets 32 \
    --n-workers 16 \
    --link-detector "$DET" \
    --layout-policy "$LAYOUT" \
    --max-grants 256
echo "[$(date)] $LANG/$STRAT done"
cat "$OUT/epoch_0/metadata.json" 2>/dev/null || true
