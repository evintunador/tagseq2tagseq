#!/bin/bash
# Precompute a graph-sparsity SUBSAMPLED epoch for one (dataset, keep_frac).
# CPU job (analytical kv). Rebuilds the SAME BFS packs as the full-density epoch
# but keeps only KEEP frac of the resolved cross-doc grants (seeded), so both the
# baked link_to_target AND kv_block_count (density buckets) reflect reduced density.
# keep=0.0 → doc_causal; keep=1.0 → full (== existing {ds}_bfs/epoch_0).
# See memory [[graph-sparsity-scaling-law]].
#
# Submit (one per dataset×keep):
#   DS=typescript DET=typescript LAYOUT=stochastic_slash_comment_prefix KEEP=0.5 \
#     sbatch scripts/precompute_sparsity_epoch.sbatch
#
#SBATCH --job-name=sparsprec
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/precompute/prec_%x_%j.out

set -euo pipefail
: "${DS:?dataset}"; : "${DET:?link-detector}"; : "${LAYOUT:?layout-policy}"; : "${KEEP:?keep-frac}"
: "${SEED:=0}"; : "${STRAT:=bfs}"; : "${NEPOCHS:=1}"

REPO=/fss/evin_t/tagseq2tagseq-sparsity   # worktree has the keep_frac precompute knob
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
TRAIN=$ART/pretokenized_datasets/${DS}/splits/train
# keep tag like keep0p50; one epoch dir per (dataset, keep)
KTAG=keep$(echo "$KEEP" | sed 's/\./p/')
OUT=$ART/sparsity_scaling/schedules/${DS}_bfs_${KTAG}
cd "$REPO"
source /fss/evin_t/tagseq2tagseq/.venv/bin/activate

echo "[$(date)] sparsity precompute $DS keep=$KEEP nepochs=$NEPOCHS (det=$DET layout=$LAYOUT) -> $OUT"
python precompute_epochs.py \
    --dataset-dir "$TRAIN" \
    --output-dir  "$OUT" \
    --n-epochs "$NEPOCHS" \
    --strategy "$STRAT" \
    --local-seq-len 32768 \
    --n-buckets 32 \
    --n-workers 16 \
    --link-detector "$DET" \
    --layout-policy "$LAYOUT" \
    --max-grants 256 \
    --keep-frac "$KEEP" \
    --keep-seed "$SEED" \
    --keep-mode edge \
    --device cpu
echo "[$(date)] $DS keep=$KEEP done"
python - "$OUT/epoch_0/metadata.json" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
print("  meta:", {k:d.get(k) for k in ("n_packs","keep_frac","keep_seed","max_grants","token_budget")})
PY
