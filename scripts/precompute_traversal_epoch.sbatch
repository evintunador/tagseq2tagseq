#!/bin/bash
# Precompute a graph-sparsity TRAVERSAL-time subsampled epoch for one (dataset, keep).
# Distinct from precompute_sparsity_epoch.sbatch (mask-time): here the graph
# ADJACENCY is thinned BEFORE traversal, so BFS co-packs a genuinely sparser
# neighborhood (packs NOT byte-identical to the full run) and grants drop
# naturally as fewer link targets get co-packed — the faithful "sparser corpus"
# analog. Validates whether mask-time edge-dropout predicts real sparser-corpus
# training (the external-validity spot-check). See _TraversalSubsampledGraph.
#
# Submit: DS=typescript DET=typescript LAYOUT=stochastic_slash_comment_prefix \
#   TKEEP=0.5 sbatch scripts/precompute_traversal_epoch.sbatch
#
#SBATCH --job-name=travprec
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/precompute/travprec_%x_%j.out

set -euo pipefail
: "${DS:?dataset}"; : "${DET:?link-detector}"; : "${LAYOUT:?layout-policy}"; : "${TKEEP:?traversal-keep-frac}"
: "${SEED:=0}"; : "${STRAT:=bfs}"; : "${NEPOCHS:=1}"

REPO=/fss/evin_t/tagseq2tagseq-sparsity   # worktree has the traversal_keep_frac precompute knob
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
TRAIN=$ART/pretokenized_datasets/${DS}/splits/train
KTAG=travkeep$(echo "$TKEEP" | sed 's/\./p/')
OUT=$ART/sparsity_scaling/schedules/${DS}_bfs_${KTAG}
cd "$REPO"
source /fss/evin_t/tagseq2tagseq/.venv/bin/activate

echo "[$(date)] TRAVERSAL precompute $DS tkeep=$TKEEP nepochs=$NEPOCHS (det=$DET layout=$LAYOUT) -> $OUT"
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
    --traversal-keep-frac "$TKEEP" \
    --traversal-keep-seed "$SEED" \
    --device cpu
echo "[$(date)] $DS tkeep=$TKEEP done"
python - "$OUT/epoch_0/metadata.json" <<'PY' 2>/dev/null || true
import json,sys
d=json.load(open(sys.argv[1]))
print("  meta:", {k:d.get(k) for k in ("n_packs","traversal_keep_frac","traversal_keep_seed","max_grants","token_budget")})
PY
