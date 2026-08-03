#!/usr/bin/env bash
# Submit the Phase-2 (train-time) graph-sparsity precompute jobs: for each chosen
# dataset, build subsampled epoch dirs at keep ∈ {0.25, 0.5, 0.75}. Endpoints are
# NOT precomputed: keep=1.0 reuses the existing {ds}_bfs/epoch_0 (full density),
# keep=0.0 ≡ doc_causal (train the doc_causal arm, no grants needed).
#
# 6 datasets × 3 interior keeps = 18 CPU jobs → sparsity_scaling/schedules/.
# Dataset set (decided 2026-08-04): the 5 strongest-signal code datasets + wiki
# (biggest merged-model mover — tests whether train-time rescues text the way
# diversity training did). thestack == python.
#
# Usage: bash scripts/submit_sparsity_precompute.sh        # submit all
#        DRYRUN=1 bash scripts/submit_sparsity_precompute.sh  # print only
set -uo pipefail
cd /fss/evin_t/tagseq2tagseq-sparsity
mkdir -p /fss-data/evin_t/tagseq2tagseq_artifacts/sparsity_scaling/precompute

# dataset : link-detector : layout-policy   (from each {ds}_bfs/epoch_0 metadata)
JOBS=(
  "javascript:javascript:stochastic_slash_comment_prefix"
  "typescript:typescript:stochastic_slash_comment_prefix"
  "rust:rust:stochastic_slash_comment_prefix"
  "dart:dart:stochastic_slash_comment_prefix"
  "thestack:python:stochastic_identifier_prefix"
  "wiki_merged:markdown:stochastic_identifier_prefix"
)
KEEPS=(0.25 0.5 0.75)

n=0
for spec in "${JOBS[@]}"; do
  IFS=: read -r ds det layout <<< "$spec"
  for keep in "${KEEPS[@]}"; do
    cmd=(sbatch --export=ALL,DS="$ds",DET="$det",LAYOUT="$layout",KEEP="$keep",SEED=0 \
         scripts/precompute_sparsity_epoch.sbatch)
    if [ "${DRYRUN:-0}" = "1" ]; then
      echo "DRYRUN: DS=$ds DET=$det LAYOUT=$layout KEEP=$keep"
    else
      "${cmd[@]}"
    fi
    n=$((n+1))
  done
done
echo "=== ${DRYRUN:+(dryrun) }enqueued $n precompute jobs ==="
