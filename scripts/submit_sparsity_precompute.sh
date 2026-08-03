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

# dataset : link-detector : layout-policy : N_EPOCHS
# N_EPOCHS = the number of distinct-seed epochs the dataset's ~15k-step (3.9B)
# cross_doc ENDPOINT trained on (verified from each endpoint's hyperparameters.json):
# big corpora reach 15k steps in 1 epoch (js/ts/thestack); smaller ones cycle
# multiple distinct-seed epochs (rust 4, dart 9, wiki 4). The interior keep-arms
# MUST see the same epoch count so density isn't confounded with training length.
# Each keep-fraction builds epoch_0..N-1 (seeds 42..42+N-1), subsampled.
JOBS=(
  "javascript:javascript:stochastic_slash_comment_prefix:1"
  "typescript:typescript:stochastic_slash_comment_prefix:1"
  "thestack:python:stochastic_identifier_prefix:1"
  "rust:rust:stochastic_slash_comment_prefix:4"
  "dart:dart:stochastic_slash_comment_prefix:9"
  "wiki_merged:markdown:stochastic_identifier_prefix:4"
)
# Interior keeps for the density line. keep=1.0 is ALSO (re)built for the
# multi-epoch datasets whose on-disk full epochs are stale/single (wiki) — pass
# KEEPS="0.25 0.5 0.75 1.0" via env to include it; default is interior only.
KEEPS=(${KEEPS:-0.25 0.5 0.75})

n=0
for spec in "${JOBS[@]}"; do
  IFS=: read -r ds det layout nep <<< "$spec"
  for keep in "${KEEPS[@]}"; do
    cmd=(sbatch --export=ALL,DS="$ds",DET="$det",LAYOUT="$layout",KEEP="$keep",SEED=0,NEPOCHS="$nep" \
         scripts/precompute_sparsity_epoch.sbatch)
    if [ "${DRYRUN:-0}" = "1" ]; then
      echo "DRYRUN: DS=$ds DET=$det LAYOUT=$layout KEEP=$keep NEPOCHS=$nep"
    else
      "${cmd[@]}"
    fi
    n=$((n+1))
  done
done
echo "=== ${DRYRUN:+(dryrun) }enqueued $n precompute jobs ==="
