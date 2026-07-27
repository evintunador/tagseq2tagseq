#!/bin/bash
# Re-eval all cross_doc_link cells at the CORRECTED max_grants=256 (was 64 pre-fix,
# commit 8317898). Only cdl cells are affected (doc_causal/concat have no grants).
# Dispatches the right cross-doc benchmark per dataset:
#   python/thestack -> repobench_cross_doc (repobench-language python) + community_pack
#   java            -> repobench_cross_doc (repobench-language java) + community_pack
#   wiki_merged     -> hotpotqa_cross_doc (the flagship +1.29 result)
#   arxiv + other code langs -> community_pack (their headline / only cross-doc metric)
# Writes eval_reeval256.json per run dir (does NOT overwrite the pre-fix eval_results.json;
# compare, then promote). Shard args: SHARD NSHARDS for parallel single-GPU packing.
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
SHARD=${1:-0}; NSHARDS=${2:-1}
mapfile -t ROWS < /tmp/cdl_cells.txt

i=0
for row in "${ROWS[@]}"; do
  if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  set -- $row; rd=$1; lang=$2; strat=$3
  ckpt=$REPO/$rd/checkpoints/best_model.pt
  [ -f "$ckpt" ] || { echo "[reeval] MISSING $ckpt"; continue; }
  # dataset dir: thestack keeps its name; others match lang
  ds=$ART/pretokenized_datasets/$lang
  benches="community_pack_perplexity"; extra=""; split="val_community"
  case "$lang" in
    thestack) benches="repobench_cross_doc community_pack_perplexity"; extra="--repobench-language python";;
    java)     benches="repobench_cross_doc community_pack_perplexity"; extra="--repobench-language java";;
    wiki_merged) benches="hotpotqa_cross_doc"; split="all";;
    arxiv)    benches="community_pack_perplexity";;
  esac
  echo "=== [$(date)] shard$SHARD REEVAL256 $rd ($lang $strat) benches=[$benches] ==="
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds" \
    --benchmarks $benches --conditions baseline experimental --split "$split" --max-docs 500 \
    $extra --output "$REPO/$rd/eval_reeval256.json" || echo "[reeval] WARN $rd"
done
echo "=== [$(date)] shard$SHARD REEVAL256 DONE ==="
