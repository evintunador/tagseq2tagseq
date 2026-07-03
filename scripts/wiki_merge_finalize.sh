#!/bin/bash
# Waits for all 8 per-dump wiki datasets to finish pretokenizing, then runs the
# downstream pipeline: merge (stage 3.5) → split (stage 4, source-stratified) →
# precompute epoch_0 (stage 5). Idempotent per stage. Meant to run detached.
set -uo pipefail

REPO=/fss/evin_t/tagseq2tagseq
PY=$REPO/.venv/bin/python
ROOT=/fss-data/evin_t/tagseq2tagseq_artifacts
PT=$ROOT/pretokenized_datasets
MERGED=$PT/wiki_merged
SCHED=$ROOT/schedules/wiki_merged_bfs
LOG=$ROOT/wiki_multi/logs/_finalize.log

cd "$REPO" || exit 1
exec >>"$LOG" 2>&1
echo "[finalize $(date +%T)] waiting for all 8 dumps to pretokenize..."

# Priority: more-useful/curated wikis win id collisions over less-useful ones.
# enwikisource (full texts) and enwiktionary (definitions) are lowest.
TAGS="simplewiki enwikinews enwikibooks enwikivoyage enwikiquote enwikiversity enwikisource enwiktionary"
PRIORITY="simplewiki,enwikivoyage,enwikibooks,enwikiquote,enwikiversity,enwikinews,enwikisource,enwiktionary"

# --- wait for all pretok metadata.json ---------------------------------------
while true; do
  missing=""
  for t in $TAGS; do
    [ -f "$PT/wiki_$t/metadata.json" ] || missing="$missing $t"
  done
  [ -z "$missing" ] && break
  echo "[finalize $(date +%T)] still waiting for:$missing"
  sleep 60
done
echo "[finalize $(date +%T)] all 8 dumps ready."

# --- Stage 3.5: merge --------------------------------------------------------
if [ -f "$MERGED/metadata.json" ]; then
  echo "[finalize] merge already done, skipping"
else
  echo "[finalize $(date +%T)] merging → $MERGED"
  INPUTS=""
  for t in $TAGS; do INPUTS="$INPUTS $t=$PT/wiki_$t"; done
  $PY $REPO/data/merge_datasets.py \
    --inputs $INPUTS \
    --output "$MERGED" \
    --priority "$PRIORITY" \
    --shard-mode hardlink || { echo "[finalize] MERGE FAILED"; exit 1; }
fi

# --- Stage 4: source-stratified split ----------------------------------------
if [ -d "$MERGED/splits/train" ]; then
  echo "[finalize] split already done, skipping"
else
  echo "[finalize $(date +%T)] splitting (source-stratified)"
  $PY $REPO/data/split_graph.py \
    --dataset-dir "$MERGED" \
    --stratify-by-source \
    --seed 42 || { echo "[finalize] SPLIT FAILED"; exit 1; }
fi

# --- Stage 5: precompute epoch_0 (mirror simplewiki schedule params) ---------
if [ -f "$SCHED/epoch_0/packs.parquet" ]; then
  echo "[finalize] precompute already done, skipping"
else
  echo "[finalize $(date +%T)] precomputing epoch_0 → $SCHED"
  $PY $REPO/precompute_epochs.py \
    --dataset-dir "$MERGED/splits/train" \
    --output-dir "$SCHED" \
    --n-epochs 1 \
    --strategy bfs \
    --local-seq-len 32768 \
    --n-buckets 32 \
    --n-workers 16 \
    --seed 42 \
    --link-detector markdown \
    --layout-policy stochastic_identifier_prefix \
    --max-grants 256 \
    --device cpu || { echo "[finalize] PRECOMPUTE FAILED"; exit 1; }
fi

echo "[finalize $(date +%T)] === PIPELINE COMPLETE ==="
echo "[finalize] merged nodes=$(wc -l < "$MERGED/tokenized_graph.jsonl")"
