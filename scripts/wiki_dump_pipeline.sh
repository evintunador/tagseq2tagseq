#!/bin/bash
# Per-dump wiki pipeline: extract → build_graph → pretokenize into a per-dump
# pretokenized dataset dir. Idempotent: each stage is skipped if its output
# marker already exists, so a killed run can be re-launched safely.
#
# Usage: wiki_dump_pipeline.sh <tag> <dump.json.gz> <procs>
set -uo pipefail

TAG="$1"
DUMP="$2"
PROCS="${3:-120}"

REPO=/fss/evin_t/tagseq2tagseq
ROOT=/fss-data/evin_t/tagseq2tagseq_artifacts/wiki_multi
EXTRACT_DIR="$ROOT/extracted/$TAG"
PRETOK_DIR=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/wiki_$TAG
PY=$REPO/.venv/bin/python

export CUDA_MODULE_LOADING=LAZY
cd "$REPO" || exit 1
mkdir -p "$EXTRACT_DIR"

echo "[$TAG $(date +%T)] === START on $(hostname) (procs=$PROCS) ==="

# --- Stage 1: extract markdown ------------------------------------------------
if [ -f "$EXTRACT_DIR/.extract_done" ]; then
  echo "[$TAG] stage1 extract: already done, skipping"
else
  echo "[$TAG $(date +%T)] stage1 extract → $EXTRACT_DIR"
  $PY -m data.wiki_graph_extractor.dump_extractor "$DUMP" -o "$EXTRACT_DIR" -p "$PROCS" \
    || { echo "[$TAG] EXTRACT FAILED"; exit 1; }
  touch "$EXTRACT_DIR/.extract_done"
fi

# --- Stage 2: build link graph ------------------------------------------------
if [ -f "$EXTRACT_DIR/graph.jsonl" ]; then
  echo "[$TAG] stage2 build_graph: graph.jsonl exists, skipping"
else
  echo "[$TAG $(date +%T)] stage2 build_graph"
  $PY -m data.wiki_graph_extractor.build_graph "$EXTRACT_DIR" -p "$PROCS" \
    || { echo "[$TAG] BUILD_GRAPH FAILED"; exit 1; }
fi

# --- Stage 3: pretokenize -----------------------------------------------------
if [ -f "$PRETOK_DIR/metadata.json" ]; then
  echo "[$TAG] stage3 pretokenize: metadata.json exists, skipping"
else
  echo "[$TAG $(date +%T)] stage3 pretokenize → $PRETOK_DIR"
  # ReproducibilityManager uses -o directly and refuses a dir that already has a
  # 'reproducibility' folder, so clear a partial attempt first.
  rm -rf "$PRETOK_DIR"
  $PY -m data.pretokenize "$EXTRACT_DIR" "$EXTRACT_DIR/graph.jsonl" \
    -o "$PRETOK_DIR" --tokenizer-name gpt2 --shard-size-gb 2.0 -p "$PROCS" \
    || { echo "[$TAG] PRETOKENIZE FAILED"; exit 1; }
fi

echo "[$TAG $(date +%T)] === DONE. nodes=$(wc -l < "$PRETOK_DIR/tokenized_graph.jsonl") ==="
