#!/usr/bin/env bash
# Build the 3.9B equal-ish merged pack set for merged_all_v2 (11 linked sources).
# Each source contributes ~11809 packs (zig gives all 921) drawn EVENLY across its 32
# density buckets → ~3.90B tok. merge_packs.py remaps each source's doc_ids into the
# merged train graph, stamps per-source layout_name, and re-buckets over the union.
# Depends on merged_all_v2/splits/train (merge_datasets output).
set -euo pipefail
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
P=$ART/pretokenized_datasets
S=$ART/schedules
MT=$P/merged_all_v2/splits/train
OUT=$P/merged_all_v2/epoch_3p9b
cd /fss/evin_t/tagseq2tagseq

# Wait for the train union to exist (merge_datasets writes metadata.json last).
until [ -f "$MT/metadata.json" ] && [ -f "$MT/tokenized_graph.jsonl" ]; do
  echo "waiting for merged_all_v2/splits/train ..."; sleep 30
done
echo "=== train union ready: $(wc -l < $MT/tokenized_graph.jsonl) nodes ==="

python -m data.merge_packs \
  --merged-train-dir "$MT" \
  --output "$OUT" \
  --n-buckets 32 --seed 42 --token-budget 32768 \
  --source "wiki=$P/wiki_merged/splits/train=$S/wiki_merged_bfs/epoch_0=11809" \
  --source "stack=$P/thestack/splits/train=$S/thestack_bfs/epoch_0=11809" \
  --source "arxiv=$P/arxiv/splits/train=$S/arxiv_bfs/epoch_0=11809" \
  --source "go=$P/go/splits/train=$S/go_bfs/epoch_0=11809" \
  --source "java=$P/java/splits/train=$S/java_bfs/epoch_0=11809" \
  --source "typescript=$P/typescript/splits/train=$S/typescript_bfs/epoch_0=11809" \
  --source "kotlin=$P/kotlin/splits/train=$S/kotlin_bfs/epoch_0=11809" \
  --source "rust=$P/rust/splits/train=$S/rust_bfs/epoch_0=11809" \
  --source "javascript=$P/javascript/splits/train=$S/javascript_bfs/epoch_0=11809" \
  --source "zig=$P/zig/splits/train=$S/zig_bfs/epoch_0=all" \
  --source "dart=$P/dart/splits/train=$S/dart_bfs/epoch_0=11809"
echo "=== merged_v2 3.9B packs built → $OUT ==="
