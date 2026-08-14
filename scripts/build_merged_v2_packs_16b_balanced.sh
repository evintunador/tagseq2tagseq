#!/usr/bin/env bash
# Build the 16B PERFECTLY-BALANCED merged pack set (multi-epoch union).
# Equal-ish share = 48466 packs (1.59B) per source, EXCEPT zig which can't reach it
# within the SAFE=4 epoch-repeat cap (maxes ~3618 packs across 4 epochs) so zig gives
# 'all' 4 epochs; its shortfall is already absorbed into the other 10's 48466 target.
# Each source unions the minimum # of distinct-seed epoch dirs (<=4) whose pooled
# packs reach the target (merge_packs balance-selects across the union). Total 16.00B.
# See TODOS.md "16B perfectly-balanced allocation". Depends on merged_all_v2/splits/train.
set -euo pipefail
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
P=$ART/pretokenized_datasets
S=$ART/schedules
MT=$P/merged_all_v2/splits/train
OUT=$P/merged_all_v2/epoch_16b_balanced
cd /fss/evin_t/tagseq2tagseq

# helper: comma-join epoch_0..(n-1) under a schedule root
ep() { local root="$1" n="$2" out=""; for i in $(seq 0 $((n-1))); do out="$out${out:+,}$root/epoch_$i"; done; echo "$out"; }

python -m data.merge_packs \
  --merged-train-dir "$MT" \
  --output "$OUT" \
  --n-buckets 32 --seed 42 --token-budget 32768 \
  --source "wiki=$P/wiki_merged/splits/train=$(ep $S/wiki_merged_bfs 2)=48466" \
  --source "stack=$P/thestack/splits/train=$(ep $S/thestack_bfs 1)=48466" \
  --source "arxiv=$P/arxiv/splits/train=$(ep $S/arxiv_bfs 1)=48466" \
  --source "go=$P/go/splits/train=$(ep $S/go_bfs 3)=48466" \
  --source "java=$P/java/splits/train=$(ep $S/java_bfs 3)=48466" \
  --source "typescript=$P/typescript/splits/train=$(ep $S/typescript_bfs 1)=48466" \
  --source "kotlin=$P/kotlin/splits/train=$(ep $S/kotlin_bfs 1)=48466" \
  --source "rust=$P/rust/splits/train=$(ep $S/rust_bfs 2)=48466" \
  --source "javascript=$P/javascript/splits/train=$(ep $S/javascript_bfs 1)=48466" \
  --source "zig=$P/zig/splits/train=$(ep $S/zig_bfs 4)=all" \
  --source "dart=$P/dart/splits/train=$(ep $S/dart_bfs 4)=48466"
echo "=== built $OUT (16B perfectly-balanced) ==="
