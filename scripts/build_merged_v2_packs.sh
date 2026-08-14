#!/usr/bin/env bash
# Build a single-epoch merged pack set for merged_all_v2 at a given per-source target.
# Usage: build_merged_v2_packs.sh <out_epoch_dir> <wiki> <stack> <arxiv> <go> <java> \
#            <typescript> <kotlin> <rust> <javascript> <zig> <dart>
# Each target is an int (#packs, balance-selected across density buckets) or 'all'.
# Reads epoch_0 per source (distinct-seed multi-epoch unions are a separate path —
# see build_merged_v2_packs_balanced.sh). Depends on merged_all_v2/splits/train.
set -euo pipefail
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
P=$ART/pretokenized_datasets
S=$ART/schedules
MT=$P/merged_all_v2/splits/train
cd /fss/evin_t/tagseq2tagseq

OUT="$1"; shift
declare -A T
T[wiki]="$1"; T[stack]="$2"; T[arxiv]="$3"; T[go]="$4"; T[java]="$5"
T[typescript]="$6"; T[kotlin]="$7"; T[rust]="$8"; T[javascript]="$9"; T[zig]="${10}"; T[dart]="${11}"

python -m data.merge_packs \
  --merged-train-dir "$MT" \
  --output "$OUT" \
  --n-buckets 32 --seed 42 --token-budget 32768 \
  --source "wiki=$P/wiki_merged/splits/train=$S/wiki_merged_bfs/epoch_0=${T[wiki]}" \
  --source "stack=$P/thestack/splits/train=$S/thestack_bfs/epoch_0=${T[stack]}" \
  --source "arxiv=$P/arxiv/splits/train=$S/arxiv_bfs/epoch_0=${T[arxiv]}" \
  --source "go=$P/go/splits/train=$S/go_bfs/epoch_0=${T[go]}" \
  --source "java=$P/java/splits/train=$S/java_bfs/epoch_0=${T[java]}" \
  --source "typescript=$P/typescript/splits/train=$S/typescript_bfs/epoch_0=${T[typescript]}" \
  --source "kotlin=$P/kotlin/splits/train=$S/kotlin_bfs/epoch_0=${T[kotlin]}" \
  --source "rust=$P/rust/splits/train=$S/rust_bfs/epoch_0=${T[rust]}" \
  --source "javascript=$P/javascript/splits/train=$S/javascript_bfs/epoch_0=${T[javascript]}" \
  --source "zig=$P/zig/splits/train=$S/zig_bfs/epoch_0=${T[zig]}" \
  --source "dart=$P/dart/splits/train=$S/dart_bfs/epoch_0=${T[dart]}"
echo "=== built $OUT ==="
