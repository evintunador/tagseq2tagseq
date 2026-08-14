#!/usr/bin/env bash
# Build the 32B BALANCED merged pack set — balance CAPPED at the memorization threshold.
# Per Muennighoff et al. 2023 ("Scaling Data-Constrained Language Models"), repeating
# data up to ~4 epochs is ~as good as fresh; beyond that returns collapse toward
# memorization. So we cap every source at SAFE=4 epochs and let the LARGER sources take
# somewhat more than the naive 1/11 share to absorb the small sources' shortfall —
# rather than forcing perfect balance by over-repeating tiny corpora (the earlier draft
# wrongly pushed zig 16x / dart 9x / java 6x, deep in the memorization regime).
#
# Water-filled targets (4-epoch cap, ~32.0B total, 976,559 packs):
#   CAPPED @4ep : go 86240, java 67788, zig 3684, dart 55672   (their 4-epoch max)
#   fill @109025: wiki(3.77x) stack arxiv typescript kotlin rust javascript
#   -> max repetition of ANY source = 3.77x (wiki); naive 1/11 would be 88778.
# Ordering is irrelevant now — training shuffles within buckets (seed=42), commit 93d542c.
set -euo pipefail
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
P=$ART/pretokenized_datasets
S=$ART/schedules
MT=$P/merged_all_v2/splits/train
OUT=$P/merged_all_v2/epoch_32b_balanced
cd /fss/evin_t/tagseq2tagseq

# helper: comma-join epoch_0..(n-1) under a schedule root
ep() { local root="$1" n="$2" out=""; for i in $(seq 0 $((n-1))); do out="$out${out:+,}$root/epoch_$i"; done; echo "$out"; }

python -m data.merge_packs \
  --merged-train-dir "$MT" \
  --output "$OUT" \
  --n-buckets 32 --seed 42 --token-budget 32768 \
  --source "wiki=$P/wiki_merged/splits/train=$(ep $S/wiki_merged_bfs 4)=109025" \
  --source "stack=$P/thestack/splits/train=$(ep $S/thestack_bfs 1)=109025" \
  --source "arxiv=$P/arxiv/splits/train=$(ep $S/arxiv_bfs 1)=109025" \
  --source "go=$P/go/splits/train=$(ep $S/go_bfs 4)=all" \
  --source "java=$P/java/splits/train=$(ep $S/java_bfs 4)=all" \
  --source "typescript=$P/typescript/splits/train=$(ep $S/typescript_bfs 1)=109025" \
  --source "kotlin=$P/kotlin/splits/train=$(ep $S/kotlin_bfs 2)=109025" \
  --source "rust=$P/rust/splits/train=$(ep $S/rust_bfs 3)=109025" \
  --source "javascript=$P/javascript/splits/train=$(ep $S/javascript_bfs 1)=109025" \
  --source "zig=$P/zig/splits/train=$(ep $S/zig_bfs 4)=all" \
  --source "dart=$P/dart/splits/train=$(ep $S/dart_bfs 4)=all"
echo "=== built $OUT (32B balanced, 4-epoch memorization cap) ==="
