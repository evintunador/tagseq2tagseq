#!/usr/bin/env bash
# Precompute per-source val schedules for the merged_all_v2 run (11 linked sources,
# NO fineweb). wiki/stack/arxiv already have *_val schedules from the original
# merged build; this fills the 8 new languages. Each val loader is single-source
# (its own splits/val_* graph + own detector + own layout), so packs carry correct
# baked link_to_target + layout_name and the mask never depends on the merged
# model's (ambiguous) config detector. All 8 new langs use the slash-comment layout.
set -euo pipefail
P=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets
S=/fss-data/evin_t/tagseq2tagseq_artifacts/schedules
DEV="${1:-cuda:1}"   # pass a free GPU as $1
cd /fss/evin_t/tagseq2tagseq

run() {  # source_dir schedule_tag detector layout split
  local sdir="$1" tag="$2" det="$3" lay="$4" split="$5"
  if [ -d "$S/${tag}_val/${split}/epoch_0" ]; then
    echo "=== SKIP $tag/$split (exists) ==="; return
  fi
  echo "=== precompute $tag/$split (detector=$det layout=$lay dev=$DEV) ==="
  python precompute_epochs.py \
    --dataset-dir "$P/$sdir/splits/$split" \
    --output-dir "$S/${tag}_val/${split}" \
    --n-epochs 1 --strategy bfs --local-seq-len 32768 --n-buckets 32 --n-workers 8 \
    --seed 42 --link-detector "$det" --layout-policy "$lay" --max-grants 256 \
    --device "$DEV" --log-level WARNING
}

# tag  dataset_dir  detector  layout — all 8 new langs use slash_comment layout.
for split in val_community val_random; do
  run go         go         go         stochastic_slash_comment_prefix "$split"
  run java       java       java       stochastic_slash_comment_prefix "$split"
  run typescript typescript typescript stochastic_slash_comment_prefix "$split"
  run kotlin     kotlin     kotlin     stochastic_slash_comment_prefix "$split"
  run rust       rust       rust       stochastic_slash_comment_prefix "$split"
  run javascript javascript javascript stochastic_slash_comment_prefix "$split"
  run zig        zig        zig        stochastic_slash_comment_prefix "$split"
  run dart       dart       dart       stochastic_slash_comment_prefix "$split"
done
echo "ALL merged_v2 NEW-LANG VAL PRECOMPUTE DONE"
