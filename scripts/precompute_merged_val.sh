#!/usr/bin/env bash
# Precompute per-source val schedules for the merged_all run.
# Each val loader is single-source (its own splits/val_* graph + own detector +
# own layout), so packs carry correct baked link_to_target + layout_name and the
# mask never depends on the merged model's (ambiguous) config detector.
# fineweb is edgeless -> only val_random (val_community is empty).
set -euo pipefail
P=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets
S=/fss-data/evin_t/tagseq2tagseq_artifacts/schedules
cd /fss/evin_t/tagseq2tagseq

run() {  # source_dir schedule_tag detector layout split
  local sdir="$1" tag="$2" det="$3" lay="$4" split="$5"
  echo "=== precompute $tag/$split (detector=$det layout=$lay) ==="
  python precompute_epochs.py \
    --dataset-dir "$P/$sdir/splits/$split" \
    --output-dir "$S/${tag}_val/${split}" \
    --n-epochs 1 --strategy bfs --local-seq-len 32768 --n-buckets 32 --n-workers 8 \
    --seed 42 --link-detector "$det" --layout-policy "$lay" \
    --device cuda:0 --log-level WARNING
}

for split in val_community val_random; do
  run wiki_merged wiki markdown stochastic_identifier_prefix     "$split"
  run thestack    stack python   stochastic_identifier_prefix     "$split"
  run arxiv       arxiv arxiv    stochastic_latex_comment_prefix  "$split"
done
# fineweb: edgeless, only val_random
run fineweb_39b   fineweb null   stochastic_identifier_prefix     val_random
echo "ALL VAL PRECOMPUTE DONE"
