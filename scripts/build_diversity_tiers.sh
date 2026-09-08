#!/usr/bin/env bash
# Diversity-COUNT curve at FIXED 3.9B total (~118446 packs), nested subsets.
# Adds datasets largest->smallest so low-diversity tiers use the best data and the
# small/noisy sets (zig,dart) only enter at the high-diversity end. Each tier splits
# the fixed 3.9B budget equally across its N domains (per-domain = 118446/N packs,
# single epoch, no repetition). Isolates DIVERSITY from total compute: more domains =
# fewer tokens/domain at constant total. cross_doc + doc_causal launched per tier.
set -euo pipefail
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
P=$ART/pretokenized_datasets; S=$ART/schedules
MT=$P/merged_all_v2/splits/train
cd /fss/evin_t/tagseq2tagseq

# tag -> "dataset_dir schedule_root"
declare -A DS=(
  [wiki]="wiki_merged wiki_merged_bfs" [stack]="thestack thestack_bfs" [arxiv]="arxiv arxiv_bfs"
  [go]="go go_bfs" [java]="java java_bfs" [typescript]="typescript typescript_bfs"
  [kotlin]="kotlin kotlin_bfs" [rust]="rust rust_bfs" [javascript]="javascript javascript_bfs"
  [zig]="zig zig_bfs" [dart]="dart dart_bfs")

build_tier() {  # $1=tier_name  $2=per_domain_target  $3..=domain tags
  local name="$1" tgt="$2"; shift 2
  local out="$P/merged_all_v2/epoch_3p9b_${name}"
  local args=()
  for tag in "$@"; do
    read -r dir root <<< "${DS[$tag]}"
    args+=(--source "$tag=$P/$dir/splits/train=$S/$root/epoch_0=$tgt")
  done
  echo "=== building $name ($# domains, $tgt packs/domain) -> $out ==="
  python -m data.merge_packs --merged-train-dir "$MT" --output "$out" \
    --n-buckets 32 --seed 42 --token-budget 32768 "${args[@]}"
}

build_tier div3 39482 arxiv javascript stack
build_tier div5 23689 arxiv javascript stack typescript kotlin
build_tier div7 16921 arxiv javascript stack typescript kotlin rust wiki
build_tier div9 13161 arxiv javascript stack typescript kotlin rust wiki go java
echo "=== ALL DIVERSITY TIERS BUILT ==="
