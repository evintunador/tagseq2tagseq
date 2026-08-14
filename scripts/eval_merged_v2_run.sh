#!/usr/bin/env bash
# Eval-only recovery for a merged_all_v2 run that crashed on pack exhaustion before
# its on-completion eval (see [[merged-corpus-build]] step-budget-exceeds-packs note).
# Runs eval_checkpoints.py PER SOURCE (11 linked sources) on the run's best_model.pt:
# held_out_perplexity (val_random) + community_pack_perplexity (val_community), under
# baseline+experimental for cross_doc runs (the within-model A/B) or doceval for
# doc_causal. Each source runs on its own GPU in parallel. Writes one JSON per source
# into <run>/eval_by_source/. (The discriminating cross-doc BENCHMARK ports —
# repobench/ase/cceval/internal — are run separately via eval/benchmark_harness.)
# Usage: eval_merged_v2_run.sh <run_dir_basename> <cdl|dc>
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
ART=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets

RD="$REPO/runs/$1"; MODE="${2:-cdl}"; CKPT_NAME="${3:-best_model.pt}"
# NOTE: prefer latest.pt for merged_v2 runs — best_model.pt was selected by the
# pre-fix biased val metric, so it's unreliable; latest.pt is the final trained model.
CKPT="$RD/checkpoints/$CKPT_NAME"
[ -f "$CKPT" ] || { echo "MISSING $CKPT"; exit 1; }
OUT="$RD/eval_by_source"; mkdir -p "$OUT"

# source_tag -> pretokenized dataset dir (wiki=wiki_merged, stack=thestack)
declare -A DS=(
  [wiki]=wiki_merged [stack]=thestack [arxiv]=arxiv [go]=go [java]=java
  [typescript]=typescript [kotlin]=kotlin [rust]=rust [javascript]=javascript
  [zig]=zig [dart]=dart)
SOURCES=(wiki stack arxiv go java typescript kotlin rust javascript zig dart)

if [ "$MODE" = cdl ]; then CONDS="baseline experimental"; else CONDS="doceval"; fi

# held_out_perplexity uses val_random; community_pack_perplexity uses val_community.
# They need different --split, so run as two invocations per source. Each on its own
# GPU, round-robin over the 8 local GPUs, staggered to avoid a shared-FS cold-load storm.
# One eval per GPU at a time (GPUs 0-7). eval flex-compiles + loads a source graph;
# >1 per GPU risks OOM/thrash. Gate concurrency to NGPU by waiting when full.
NGPU="${NGPU:-8}"
launch() {  # gpu src benchmark dataset_subpath split
  local g="$1" src="$2" bench="$3" subpath="$4" split="$5"
  local ds="$ART/${DS[$src]}${subpath}"
  echo "=== eval $src/$bench (ds=$ds split=$split) on cuda:$g conds=[$CONDS] ==="
  CUDA_VISIBLE_DEVICES=$g nohup python eval_checkpoints.py \
    --checkpoints "$CKPT" --dataset "$ds" \
    --benchmarks "$bench" --conditions $CONDS --max-docs 500 \
    --split "$split" \
    --output "$OUT/${src}__${bench}.json" > "$OUT/${src}__${bench}.log" 2>&1
}
# build the job list (src bench dataset_subpath split), then run NGPU at a time.
# held_out_perplexity: point --dataset at the split SUBDIR (its own tokenized_graph.jsonl,
#   no per-node split annotations) and sample all → --split all. Uses score_docs_batched,
#   unaffected by the 2048-budget bug.
# community_pack_perplexity: point --dataset at the SOURCE dir; the benchmark reads
#   splits/val_community internally. Budget now resolves to 32768 (2048 bug fixed).
# each job line: "src bench subpath split" (subpath "-" = none, i.e. the source dir)
JOBS=()
for src in "${SOURCES[@]}"; do
  JOBS+=("$src held_out_perplexity /splits/val_random all")
  JOBS+=("$src community_pack_perplexity - val_community")
done
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in $(seq 0 $((NGPU-1))); do
    [ $i -lt ${#JOBS[@]} ] || break
    read -r s b sub sp <<< "${JOBS[$i]}"
    [ "$sub" = "-" ] && sub=""
    launch "$g" "$s" "$b" "$sub" "$sp" &
    sleep 6
    i=$((i+1))
  done
  wait  # barrier: finish this wave before the next (keeps ≤NGPU concurrent)
done
echo "=== ALL PER-SOURCE EVAL DONE for $1 -> $OUT ==="
