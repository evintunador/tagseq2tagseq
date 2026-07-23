#!/bin/bash
# Eval-only recovery for the 15 near-complete code runs that stalled ~98% through
# training (schedule exhausted before the old hard step-cap) and never fired
# end-of-training eval. They have a valid best_model.pt. Run eval_checkpoints.py on
# each; no retraining. One GPU each (eval is single-GPU). Writes eval_results.json
# into each run dir.
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
ART=/fss-data/evin_t/tagseq2tagseq_artifacts

# run_dir  lang  is_cdl(1/0)
# benchmarks: all code runs get held_out_perplexity + community_pack_perplexity;
# cdl runs add repobench_cross_doc (java only, python-hardcoded otherwise); +humaneval where supported.
evalrun () {
  local rd=$1 lang=$2 cdl=$3
  local dir=$REPO/runs/$rd
  local ckpt=$dir/checkpoints/best_model.pt
  [ -f "$ckpt" ] || { echo "[eval] MISSING $ckpt"; return; }
  local ds=$ART/pretokenized_datasets/$lang
  # benchmark set
  local benches="held_out_perplexity community_pack_perplexity"
  local extra=""
  # humaneval only where HumanEvalPack has the lang
  case "$lang" in java|rust|go) benches="$benches humaneval_buggy"; extra="$extra --humaneval-language $lang";; esac
  # repobench_cross_doc only for java cdl (python-hardcoded; java now supported)
  if [ "$cdl" = 1 ] && [ "$lang" = java ]; then benches="$benches repobench_cross_doc"; extra="$extra --repobench-language java"; fi
  local conds="doceval"; [ "$cdl" = 1 ] && conds="baseline experimental"
  echo "=== [$(date)] eval $rd ($lang cdl=$cdl) benches=[$benches] ==="
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds" \
    --benchmarks $benches --conditions $conds --max-docs 500 $extra \
    --output "$dir/eval_results.json" || echo "[eval] WARN $rd nonzero"
}

# JAVA (7): sweep dc/cdl/concat/concatlink + ablation dfs-cdl/dfs-dc/rw-cdl
evalrun run_20260722_063905_465684 java 1   # sweep cdl
evalrun run_20260722_064306_874871 java 0   # sweep dc
evalrun run_20260722_065208_373136 java 0   # sweep concat
evalrun run_20260722_070055_225903 java 0   # sweep concatlink
evalrun run_20260722_191916_590119 java 1   # abl dfs cdl
evalrun run_20260722_193422_781377 java 0   # abl dfs dc
evalrun run_20260722_194928_381368 java 1   # abl rw cdl
# DART (4)
evalrun run_20260722_172933_276517 dart 1   # cdl
evalrun run_20260722_173334_695764 dart 0   # dc
evalrun run_20260722_174840_254323 dart 0   # concat
evalrun run_20260722_180345_944331 dart 0   # concatlink
# ZIG (4)
evalrun run_20260722_181852_210934 zig 1    # cdl
evalrun run_20260722_183358_354915 zig 0    # dc
evalrun run_20260722_184904_248598 zig 0    # concat
evalrun run_20260722_190409_782860 zig 0    # concatlink
echo "=== [$(date)] STALLED-RUN EVAL DONE ==="
