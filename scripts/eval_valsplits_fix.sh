#!/bin/bash
# Fix the val-split metrics the 47851 recovery botched (passed --split all → community_pack n=0/NaN).
# For each of the 15 recovered runs, re-run:
#   - held_out_perplexity on val_random   (proper held-out set)
#   - community_pack_perplexity on val_community AND val_random (headline cross-doc metric for code)
# Writes into a merged eval_valsplits.json per run dir (keeps the good repobench/humaneval already in eval_results.json).
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
ART=/fss-data/evin_t/tagseq2tagseq_artifacts

vs () { # run_dir lang is_cdl
  local rd=$1 lang=$2 cdl=$3
  local dir=$REPO/runs/$rd ckpt=$REPO/runs/$rd/checkpoints/best_model.pt
  [ -f "$ckpt" ] || { echo "[vs] MISSING $ckpt"; return; }
  local ds=$ART/pretokenized_datasets/$lang
  local conds="doceval"; [ "$cdl" = 1 ] && conds="baseline experimental"
  echo "=== [$(date)] $rd ($lang cdl=$cdl) held_out(val_random)+community_pack(val_community,val_random) ==="
  # held_out on val_random
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds" \
    --benchmarks held_out_perplexity --conditions doceval --split val_random --max-docs 500 \
    --output "$dir/eval_heldout_valrandom.json" || echo "[vs] WARN heldout $rd"
  # community_pack on val_community (headline)
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds" \
    --benchmarks community_pack_perplexity --conditions $conds --split val_community --max-docs 500 \
    --output "$dir/eval_commpack_valcommunity.json" || echo "[vs] WARN cp_vc $rd"
  # community_pack on val_random
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds" \
    --benchmarks community_pack_perplexity --conditions $conds --split val_random --max-docs 500 \
    --output "$dir/eval_commpack_valrandom.json" || echo "[vs] WARN cp_vr $rd"
}

vs run_20260722_063905_465684 java 1
vs run_20260722_064306_874871 java 0
vs run_20260722_065208_373136 java 0
vs run_20260722_070055_225903 java 0
vs run_20260722_191916_590119 java 1
vs run_20260722_193422_781377 java 0
vs run_20260722_194928_381368 java 1
vs run_20260722_172933_276517 dart 1
vs run_20260722_173334_695764 dart 0
vs run_20260722_174840_254323 dart 0
vs run_20260722_180345_944331 dart 0
vs run_20260722_181852_210934 zig 1
vs run_20260722_183358_354915 zig 0
vs run_20260722_184904_248598 zig 0
vs run_20260722_190409_782860 zig 0
echo "=== [$(date)] VAL-SPLIT EVAL FIX DONE ==="
