#!/bin/bash
# Eval the 12 TRAINED-NO-EVAL ablation cells (java 1, kotlin 2, dart 3, zig 6).
# Correct split handling (learned from the 47851/47924 bugs):
#   - held_out: point --dataset at splits/val_random + --split all (named-split lookup fails on split-dir datasets)
#   - community_pack: --dataset at top-level dataset + --split val_community (headline cross-doc for code)
#   - humaneval: rust/js/go/java only (HumanEvalPack langs); none of these 12 except java qualify
# Writes eval_results.json (community_pack, main) + eval_heldout_valrandom.json per run dir.
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
ART=/fss-data/evin_t/tagseq2tagseq_artifacts

ev () { # run_dir lang cdl
  local rd=$1 lang=$2 cdl=$3
  local dir=$REPO/runs/$rd ckpt=$REPO/runs/$rd/checkpoints/best_model.pt
  [ -f "$ckpt" ] || { echo "[tne] MISSING $ckpt"; return; }
  local ds=$ART/pretokenized_datasets/$lang
  local conds="doceval"; [ "$cdl" = 1 ] && conds="baseline experimental"
  local hb=""; case "$lang" in java) hb="humaneval_buggy"; esac
  echo "=== [$(date)] $rd ($lang cdl=$cdl) ==="
  # community_pack (val_community) + humaneval (if any) -> main eval_results.json
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds" \
    --benchmarks community_pack_perplexity $hb --conditions $conds --split val_community --max-docs 500 \
    ${hb:+--humaneval-language $lang} --output "$dir/eval_results.json" || echo "[tne] WARN cp $rd"
  # held_out on val_random split-subdir
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds/splits/val_random" \
    --benchmarks held_out_perplexity --conditions doceval --split all --max-docs 500 \
    --output "$dir/eval_heldout_valrandom.json" || echo "[tne] WARN ho $rd"
}

ev run_20260722_200434_279375 java 0
ev run_20260723_235106_530994 kotlin 1
ev run_20260724_000439_420779 kotlin 1
ev run_20260723_235106_530901 dart 1
ev run_20260724_000354_604732 dart 1
ev run_20260724_041049_444760 dart 1
ev run_20260723_235508_062681 zig 0
ev run_20260723_235106_530875 zig 1
ev run_20260724_000726_023385 zig 0
ev run_20260724_000339_593067 zig 1
ev run_20260724_003118_515308 zig 0
ev run_20260724_080058_124394 zig 1
echo "=== [$(date)] TNE EVAL DONE ==="
