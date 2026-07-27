#!/bin/bash
# Eval the 19 TRAINED-NO-EVAL ablation cells (rust/kotlin/dart/javascript × dfs/rw/random).
# All trained to completion over the weekend; end-of-training eval never fired. Eval-only.
# Correct split handling: community_pack on val_community (headline); held_out via
# splits/val_random subdir + --split all; humaneval for rust/js (HumanEvalPack langs).
# Takes a shard arg (0-3) + nshards to split the 19 across parallel single-GPU jobs.
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
SHARD=${1:-0}; NSHARDS=${2:-1}

mapfile -t ROWS < <(cat <<'EOF'
runs/run_20260723_235508_062910 dart doc_causal
runs/run_20260724_000811_043374 dart doc_causal
runs/run_20260724_154844_853638 dart doc_causal
runs/run_20260723_235106_530999 javascript cross_doc_link
runs/run_20260723_235852_969868 javascript doc_causal
runs/run_20260724_001039_549882 javascript cross_doc_link
runs/run_20260724_095721_469921 javascript doc_causal
runs/run_20260724_154608_233719 javascript cross_doc_link
runs/run_20260724_005556_813826 javascript doc_causal
runs/run_20260723_235537_918590 kotlin doc_causal
runs/run_20260724_000912_260241 kotlin doc_causal
runs/run_20260724_095209_785799 kotlin cross_doc_link
runs/run_20260724_153821_125117 kotlin doc_causal
runs/run_20260723_235106_530909 rust cross_doc_link
runs/run_20260723_235522_920865 rust doc_causal
runs/run_20260724_000355_319847 rust cross_doc_link
runs/run_20260724_000758_678039 rust doc_causal
runs/run_20260724_052039_006085 rust cross_doc_link
runs/run_20260724_125820_815434 rust doc_causal
EOF
)

i=0
for row in "${ROWS[@]}"; do
  if [ $((i % NSHARDS)) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  set -- $row; rd=$1; lang=$2; mask=$3
  ckpt=$REPO/$rd/checkpoints/best_model.pt
  [ -f "$ckpt" ] || { echo "[tne19] MISSING $ckpt"; continue; }
  ds=$ART/pretokenized_datasets/$lang
  cdl=0; [ "$mask" = cross_doc_link ] && cdl=1
  conds="doceval"; [ "$cdl" = 1 ] && conds="baseline experimental"
  hb=""; case "$lang" in rust|javascript) hb="humaneval_buggy"; esac
  hl=""; [ "$lang" = javascript ] && hl="js"; [ "$lang" = rust ] && hl="rust"
  echo "=== [$(date)] shard$SHARD $rd ($lang $mask) ==="
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds" \
    --benchmarks community_pack_perplexity $hb --conditions $conds --split val_community --max-docs 500 \
    ${hl:+--humaneval-language $hl} --output "$REPO/$rd/eval_results.json" || echo "[tne19] WARN cp $rd"
  python eval_checkpoints.py --checkpoints "$ckpt" --dataset "$ds/splits/val_random" \
    --benchmarks held_out_perplexity --conditions doceval --split all --max-docs 500 \
    --output "$REPO/$rd/eval_heldout_valrandom.json" || echo "[tne19] WARN ho $rd"
done
echo "=== [$(date)] shard$SHARD DONE ==="
