#!/usr/bin/env bash
# Run the cross-doc benchmark PORTS (Tier 0/1/2) on completed merged_v2 cross_doc
# checkpoints, using the free LOCAL GPUs on the login node (GPU 0 is in use → use 1-7).
# One port-audit per GPU, gated to NGPU concurrent. Writes <run>/port_eval/<port>.json.
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
GPUS=(${GPUS_OVERRIDE:-1 2 3 4 5 6 7})
MAXEX="${MAXEX:-500}"
SCOPE="${SCOPE:-native}"
PORTS=(repobench_python repobench_java ase_kotlin crosscodeeval_ts \
       internal_python internal_go internal_java internal_javascript \
       internal_kotlin internal_rust internal_typescript internal_zig internal_dart)
# checkpoint label -> run dir (cross_doc completed runs; latest.pt = fully-cooled final)
declare -A CK=(
  [8b_cdl]=runs/run_20260813_144916_125137
  [16bnat_cdl]=runs/run_20260813_182257_104861
)
JOBS=()
for lbl in "${!CK[@]}"; do
  for p in "${PORTS[@]}"; do JOBS+=("$lbl $p"); done
done
echo "=== ${#JOBS[@]} port-audits across ${#GPUS[@]} GPUs (max-examples $MAXEX) ==="
i=0
while [ $i -lt ${#JOBS[@]} ]; do
  for g in "${GPUS[@]}"; do
    [ $i -ge ${#JOBS[@]} ] && break
    set -- ${JOBS[$i]}; lbl="$1"; port="$2"
    rd="${CK[$lbl]}"; out="$rd/port_eval"; mkdir -p "$out"
    ck="$rd/checkpoints/latest.pt"
    echo "[$i] $lbl/$port -> cuda:$g"
    CUDA_VISIBLE_DEVICES=$g nohup python -m eval.benchmark_harness.run_port_audit \
      --port "$port" --tiers 0 1 2 --checkpoint "$ck" --scope "$SCOPE" \
      --max-examples "$MAXEX" --device cuda --out "$out/${port}__${SCOPE}.json" \
      > "$out/${port}__${SCOPE}.log" 2>&1 &
    i=$((i+1)); sleep 3
  done
  wait   # barrier: let this batch of NGPU finish before the next
done
echo "=== ALL PORT AUDITS DONE ==="
