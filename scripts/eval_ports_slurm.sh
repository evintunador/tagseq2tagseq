#!/usr/bin/env bash
# Run the 13 cross-doc benchmark PORTS (Tier 0/1/2, use_line scope) for ONE finished
# cross_doc checkpoint on ONE whole SLURM node (8 GPUs, ~30 min). Writes
# <run_dir>/port_eval/<port>__<scope>.json + .log, the same layout eval_ports_local.sh
# produces on the login node, so RESULTS tables read either.
#
# Usage: scripts/eval_ports_slurm.sh <label> <run_dir> [ckpt_name=latest.pt]
#   label    short tag for the job name (ts2ts_ports_<label>)
#   run_dir  absolute run dir holding checkpoints/<ckpt_name>
# Env: SCOPE (use_line), MAXEX (500), PARTITION (compute), TIME (06:00:00), NODELIST (none)
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq
label="$1"; rundir="$2"; ckname="${3:-latest.pt}"
ck="$rundir/checkpoints/$ckname"
[ -f "$ck" ] || { echo "MISSING $ck"; exit 1; }
SCOPE="${SCOPE:-use_line}"; MAXEX="${MAXEX:-500}"
out="$rundir/port_eval"; mkdir -p "$out"
nodelist_line=""; [ -n "${NODELIST:-}" ] && nodelist_line="#SBATCH --nodelist=${NODELIST}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ts2ts_ports_${label}
#SBATCH --partition=${PARTITION:-compute}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512GB
#SBATCH --time=${TIME:-06:00:00}
#SBATCH --output=${out}/slurm_%j.out
#SBATCH --error=${out}/slurm_%j.err
${nodelist_line}
set -uo pipefail
cd "$REPO"
PORTS=(repobench_python repobench_java ase_kotlin crosscodeeval_ts \\
       internal_python internal_go internal_java internal_javascript \\
       internal_kotlin internal_rust internal_typescript internal_zig internal_dart)
echo "=== \${#PORTS[@]} port audits for $label ($ck) scope=$SCOPE max-examples=$MAXEX ==="
i=0
while [ \$i -lt \${#PORTS[@]} ]; do
  for g in 0 1 2 3 4 5 6 7; do
    [ \$i -ge \${#PORTS[@]} ] && break
    port="\${PORTS[\$i]}"
    echo "[\$i] \$port -> cuda:\$g"
    CUDA_VISIBLE_DEVICES=\$g "$REPO/.venv/bin/python" -m eval.benchmark_harness.run_port_audit \\
      --port "\$port" --tiers 0 1 2 --checkpoint "$ck" --scope "$SCOPE" \\
      --max-examples "$MAXEX" --device cuda --out "$out/\${port}__${SCOPE}.json" \\
      > "$out/\${port}__${SCOPE}.log" 2>&1 &
    i=\$((i+1)); sleep 3
  done
  wait   # finish this wave of 8 before the next
done
echo "=== ALL PORT AUDITS DONE for $label ==="
ls -la "$out"/*.json | wc -l
EOF
