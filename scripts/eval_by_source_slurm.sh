#!/usr/bin/env bash
# Run scripts/eval_merged_v2_run.sh (per-source held-out + community-pack perplexity, 11
# sources, 22 evals) for ONE finished merged_v2 checkpoint on ONE whole SLURM node
# (8 GPUs, ~1 h). Output: <run_dir>/eval_by_source/<src>__<bench>.json.
#
# Usage: scripts/eval_by_source_slurm.sh <label> <run_dir> <mode: cdl|dc> [ckpt_name=latest.pt]
#   mode cdl  -> conditions "baseline experimental" (cross-doc mask off/on)
#   mode dc   -> condition "doceval" (doc_causal / concat arms have no cross-doc mask)
# Env: PARTITION (compute), TIME (08:00:00), NODELIST (none)
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq
label="$1"; rundir="$2"; mode="$3"; ckname="${4:-latest.pt}"
[ -f "$rundir/checkpoints/$ckname" ] || { echo "MISSING $rundir/checkpoints/$ckname"; exit 1; }
mkdir -p "$rundir/eval_by_source"
nodelist_line=""; [ -n "${NODELIST:-}" ] && nodelist_line="#SBATCH --nodelist=${NODELIST}"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ts2ts_bysrc_${label}
#SBATCH --partition=${PARTITION:-compute}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512GB
#SBATCH --time=${TIME:-08:00:00}
#SBATCH --output=${rundir}/eval_by_source/slurm_%j.out
#SBATCH --error=${rundir}/eval_by_source/slurm_%j.err
${nodelist_line}
cd "$REPO"
NGPU=8 "$REPO/scripts/eval_merged_v2_run.sh" "$rundir" "$mode" "$ckname"
ls "$rundir/eval_by_source"/*.json | wc -l
EOF
