#!/bin/bash
# Launch a torchrun training job on a SLURM-detached "down" node
# (GPU-269/423/694 — broken 8th GPU, not in SLURM, ssh-only).
#
# Parameterized so multiple jobs can share a node on disjoint GPU sets:
#   thestack : 4 GPUs (0-3) x accum 2  = 262144 tok/step
#   arxiv    : 2 GPUs      x accum 4   = 262144 tok/step
#   wiki     : 1 GPU       x accum 8   = 262144 tok/step
#
# Usage (run ON the node):
#   launch_downnode.sh <config> <run_tag> <gpus> <accum> <max_opt_steps> [extra main.py args...]
# where <gpus> is a CUDA_VISIBLE_DEVICES list, e.g. "0,1,2,3" or "4,5" or "6".
# Back-compat: if <gpus>/<accum>/<max_opt_steps> omitted → thestack defaults
# (0,1,2,3 / 2 / 27488).
set -euo pipefail

REPO=/fss/evin_t/tagseq2tagseq
VENV=$REPO/.venv
ART=/fss-data/evin_t/tagseq2tagseq_artifacts

CONFIG="$1"; shift
RUN_TAG="$1"; shift
GPUS="${1:-0,1,2,3}"; [ $# -gt 0 ] && shift || true
ACCUM="${1:-2}";      [ $# -gt 0 ] && shift || true
MAX_STEPS="${1:-27488}"; [ $# -gt 0 ] && shift || true
EXTRA_ARGS="$@"

# nproc = number of GPUs in the list
NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")

cd "$REPO"

# --- env: replicate launch_slurm.py's critical settings ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TORCH_DIST_TIMEOUT_SECONDS=1800
export NCCL_TIMEOUT=1800
# Single-node intra-host: NVLink, no IB needed.
# Prevent concurrent-compile deadlock (CLAUDE.md): synchronous in-process compile.
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
# Compile cache is a NODE-LOCAL shared dir keyed by RUN_TAG (set in
# downnode_rank_wrapper.sh) — distinct RUN_TAGs get distinct caches, so jobs
# sharing a node do not collide.

export CUDA_VISIBLE_DEVICES="$GPUS"

LOGDIR=$ART/pipeline_logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/downnode_${RUN_TAG}_$(hostname).log

TOK_PER_STEP=$(( NPROC * ACCUM * 32768 ))
echo "[launch_downnode] host=$(hostname) config=$CONFIG tag=$RUN_TAG gpus=$GPUS nproc=$NPROC accum=$ACCUM max_steps=$MAX_STEPS log=$LOG"
echo "[launch_downnode] $NPROC GPUs x accum_steps=$ACCUM x 32768 = $TOK_PER_STEP tokens/step"

# Random-ish port so jobs sharing a node don't clash on the rendezvous port.
PORT=$(( 29500 + (RANDOM % 6000) ))

nohup "$VENV/bin/torchrun" \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NPROC \
    --master_port=$PORT \
    --no-python \
    "$REPO/downnode_rank_wrapper.sh" "$RUN_TAG" \
    --config "$CONFIG" \
    --train_loop.max_optimizer_steps $MAX_STEPS \
    --train_loop.atomic_feature_kwargs.accum_steps $ACCUM \
    $EXTRA_ARGS \
    > "$LOG" 2>&1 &

echo "[launch_downnode] PID=$! LOG=$LOG"
echo "$LOG"
