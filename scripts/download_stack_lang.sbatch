#!/bin/bash
# SLURM array downloader for a full the-stack-dedup language subset.
#
# Each array task is one worker that downloads a round-robin slice of the
# language's parquet files (see data/stack_sharded_download.py). Set the array
# size to the parquet-file count (get it with `--list-only`).
#
# Submit — THROTTLE concurrency with %N (HF returns 429 if too many parquet
# fetches hit at once; keep simultaneous tasks <= ~12):
#   LANG=rust NPARQ=21 sbatch --array=0-20%12 scripts/download_stack_lang.sbatch
#   LANG=kotlin NPARQ=15 sbatch --array=0-14%12 scripts/download_stack_lang.sbatch
#   LANG=typescript NPARQ=71 sbatch --array=0-70%12 scripts/download_stack_lang.sbatch
#
# Only ONE language's array should run at a time (don't stack languages onto the
# same endpoint concurrently). Wait for one to drain before submitting the next.
#
# Output: one JSONL per parquet shard under
#   /fss-data/evin_t/tagseq2tagseq_artifacts/raw/<lang>/shards/
# (never /fss for bulk I/O — CLAUDE.md). The graph builder reads them as a glob.
#SBATCH --job-name=stack_dl
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/raw/%x_%A_%a.out

set -euo pipefail

: "${LANG:?set LANG=rust|kotlin|typescript}"
: "${NPARQ:?set NPARQ=<parquet file count> (== array size)}"

REPO=/fss/evin_t/tagseq2tagseq
OUTDIR=/fss-data/evin_t/tagseq2tagseq_artifacts/raw/${LANG}/shards
cd "$REPO"

# Keep the HF parquet cache on /fss-data (never /fss home) — parquet files are
# large (~2GB each for TS) and hf_hub_download caches the whole file.
export HF_HOME=/fss-data/evin_t/tagseq2tagseq_artifacts/.hf_cache
mkdir -p "$HF_HOME"

# Activate the project venv (editable install lives here).
source "$REPO/.venv/bin/activate"

echo "[$(date)] LANG=$LANG worker=${SLURM_ARRAY_TASK_ID}/${NPARQ} -> $OUTDIR"
python -m data.stack_sharded_download \
    --lang "$LANG" \
    --out-dir "$OUTDIR" \
    --num-workers "$NPARQ" \
    --worker-id "${SLURM_ARRAY_TASK_ID}" \
    --stagger 4
echo "[$(date)] worker ${SLURM_ARRAY_TASK_ID} done"
