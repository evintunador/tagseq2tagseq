#!/bin/bash
# SLURM array downloader for a full the-stack-dedup language subset.
#
# Each array task is one worker that downloads a round-robin slice of the
# language's parquet files (see data/stack_sharded_download.py). Set the array
# size to the parquet-file count (get it with `--list-only`).
#
# Submit (example — Rust has 21 parquet files):
#   LANG=rust NPARQ=21 sbatch --array=0-20 scripts/download_stack_lang.sbatch
#   LANG=kotlin NPARQ=15 sbatch --array=0-14 scripts/download_stack_lang.sbatch
#   LANG=typescript NPARQ=71 sbatch --array=0-70 scripts/download_stack_lang.sbatch
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

# Activate the project venv (editable install lives here).
source "$REPO/.venv/bin/activate"

echo "[$(date)] LANG=$LANG worker=${SLURM_ARRAY_TASK_ID}/${NPARQ} -> $OUTDIR"
python -m data.stack_sharded_download \
    --lang "$LANG" \
    --out-dir "$OUTDIR" \
    --num-workers "$NPARQ" \
    --worker-id "${SLURM_ARRAY_TASK_ID}"
echo "[$(date)] worker ${SLURM_ARRAY_TASK_ID} done"
