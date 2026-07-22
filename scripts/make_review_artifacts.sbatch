#!/bin/bash
# SLURM wrapper for scripts/make_review_artifacts.sh (no wall-clock pressure —
# loading a multi-million-node GraphIndex + sample-dump scan takes minutes).
#
# Submit:  LANG=rust DET=rust sbatch scripts/make_review_artifacts.sbatch
#SBATCH --job-name=stack_artifacts
#SBATCH --partition=compute
#SBATCH --exclude=GPU-954,GPU-495   # known-bad nodes (instant-fail exit 0:53); see project memory
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/review_artifacts_%x_%j.out

set -uo pipefail
: "${LANG:?set LANG}"
: "${DET:?set DET (link detector name)}"
REPO=/fss/evin_t/tagseq2tagseq
cd "$REPO"
source "$REPO/.venv/bin/activate"
bash scripts/make_review_artifacts.sh "$LANG" "$DET"
