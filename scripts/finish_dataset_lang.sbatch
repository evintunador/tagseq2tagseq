#!/bin/bash
# SLURM CPU job: pretokenize -> symlink -> split -> audit for a built language graph.
# Runs after scripts/build_graph_lang.sbatch has produced graphs/<lang>/{graph,content}.jsonl.
#
# Submit (one language at a time):
#   LANG=rust sbatch scripts/finish_dataset_lang.sbatch
#SBATCH --job-name=stack_finish
#SBATCH --partition=compute
#SBATCH --exclude=GPU-954,GPU-495   # known-bad nodes (instant-fail exit 0:53); see project memory
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/%x_%j.out

set -euo pipefail
: "${LANG:?set LANG=rust|kotlin|typescript}"

REPO=/fss/evin_t/tagseq2tagseq
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
GRAPH=$ART/graphs/${LANG}
RUN=$ART/pretokenized_datasets/${LANG}_run
LINK=$ART/pretokenized_datasets/${LANG}
cd "$REPO"
source "$REPO/.venv/bin/activate"

echo "[$(date)] pretokenize $LANG: $GRAPH -> $RUN"
python -m data.pretokenize_${LANG} \
    "$GRAPH/content.jsonl" "$GRAPH/graph.jsonl" \
    -o "$RUN" -p 32

# Symlink <lang> -> <lang>_run (configs point at <lang>); replace any stale link.
rm -f "$LINK"
ln -s "$(basename "$RUN")" "$LINK"
echo "[$(date)] symlinked $LINK -> $RUN"

echo "[$(date)] split $LANG"
python -m data.split_graph --dataset-dir "$LINK"

echo "[$(date)] audit $LANG (full graph)"
python -m data.graph_harness.run_audit "$LINK"
echo "[$(date)] audit $LANG (train split)"
python -m data.graph_harness.run_audit "$LINK/splits/train"
echo "[$(date)] $LANG dataset COMPLETE"
