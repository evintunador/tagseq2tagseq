#!/bin/bash
# SLURM CPU job: build the intra-repo import graph for a full language subset
# from its downloaded JSONL shards (data/<lang>_graph_extractor/build_<lang>_graph.py).
#
# Single-threaded tree-sitter over millions of files — the long pole. STAGGER:
# run one language's build at a time (CLAUDE.md launch discipline).
#
# Submit:
#   LANG=rust       sbatch scripts/build_graph_lang.sbatch
#   LANG=kotlin     sbatch scripts/build_graph_lang.sbatch
#   LANG=typescript sbatch scripts/build_graph_lang.sbatch
#SBATCH --job-name=stack_build
#SBATCH --partition=compute
#SBATCH --exclude=GPU-954,GPU-495   # known-bad nodes (instant-fail exit 0:53); see project memory
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/fss-data/evin_t/tagseq2tagseq_artifacts/graphs/%x_%j.out

set -euo pipefail
: "${LANG:?set LANG=rust|kotlin|typescript}"

REPO=/fss/evin_t/tagseq2tagseq
ART=/fss-data/evin_t/tagseq2tagseq_artifacts
SHARDS=$ART/raw/${LANG}/shards
OUT=$ART/graphs/${LANG}
cd "$REPO"
source "$REPO/.venv/bin/activate"
mkdir -p "$OUT"

echo "[$(date)] build $LANG graph from $SHARDS -> $OUT"
python -m data.${LANG}_graph_extractor.build_${LANG}_graph "$SHARDS" -o "$OUT" --min-links 1
echo "[$(date)] $LANG build done"
cat "$OUT/graph_stats.json"
