#!/usr/bin/env bash
# Runs a sequential queue of eval jobs on ONE GPU (this node). Invoked remotely by
# orchestrate.sh via ssh+nohup. Args: <gpu_index> <jobspec> [<jobspec> ...]
# jobspec is one pipe-delimited line from jobs.tsv:
#   run_id|dataset_subdir|benchmarks|conditions|split|max_docs|extra_flags
# Idempotent: a job whose <log>.done marker exists is skipped, so the queue can be
# safely relaunched to fill in skipped/failed jobs.
set -uo pipefail
IDX="$1"; shift
WT=/fss/evin_t/tagseq2tagseq-evaltrack
PY=/fss/evin_t/tagseq2tagseq/.venv/bin/python
RUNS=/fss/evin_t/tagseq2tagseq/runs
ART=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets
export TS2TS_EVALS_ROOT=/fss-data/evin_t/tagseq2tagseq_artifacts/evals
LOGDIR=$WT/scripts/rerun/logs
cd "$WT" || exit 3

for spec in "$@"; do
  IFS='|' read -r run dss bench conds split mdocs extra <<< "$spec"
  tag="${run}__${bench}"
  log="$LOGDIR/${tag}.log"
  if [ -f "$log.done" ]; then echo "[$(date +%T)] SKIP-done $tag" >> "$log"; continue; fi
  echo "[$(date +%T)] START $tag gpu=$IDX host=$(hostname)" >> "$log"
  # shellcheck disable=SC2086  (conds/extra intentionally word-split into flags)
  CUDA_VISIBLE_DEVICES=$IDX $PY eval_checkpoints.py \
    --checkpoints "$RUNS/$run/checkpoints/best_model.pt" \
    --dataset "$ART/$dss" \
    --benchmarks $bench --conditions $conds --split $split --max-docs $mdocs $extra \
    >> "$log" 2>&1
  rc=$?
  echo "[$(date +%T)] END $tag rc=$rc" >> "$log"
  if [ $rc -eq 0 ]; then touch "$log.done"; else touch "$log.fail"; fi
done
