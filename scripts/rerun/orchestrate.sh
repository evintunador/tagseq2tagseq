#!/usr/bin/env bash
# Distribute the eval rerun jobs across drained-but-idle GPUs and launch them.
# Round-robins jobs.tsv across a pool of node:gpu slots; each slot runs its queue
# sequentially via slot_runner.sh. Live-checks each GPU the MOMENT before launching
# its queue (availability on drained nodes drifts) and staggers launches so cold
# torch.compile doesn't thrash a shared node. Idempotent via slot_runner's .done
# markers, so re-running fills in only skipped/failed jobs.
set -uo pipefail
WT=/fss/evin_t/tagseq2tagseq-evaltrack
LOGDIR=$WT/scripts/rerun/logs; mkdir -p "$LOGDIR"
JOBS_FILE=$WT/scripts/rerun/jobs.tsv
RUNNER=$WT/scripts/rerun/slot_runner.sh
STATUS=$WT/scripts/rerun/status.txt
# node:gpu candidates. Avoid GPU-159 (vLLM endpoint) and GPU-670 (MFS monitor).
POOL=(${RERUN_POOL:-GPU-658:0 GPU-658:2 GPU-658:3 GPU-658:4 GPU-658:5 GPU-652:5 GPU-652:6})
FREE_MEM_MIB="${FREE_MEM_MIB:-2000}"
STAGGER="${STAGGER:-45}"

mapfile -t JOBS < <(grep -v '^#' "$JOBS_FILE" | grep -v '^[[:space:]]*$')
ns=${#POOL[@]}
declare -A SLOT_JOBS
for i in "${!JOBS[@]}"; do
  s=$((i % ns)); SLOT_JOBS[$s]+="${JOBS[$i]}"$'\t'
done

{ echo "=== orchestrate $(date '+%F %T') ==="; } | tee -a "$STATUS"
for s in "${!POOL[@]}"; do
  slot="${POOL[$s]}"; node="${slot%%:*}"; idx="${slot##*:}"
  IFS=$'\t' read -r -a specs <<< "${SLOT_JOBS[$s]:-}"
  args=(); for sp in "${specs[@]}"; do [ -n "$sp" ] && args+=("$sp"); done
  [ ${#args[@]} -eq 0 ] && continue
  # LIVE availability check right before submit.
  used=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $idx" 2>/dev/null | tr -dc '0-9')
  if [ -z "$used" ] || [ "$used" -ge "$FREE_MEM_MIB" ]; then
    echo "SKIP $slot busy(used=${used:-NA}) njobs=${#args[@]}" | tee -a "$STATUS"; continue
  fi
  # Build remote command, single-quoting each jobspec (they contain spaces).
  q=""; for a in "${args[@]}"; do q+=" '${a}'"; done
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
    "nohup bash $RUNNER $idx $q >/dev/null 2>&1 & echo pid=\$!" 2>&1 | tail -1)
  echo "LAUNCH $slot $out njobs=${#args[@]} [${args[*]//|/ }]" | tee -a "$STATUS"
  sleep "$STAGGER"
done
echo "=== submit pass done $(date '+%T') ===" | tee -a "$STATUS"
