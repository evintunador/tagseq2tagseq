#!/bin/bash
# Preflight a compute node before launching a training job on it.
# Checks BOTH conditions that have silently killed sweep runs:
#   1. GPU 0 is empty (a foreign co-tenant on rank-0's GPU OOMs the run — see
#      SLURM_DDP_HANG_HANDOFF.md: the "DDP hang" was really a rank-0 CUDA OOM).
#   2. The dataset dir on /fss-data is visible (GPU-954 was missing this mount
#      entirely 2026-07-25 -> 2026-07-25, since fixed with `sudo mkdir -p
#      /fss-data && sudo mount /fss-data`; a run on an unmounted node dies in
#      ~77s with "Dataset directory not found"). Kept as a general safeguard
#      since any node could silently lose this mount, not just GPU-954.
#
# Usage:   scripts/preflight_node.sh GPU-1006 [GPU-229 ...]
# Prints one line per node ending in CLEAN or SKIP:<reason>. Exit 0 if ALL clean.
set -uo pipefail
DS=/fss-data/evin_t/tagseq2tagseq_artifacts/pretokenized_datasets/wiki_merged
GPU0_MAX_MIB="${GPU0_MAX_MIB:-500}"
all_clean=0
for n in "$@"; do
  res=$(timeout 35 srun --nodes=1 --nodelist="$n" --ntasks=1 --gpus-per-node=1 \
        --cpus-per-task=1 --time=00:01:00 --partition=compute \
        bash -c "m=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | tr -dc 0-9); [ -d $DS ] && nfs=OK || nfs=MISSING; echo \"\${m:-99999} \$nfs\"" 2>/dev/null)
  mem=$(echo "$res" | awk '{print $1}'); nfs=$(echo "$res" | awk '{print $2}')
  if [ -z "$res" ]; then echo "$n: SKIP:probe_failed"; all_clean=1
  elif [ "$nfs" != "OK" ]; then echo "$n: SKIP:nfs_$nfs"; all_clean=1
  elif [ "${mem:-99999}" -ge "$GPU0_MAX_MIB" ]; then echo "$n: SKIP:gpu0_busy_${mem}MiB"; all_clean=1
  else echo "$n: CLEAN (gpu0=${mem}MiB nfs=OK)"; fi
done
exit $all_clean
