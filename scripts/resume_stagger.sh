#!/bin/bash
# Resume a list of stalled runs from their latest.pt, staggered (wait for each to reach
# a training step before launching the next). Args: pairs of "config_path run_dir".
set -uo pipefail
REPO=/fss/evin_t/tagseq2tagseq; cd "$REPO"; source .venv/bin/activate 2>/dev/null || true
EXCLUDE="GPU-954,GPU-749,GPU-495,GPU-386"
while [ $# -ge 2 ]; do
  cfg="$1"; rd="$2"; shift 2
  ckpt="$REPO/runs/$rd/checkpoints/latest.pt"
  if [ ! -f "$ckpt" ]; then echo "[resume] MISSING ckpt $ckpt — skip"; continue; fi
  echo "[resume] $cfg  from runs/$rd/checkpoints/latest.pt"
  out=$(python launch_slurm.py --nodes 1 --gpus-per-node 8 --config "$cfg" \
        --time 96:00:00 --no-tail --exclude "$EXCLUDE" --resume-from "$ckpt" 2>&1)
  jid=$(echo "$out"|grep -oE 'Job ID *: [0-9]+'|grep -oE '[0-9]+')
  nrd=$(echo "$out"|grep -oE 'runs/run_[0-9_]+'|head -1)
  echo "$jid  RESUME($rd)  $nrd  $cfg" >> runs/CODE_SWEEP_RUNMAP.txt
  echo "[resume] -> job $jid $nrd"
  for i in $(seq 1 60); do
    grep -q "smart_compiled_loop.*Training step" "$nrd/logs/stderr.txt" 2>/dev/null && { echo "[resume] $jid TRAINING"; break; }
    grep -qiE "Traceback|out of range|Error:|Process group cannot be None" "$nrd/logs/stderr.txt" "$nrd"/.slurm/*_0_log.err 2>/dev/null && { echo "[resume] $jid ERROR"; grep -iE "Error|Traceback" "$nrd"/.slurm/${jid}_0_log.err 2>/dev/null|tail -3; break; }
    sleep 15
  done
done
echo "[resume] all done."
