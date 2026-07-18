#!/bin/bash
# LR-sweep courtesy watcher (node-demand-aware).
#
# Frees nodes for coworkers: when OTHER users have jobs PENDING and blocked on node
# availability (REASON Resources/Priority), cancels our lowest-priority sweep jobs
# (name-prefixed, default ts2ts_) YOUNGEST-first, freeing as many nodes as the
# waiting jobs collectively request (capped at how many we're running). All sweep
# runs checkpoint latest.pt every 250 steps → resumable via --resume-from.
#
# The user (evin_t) explicitly authorized auto-kill: "do not hesitate to kill if
# somebody's waiting in line." Personal, non-urgent project.
#
# The decision logic is factored into `decide_cancellations` (a PURE function:
# reads `squeue`-style text on stdin, prints the job IDs to cancel) so it can be
# unit-tested offline with no SLURM calls. The main loop only adds polling + the
# real `scancel` + notifications.
#
# ALSO auto-RESUMES yielded jobs: when this watcher kills a run for a coworker, it
# records (run_dir, config) to a ledger. On later polls, if a node has been idle
# for >= IDLE_RELAUNCH_MIN minutes (default 30) AND nobody is waiting, it relaunches
# a yielded job with --resume-from <run_dir>/checkpoints/latest.pt (exact Muon+AdamW
# + dataset-position resume), or fresh if no checkpoint. The 30-min idle gate avoids
# racing a coworker who is rapidly cycling failing/short debug jobs on a node.
#
# Usage:
#   nohup scripts/sweep_yield_watcher.sh >> <log> 2>&1 &     # run the watcher
#   scripts/sweep_yield_watcher.sh --selftest                # run offline tests
# Env: POLL_SECS (120), JOB_NAME_PREFIX (ts2ts_), ME (default $USER),
#      MAX_KILL (8), IDLE_RELAUNCH_MIN (30), AUTO_RELAUNCH (1; set 0 to disable).
set -uo pipefail

ME_DEFAULT="${USER:-evin_t}"
POLL_SECS="${POLL_SECS:-120}"
PREFIX="${JOB_NAME_PREFIX:-ts2ts_}"
MAX_KILL="${MAX_KILL:-8}"
IDLE_RELAUNCH_MIN="${IDLE_RELAUNCH_MIN:-30}"
AUTO_RELAUNCH="${AUTO_RELAUNCH:-1}"
REPO="/fss/evin_t/tagseq2tagseq"
NOTIFY="/fss-data/evin_t/tagseq2tagseq_artifacts/pipeline_logs/SWEEP_YIELD_NOTIFY.log"
STATE_DIR="/fss-data/evin_t/tagseq2tagseq_artifacts/pipeline_logs/watcher_state"
YIELD_LEDGER="$STATE_DIR/yielded_jobs.tsv"        # run_dir<TAB>config<TAB>killed_epoch<TAB>status
IDLE_SINCE="$STATE_DIR/node_idle_since.tsv"        # node<TAB>first_idle_epoch

# --- PURE DECISION FUNCTION -------------------------------------------------
# Input (stdin): lines "USER|STATE|REASON|NODES|JOBID|NAME" (squeue -o "%u|%T|%r|%D|%i|%j").
# Args: $1=me (my username), $2=prefix (my sweep job-name prefix), $3=max_kill.
# Output (stdout): job IDs to cancel, one per line, youngest-first, sized to the
#   total nodes demanded by other users' blocked-pending jobs (capped at max_kill
#   and at how many sweep jobs I actually have running). Empty output = do nothing.
# Also prints, to stderr, a human summary line prefixed "SUMMARY:".
decide_cancellations() {
  local me="$1" prefix="$2" max_kill="$3"
  local idle="${4:-0}"
  awk -F'|' -v me="$me" -v prefix="$prefix" -v maxkill="$max_kill" -v idle="$idle" '
    function elapsed_to_secs(e,   n,a) {
      n = split(e, a, /[-:]/)
      if (n==4) return ((a[1]*24+a[2])*60+a[3])*60+a[4]
      else if (n==3) return ((a[1]*60)+a[2])*60+a[3]
      else if (n==2) return (a[1]*60)+a[2]
      else return a[1]+0
    }
    {
      user=$1; state=$2; reason=$3; nodes=$4+0; jobid=$5; name=$6; elapsed=$7
      # Other users blocked in the queue waiting for nodes:
      if (user!=me && state=="PENDING" && (reason=="Resources"||reason=="Priority")) {
        demand += nodes
        waiters = waiters " " jobid "(" nodes "n:" reason ")"
      }
      # My running sweep jobs (name starts with prefix): collect for ranking.
      if (user==me && state=="RUNNING" && index(name,prefix)==1) {
        n_mine++
        mine_secs[n_mine]=elapsed_to_secs(elapsed)
        mine_jid[n_mine]=jobid
        mine_nodes[n_mine]=nodes
      }
    }
    END {
      if (demand<=0) { print "SUMMARY: no blocked waiters; no action" > "/dev/stderr"; exit 0 }
      # A pending job can be blocked while idle nodes exist (scheduler just has not
      # placed it yet, or reservation/backfill timing). Do NOT kill for demand the
      # cluster can already satisfy: net_demand = demand - idle_nodes_available.
      net = demand - idle
      if (net<=0) {
        printf("SUMMARY: %d node(s) demanded by%s but %d idle node(s) available (net %d) -> no action (scheduler will place them)\n", demand, waiters, idle, net) > "/dev/stderr"
        exit 0
      }
      if (n_mine<=0) { printf("SUMMARY: net %d node(s) demanded by%s (demand %d - idle %d) but I have no %s jobs to yield\n", net, waiters, demand, idle, prefix) > "/dev/stderr"; exit 0 }
      # sort my jobs youngest-first (ascending elapsed) via simple insertion sort on index
      for (i=1;i<=n_mine;i++) ord[i]=i
      for (i=2;i<=n_mine;i++){ key=ord[i]; j=i-1; while(j>=1 && mine_secs[ord[j]]>mine_secs[key]){ord[j+1]=ord[j];j--} ord[j+1]=key }
      freed=0; killed=0
      for (i=1;i<=n_mine && freed<net && killed<maxkill;i++){
        k=ord[i]
        print mine_jid[k]                      # <-- job to cancel (stdout)
        freed += mine_nodes[k]; killed++
      }
      printf("SUMMARY: net %d node(s) needed (demand %d - idle %d) by%s -> cancelling %d youngest sweep job(s) freeing %d node(s)\n", net, demand, idle, waiters, killed, freed) > "/dev/stderr"
    }
  '
}

# --- PURE: which nodes have been idle long enough to safely relaunch on? -----
# Args: $1=now_epoch, $2=min_idle_secs.
# stdin: current idle-since ledger lines "node<TAB>first_idle_epoch".
# stdout: nodes whose (now - first_idle_epoch) >= min_idle_secs, one per line.
qualified_idle_nodes() {
  local now="$1" min_secs="$2"
  awk -F'\t' -v now="$now" -v m="$min_secs" '
    NF>=2 && (now - $2) >= m { print $1 }'
}

# --- PURE: update idle-since ledger given the current idle-node set ----------
# Args: $1=now_epoch. stdin line 1: space-separated CURRENT idle nodes.
#       remaining stdin lines: existing ledger "node<TAB>epoch".
# stdout: new ledger — nodes still idle keep their original first_idle_epoch;
#         newly-idle nodes get `now`; no-longer-idle nodes are dropped.
update_idle_ledger() {
  local now="$1"
  awk -F'\t' -v now="$now" '
    NR==1 { n=split($0, cur, /[ ]+/); for(i=1;i<=n;i++) if(cur[i]!="") isidle[cur[i]]=1; next }
    NF>=2 { since[$1]=$2 }
    END { for (nd in isidle) print nd "\t" (since[nd] ? since[nd] : now) }
  '
}

# --- offline self-test ------------------------------------------------------
run_selftest() {
  local pass=0 fail=0
  _check() { # name, expected (space-joined ids), actual (newline ids)
    local name="$1" exp="$2" act; act="$(echo "$3" | tr '\n' ' ' | sed 's/ *$//;s/^ *//')"
    if [ "$act" = "$exp" ]; then echo "PASS: $name"; pass=$((pass+1)); else echo "FAIL: $name  expected=[$exp] got=[$act]"; fail=$((fail+1)); fi
  }
  # Scenario A: nobody waiting -> no kills
  local A; A="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|23:04' \
    'dawei_gui|RUNNING|None|2|45039|mic_b|17:11' \
    | decide_cancellations evin_t ts2ts_ 8 2>/dev/null)"
  _check "A nobody-waiting" "" "$A"
  # Scenario B: coworker pending needs 1 node -> kill my 1 youngest sweep job
  local B; B="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|23:04' \
    'evin_t|RUNNING|None|1|45052|ts2ts_b|0:28' \
    'dawei_gui|PENDING|Resources|1|45099|mic_wait|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 2>/dev/null)"
  _check "B need1-kill-youngest" "45052" "$B"
  # Scenario C: coworker pending needs 3 nodes -> kill youngest until >=3 freed (each mine=1 node)
  local C; C="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|30:00' \
    'evin_t|RUNNING|None|1|45052|ts2ts_b|0:28' \
    'evin_t|RUNNING|None|1|45055|ts2ts_c|5:00' \
    'evin_t|RUNNING|None|1|45056|ts2ts_d|10:00' \
    'dawei_gui|PENDING|Priority|3|45099|mic_wait|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 2>/dev/null)"
  _check "C need3-kill-3-youngest" "45052 45055 45056" "$C"
  # Scenario D: waiter demand exceeds my jobs -> kill all mine (don't over-promise)
  local D; D="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|30:00' \
    'evin_t|RUNNING|None|1|45052|ts2ts_b|0:28' \
    'dawei_gui|PENDING|Resources|6|45099|mic_wait|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 2>/dev/null)"
  _check "D demand>mine-kill-all" "45052 45048" "$D"
  # Scenario E: my OWN pending job must NOT trigger kills (self-discrimination)
  local E; E="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|23:04' \
    'evin_t|PENDING|Resources|1|45060|ts2ts_pending|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 2>/dev/null)"
  _check "E own-pending-ignored" "" "$E"
  # Scenario F: pending but NOT resource-blocked (e.g. Dependency) -> ignore
  local F; F="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|23:04' \
    'dawei_gui|PENDING|Dependency|2|45099|mic_dep|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 2>/dev/null)"
  _check "F non-resource-pending-ignored" "" "$F"
  # Scenario G: non-sweep job of mine (different prefix) must NOT be cancelled
  local G; G="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|30:00' \
    'evin_t|RUNNING|None|1|45070|important_other|0:10' \
    'dawei_gui|PENDING|Resources|1|45099|mic_wait|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 2>/dev/null)"
  _check "G other-prefix-protected" "45048" "$G"
  # Scenario H: waiter needs 2 but 2 idle nodes exist -> net 0 -> NO kill
  local H; H="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|30:00' \
    'evin_t|RUNNING|None|1|45052|ts2ts_b|0:28' \
    'dawei_gui|PENDING|Resources|2|45099|mic_wait|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 2 2>/dev/null)"
  _check "H idle-covers-demand-no-kill" "" "$H"
  # Scenario I: waiter needs 3, 1 idle -> net 2 -> kill 2 youngest
  local I; I="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|30:00' \
    'evin_t|RUNNING|None|1|45052|ts2ts_b|0:28' \
    'evin_t|RUNNING|None|1|45055|ts2ts_c|5:00' \
    'dawei_gui|PENDING|Resources|3|45099|mic_wait|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 1 2>/dev/null)"
  _check "I partial-idle-net2" "45052 45055" "$I"
  # Scenario J: waiter is a self-dependency (their own job chain) -> Dependency reason -> ignore
  local J; J="$(printf '%s\n' \
    'evin_t|RUNNING|None|1|45048|ts2ts_a|30:00' \
    'dawei_gui|PENDING|Dependency|4|45099|mic_afterok|0:00' \
    | decide_cancellations evin_t ts2ts_ 8 0 2>/dev/null)"
  _check "J dependency-blocked-ignored" "" "$J"
  # Scenario K: idle-duration gate. now=1000, min=1800s(30m). Node idle 40m qualifies, 10m doesn't.
  local TAB; TAB="$(printf '\t')"
  local K; K="$(printf "GPU-A${TAB}%s\nGPU-B${TAB}%s\n" "$((1000-2400))" "$((1000-600))" \
    | qualified_idle_nodes 1000 1800)"
  _check "K idle>=30min-only" "GPU-A" "$K"
  # Scenario L: idle-ledger update. current idle = "GPU-A GPU-C"; GPU-A was idle since 500,
  #   GPU-B (now gone) dropped, GPU-C newly idle -> gets `now`=1000.
  local L; L="$(printf "GPU-A GPU-C\nGPU-A${TAB}500\nGPU-B${TAB}400\n" | update_idle_ledger 1000 | sort)"
  local Lexp="GPU-A${TAB}500
GPU-C${TAB}1000"
  if [ "$L" = "$Lexp" ]; then echo "PASS: L idle-ledger-carryover"; pass=$((pass+1)); else echo "FAIL: L idle-ledger  got=[$L]"; fail=$((fail+1)); fi
  echo "---- selftest: $pass passed, $fail failed ----"
  [ "$fail" -eq 0 ]
}

# --- main loop --------------------------------------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "[$(ts)] $*"; echo "[$(ts)] $*" >> "$NOTIFY"; }

# Map a running job id -> "run_dir<TAB>config". Job name is ts2ts_<rundirbase>;
# config comes from that run's reproducibility/run_invocation.json (argv --config).
job_to_rundir_config() {
  local jid="$1"
  local name; name="$(squeue -h -j "$jid" -o '%j' 2>/dev/null)"
  [ -z "$name" ] && return 1
  local rundir="$REPO/runs/${name#${PREFIX}}"
  [ -d "$rundir" ] || return 1
  local inv; inv="$(find "$rundir/reproducibility" -name run_invocation.json 2>/dev/null | head -1)"
  local cfg=""
  [ -n "$inv" ] && cfg="$(python3 -c "import json,sys; a=json.load(open('$inv'))['argv']; print(a[a.index('--config')+1] if '--config' in a else '')" 2>/dev/null)"
  printf '%s\t%s\n' "$rundir" "$cfg"
}

# Relaunch one yielded job on a given clean node, resuming if a checkpoint exists.
relaunch_yielded() {
  local rundir="$1" cfg="$2" node="$3"
  [ -z "$cfg" ] && { note "  relaunch SKIP ($rundir): no config recorded"; return 1; }
  local ck="$rundir/checkpoints/latest.pt"
  local resume_args=""
  if [ -f "$ck" ]; then resume_args="--resume-from $ck"; fi
  local tag; tag="$(basename "$cfg" .yaml)"
  note "  RELAUNCH $tag on $node $([ -n "$resume_args" ] && echo "(resume from $(basename "$rundir"))" || echo "(fresh — no ckpt)")"
  TS2TS_SHARED_COMPILE_CACHE="/tmp/ts2ts_relaunch_$(basename "$rundir")" \
    "$REPO/.venv/bin/python" "$REPO/launch_slurm.py" --nodes 1 --gpus-per-node 8 \
    --nodelist "$node" --config "$cfg" --time 96:00:00 --no-tail $resume_args \
    >> "$STATE_DIR/relaunch.log" 2>&1
}

if [ "${1:-}" = "--selftest" ]; then run_selftest; exit $?; fi

ME="${ME:-$ME_DEFAULT}"
mkdir -p "$STATE_DIR"
touch "$YIELD_LEDGER" "$IDLE_SINCE"
note "yield-watcher started (me=$ME, poll=${POLL_SECS}s, max_kill=$MAX_KILL, idle_relaunch_min=$IDLE_RELAUNCH_MIN, auto_relaunch=$AUTO_RELAUNCH)"

prev_had_net=0   # was there unmet net demand on the PREVIOUS poll? (persistence gate)
while true; do
  now="$(date +%s)"
  snapshot="$(squeue -h -o '%u|%T|%r|%D|%i|%j|%M' 2>/dev/null)"
  idle_nodes="$(sinfo -h -t idle -o '%n' -p compute 2>/dev/null | grep -vE 'GPU-954|GPU-749' | tr '\n' ' ')"
  idle="$(echo "$idle_nodes" | wc -w)"

  # ---- YIELD (kill) logic ----
  to_kill="$(printf '%s\n' "$snapshot" | decide_cancellations "$ME" "$PREFIX" "$MAX_KILL" "$idle" 2>/dev/null)"
  summary="$(printf '%s\n' "$snapshot" | decide_cancellations "$ME" "$PREFIX" "$MAX_KILL" "$idle" 2>&1 >/dev/null | grep '^SUMMARY:')"
  someone_waiting=0; [ -n "$to_kill" ] && someone_waiting=1
  if [ -n "$to_kill" ]; then
    if [ "$prev_had_net" -eq 1 ]; then
      note "$summary"
      while read -r jid; do
        [ -z "$jid" ] && continue
        # Record run_dir + config to the ledger BEFORE scancel (squeue still knows the job).
        rc="$(job_to_rundir_config "$jid" 2>/dev/null)"
        note "  scancel $jid (yielding; will auto-resume when a node is idle >= ${IDLE_RELAUNCH_MIN}min)"
        scancel "$jid" 2>/dev/null
        [ -n "$rc" ] && printf '%s\t%s\tyielded\n' "$rc" "$now" >> "$YIELD_LEDGER"
      done <<< "$to_kill"
      note "  remaining sweep jobs: $(squeue -h -u "$ME" -t RUNNING -o '%i' 2>/dev/null | tr '\n' ' ')"
      prev_had_net=0
      sleep $(( POLL_SECS * 3 ))
      continue
    else
      note "unmet demand detected (idle=$idle); confirming persistence before yielding. [$summary]"
      prev_had_net=1
    fi
  else
    prev_had_net=0
  fi

  # ---- RELAUNCH logic: only when NOBODY is waiting, on nodes idle >= threshold ----
  # Update the idle-duration ledger (carry forward first-idle timestamps).
  new_ledger="$(printf '%s\n' "$idle_nodes"; cat "$IDLE_SINCE" 2>/dev/null)"
  printf '%s\n' "$new_ledger" | update_idle_ledger "$now" > "$IDLE_SINCE.tmp" && mv "$IDLE_SINCE.tmp" "$IDLE_SINCE"

  if [ "$AUTO_RELAUNCH" = "1" ] && [ "$someone_waiting" -eq 0 ]; then
    min_secs=$(( IDLE_RELAUNCH_MIN * 60 ))
    ready_nodes="$(qualified_idle_nodes "$now" "$min_secs" < "$IDLE_SINCE" | tr '\n' ' ')"
    pending_yield="$(awk -F'\t' '$4=="yielded"' "$YIELD_LEDGER" 2>/dev/null)"
    if [ -n "$ready_nodes" ] && [ -n "$pending_yield" ]; then
      for node in $ready_nodes; do
        # take the oldest un-relaunched yielded job
        line="$(awk -F'\t' '$4=="yielded"{print; exit}' "$YIELD_LEDGER" 2>/dev/null)"
        [ -z "$line" ] && break
        rundir="$(echo "$line" | cut -f1)"; cfg="$(echo "$line" | cut -f2)"
        # preflight the node (GPU0 empty + NFS) before using it
        if "$REPO/scripts/preflight_node.sh" "$node" >/dev/null 2>&1; then
          if relaunch_yielded "$rundir" "$cfg" "$node"; then
            # mark this ledger line relaunched (first match only)
            awk -F'\t' -v rd="$rundir" 'BEGIN{done=0} $1==rd && $4=="yielded" && !done{$4="relaunched"; done=1} {print $1"\t"$2"\t"$3"\t"$4}' OFS='\t' "$YIELD_LEDGER" > "$YIELD_LEDGER.tmp" && mv "$YIELD_LEDGER.tmp" "$YIELD_LEDGER"
            # reset that node's idle clock so we don't reuse it next poll
            grep -v "^${node}	" "$IDLE_SINCE" > "$IDLE_SINCE.tmp" 2>/dev/null; mv "$IDLE_SINCE.tmp" "$IDLE_SINCE"
          fi
        else
          note "  relaunch SKIP: $node failed preflight (GPU0 busy or NFS)"
        fi
      done
    fi
  fi

  sleep "$POLL_SECS"
done
