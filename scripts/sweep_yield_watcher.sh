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
# Usage:
#   nohup scripts/sweep_yield_watcher.sh >> <log> 2>&1 &     # run the watcher
#   scripts/sweep_yield_watcher.sh --selftest                # run offline tests
# Env: POLL_SECS (120), JOB_NAME_PREFIX (ts2ts_), ME (default $USER),
#      MAX_KILL (safety ceiling per episode, default 8).
set -uo pipefail

ME_DEFAULT="${USER:-evin_t}"
POLL_SECS="${POLL_SECS:-120}"
PREFIX="${JOB_NAME_PREFIX:-ts2ts_}"
MAX_KILL="${MAX_KILL:-8}"
NOTIFY="/fss-data/evin_t/tagseq2tagseq_artifacts/pipeline_logs/SWEEP_YIELD_NOTIFY.log"

# --- PURE DECISION FUNCTION -------------------------------------------------
# Input (stdin): lines "USER|STATE|REASON|NODES|JOBID|NAME" (squeue -o "%u|%T|%r|%D|%i|%j").
# Args: $1=me (my username), $2=prefix (my sweep job-name prefix), $3=max_kill.
# Output (stdout): job IDs to cancel, one per line, youngest-first, sized to the
#   total nodes demanded by other users' blocked-pending jobs (capped at max_kill
#   and at how many sweep jobs I actually have running). Empty output = do nothing.
# Also prints, to stderr, a human summary line prefixed "SUMMARY:".
decide_cancellations() {
  local me="$1" prefix="$2" max_kill="$3"
  awk -F'|' -v me="$me" -v prefix="$prefix" -v maxkill="$max_kill" '
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
      if (n_mine<=0) { printf("SUMMARY: %d node(s) demanded by%s but I have no %s jobs to yield\n", demand, waiters, prefix) > "/dev/stderr"; exit 0 }
      # sort my jobs youngest-first (ascending elapsed) via simple insertion sort on index
      for (i=1;i<=n_mine;i++) ord[i]=i
      for (i=2;i<=n_mine;i++){ key=ord[i]; j=i-1; while(j>=1 && mine_secs[ord[j]]>mine_secs[key]){ord[j+1]=ord[j];j--} ord[j+1]=key }
      freed=0; killed=0
      for (i=1;i<=n_mine && freed<demand && killed<maxkill;i++){
        k=ord[i]
        print mine_jid[k]                      # <-- job to cancel (stdout)
        freed += mine_nodes[k]; killed++
      }
      printf("SUMMARY: %d node(s) demanded by%s -> cancelling %d youngest sweep job(s) freeing %d node(s)\n", demand, waiters, killed, freed) > "/dev/stderr"
    }
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
  echo "---- selftest: $pass passed, $fail failed ----"
  [ "$fail" -eq 0 ]
}

# --- main loop --------------------------------------------------------------
ts() { date '+%Y-%m-%d %H:%M:%S'; }
note() { echo "[$(ts)] $*"; echo "[$(ts)] $*" >> "$NOTIFY"; }

if [ "${1:-}" = "--selftest" ]; then run_selftest; exit $?; fi

ME="${ME:-$ME_DEFAULT}"
mkdir -p "$(dirname "$NOTIFY")"
note "yield-watcher started (me=$ME, poll=${POLL_SECS}s, prefix=$PREFIX, max_kill=$MAX_KILL)"

while true; do
  snapshot="$(squeue -h -o '%u|%T|%r|%D|%i|%j|%M' 2>/dev/null)"
  # decide_cancellations wants USER|STATE|REASON|NODES|JOBID|NAME|ELAPSED (7 fields)
  to_kill="$(printf '%s\n' "$snapshot" | decide_cancellations "$ME" "$PREFIX" "$MAX_KILL" 2> >(grep '^SUMMARY:' >&2))"
  summary="$(printf '%s\n' "$snapshot" | decide_cancellations "$ME" "$PREFIX" "$MAX_KILL" 2>&1 >/dev/null | grep '^SUMMARY:')"
  if [ -n "$to_kill" ]; then
    note "$summary"
    while read -r jid; do
      [ -z "$jid" ] && continue
      note "  scancel $jid (yielding to waiter; resumable from latest.pt)"
      scancel "$jid" 2>/dev/null
    done <<< "$to_kill"
    note "  remaining sweep jobs: $(squeue -h -u "$ME" -t RUNNING -o '%i' 2>/dev/null | tr '\n' ' ')"
    sleep $(( POLL_SECS * 3 ))   # back off so the waiter can schedule
  fi
  sleep "$POLL_SECS"
done
