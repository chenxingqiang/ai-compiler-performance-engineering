#!/bin/bash
# GB300 inventory heartbeat-freeze watchdog.
#
# Why: a hung benchmark (e.g. an intermittent cluster-launch + CUDA-graph tcgen05
# replay on sm_103) is not reaped quickly by the harness. The default per-target
# bound is BenchmarkDefaults.measurement_timeout_seconds=1200 times the timeout
# multiplier (~3600s effective), so a single hung sub-second microbench can wedge a
# multi-hour `bench run` inventory for tens of minutes before it is killed.
#
# This watchdog reaps a hung run the way the harness's own progress heartbeat lets
# you detect it: the run-progress snapshot's embedded `timestamp` STOPS advancing
# while the kernel is stuck (a slow-but-progressing target, including a long
# training lab, keeps advancing it). When the embedded timestamp is older than
# FREEZE_LIMIT and an isolated_runner is alive, kill the runner so the parent
# records a failure and the loop advances to the next target. It NEVER kills the
# parent loop, only the hung worker.
#
# This is the precise, low-false-positive signal: it distinguishes "hung" from
# "slow", so it will not kill a legitimately long target. The trade-off is it does
# not catch a hang that occurs before the first progress update (timestamp still
# null) -- that early-setup window is covered by setup_timeout_seconds instead.
#
# Usage:
#   bash gb300-inventory-watchdog.sh &
# Env:
#   FREEZE_LIMIT     seconds of frozen progress before reaping (default 600)
#   AISP_RUNS_DIR    runs artifacts dir (default <repo>/artifacts/runs)
#   AISP_RUN_GLOB    glob of run-ids to watch (default '*')
#   AISP_WATCHDOG_LOG  log path (default /tmp/aisp-inventory-watchdog.log)
set -u

FREEZE_LIMIT=${FREEZE_LIMIT:-600}
AISP_HOME=${AISP_HOME:-$(cd "$(dirname "$0")/.." && pwd)}
RUNS=${AISP_RUNS_DIR:-"$AISP_HOME/artifacts/runs"}
RUN_GLOB=${AISP_RUN_GLOB:-'*'}
LOG=${AISP_WATCHDOG_LOG:-/tmp/aisp-inventory-watchdog.log}

echo "$(date -u +%FT%TZ) watchdog start (freeze_limit=${FREEZE_LIMIT}s runs=$RUNS glob=$RUN_GLOB)" >> "$LOG"

while true; do
  sleep 60
  # The active run-id is the one whose `bench run --targets` process is alive.
  RUNID=$(ps -eo args 2>/dev/null | grep -E "bench run --targets" | grep -v grep \
            | grep -oE "[A-Za-z0-9_.-]*${RUN_GLOB#\*}[A-Za-z0-9_.-]*" | head -1)
  [ -z "${RUNID:-}" ] && continue
  PJ="$RUNS/$RUNID/progress/run_progress.json"
  [ -f "$PJ" ] || continue
  TS=$(python3 -c "
import json, datetime
try:
    d = json.load(open('$PJ')); t = d.get('timestamp')
    print(int(datetime.datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()) if t else 0)
except Exception:
    print(0)
" 2>/dev/null)
  [ -z "${TS:-}" ] && continue
  [ "${TS:-0}" = "0" ] && continue   # null timestamp (target starting) -> skip
  NOW=$(date -u +%s); AGE=$((NOW - TS))
  if [ "$AGE" -gt "$FREEZE_LIMIT" ]; then
    STEP=$(python3 -c "import json; print(json.load(open('$PJ')).get('step', '?'))" 2>/dev/null)
    PIDS=$(pgrep -f "core.harness.isolated_runner" 2>/dev/null | tr '\n' ' ')
    if [ -n "${PIDS// /}" ]; then
      echo "$(date -u +%FT%TZ) WATCHDOG: $RUNID progress frozen ${AGE}s at step=$STEP; reaping hung runner(s): $PIDS" >> "$LOG"
      kill -9 $PIDS 2>/dev/null
      sleep 5
    fi
  fi
done
