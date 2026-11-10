#!/usr/bin/env bash
# Show whether a detached run is alive and the tail of its log.  Usage: tools/status.sh <run-name> [n-lines]
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$1"; N="${2:-20}"
LOGDIR="${RUNS_ROOT:-$WS/runs}/$RUN/logs"
[ -f "$LOGDIR/pid" ] || { echo "no pid file for $RUN"; exit 1; }
PID=$(cat "$LOGDIR/pid")
if kill -0 "$PID" 2>/dev/null; then echo "run=$RUN pid=$PID RUNNING"; else echo "run=$RUN pid=$PID FINISHED (exit code in log if the command printed it)"; fi
tail -n "$N" "$LOGDIR/stdout.log"
