#!/usr/bin/env bash
# Launch a long-running command detached from the terminal (survives SSH / IDE disconnects).
# Usage: tools/detach.sh <run-name> <command...>
#   logs : $RUNS_ROOT/<run-name>/logs/stdout.log  (RUNS_ROOT defaults to <repo>/runs)
#   pid  : $RUNS_ROOT/<run-name>/logs/pid          monitor: tools/status.sh <run-name>
# Memory policy: when systemd-run is available the command runs in its own cgroup scope with a hard memory cap
# (DETACH_MEM_MAX, default 32G, page cache included) inside a slice capped as a whole (DETACH_TOTAL_MEM_MAX, default 64G),
# without swap. A job that outgrows its cap is OOM-killed inside its own scope instead of filling the host RAM.
set -euo pipefail
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$1"; shift
[ $# -gt 0 ] || { echo "usage: $0 <run-name> <command...>" >&2; exit 2; }
LOGDIR="${RUNS_ROOT:-$WS/runs}/$RUN/logs"; mkdir -p "$LOGDIR"
{ echo "# $(date -Is) cwd=$PWD"; echo "# cmd: $*"; } >> "$LOGDIR/command.txt"
MEM_MAX="${DETACH_MEM_MAX:-32G}"; TOT_MAX="${DETACH_TOTAL_MEM_MAX:-64G}"; SLICE="${DETACH_SLICE:-em2djobs.slice}"
WRAP=()
if systemd-run --user --scope -q --slice="$SLICE" -p MemoryMax=1G -p MemorySwapMax=0 true 2>/dev/null; then
  systemctl --user set-property --runtime "$SLICE" MemoryMax="$TOT_MAX" MemorySwapMax=0 2>/dev/null || true
  WRAP=(systemd-run --user --scope -q --slice="$SLICE" -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 -p ManagedOOMPreference=avoid)
  echo "# scope: $SLICE (total $TOT_MAX), MemoryMax=$MEM_MAX, MemorySwapMax=0" >> "$LOGDIR/command.txt"
else
  echo "WARNING: systemd-run unavailable; running WITHOUT a memory cap" >&2
fi
setsid nohup "${WRAP[@]}" "$@" > "$LOGDIR/stdout.log" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$LOGDIR/pid"
echo "started run=$RUN pid=$PID"
echo "log: $LOGDIR/stdout.log"
