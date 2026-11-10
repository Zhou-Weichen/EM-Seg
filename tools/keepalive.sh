#!/usr/bin/env bash
# Resume a training run from ckpt/last.pt whenever its process dies before train_done (shared GPUs, OOM kills).
# Usage: tools/keepalive.sh <RUN> <command...>     (the exact launch command, without --resume; env vars included)
# Start it detached next to the run:  tools/detach.sh keepalive-<RUN> tools/keepalive.sh <RUN> env CUDA_VISIBLE_DEVICES=0,1 ... python -m torch.distributed.run ...
WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; RUN="$1"; shift; D="${RUNS_ROOT:-$WS/runs}/$RUN"
while true; do
  sleep 300
  grep -q train_done "$D/log.jsonl" 2>/dev/null && { echo "$(date -Is) $RUN done"; exit 0; }
  PID=$(cat "$D/logs/pid" 2>/dev/null); kill -0 "$PID" 2>/dev/null && continue
  [ -f "$D/ckpt/last.pt" ] || { echo "$(date -Is) $RUN dead, no checkpoint - not resuming"; exit 1; }
  echo "$(date -Is) $RUN dead (pid $PID) - resuming from last.pt"
  cd "$WS" && setsid nohup "$@" --resume "$D/ckpt/last.pt" >> "$D/logs/stdout.log" 2>&1 < /dev/null &
  echo $! > "$D/logs/pid"; sleep 120
done
