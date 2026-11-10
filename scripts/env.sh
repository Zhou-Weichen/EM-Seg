#!/usr/bin/env bash
# Common environment for every script: source it (`source scripts/env.sh`) or let the scripts source it.
# Override any variable before sourcing, e.g.  EMDATA_ROOT=/mnt/em/data RUNS_ROOT=/mnt/em/runs source scripts/env.sh
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO
export EMDATA_ROOT="${EMDATA_ROOT:-$REPO/data}"          # corpus, unlabelled pool, pseudo-labels, benchmarks
export RUNS_ROOT="${RUNS_ROOT:-$REPO/runs}"              # one directory per run: config.json, log.jsonl, ckpt/, pred/, done.json
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON="${PYTHON:-python}"                        # interpreter of the active conda env (em2d-train or em2d-data)
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"         # multi-GPU NCCL hangs at the first collective with P2P on some hosts
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$EMDATA_ROOT" "$RUNS_ROOT"
