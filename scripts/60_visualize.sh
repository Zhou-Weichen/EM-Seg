#!/usr/bin/env bash
# Figures: predictions of a run (EM | GT boundaries | GT | prediction) and random pseudo-label panels of a pass.
#   scripts/60_visualize.sh run zeroshot-student60k-snemi [out-dir]      -> <out>/snemi_zXXXX.png
#   scripts/60_visualize.sh pseudo teacher40k [out-dir]                  -> <out>/<dataset>_<i>.png + summary.csv
set -euo pipefail
source "$(dirname "$0")/env.sh"
case "$1" in
  run)    "$PYTHON" "$REPO/tools/vis_predictions.py" --run "$2" --out "${3:-$RUNS_ROOT/$2/vis}" --n "${N:-4}";;
  pseudo) "$PYTHON" "$REPO/tools/pseudo_vis.py" "$EMDATA_ROOT/pseudo/$2" "${3:-$EMDATA_ROOT/pseudo/$2/vis}" --per-dataset "${N:-2}";;
  *) echo "usage: $0 run <run-name> [out] | pseudo <tag> [out]"; exit 2;;
esac
