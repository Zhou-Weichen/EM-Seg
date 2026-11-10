#!/usr/bin/env bash
# Teacher pseudo-labels for the unlabelled pool: one shard per GPU, then per-row statistics for the student's filter.
# Pool = the corpus' non-cell neural datasets + the raw connectome slabs (manifests/unlab.csv), at most 6000 rows per dataset.
# ~130k rows, 35 GB, 4-5 h per shard on 3 GPUs.
#   GPUS=0,1,2 scripts/30_pseudolabel.sh $RUNS_ROOT/teacher/ckpt/step_040000.pt teacher40k
set -euo pipefail
source "$(dirname "$0")/env.sh"
CK="$1"; TAG="${2:-teacher40k}"; OUT="$EMDATA_ROOT/pseudo/$TAG"
GPUS="${GPUS:-0}"; IFS=',' read -ra G <<< "$GPUS"; N=${#G[@]}; mkdir -p "$OUT"
UNLAB=$("$PYTHON" -c "import pandas as pd; print(','.join(sorted(pd.read_csv('$EMDATA_ROOT/manifests/unlab.csv', low_memory=False).dataset.unique())))")
DS="mitoemv2_pyra,mitoemv2_mossy,mitoem_H,mitoem_R,lucchi,cem_fly_brain,$UNLAB"
MAN="$EMDATA_ROOT/manifests/em_only.csv,$EMDATA_ROOT/manifests/unlab.csv"
PIDS=()
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} "$PYTHON" -m emseg.pseudolabel --teacher "$CK" --out "$OUT" --manifest "$MAN" --datasets "$DS" \
    --per-dataset 6000 --shard "$i" --n-shards "$N" --workers "${WORKERS:-8}" > "$OUT.shard$i.log" 2>&1 &
  PIDS+=($!); echo "shard $i on GPU ${G[$i]} (pid $!, log $OUT.shard$i.log)"
done
wait "${PIDS[@]}"
"$PYTHON" "$REPO/tools/pseudo_stats.py" "$OUT" --workers "${WORKERS:-16}"
echo "pseudo-labels: $OUT (rows_*.csv, stats.csv, config.json)"
