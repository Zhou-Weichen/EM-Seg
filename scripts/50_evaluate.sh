#!/usr/bin/env bash
# Zero-shot evaluation (checkpoint's own affinity head + mutex watershed, MWS grid selected on validation, metrics on test)
# on SNEMI (AC3) and CREMI B, plus mouse30um when the in-house data is present; optionally the 20k-step fine-tune.
#   GPUS=0 scripts/50_evaluate.sh $RUNS_ROOT/student-L-v2a-T40k/ckpt/step_060000.pt student60k            # zero-shot
#   GPUS=0 FINETUNE=1 scripts/50_evaluate.sh <ckpt> <name>                                                 # + fine-tune per domain
# Results: $RUNS_ROOT/zeroshot-<name>-<domain>/done.json (test_summary.voi_sum etc.), test_per_slice.csv, pred/test/*.tif
set -euo pipefail
source "$(dirname "$0")/env.sh"
CK="$1"; NAME="$2"; export CUDA_VISIBLE_DEVICES="${GPUS:-0}"; OMP_NUM_THREADS=4
DOMS=("snemi" "cremi --sample B")
[ -d "${EM2DSEG_DATA_ROOT:-$EMDATA_ROOT/benchmarks}/mouse_30um" ] && DOMS+=("mouse30um")
for d in "${DOMS[@]}"; do
  tag=$(echo "$d" | tr -d ' -' | sed 's/sampleB/B/')
  "$PYTHON" -m emseg.finetune --zero-shot --ckpt "$CK" --run "zeroshot-$NAME-$tag" --domain $d
  [ "${FINETUNE:-0}" = 1 ] && "$PYTHON" -m emseg.finetune --ckpt "$CK" --run "ft-$NAME-$tag" --domain $d --seed "${SEED:-0}"
done
"$PYTHON" - "$RUNS_ROOT" "$NAME" <<'PY'
import json, sys, glob, os
root, name = sys.argv[1:]
for f in sorted(glob.glob(f"{root}/*-{name}-*/done.json")):
    s = json.load(open(f))["test_summary"]; print(f"{os.path.basename(os.path.dirname(f)):40s} VOI {s['voi_sum']:.3f} (split {s['voi_split']:.3f} merge {s['voi_merge']:.3f})  ARAND {s['arand']:.3f}  F1@50 {s['f1_50']:.3f}  PQ {s['pq50']:.3f}")
PY
