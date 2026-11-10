#!/usr/bin/env bash
# Student (recipe V2-A): ConvNeXt-V2-L, ImageNet-22k init, joint training on the teacher's 22 gold datasets (50 % of the
# samples) and the teacher's pseudo-labels (50 %), per-source loss normalisation, pseudo-label weight ramped over 4k steps,
# 60k steps. 2 GPUs x batch 16, ~25 h. Resumable with --resume $RUNS_ROOT/<run>/ckpt/last.pt.
#   GPUS=0,1 scripts/40_train_student.sh T40k [run-name] [extra train_inst args]
# Smaller / faster variant (recipe I3-base, ConvNeXt-V2-B initialised from a teacher checkpoint, 40k steps):
#   GPUS=0,1 scripts/40_train_student.sh T40k student-B --arch convnextv2_base --init $RUNS_ROOT/teacher-L-gt22/ckpt/step_040000.pt --iters 40000 --gt-sched 1.0:0.6,0.2
set -euo pipefail
source "$(dirname "$0")/env.sh"
TAG="${1:-T40k}"; RUN="${2:-student-L-v2a-$TAG}"; [ $# -ge 2 ] && shift 2 || shift $#
GPUS="${GPUS:-0,1}"; N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
EXCLUDE=snemi3d,tumor_spheroid,emneuron_valid_fib25-2,emneuron_mira-scn,emneuron_axonemm-sstem,emneuron_valid_fafb-1,emneuron_valid_fafb-2,emneuron_mira-fish
MAN="$EMDATA_ROOT/manifests/em_only.csv,$EMDATA_ROOT/manifests/unlab.csv"
CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" -m torch.distributed.run --nproc_per_node "$N" --master_port "${PORT:-29660}" -m emseg.train.teacher_student \
  --run "$RUN" --exp I3 --arch convnextv2_large --pretrained --pseudo-root "$EMDATA_ROOT/pseudo/$TAG" --manifest "$MAN" --exclude "$EXCLUDE" \
  --gt-sched 0.5:1.0,0.5 --gt-power 0.5 --pl-power 0.5 --src-norm --pl-weight 1.0 --pl-ramp 4000 \
  --iters 60000 --batch 16 --lr 2e-4 --workers "${WORKERS:-3}" --prefetch 2 --snap-every 5000 "$@"
echo "checkpoints: $RUNS_ROOT/$RUN/ckpt/"
