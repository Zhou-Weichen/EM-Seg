#!/usr/bin/env bash
# Teacher: ConvNeXt-V2-L (ImageNet-22k init) on the 22 audited gold datasets, affinity + contrastive loss, 40k steps.
# 2 GPUs x batch 16 (~43 GB each), ~16 h on RTX PRO 6000. Resumable: add --resume $RUNS_ROOT/<run>/ckpt/last.pt.
#   GPUS=0,1 scripts/20_train_teacher.sh [run-name] [extra train_inst args]
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN="${1:-teacher-L-gt22}"; shift || true
GPUS="${GPUS:-0,1}"; N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
# datasets excluded by the label audit (tools/label_audit.py) and the benchmark hold-outs
EXCLUDE=snemi3d,tumor_spheroid,emneuron_valid_fib25-2,emneuron_mira-scn,emneuron_axonemm-sstem,emneuron_valid_fafb-1,emneuron_valid_fafb-2,emneuron_mira-fish
CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" -m torch.distributed.run --nproc_per_node "$N" --master_port "${PORT:-29617}" -m emseg.train.teacher_student \
  --run "$RUN" --exp I2 --arch convnextv2_large --pretrained --exclude "$EXCLUDE" \
  --iters 40000 --batch 16 --lr 2e-4 --workers "${WORKERS:-8}" --snap-every 5000 "$@"
echo "checkpoints: $RUNS_ROOT/$RUN/ckpt/step_040000.pt"
