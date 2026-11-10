#!/usr/bin/env bash
# Create the two conda environments (training and data). Re-run to update them.
set -euo pipefail
cd "$(dirname "$0")/.."
conda env create -f envs/train.yml 2>/dev/null || conda env update -f envs/train.yml --prune
conda env create -f envs/data.yml  2>/dev/null || conda env update -f envs/data.yml  --prune
echo "done. activate with:  conda activate em2d-train   (training / evaluation)   or   conda activate em2d-data   (data pipeline)"
echo "then:  source scripts/env.sh"
