#!/usr/bin/env bash
# Evaluation benchmarks under $EM2DSEG_DATA_ROOT (default $EMDATA_ROOT/benchmarks):
#   snemi/snemi_train.h5         SNEMI3D AC4 (train z0-79 / val z80-99), torch_em mirror of the challenge files
#   snemi/ac3_image.tif + ac3_labels.tif   AC3 test volume with labels, from the EMNeuron release (labeled/Kasthuri-atum/0*.tif)
#   cremi/sample_B_20160501.hdf  CREMI sample B (train z0-74 / val 75-99 / test 100-124)
# Files already fetched by the corpus pipeline ($EMDATA_ROOT/raw_downloads) are reused instead of downloaded again.
set -euo pipefail
source "$(dirname "$0")/env.sh"
B="${EM2DSEG_DATA_ROOT:-$EMDATA_ROOT/benchmarks}"; RAW="$EMDATA_ROOT/raw_downloads/em"; mkdir -p "$B/snemi" "$B/cremi"
PDL="$PYTHON $REPO/tools/pdl.py"
get() { # get <dest> <url> <sha256> [pdl flags]
  [ -f "$1" ] && { echo "have $1"; return; }
  $PDL "$2" "$1" --conn 4 --sha256 "$3" ${4:-}
}
# SNEMI AC4
if [ -f "$RAW/snemi/snemi_train.h5" ]; then ln -f "$RAW/snemi/snemi_train.h5" "$B/snemi/snemi_train.h5" 2>/dev/null || cp "$RAW/snemi/snemi_train.h5" "$B/snemi/"; fi
get "$B/snemi/snemi_train.h5" https://oc.embl.de/index.php/s/43iMotlXPyAB39z/download 5b130a24d9eb23d972fede0f1a403bc05f6808b361cfa22eff23b930b12f0615
# SNEMI AC3 (EMNeuron Kasthuri-atum/0)
K="$RAW/emneuron/extracted/labeled/Kasthuri-atum"
if [ ! -f "$B/snemi/ac3_labels.tif" ]; then
  if [ ! -f "$K/0_MaskIns.tif" ]; then
    mkdir -p "$RAW/emneuron"
    [ -f "$RAW/emneuron/labeled.rar" ] || $PDL https://huggingface.co/datasets/yanchaoz/EMNeuron/resolve/main/labeled.rar "$RAW/emneuron/labeled.rar" --conn 4
    mkdir -p "$RAW/emneuron/extracted"; bsdtar -xf "$RAW/emneuron/labeled.rar" -C "$RAW/emneuron/extracted" 'labeled/Kasthuri-atum/*'
  fi
  ln -f "$K/0.tif" "$B/snemi/ac3_image.tif" 2>/dev/null || cp "$K/0.tif" "$B/snemi/ac3_image.tif"
  ln -f "$K/0_MaskIns.tif" "$B/snemi/ac3_labels.tif" 2>/dev/null || cp "$K/0_MaskIns.tif" "$B/snemi/ac3_labels.tif"
fi
# CREMI B (cremi.org serves an expired TLS certificate; integrity comes from the published sha256)
if [ -f "$RAW/cremi/sample_B_20160501.hdf" ]; then ln -f "$RAW/cremi/sample_B_20160501.hdf" "$B/cremi/" 2>/dev/null || cp "$RAW/cremi/sample_B_20160501.hdf" "$B/cremi/"; fi
get "$B/cremi/sample_B_20160501.hdf" https://cremi.org/static/data/sample_B_20160501.hdf 887e85521e00deead18c94a21ad71f278d88a5214c7edeed943130a1f4bb48b8 --insecure
echo "benchmarks at $B"; EM2DSEG_DATA_ROOT="$B" "$PYTHON" -m emseg.eval.cli splits --domain snemi; EM2DSEG_DATA_ROOT="$B" "$PYTHON" -m emseg.eval.cli splits --domain cremi --sample B
