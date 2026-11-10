#!/usr/bin/env bash
# Build the labelled corpus from the public releases (conda env em2d-data).
#   scripts/10_build_corpus.sh            all EM sources (no COCO), every stage; resumable
#   scripts/10_build_corpus.sh qa dedup   only the given stages
#   WITH_COCO=1 scripts/10_build_corpus.sh   also download COCO panoptic (21 GB) for the em_plus_natural manifest
# Long-running (hours to days, depends on the hosts): tools/detach.sh DATA-corpus scripts/10_build_corpus.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO"
if [ $# -gt 0 ]; then "$PYTHON" -m emseg.data.pipeline "$@"; exit; fi
if [ "${WITH_COCO:-0}" = 1 ]; then
  "$PYTHON" -m emseg.data.pipeline download convert qa dedup manifest report
else
  SRC=$("$PYTHON" -c "from emseg.data.registry import SOURCES; print(' '.join(s.id for s in SOURCES if s.default and s.id != 'coco_panoptic'))")
  "$PYTHON" -m emseg.data.stages.download $SRC
  "$PYTHON" -m emseg.data.stages.convert $SRC
  "$PYTHON" -m emseg.data.pipeline qa dedup manifest report
fi
echo "corpus at $EMDATA_ROOT  (manifests/em_only.csv, README.md, STATUS.md)"
