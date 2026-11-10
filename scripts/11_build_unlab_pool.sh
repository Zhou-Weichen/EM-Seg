#!/usr/bin/env bash
# Unlabelled pool for pseudo-labelling: raw 1024x1024x4 slabs streamed anonymously from public connectome volumes
# (conda env em2d-data; cloud-volume). Resumable; one source at a time. ~106 GB and about a day for the full pool.
#   scripts/11_build_unlab_pool.sh                 the 6 built-in sources (1500 slabs each) + 36 BossDB/other sources (600 slabs each)
#   scripts/11_build_unlab_pool.sh hemibrain h01   only these sources
set -uo pipefail
source "$(dirname "$0")/env.sh"
BIG="${BIG_SLABS:-1500}"; SMALL="${SMALL_SLABS:-600}"; JSON="$REPO/configs/unlab_sources_public.json"
if [ $# -gt 0 ]; then
  for s in "$@"; do "$PYTHON" -m emseg.data.unlab "$s" --slabs "$BIG" --sources-json "$JSON"; done; exit
fi
for s in hemibrain manc malecns h01 minnie65 flywire; do
  "$PYTHON" -m emseg.data.unlab "$s" --slabs "$BIG" || echo "source $s failed, continuing"
done
for s in $("$PYTHON" -c "import json; print(' '.join(json.load(open('$JSON'))))"); do
  "$PYTHON" -m emseg.data.unlab "$s" --slabs "$SMALL" --sources-json "$JSON" || echo "source $s failed, continuing"
done
echo "pool at $EMDATA_ROOT/unlab, manifest $EMDATA_ROOT/manifests/unlab.csv"
