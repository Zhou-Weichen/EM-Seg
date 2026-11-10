# 🧰 tools

| tool | purpose |
|---|---|
| `detach.sh <name> <cmd…>` | run a command detached from the terminal; log and pid under `$RUNS_ROOT/<name>/logs/`; memory-capped cgroup scope when `systemd-run` exists (`DETACH_MEM_MAX`, `DETACH_TOTAL_MEM_MAX`) |
| `status.sh <name> [n]` | alive / finished + log tail of a detached run |
| `keepalive.sh <run> <cmd…>` | relaunch a training run from `ckpt/last.pt` whenever its process dies before `train_done` |
| `pdl.py <url> <file> [--conn N --sha256 H --insecure]` | resumable parallel HTTP downloader used by `emseg.data.stages.download` |
| `pseudo_stats.py <pseudo_root>` | per-section statistics of a pseudo-label pass → `stats.csv` (student filter) |
| `pseudo_vis.py <pseudo_root> <out>` | panels EM / boundaries / instances / teacher foreground per dataset |
| `vis_predictions.py --run <run> --out <dir>` | panels EM / GT boundaries / GT / prediction for a run's test predictions |
| `label_audit.py <out.json> [N]` | edge-alignment audit of the cell-labelled datasets (used to pick the teacher's 22 datasets) |
