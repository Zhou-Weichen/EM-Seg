"""Pipeline driver: runs the stages in order, each as a subprocess, logs stage start/end and refreshes STATUS.md after
every stage.  usage: python -m emseg.data.pipeline all | <stage> [<stage> ...]
Stages: download -> convert -> qa -> dedup -> manifest -> report -> export_hf (see emdata/README.md)."""
import os, subprocess, sys, time
from pathlib import Path

from emseg.data.log import event

PY = sys.executable
W = str(min(48, os.cpu_count() or 8))
STAGES = {
    "download": [PY, "-m", "emseg.data.stages.download"],
    "convert": [PY, "-m", "emseg.data.stages.convert"],
    "qa": [PY, "-m", "emseg.data.stages.qa", "--workers", W],
    "dedup": [PY, "-m", "emseg.data.stages.dedup", "--workers", W],
    "manifest": [PY, "-m", "emseg.data.stages.manifest", "--nat-frac", "0.05"],
    "report": [PY, "-m", "emseg.data.stages.report", "--checksums"],
    "export_hf": [PY, "-m", "emseg.data.stages.export_hf"],
}


def run(stage):
    t0 = time.time(); event("pipeline", "stage_start", step=stage)
    rc = subprocess.run(STAGES[stage], cwd=Path(__file__).resolve().parents[2]).returncode
    event("pipeline", "stage_end", step=stage, returncode=rc, seconds=round(time.time() - t0))
    if stage != "report":
        subprocess.run([PY, "-m", "emseg.data.stages.report"], cwd=Path(__file__).resolve().parents[2])     # live STATUS.md
    return rc


def main():
    args = sys.argv[1:] or ["all"]
    if args[0] in ("-h", "--help") or any(a not in STAGES for a in args if a != "all"):
        print(f"usage: python -m emseg.data.pipeline all | <stage> [<stage> ...]   stages: {' '.join(STAGES)}"); sys.exit(0 if args[0] in ("-h", "--help") else 2)
    stages = list(STAGES) if args == ["all"] else args
    for s in stages:
        rc = run(s)
        if rc != 0 and s not in ("download",):          # download may be partial (throttled hosts): continue with what is there
            print(f"stage {s} failed (rc={rc}); stopping"); sys.exit(rc)
    print("exit=0")


if __name__ == "__main__":
    main()
