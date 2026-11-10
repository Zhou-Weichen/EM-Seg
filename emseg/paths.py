"""Root directories, relative to the repository unless overridden by environment variables.

  EMDATA_ROOT        corpus built by emseg.data                   (default <repo>/data)
  EM2DSEG_DATA_ROOT  benchmark volumes for evaluation/fine-tuning (default <EMDATA_ROOT>/benchmarks)
  RUNS_ROOT          run directories: checkpoints, predictions, logs (default <repo>/runs)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def emdata_root() -> Path:
    return Path(os.environ.get("EMDATA_ROOT", REPO / "data"))


def data_root() -> Path:
    return Path(os.environ.get("EM2DSEG_DATA_ROOT", emdata_root() / "benchmarks"))


def mouse30um_root() -> Path:
    return data_root() / "mouse_30um"


def runs_root() -> Path:
    return Path(os.environ.get("RUNS_ROOT", REPO / "runs"))
