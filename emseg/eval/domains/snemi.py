"""SNEMI3D / Kasthuri AC4-AC3 (standard protocol):

  train: AC4 z 0-79   <benchmarks>/snemi/snemi_train.h5  (volumes/raw, volumes/labels/neuron_ids; torch_em mirror of the challenge data)
  val:   AC4 z 80-99
  test:  AC3 z 0-99   <benchmarks>/snemi/ac3_image.tif + ac3_labels.tif  (EMNeuron release, labeled/Kasthuri-atum/0.tif + 0_MaskIns.tif)

Slice indices: AC4 z -> z; AC3 z -> 1000 + z, so that all indices are plain integers (see scripts/12_get_benchmarks.sh).
"""
from __future__ import annotations

import h5py
import numpy as np
import tifffile

from emseg.paths import data_root

AC3_OFFSET = 1000


class Snemi:
    name = "snemi"

    def __init__(self, root=None):
        root = (root or data_root()) / "snemi"
        self.ac4 = root / "snemi_train.h5"
        self.ac3_img = root / "ac3_image.tif"
        self.ac3_lab = root / "ac3_labels.tif"
        self._cache = {}

    def _vol(self, key):
        if key not in self._cache:
            if key.startswith("ac4"):
                f = h5py.File(self.ac4, "r")
                self._cache["ac4_img"], self._cache["ac4_lab"] = f["volumes/raw"], f["volumes/labels/neuron_ids"]
            else:
                self._cache[key] = tifffile.memmap(getattr(self, key), mode="r")
        return self._cache[key]

    def n_slices(self) -> int:
        return 100 + 100

    @staticmethod
    def _where(idx: int):
        return ("ac3", idx - AC3_OFFSET) if idx >= AC3_OFFSET else ("ac4", idx)

    def load_image(self, idx: int) -> np.ndarray:
        vol, z = self._where(idx)
        return np.array(self._vol(f"{vol}_img")[z])

    def load_label(self, idx: int) -> np.ndarray:
        vol, z = self._where(idx)
        return np.array(self._vol(f"{vol}_lab")[z]).astype(np.int64)

    @staticmethod
    def splits(scheme: str = "protocol") -> dict:
        if scheme != "protocol":
            raise ValueError("snemi only has the standard 'protocol' scheme")
        return {"train": list(range(0, 80)), "val": list(range(80, 100)),
                "test": [AC3_OFFSET + z for z in range(0, 100)]}
