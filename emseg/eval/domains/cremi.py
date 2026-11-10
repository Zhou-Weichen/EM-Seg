"""CREMI A/B/C (125 x 1250 x 1250, volumes/labels/neuron_ids), <benchmarks>/cremi/sample_<X>_20160501.hdf. One domain per sample.

Interim scheme: train z 0-74, val z 75-99, test z 100-124.
"""
from __future__ import annotations

import h5py
import numpy as np

from emseg.paths import data_root

FILES = {"A": "sample_A_20160501.hdf", "B": "sample_B_20160501.hdf", "C": "sample_C_20160501.hdf"}


class Cremi:
    name = "cremi"

    def __init__(self, sample: str = "A", root=None):
        if sample not in FILES:
            raise ValueError("sample must be A, B or C")
        self.sample = sample
        self.path = (root or data_root()) / "cremi" / FILES[sample]
        self._f = None

    def _file(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
        return self._f

    def n_slices(self) -> int:
        return int(self._file()["volumes/raw"].shape[0])

    def load_image(self, z: int) -> np.ndarray:
        return self._file()["volumes/raw"][z]

    def load_label(self, z: int) -> np.ndarray:
        return self._file()["volumes/labels/neuron_ids"][z].astype(np.int64)

    # Entirely blank (all-zero) sections of sample C (z=14, z=74) are excluded from training.
    BLANK_TRAIN = {"A": (), "B": (), "C": (14, 74)}

    def splits(self, scheme: str = "interim") -> dict:
        if scheme != "interim":
            raise ValueError("cremi only has the 'interim' scheme")
        blank = self.BLANK_TRAIN[self.sample]
        return {"train": [z for z in range(0, 75) if z not in blank], "val": list(range(75, 100)), "test": list(range(100, 125))}
