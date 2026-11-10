"""mouse_30um: in-house target domain (not distributed with this repository). 100 exported slices (z 0..99) at mip1..mip4
under <benchmarks>/mouse_30um; seg is RGB-encoded 3D ids.

Schemes:
  "protocol": train z 0-79 (excluding BLANK), val z 80-99, test = external block (not yet exported)
  "interim":  train z 0-69 (excluding BLANK), val z 70-79, test z 80-99  (adjacent blocks; optimistic)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from emseg.paths import mouse30um_root
from emseg.eval.gt import decode_rgb

BLANK_Z = (8,)          # EM slice is constant 128; labels still present -> always excluded
N_SLICES = 100
MIP_SIZES = {1: 2048, 2: 1024, 3: 512, 4: 256}


class Mouse30um:
    name = "mouse30um"

    def __init__(self, mip: int = 1, root=None):
        if mip not in MIP_SIZES:
            raise ValueError(f"mip must be one of {sorted(MIP_SIZES)}")
        self.mip = mip
        self.root = root or mouse30um_root()

    def n_slices(self) -> int:
        return N_SLICES

    def image_path(self, z: int):
        return self.root / "em" / f"mip{self.mip}" / f"im{z:04d}.png"

    def label_path(self, z: int):
        return self.root / "seg" / f"mip{self.mip}" / f"seg{z:04d}.png"

    def load_image(self, z: int) -> np.ndarray:
        return np.asarray(Image.open(self.image_path(z)))

    def load_label(self, z: int) -> np.ndarray:
        return decode_rgb(np.asarray(Image.open(self.label_path(z)).convert("RGB")))

    @staticmethod
    def splits(scheme: str = "interim") -> dict:
        def rng(a, b):
            return [z for z in range(a, b) if z not in BLANK_Z]
        if scheme == "protocol":
            return {"train": rng(0, 80), "val": rng(80, 100), "test": []}
        if scheme == "interim":
            return {"train": rng(0, 70), "val": rng(70, 80), "test": rng(80, 100)}
        raise ValueError(f"unknown scheme {scheme!r}")
