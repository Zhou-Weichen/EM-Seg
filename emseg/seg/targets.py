"""Target definitions and post-processing for the three U-Net baselines.

METHODS:
  fg_cc        : 1 channel  [foreground]                     -> threshold + connected components
  boundary_ws  : 2 channels [foreground, boundary]           -> torch_em.watershed_from_components
  affinity_mws : 1 + K ch.  [foreground, disaffinities(K)]   -> torch_em.mutex_watershed (fg mask)
Disaffinity convention (torch_em): 1 = boundary/repulsive, 0 = same object.
"""
from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
from skimage.measure import label as sk_label
from torch_em.transform.label import AffinityTransform, BoundaryTransform
from torch_em.util.segmentation import mutex_watershed, watershed_from_components, size_filter

OFFSETS_2D: List[List[int]] = [[-1, 0], [0, -1], [-3, 0], [0, -3], [-9, 0], [0, -9]]


class Target:
    def __init__(self, method: str, scale: int = 1, offsets=None):
        """scale: input up-sampling factor. Affinity offsets are multiplied by `scale` so that
        sub-sampling the prediction back to the original grid ([::scale, ::scale]) yields the standard offsets.
        offsets: affinity offsets override; default OFFSETS_2D."""
        if method not in METHOD_CHANNELS:
            raise ValueError(f"unknown method {method}")
        self.method = method
        self.scale = int(scale)
        base = OFFSETS_2D if offsets is None else [list(o) for o in offsets]
        self.n_out = METHOD_CHANNELS[method] if offsets is None else 1 + len(base)
        self.offsets = [[o * self.scale for o in off] for off in base]
        if method == "boundary_ws":
            self._bt = BoundaryTransform(mode="thick", add_binary_target=True, ndim=2)
        elif method == "affinity_mws":
            self._at = AffinityTransform(self.offsets, ignore_label=None, add_binary_target=True, add_mask=True)

    def __call__(self, label: np.ndarray):
        """label (H, W) int -> (target (C, H, W) float32, mask (C, H, W) float32)."""
        if self.method == "fg_cc":
            t = (label > 0)[None].astype(np.float32)
            return t, np.ones_like(t)
        if self.method == "boundary_ws":
            t = self._bt(label).astype(np.float32)
            return t, np.ones_like(t)
        out = self._at(label)                        # [fg, affs(K), mask_fg, mask_affs(K)]
        c = out.shape[0] // 2
        return out[:c].astype(np.float32), out[c:].astype(np.float32)


METHOD_CHANNELS = {"fg_cc": 1, "boundary_ws": 2, "affinity_mws": 1 + len(OFFSETS_2D)}

# Fixed post-processing grids (searched on the validation split only).
GRIDS: Dict[str, List[dict]] = {
    "fg_cc": [{"threshold": t} for t in (0.3, 0.5, 0.7)],
    "boundary_ws": [{"threshold1": t1, "threshold2": t2, "min_size": m}
                    for t1, t2, m in itertools.product((0.3, 0.5, 0.7), (0.5,), (0, 20))],
    "affinity_mws": [{"stride": s, "fg_threshold": f, "min_size": m}
                     for s, f, m in itertools.product((1, 2), (0.0, 0.5), (0, 25, 100))],
}
DEFAULTS = {"fg_cc": {"threshold": 0.5}, "boundary_ws": {"threshold1": 0.5, "threshold2": 0.5, "min_size": 0},
            "affinity_mws": {"stride": 2, "fg_threshold": 0.5, "min_size": 25}}


def postprocess(method: str, pred: np.ndarray, params: dict) -> np.ndarray:
    """pred: (C, H, W) probabilities (after sigmoid). Returns (H, W) instance labels."""
    if method == "fg_cc":
        return sk_label(pred[0] > params["threshold"], background=0, connectivity=1).astype(np.int64)
    if method == "boundary_ws":
        fg, bd = pred[0], pred[1]
        seg = watershed_from_components(bd, fg, min_size=params["min_size"],
                                        threshold1=params["threshold1"], threshold2=params["threshold2"])
        return seg.astype(np.int64)
    if method == "affinity_mws":
        fg, affs = pred[0], pred[1:].copy()
        mask = (fg > params["fg_threshold"]) if params["fg_threshold"] > 0 else None
        s = params["stride"]
        seg = mutex_watershed(affs, OFFSETS_2D, strides=[s, s], randomize_strides=False, mask=mask)
        if params.get("min_size", 0) > 0:
            # merge fragments smaller than min_size into neighbours via watershed on the short-range disaffinities
            seg = size_filter(seg.astype(np.uint64), params["min_size"], hmap=pred[1:3].astype(np.float32))
        return seg.astype(np.int64)
    raise ValueError(method)
