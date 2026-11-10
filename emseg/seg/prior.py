"""Membrane prior for mask sampling and loss weighting.

EM membranes are thin dark ridges. For a dark ridge the Hessian has one large positive eigenvalue (intensity curves
upwards across the ridge), so ridge-ness = max(0, lambda_max) at sigma in {1, 2} px, max over sigma, normalised by the
image's 99th percentile to [0, 1]. Labelled crops may instead use the dilated instance boundary (`boundary_prior`).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def ridge_prior(img: np.ndarray, sigmas=(1.0, 2.0), pct: float = 99.0) -> np.ndarray:
    """img (H,W) float (any normalisation). Returns ridge-ness in [0,1], float32."""
    x = img.astype(np.float32)
    best = None
    for s in sigmas:
        hxx = ndi.gaussian_filter(x, s, order=(2, 0))
        hyy = ndi.gaussian_filter(x, s, order=(0, 2))
        hxy = ndi.gaussian_filter(x, s, order=(1, 1))
        tr = hxx + hyy
        det_root = np.sqrt(np.maximum((hxx - hyy) ** 2 / 4 + hxy ** 2, 0))
        lmax = tr / 2 + det_root                              # largest eigenvalue, scale-normalised by s^2
        r = np.maximum(lmax, 0) * (s * s)
        best = r if best is None else np.maximum(best, r)
    hi = np.percentile(best, pct)
    if hi <= 0:
        return np.zeros_like(best, dtype=np.float32)
    return np.clip(best / hi, 0, 1).astype(np.float32)


def boundary_prior(label: np.ndarray, dilate: int = 2) -> np.ndarray:
    """label (H,W) int instances -> dilated label-transition map in {0,1} (background/label 0 transitions included)."""
    b = np.zeros(label.shape, bool)
    b[:-1] |= label[:-1] != label[1:]; b[1:] |= label[:-1] != label[1:]
    b[:, :-1] |= label[:, :-1] != label[:, 1:]; b[:, 1:] |= label[:, :-1] != label[:, 1:]
    if dilate > 0:
        b = ndi.binary_dilation(b, iterations=dilate)
    return b.astype(np.float32)
