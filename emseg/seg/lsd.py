"""2D Local Shape Descriptors (Sheridan et al. 2023) — numpy re-implementation of funkelab/lsd's
`local_shape_descriptor.py` (gaussian mode) for 2D labels. No gunpowder dependency.

Per object, a Gaussian window (sigma, pixels) is intersected with the object mask and the following local
statistics are computed at every object pixel p:
    count(p)      = sum_q G(p-q) * mask(q)                              (in [0,1]; Gaussian is normalised)
    mean(p)       = sum_q G(p-q) * mask(q) * q / count(p)               (local centre of mass)
    offset(p)     = mean(p) - p
    cov(p)        = E[q q^T] - mean mean^T  (2x2)
Channel order (6): [offset_y, offset_x, var_yy, var_xx, pearson_yx, size]
Normalisation (funkelab convention): offset / sigma * 0.5 + 0.5; var / sigma^2 clipped to [0,1];
pearson * 0.5 + 0.5; size = count. Pixels outside objects are 0 in all channels.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

N_LSD = 6


def lsd2d(label: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """label (H, W) int (0 = background) -> (6, H, W) float32 descriptors in [0, 1]."""
    H, W = label.shape
    out = np.zeros((N_LSD, H, W), np.float32)
    if sigma <= 0:
        return out
    margin = int(math.ceil(truncate * sigma))
    objs = ndimage.find_objects(label)
    for idx, sl in enumerate(objs):
        if sl is None:
            continue
        oid = idx + 1
        ys, ye = max(0, sl[0].start - margin), min(H, sl[0].stop + margin)
        xs, xe = max(0, sl[1].start - margin), min(W, sl[1].stop + margin)
        m = (label[ys:ye, xs:xe] == oid).astype(np.float64)
        if not m.any():
            continue
        yy, xx = np.mgrid[0:ye - ys, 0:xe - xs].astype(np.float64)   # local coordinates (translation invariant)
        g = lambda a: ndimage.gaussian_filter(a, sigma, mode="constant", truncate=truncate)
        cnt = g(m)
        inside = m > 0
        c = cnt[inside]
        my, mx = g(m * yy)[inside] / c, g(m * xx)[inside] / c
        vyy = g(m * yy * yy)[inside] / c - my * my
        vxx = g(m * xx * xx)[inside] / c - mx * mx
        vyx = g(m * yy * xx)[inside] / c - my * mx
        vyy, vxx = np.maximum(vyy, 0.0), np.maximum(vxx, 0.0)
        pear = vyx / np.sqrt(vyy * vxx + 1e-12)
        oy, ox = my - yy[inside], mx - xx[inside]
        sub = out[:, ys:ye, xs:xe]
        sub[0][inside] = np.clip(oy / sigma * 0.5 + 0.5, 0.0, 1.0)
        sub[1][inside] = np.clip(ox / sigma * 0.5 + 0.5, 0.0, 1.0)
        sub[2][inside] = np.clip(vyy / sigma ** 2, 0.0, 1.0)
        sub[3][inside] = np.clip(vxx / sigma ** 2, 0.0, 1.0)
        sub[4][inside] = np.clip(pear * 0.5 + 0.5, 0.0, 1.0)
        sub[5][inside] = np.clip(c, 0.0, 1.0)
    return out


def augment_lsd(lsd: np.ndarray, mask: np.ndarray, k90: int, flip_x: bool, flip_y: bool) -> np.ndarray:
    """Apply the dataset augmentation (np.rot90 k times, then x-flip, then y-flip) to LSD channels including the
    vector/tensor components. mask (H, W) bool = object pixels of the *un-augmented* label (negations are applied
    only inside objects so background stays 0). Returns the augmented (6, H', W') array."""
    L = lsd.copy(); M = mask.copy()
    def neg(ch):  # negate a normalised signed channel (v*0.5+0.5 -> 1-v)
        L[ch][M] = 1.0 - L[ch][M]
    for _ in range(k90 % 4):
        # np.rot90: new[i, j] = old[j, W-1-i]; a vector (dy, dx) -> (-dx, dy); var swap; cov -> -cov
        L = np.stack([np.rot90(L[c]) for c in range(L.shape[0])]); M = np.rot90(M)
        L = L[[1, 0, 3, 2, 4, 5]].copy(); neg(0); neg(4)
    if flip_x:
        L = L[:, :, ::-1].copy(); M = M[:, ::-1]; neg(1); neg(4)
    if flip_y:
        L = L[:, ::-1, :].copy(); M = M[::-1, :]; neg(0); neg(4)
    return np.ascontiguousarray(L)
