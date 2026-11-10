"""Superpixels from the membrane ridge prior (label-free pseudo-instances).

superpixels(img): distance-transform watershed on the ridge prior (Beier et al. 2017 procedure, elf), then region
adjacency merging of all edges whose mean ridge value is below `merge_t` (weak internal borders -> fewer false
"different object" pairs). Returns int32 labels >= 1 (full partition, no background).
"""
from __future__ import annotations

import numpy as np

from emseg.seg.prior import ridge_prior


def _merge_weak(seg: np.ndarray, bmap: np.ndarray, merge_t: float) -> np.ndarray:
    """Merge adjacent regions whose shared border has mean boundary value < merge_t (numpy RAG + connected components)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    _, s = np.unique(seg, return_inverse=True); s = s.reshape(seg.shape); n = int(s.max()) + 1
    us, vs, ws = [], [], []
    for a, b, wa, wb in ((s[:, :-1], s[:, 1:], bmap[:, :-1], bmap[:, 1:]), (s[:-1], s[1:], bmap[:-1], bmap[1:])):
        d = a != b
        u, v = np.minimum(a[d], b[d]), np.maximum(a[d], b[d]); us.append(u); vs.append(v); ws.append(0.5 * (wa[d] + wb[d]))
    u, v, w = np.concatenate(us), np.concatenate(vs), np.concatenate(ws)
    if len(u) == 0:
        return s + 1
    key = u.astype(np.int64) * n + v
    uk, inv = np.unique(key, return_inverse=True)
    mean = np.bincount(inv, weights=w) / np.bincount(inv)
    weak = uk[mean < merge_t]
    g = coo_matrix((np.ones(len(weak)), (weak // n, weak % n)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    return comp[s].astype(np.int32)


def superpixels(img: np.ndarray, threshold: float = 0.25, sigma_seeds: float = 2.0, min_size: int = 50, merge_t: float = 0.15,
                prior: np.ndarray | None = None) -> np.ndarray:
    """img (H,W) normalised float -> (H,W) int32 superpixel ids (1..n)."""
    from elf.segmentation.watershed import distance_transform_watershed
    b = ridge_prior(img) if prior is None else prior
    seg, _ = distance_transform_watershed(b.astype(np.float32), threshold=threshold, sigma_seeds=sigma_seeds, min_size=min_size)
    seg = seg.astype(np.uint32)
    if merge_t > 0:
        seg = _merge_weak(seg, b.astype(np.float32), merge_t)
    _, inv = np.unique(seg, return_inverse=True)
    return (inv.reshape(seg.shape) + 1).astype(np.int32)
