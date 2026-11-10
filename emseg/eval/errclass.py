"""Error-class decomposition of one 2D segmentation.

All entropies in bits, normalised by the number of GT-foreground pixels N (GT 0 = ignore; prediction clipped to GT fg).
  H_merge  = H(GT | Pred)   = VOI merge ;  H_split = H(Pred | GT) = VOI split
  M2 (chain)      : part of H_merge contributed by predicted segments that absorb >= 3 GT objects (each >= 20 px inside)
  M_pair          : part of H_merge from segments absorbing exactly 2 such GT objects
  M1 (invisible)  : sum of ΔH over merged GT-adjacent pairs whose shared border is mostly undetected (frac of crossings
                    with barrier < 0.5 is > 0.5); M1b (visible/broken) the rest.  ΔH = [(a+b)log(a+b) - a log a - b log b]/N
                    merged := both objects have the same majority predicted label (>0)
  M3 (non-adjacent) : fraction of merged object pairs (sharing a segment, both >= 20 px in it) that are not 4-adjacent in GT
  S1 (true cuts)  : part of H_split from GT objects cut into >= 2 predicted pieces of >= 20 px
  S2 (slivers)    : remaining H_split
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _xlogx(x):
    x = np.asarray(x, np.float64)
    return np.where(x > 0, x * np.log2(np.maximum(x, 1e-300)), 0.0)


def error_classes(seg: np.ndarray, gt: np.ndarray, barrier: np.ndarray = None, min_piece: int = 20) -> Dict[str, float]:
    """seg, gt: 2D int labels (gt 2D-relabelled, 0 ignored). barrier: (2,H,W) short-range disaffinities [d(-1,0), d(0,-1)]
    used to classify membrane visibility of merged adjacent pairs (optional)."""
    fg = gt > 0
    N = float(fg.sum())
    out = {}
    if N == 0:
        return out
    s = seg[fg].astype(np.int64); g = gt[fg].astype(np.int64)
    S = int(s.max()) + 1; key = s * (int(g.max()) + 1) + g
    u, c = np.unique(key, return_counts=True); si, gi = u // (int(g.max()) + 1), u % (int(g.max()) + 1); c = c.astype(np.float64)
    a = np.bincount(si, weights=c, minlength=S)                       # predicted segment sizes (within GT fg)
    b = np.bincount(gi, weights=c, minlength=int(g.max()) + 1)        # GT sizes
    # H(GT|Pred) per predicted segment, H(Pred|GT) per GT object
    hm_seg = np.bincount(si, weights=-c * np.log2(c / a[si]), minlength=S) / N
    hs_gt = np.bincount(gi, weights=-c * np.log2(c / b[gi]), minlength=len(b)) / N
    big = c >= min_piece
    n_gt_in_seg = np.bincount(si[big], minlength=S)
    n_seg_in_gt = np.bincount(gi[big], minlength=len(b))
    out["H_merge"] = float(hm_seg.sum()); out["H_split"] = float(hs_gt.sum())
    out["M2_chain"] = float(hm_seg[n_gt_in_seg >= 3].sum()); out["M_pair"] = float(hm_seg[n_gt_in_seg == 2].sum())
    out["S1_cut"] = float(hs_gt[n_seg_in_gt >= 2].sum()); out["S2_sliver"] = out["H_split"] - out["S1_cut"]
    # majority predicted label per GT object
    order = np.lexsort((c, gi)); last = np.r_[gi[order][1:] != gi[order][:-1], True]
    maj = np.zeros(len(b), np.int64); maj[gi[order][last]] = si[order][last]
    # GT adjacency (4-neighbourhood) with crossing barriers
    pairs = []; vals = []
    for (A, B_, bar) in ((gt[:-1, :], gt[1:, :], None if barrier is None else barrier[0][1:, :]),
                         (gt[:, :-1], gt[:, 1:], None if barrier is None else barrier[1][:, 1:])):
        m = (A != B_) & (A > 0) & (B_ > 0)
        lo, hi = np.minimum(A[m], B_[m]), np.maximum(A[m], B_[m])
        pairs.append(lo.astype(np.int64) * (len(b) + 1) + hi)
        vals.append(np.zeros(m.sum()) if bar is None else (bar[m] < 0.5).astype(np.float64))
    pk = np.concatenate(pairs); below = np.concatenate(vals)
    uk, inv = np.unique(pk, return_inverse=True)
    frac_below = np.bincount(inv, weights=below) / np.bincount(inv)
    pa, pb = uk // (len(b) + 1), uk % (len(b) + 1)
    merged = (maj[pa] == maj[pb]) & (maj[pa] > 0)
    dH = (_xlogx(b[pa] + b[pb]) - _xlogx(b[pa]) - _xlogx(b[pb])) / N
    if barrier is not None:
        out["M1_invisible"] = float(dH[merged & (frac_below > 0.5)].sum())
        out["M1b_visible"] = float(dH[merged & (frac_below <= 0.5)].sum())
    out["n_adjacent_pairs"] = float(len(uk)); out["adjacent_merge_rate"] = float(merged.mean()) if len(uk) else 0.0
    # non-adjacent merged pairs among big pieces
    adj = set(uk.tolist()); n_pairs = n_non = 0
    sb, gb = si[big], gi[big]
    o = np.argsort(sb, kind="stable"); sb, gb = sb[o], gb[o]
    starts = np.r_[0, np.nonzero(sb[1:] != sb[:-1])[0] + 1, len(sb)]
    for i0, i1 in zip(starts[:-1], starts[1:]):
        if i1 - i0 < 2:
            continue
        gs = np.sort(gb[i0:i1])
        for x in range(len(gs)):
            for y in range(x + 1, len(gs)):
                n_pairs += 1; n_non += (int(gs[x]) * (len(b) + 1) + int(gs[y])) not in adj
    out["merged_pairs"] = float(n_pairs); out["M3_nonadjacent_frac"] = float(n_non / n_pairs) if n_pairs else 0.0
    return out
