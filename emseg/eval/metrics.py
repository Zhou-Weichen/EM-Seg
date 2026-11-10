"""Per-slice metrics of the 2D protocol.

Primary  : voi_split, voi_merge (base-2 bits, GT background 0 ignored), voi_sum,
           arand (adapted Rand error, GT background 0 ignored)  -- via elf.evaluation (CREMI defs).
Secondary: msa (mean segmentation accuracy, IoU 0.5..0.95 step 0.05; DSB-2018 style, via elf),
           at IoU 0.5 and 0.75: precision, recall, f1, ap (= TP/(TP+FP+FN), DSB "average precision"),
           pq (panoptic quality = sum IoU(matched) / (TP + FP/2 + FN/2)), computed from a
           contingency-table matching implemented here and cross-checked against elf.matching.
Validity : dice_fg (foreground Dice), dice_boundary (Dice of 1-px boundary maps).
Extra EM metrics (em_extra_metrics; all on GT-foreground pixels only, GT 0 ignored):
           vrand_split / vrand_merge / vrand_f   (ISBI 2012 / SNEMI3D Rand scores, Arganda-Carreras 2015; vrand_f = 1 - arand)
           vinfo_split / vinfo_merge / vinfo_f   (information-theoretic scores: I(S;T)/H(S), I(S;T)/H(T), harmonic mean)
           dice_obj   (object-level Dice, Sirinukunwattana 2017 / GlaS: area-weighted, symmetric, best-overlap partner)
           dice_inst  (mean over GT objects of Dice with the best-overlapping predicted object, unweighted)
           aji        (aggregated Jaccard index, Kumar 2017)
           In these, predicted label 0 counts as a segment for Rand/VI but never as an object for dice_obj/dice_inst/aji,
           and predicted objects are clipped to GT foreground (so SNEMI membrane / masked regions do not count).

Both inputs must be 2D integer label images with 0 = background (already 2D-relabelled GT).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import ndimage
from elf.evaluation import variation_of_information, rand_index, mean_segmentation_accuracy

IOU_THRESHOLDS = (0.5, 0.75)


def _check(seg: np.ndarray, gt: np.ndarray):
    if seg.shape != gt.shape or seg.ndim != 2:
        raise ValueError(f"shape mismatch or not 2D: seg {seg.shape}, gt {gt.shape}")


def contingency(seg: np.ndarray, gt: np.ndarray, ignore_label: int = 0):
    """Contingency table following stardist/elf semantics.

    Returns (seg_ids, gt_ids, overlap, seg_sizes, gt_sizes) where the ignore label has been removed
    from the id lists and from the overlap rows/cols, but object sizes are computed on the full
    image (i.e. include pixels overlapping the other image's ignore label).
    """
    s = seg.ravel().astype(np.int64)
    g = gt.ravel().astype(np.int64)
    s_ids, s_inv = np.unique(s, return_inverse=True)
    g_ids, g_inv = np.unique(g, return_inverse=True)
    overlap = np.zeros((len(s_ids), len(g_ids)), dtype=np.int64)
    np.add.at(overlap, (s_inv, g_inv), 1)
    s_sizes = overlap.sum(axis=1)
    g_sizes = overlap.sum(axis=0)
    s_keep = s_ids != ignore_label
    g_keep = g_ids != ignore_label
    return (s_ids[s_keep], g_ids[g_keep], overlap[np.ix_(s_keep, g_keep)],
            s_sizes[s_keep], g_sizes[g_keep])


def matching_stats(seg: np.ndarray, gt: np.ndarray, thresholds: Sequence[float] = IOU_THRESHOLDS,
                   ignore_label: int = 0) -> Dict[str, float]:
    """One-to-one greedy-free matching: a pair matches iff IoU >= thr (at thr >= 0.5 this is unique)."""
    _check(seg, gt)
    s_ids, g_ids, ov, s_sz, g_sz = contingency(seg, gt, ignore_label)
    n_pred, n_true = len(s_ids), len(g_ids)
    out = {"n_pred": float(n_pred), "n_true": float(n_true)}
    if n_pred == 0 or n_true == 0:
        iou = np.zeros((n_pred, n_true))
    else:
        union = s_sz[:, None] + g_sz[None, :] - ov
        iou = np.where(union > 0, ov / np.maximum(union, 1), 0.0)
    for thr in thresholds:
        if thr < 0.5:
            raise ValueError("thresholds must be >= 0.5 for unique matching")
        matched = iou >= thr
        tp = int(matched.sum())
        sum_iou = float(iou[matched].sum())
        fp, fn = n_pred - tp, n_true - tp
        key = f"{int(round(thr * 100))}"
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[f"precision{key}"] = prec
        out[f"recall{key}"] = rec
        out[f"f1_{key}"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out[f"ap{key}"] = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
        out[f"pq{key}"] = sum_iou / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
    return out


def boundary_map(label: np.ndarray) -> np.ndarray:
    """1-pixel boundaries between different labels (4-neighbourhood)."""
    b = np.zeros(label.shape, dtype=bool)
    b[:-1, :] |= label[:-1, :] != label[1:, :]
    b[1:, :] |= label[:-1, :] != label[1:, :]
    b[:, :-1] |= label[:, :-1] != label[:, 1:]
    b[:, 1:] |= label[:, :-1] != label[:, 1:]
    return b


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else 2.0 * float((a & b).sum()) / float(denom)


def _entropy(counts: np.ndarray, n: float) -> float:
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def em_extra_metrics(seg: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Rand / VI split-merge scores, object Dice, instance Dice and AJI on GT-foreground pixels (see module doc)."""
    _check(seg, gt)
    fg = gt > 0
    out = {k: float("nan") for k in ("vrand_split", "vrand_merge", "vrand_f", "vinfo_split", "vinfo_merge", "vinfo_f", "dice_obj", "dice_inst", "aji")}
    if not fg.any():
        return out
    s = seg[fg].astype(np.int64); g = gt[fg].astype(np.int64); n = float(s.size)
    s_ids, s_inv = np.unique(s, return_inverse=True); g_ids, g_inv = np.unique(g, return_inverse=True)
    M = len(g_ids); key = s_inv.astype(np.int64) * M + g_inv
    u, c = np.unique(key, return_counts=True); si, gi = u // M, u % M; c = c.astype(np.float64)
    a = np.bincount(si, weights=c, minlength=len(s_ids)); b = np.bincount(gi, weights=c, minlength=M)
    # Rand (pixel pairs, squared counts)
    sp = (c ** 2).sum(); out["vrand_split"] = float(sp / (b ** 2).sum()); out["vrand_merge"] = float(sp / (a ** 2).sum())
    out["vrand_f"] = float(2 * out["vrand_split"] * out["vrand_merge"] / (out["vrand_split"] + out["vrand_merge"]))
    # information theoretic
    hs, ht, hst = _entropy(a, n), _entropy(b, n), _entropy(c, n); mi = hs + ht - hst
    out["vinfo_split"] = float(mi / hs) if hs > 0 else 1.0; out["vinfo_merge"] = float(mi / ht) if ht > 0 else 1.0
    out["vinfo_f"] = float(2 * out["vinfo_split"] * out["vinfo_merge"] / (out["vinfo_split"] + out["vinfo_merge"])) if (out["vinfo_split"] + out["vinfo_merge"]) > 0 else 0.0
    # object-level: predicted 0 is not an object
    obj = s_ids[si] != 0
    si_o, gi_o, c_o = si[obj], gi[obj], c[obj]
    a_o = a.copy(); a_o[s_ids == 0] = 0
    # best partner per GT object (max overlap) and per predicted object
    best_s = np.full(M, -1); best_s_ov = np.zeros(M)
    order = np.argsort(c_o)                              # ascending -> later (larger) overwrite
    best_s[gi_o[order]] = si_o[order]; best_s_ov[gi_o[order]] = c_o[order]
    best_g = np.full(len(s_ids), -1); best_g_ov = np.zeros(len(s_ids))
    best_g[si_o[order]] = gi_o[order]; best_g_ov[si_o[order]] = c_o[order]
    dice_g = np.where(best_s >= 0, 2 * best_s_ov / (b + np.where(best_s >= 0, a_o[np.maximum(best_s, 0)], 0)), 0.0)
    pv = (s_ids != 0) & (a_o > 0)
    dice_s = np.where(best_g >= 0, 2 * best_g_ov / (a_o + np.where(best_g >= 0, b[np.maximum(best_g, 0)], 0) + 1e-12), 0.0)
    out["dice_inst"] = float(dice_g.mean())
    out["dice_obj"] = float(0.5 * ((b / b.sum()) * dice_g).sum() + 0.5 * ((a_o[pv] / a_o[pv].sum()) * dice_s[pv]).sum()) if pv.any() else float(0.5 * ((b / b.sum()) * dice_g).sum())
    # AJI: per GT object the predicted object with max IoU
    iou = c_o / (b[gi_o] + a_o[si_o] - c_o)
    best_j = np.full(M, -1); order = np.argsort(iou); best_j[gi_o[order]] = si_o[order]
    inter_lookup = {}
    for x, y, v in zip(si_o.tolist(), gi_o.tolist(), c_o.tolist()):
        inter_lookup[(x, y)] = v
    C = U = 0.0; used = np.zeros(len(s_ids), bool)
    for gidx in range(M):
        j = best_j[gidx]
        if j < 0:
            U += b[gidx]; continue
        inter = inter_lookup[(j, gidx)]; C += inter; U += b[gidx] + a_o[j] - inter; used[j] = True
    U += a_o[(~used) & pv].sum()
    out["aji"] = float(C / U) if U > 0 else 0.0
    return out


def slice_metrics(seg: np.ndarray, gt: np.ndarray, ignore_gt_background: bool = True) -> Dict[str, float]:
    """All protocol metrics for one slice. ``gt`` should already be 2D-relabelled."""
    _check(seg, gt)
    seg = seg.astype(np.int64); gt = gt.astype(np.int64)
    ignore_gt = (0,) if ignore_gt_background else None
    vs, vm = variation_of_information(seg, gt, ignore_gt=ignore_gt, use_log2=True)
    arand, ri = rand_index(seg, gt, ignore_gt=ignore_gt)
    m = {"voi_split": float(vs), "voi_merge": float(vm), "voi_sum": float(vs + vm),
         "arand": float(arand), "rand_index": float(ri),
         "cremi_score": float(np.sqrt(max(vs + vm, 0.0) * max(arand, 0.0)))}
    m["msa"] = float(mean_segmentation_accuracy(seg, gt, ignore_label=0))
    m.update(matching_stats(seg, gt))
    m["betti0_err"] = abs(m["n_pred"] - m["n_true"])   # |#predicted objects - #GT objects| per slice (Betti-0 error)
    m["dice_fg"] = dice(seg > 0, gt > 0)
    m["dice_boundary"] = dice(boundary_map(seg), boundary_map(gt))
    m.update(em_extra_metrics(seg, gt))
    return m


METRIC_COLUMNS = ["voi_split", "voi_merge", "voi_sum", "arand", "rand_index", "cremi_score", "msa",
                  "precision50", "recall50", "f1_50", "ap50", "pq50",
                  "precision75", "recall75", "f1_75", "ap75", "pq75",
                  "betti0_err", "dice_fg", "dice_boundary", "n_pred", "n_true",
                  "vrand_split", "vrand_merge", "vrand_f", "vinfo_split", "vinfo_merge", "vinfo_f", "dice_obj", "dice_inst", "aji"]
