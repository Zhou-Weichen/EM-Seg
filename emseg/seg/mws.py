"""Mutex-watershed decoding and evaluation helpers shared by fine-tuning, zero-shot evaluation and pseudo-labelling."""
from __future__ import annotations

import numpy as np

from emseg.eval.evaluate import missing_mask
from emseg.eval.gt import relabel_2d

SCHEME = {"mouse30um": "interim", "snemi": "protocol", "cremi": "interim"}


def mws(affs, offsets, params, fg=None):
    """affs (K,H,W) disaffinities (1 = boundary), fg (H,W) foreground probability -> (H,W) int64 instance labels."""
    from torch_em.util.segmentation import mutex_watershed, size_filter
    mask = (fg > params["fg_threshold"]) if (params["fg_threshold"] > 0 and fg is not None) else None
    s = params["stride"]
    seg = mutex_watershed(affs, offsets, strides=[s, s], randomize_strides=False, mask=mask)
    if params.get("min_size", 0) > 0:
        seg = size_filter(seg.astype(np.uint64), params["min_size"], hmap=affs[:2].astype(np.float32))
    return seg.astype(np.int64)


def gt_of(dom, z, mask):
    """2D-relabelled ground truth of slice z; mask=True ignores GT where the image is missing (dilated by 8 px)."""
    g = dom.load_label(z).copy()
    if mask:
        g[missing_mask(dom.load_image(z), 8)] = 0
    return relabel_2d(g)


def voi_sum(seg, gt):
    from elf.evaluation import variation_of_information
    vs, vm = variation_of_information(seg, gt, ignore_gt=[0]); return float(vs + vm)
