"""Slice-wise evaluation driver: predictions + domain/split -> per-slice CSV and summary JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from emseg.eval.gt import relabel_2d
from emseg.eval.metrics import slice_metrics, METRIC_COLUMNS

PRED_EXTS = (".tif", ".tiff", ".npy", ".png")


def load_prediction(pred_dir: Path, idx: int) -> np.ndarray:
    for ext in PRED_EXTS:
        p = pred_dir / f"pred_{idx:04d}{ext}"
        if p.exists():
            if ext == ".npy":
                return np.load(p)
            if ext == ".png":
                return np.asarray(Image.open(p)).astype(np.int64)
            return tifffile.imread(p)
    raise FileNotFoundError(f"no prediction for slice {idx} in {pred_dir} (expected pred_{idx:04d}.tif/.npy/.png)")


def missing_mask(image: np.ndarray, dilate: int, min_area: int = 1024) -> np.ndarray:
    """Regions where the EM image is missing: connected components of zero-valued pixels with area >= min_area
    (stitching gaps / padding), dilated by `dilate` px. Isolated zero pixels inside tissue (e.g. SNEMI) are not missing data."""
    from scipy import ndimage
    m = image == 0
    if not m.any():
        return m
    cc, n = ndimage.label(m)
    sizes = np.bincount(cc.ravel()); keep = sizes >= min_area; keep[0] = False
    m = keep[cc]
    return ndimage.binary_dilation(m, iterations=dilate) if (dilate > 0 and m.any()) else m


def evaluate_slices(gt_fn: Callable[[int], np.ndarray], pred_fn: Callable[[int], np.ndarray],
                    indices: Iterable[int], min_size: int = 0, relabel_pred: bool = True,
                    progress: Optional[Callable[[str], None]] = None,
                    image_fn: Optional[Callable[[int], np.ndarray]] = None, mask_missing: int = 0) -> pd.DataFrame:
    """mask_missing > 0: GT is set to 0 (ignored by VOI/ARAND/matching) where the image is missing, dilated by that many px."""
    rows = []
    for idx in indices:
        gt_raw = gt_fn(idx)
        if mask_missing > 0 and image_fn is not None:
            gt_raw = gt_raw.copy(); gt_raw[missing_mask(image_fn(idx), mask_missing)] = 0
        gt = relabel_2d(gt_raw, min_size=min_size)
        pred = pred_fn(idx)
        if relabel_pred:
            pred = relabel_2d(pred, min_size=0)   # predictions are never size-filtered by the protocol
        m = slice_metrics(pred, gt)
        m["slice"] = idx
        rows.append(m)
        if progress:
            progress(f"slice {idx}: voi_sum={m['voi_sum']:.3f} arand={m['arand']:.3f} msa={m['msa']:.3f}")
    df = pd.DataFrame(rows)
    return df[["slice"] + METRIC_COLUMNS]


def summarize(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
    for c in METRIC_COLUMNS:
        out[c] = {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1)) if len(df) > 1 else 0.0,
                  "n": int(len(df))}
    return out


def write_outputs(df: pd.DataFrame, out_dir: Path, meta: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_slice.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"meta": meta, "summary": summarize(df)}, f, indent=2)
