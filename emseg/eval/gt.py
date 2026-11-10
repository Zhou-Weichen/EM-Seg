"""Ground-truth handling for the 2D protocol.

- decode_rgb: mouse_30um seg PNG (RGB) -> instance id = R<<16 | G<<8 | B, 0 = background.
- relabel_2d: split every instance id into its 2D connected components and relabel
  consecutively (1..K); optional removal of fragments smaller than ``min_size`` pixels
  (set to background 0). ``min_size=0`` keeps everything. Background 0 is preserved.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import label as sk_label



def decode_rgb(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H, W, 3) -> uint32 instance ids."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected (H, W, 3) RGB array, got {rgb.shape}")
    a = rgb.astype(np.uint32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def encode_rgb(ids: np.ndarray) -> np.ndarray:
    """Inverse of decode_rgb (for tests / round trips)."""
    ids = ids.astype(np.uint32)
    return np.stack([(ids >> 16) & 255, (ids >> 8) & 255, ids & 255], axis=-1).astype(np.uint8)


def relabel_2d(label: np.ndarray, min_size: int = 0, connectivity: int = 1) -> np.ndarray:
    """Relabel a 2D label image so that every connected component gets its own id.

    Uses skimage.measure.label on the integer image: pixels are connected iff they are adjacent
    AND carry the same original id, so each original id is split into its 2D components in one pass.

    Args:
        label: (H, W) integer label image, 0 = background.
        min_size: components with fewer pixels than this are set to 0. 0 disables.
        connectivity: 1 (4-neighbourhood) or 2 (8-neighbourhood).
    Returns:
        (H, W) int64 image with consecutive ids 1..K, 0 = background.
    """
    if label.ndim != 2:
        raise ValueError(f"expected 2D label image, got {label.shape}")
    if connectivity not in (1, 2):
        raise ValueError("connectivity must be 1 or 2")
    cc = sk_label(np.asarray(label), background=0, connectivity=connectivity).astype(np.int64)
    if min_size > 0:
        sizes = np.bincount(cc.ravel())
        keep = sizes >= min_size
        keep[0] = False
        lut = np.zeros(len(sizes), dtype=np.int64)
        lut[keep] = np.arange(1, int(keep.sum()) + 1)
        cc = lut[cc]
    return cc


def relabel_consecutive(label: np.ndarray) -> np.ndarray:
    """Map arbitrary ids to 0..K (0 stays 0) without touching connectivity."""
    ids, inv = np.unique(label, return_inverse=True)
    inv = inv.reshape(label.shape)
    if ids[0] != 0:
        inv = inv + 1
    return inv.astype(np.int64)
