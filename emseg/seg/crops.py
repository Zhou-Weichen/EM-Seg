"""In-memory random-crop dataset over a list of slices of a domain."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from emseg.seg.targets import Target
from emseg.seg.lsd import lsd2d, augment_lsd, N_LSD
from emseg.seg.unet import upsample_image, upsample_label


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-6)


class SliceCropDataset(Dataset):
    """Returns (img[1,H,W], target[C(+6),H,W], mask[C(+6),H,W], label[H,W]).
    scale > 1: slices are up-sampled once at load time (image bilinear, label nearest) and the
    affinity offsets are multiplied by `scale`; the crop size in pixels stays `patch` (field of view / scale).
    lsd_sigma > 0: 6 LSD channels (mask = foreground) are appended to target/mask. LSDs are computed
    on an enlarged crop (margin 4*sigma) before augmentation so that crop borders do not truncate the Gaussian window,
    then trimmed and augmented with the proper vector transforms (lsd.augment_lsd)."""

    def __init__(self, domain, indices, method: str, patch: int = 512, n_samples: int = 10_000,
                 augment: bool = True, seed: int = 0, min_std: float = 0.05, max_tries: int = 50,
                 scale: int = 1, lsd_sigma: float = 0.0, offsets=None, neighbour: bool = False):
        self.scale, self.lsd_sigma = int(scale), float(lsd_sigma)
        self.images = [upsample_image(normalize(domain.load_image(i)), self.scale) for i in indices]
        # int32 in memory (halves RAM; the x2 arm holds 69 x 4096^2 labels); crops are cast to int64 on output
        self.labels = [upsample_label(domain.load_label(i).astype(np.int32), self.scale) for i in indices]
        # neighbour=True: also return the same crop of an adjacent training slice
        # (z+1 if it is in the training list, else z-1, else the slice itself); raw ids are 3D-consistent in all domains.
        self.neighbour = neighbour
        self.nb = []
        for k, i in enumerate(indices):
            j = k + 1 if k + 1 < len(indices) and indices[k + 1] == i + 1 else (k - 1 if k > 0 and indices[k - 1] == i - 1 else k)
            self.nb.append(j)
        self.patch, self.n, self.augment = patch, n_samples, augment
        self.target = Target(method, scale=self.scale, offsets=offsets)
        self.rng = np.random.default_rng(seed)
        # Reject (near-)constant crops: InstanceNorm2d yields NaN gradients on zero-variance inputs
        # Such crops come from zero-padded regions / large uniform somata.
        self.min_std, self.max_tries, self.n_rejected = min_std, max_tries, 0
        self.margin = int(np.ceil(4 * self.lsd_sigma)) if self.lsd_sigma > 0 else 0

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        for _ in range(self.max_tries):
            k = self.rng.integers(len(self.images))
            img_full, lab_full = self.images[k], self.labels[k]
            H, W = img_full.shape
            y = self.rng.integers(0, H - self.patch + 1); x = self.rng.integers(0, W - self.patch + 1)
            img = img_full[y:y + self.patch, x:x + self.patch]; lab = lab_full[y:y + self.patch, x:x + self.patch]
            if float(img.std()) >= self.min_std:
                break
            self.n_rejected += 1
        lsd = None
        if self.lsd_sigma > 0:
            M = self.margin
            ys, ye, xs, xe = max(0, y - M), min(H, y + self.patch + M), max(0, x - M), min(W, x + self.patch + M)
            big = lsd2d(lab_full[ys:ye, xs:xe], self.lsd_sigma)
            lsd = big[:, y - ys:y - ys + self.patch, x - xs:x - xs + self.patch]
        k90, fx, fy = 0, False, False
        if self.augment:
            k90 = int(self.rng.integers(4)); fx = bool(self.rng.random() < 0.5); fy = bool(self.rng.random() < 0.5)
            if lsd is not None:
                lsd = augment_lsd(lsd, lab > 0, k90, fx, fy)
            img, lab = np.rot90(img, k90), np.rot90(lab, k90)
            if fx:
                img, lab = img[:, ::-1], lab[:, ::-1]
            if fy:
                img, lab = img[::-1], lab[::-1]
            # light intensity augmentation
            img = img * self.rng.uniform(0.9, 1.1) + self.rng.uniform(-0.1, 0.1)
        img, lab = np.ascontiguousarray(img), np.ascontiguousarray(lab)
        t, m = self.target(lab)
        extra = ()
        if self.neighbour:
            j = self.nb[k]
            img2 = self.images[j][y:y + self.patch, x:x + self.patch]; lab2 = self.labels[j][y:y + self.patch, x:x + self.patch]
            img2, lab2 = np.rot90(img2, k90), np.rot90(lab2, k90)
            if fx:
                img2, lab2 = img2[:, ::-1], lab2[:, ::-1]
            if fy:
                img2, lab2 = img2[::-1], lab2[::-1]
            extra = (torch.from_numpy(np.ascontiguousarray(img2)[None].astype(np.float32)), torch.from_numpy(np.ascontiguousarray(lab2).astype(np.int64)))
        if lsd is not None:
            fg = (lab > 0)[None].astype(np.float32)
            t = np.concatenate([t, lsd.astype(np.float32)], 0)
            m = np.concatenate([m, np.repeat(fg, N_LSD, axis=0)], 0)
        return (torch.from_numpy(img[None].astype(np.float32)), torch.from_numpy(t), torch.from_numpy(m), torch.from_numpy(lab.astype(np.int64))) + extra
