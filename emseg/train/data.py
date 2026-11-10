"""Instance-relation crops (P15: I1 superpixels / I2 ground truth / I3 teacher pseudo-labels).

Each item gives an augmented crop and two id maps that define pair relations:
  pos (H,W) int: pixels with equal non-negative pos id are "same object"   (-1 = no positive information)
  neg (H,W) int: pixels with different neg ids (both >= 0) are "different objects"; pairs that are neither are ignored
  fg (H,W) float + fg_valid (H,W) bool: foreground target (ground truth / teacher only)
  valid (H,W) bool: pixels that carry any supervision
Sources: "gt" (labels/cell + masks/cell; pos = neg = label incl. background 0), "sp" (pos = fine superpixels, neg = coarse
regions after merging borders with mean ridge < merge_t), "pl" (teacher pseudo-labels from emseg.pseudolabel; pos =
neg = pseudo id incl. background 0, fg from the teacher).
"""
from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import Dataset

from emseg.seg.prior import ridge_prior
from emseg.seg.superpixels import superpixels, _merge_weak


def _res(v) -> float:
    """Pixel size in nm from a manifest cell; NaN when unknown."""
    try:
        v = float(v)
    except Exception:
        return float("nan")
    return v if v > 0 else float("nan")


def _resize(a: np.ndarray, size: int, mode: str) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None]
    down = size < min(a.shape)
    t = F.interpolate(t, size=(size, size), mode=mode, antialias=(down and mode == "bilinear"), align_corners=False if mode == "bilinear" else None)
    return t[0, 0].numpy()


def _photometric(a: np.ndarray, rng, strong: bool) -> np.ndarray:
    """a normalised (mean 0, std 1). Contrast/gamma/brightness, blur, noise, illumination gradient, line artefacts."""
    g = rng.uniform(0.6, 1.5) if strong else rng.uniform(0.85, 1.15)
    a = a * g + rng.uniform(-0.3, 0.3) * (1 if strong else 0.3)
    if strong and rng.random() < 0.5:                                    # gamma on a [0,1] rescale
        lo, hi = np.percentile(a, (0.5, 99.5)); u = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)
        a = (u ** rng.uniform(0.7, 1.4)) * (hi - lo) + lo
    if rng.random() < (0.3 if strong else 0.1):
        a = ndi.gaussian_filter(a, rng.uniform(0.3, 1.5 if strong else 0.7))
    if rng.random() < (0.5 if strong else 0.2):
        a = a + rng.normal(0, rng.uniform(0.02, 0.2 if strong else 0.08), a.shape).astype(np.float32)
    if strong and rng.random() < 0.3:                                    # smooth illumination / charging gradient
        H, W = a.shape; yy, xx = np.mgrid[0:H, 0:W] / max(H, W)
        a = a + rng.uniform(-0.6, 0.6) * (np.cos(rng.uniform(0, 2 * np.pi)) * yy + np.sin(rng.uniform(0, 2 * np.pi)) * xx)
    if strong and rng.random() < 0.1:                                    # scan-line artefacts
        rows = rng.random(a.shape[0]) < 0.02; a[rows] += rng.normal(0, 0.8, (rows.sum(), 1))
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a.astype(np.float32)


def parse_sched(spec: str):
    """"f0:t,f1" -> (f0, t, f1): GT sampling fraction f0 while progress < t, then f1."""
    a, f1 = spec.split(","); f0, t = a.split(":")
    return float(f0), float(t), float(f1)


class InstIndex:
    """rows: (file, key, z, H, W, res, weight, dataset, source)."""
    def __init__(self, manifest_csv: str, sources: dict[str, str], exclude=("snemi3d",), split: str = "train", gt_frac: float | None = None,
                 pseudo_root: str = "", pl_power: float | None = None, pl_min_inst: int = 8, pl_max_frac: float = 0.6, gt_power: float | None = None):
        """sources: {'gt': 'cell' | '', 'other': 'sp' | 'pl' | ''} — rows with the cell class use sources['gt'] when set,
        all other rows use sources['other'] when set; rows with an empty source are dropped. gt_frac: total sampling
        probability of the ground-truth rows (None = manifest weights). manifest_csv may be a comma-separated list.
        pl_power: if set, pseudo-label rows are re-weighted so that each dataset has mass n_rows**pl_power (manifest weights
        of unlabelled pools are not comparable with the corpus weights)."""
        m = pd.concat([pd.read_csv(f, low_memory=False) for f in manifest_csv.split(",")], ignore_index=True)
        m = m[(m.split == split) & ~m.dataset.isin(list(exclude))]
        is_cell = m.classes.str.contains("cell", na=False)
        pl = None
        if sources.get("other") == "pl":                 # only rows that were pseudo-labelled (emseg.pseudolabel rows_*.csv)
            r_ = pd.concat([pd.read_csv(f) for f in sorted(Path(pseudo_root).glob("rows_*.csv"))]).fillna({"key": ""})
            st = Path(pseudo_root) / "stats.csv"                        # tools/p15_pseudo_stats.py: drop empty / noise slabs
            if st.exists():
                s_ = pd.read_csv(st).fillna({"key": ""}); bad = (s_.n_inst < pl_min_inst) | (s_.max_frac > pl_max_frac)
                drop = set(zip(s_.file[bad], s_.key[bad].astype(str), s_.z[bad].astype(int)))
                print(f"InstIndex: pseudo stats {len(s_)} rows, dropping {len(drop)} (n_inst<{pl_min_inst} or max_frac>{pl_max_frac})", flush=True)
            else:
                drop = set()
            pl = set(zip(r_.file, r_.key.astype(str), r_.z.astype(int))) - drop
        rows = []
        for r, c in zip(m.itertuples(), is_cell):
            src = ("gt" if sources.get("gt") else sources.get("other")) if c else sources.get("other")
            if not src:
                continue
            if src == "pl" and (r.file, r.key if isinstance(r.key, str) else "", int(r.z) if r.z == r.z else -1) not in pl:
                continue
            rows.append((r.file, r.key if isinstance(r.key, str) else "", int(r.z) if r.z == r.z else -1, int(r.H), int(r.W),
                         _res(r.res_xy_nm), float(r.weight), r.dataset, src))
        self.rows = rows
        w = np.array([r[6] for r in rows], np.float64)
        g = np.array([r[8] == "gt" for r in rows], bool)
        self.n_per_dataset = {}
        for r in rows:
            self.n_per_dataset[r[7]] = self.n_per_dataset.get(r[7], 0) + 1
        if pl_power is not None and (~g).any():
            ispl = np.array([r[8] == "pl" for r in rows], bool)
            if ispl.any():
                n = np.array([self.n_per_dataset[r[7]] for r in rows], np.float64)
                w[ispl] = n[ispl] ** pl_power / n[ispl]
        if gt_power is not None and g.any():                                  # dataset mass ∝ n_rows**gt_power; manifest weights only shape the within-dataset distribution
            for d in set(r[7] for r, gg in zip(rows, g) if gg):
                sel = np.array([r[7] == d for r in rows], bool) & g
                w[sel] = w[sel] / max(w[sel].sum(), 1e-12) * sel.sum() ** gt_power
        self.is_gt = g
        self.p_gt = w * g; self.p_other = w * ~g                            # per-group sampling distributions (for schedules)
        if g.any(): self.p_gt = self.p_gt / self.p_gt.sum()
        if (~g).any(): self.p_other = self.p_other / self.p_other.sum()
        if gt_frac is not None and g.any() and (~g).any():
            w[g] *= gt_frac / w[g].sum(); w[~g] *= (1 - gt_frac) / w[~g].sum()
        self.p = w / w.sum()
        self.datasets = sorted(set(r[7] for r in rows)); self.ds_id = {d: i for i, d in enumerate(self.datasets)}


class InstCropDataset(Dataset):
    def __init__(self, index: InstIndex, data_root: str = "data", pseudo_root: str = "", patch: int = 512, n_samples: int = 10_000,
                 seed: int = 0, res_lo: float = 4.0, res_hi: float = 16.0, sp_fine: dict | None = None, sp_merge: float = 0.6,
                 gt_sched: tuple | None = None):
        """gt_sched (f0, t, f1): GT sampling fraction f0 while i / n_samples < t, then f1 (i ~ iteration * batch)."""
        self.idx, self.root, self.proot, self.patch, self.n, self.seed = index, Path(data_root), Path(pseudo_root) if pseudo_root else None, patch, n_samples, seed
        self.gt_sched = gt_sched; self.offset = 0                      # offset: samples already consumed before a resume (iter * batch)
        self.res_lo, self.res_hi = res_lo, res_hi
        self.sp_fine = sp_fine or dict(threshold=0.3, sigma_seeds=2.5, min_size=50, merge_t=0.0); self.sp_merge = sp_merge
        self._files: dict = {}; self.rng = None

    def __len__(self):
        return self.n

    def __getstate__(self):
        d = self.__dict__.copy(); d["_files"] = {}; d["rng"] = None; return d

    max_open = 32      # bounded LRU of open HDF5 handles per worker. The pool is one file per slab (>100k raw + >100k pseudo files);
                       # an unbounded dict would keep every handle (~1-3 MB each) open and exhaust host memory.

    def _h5(self, p: Path):
        k = str(p); h = self._files.pop(k, None)
        if h is None:
            while len(self._files) >= self.max_open:
                self._files.pop(next(iter(self._files))).close()
            h = h5py.File(p, "r")
        self._files[k] = h                                       # re-insert at the end: dict order = recency
        return h

    def _read(self, h, key, z, name, y, x, c):
        d = h[f"{key}/{name}"] if key else h[name]
        return d[y:y + c, x:x + c] if z < 0 else d[z, y:y + c, x:x + c]

    def __getitem__(self, i):
        if self.rng is None:
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(self.seed * 100_003 + (info.id if info else 0) * 7919 + i)
        rng, P = self.rng, self.patch
        for _ in range(20):
            if self.gt_sched is not None and self.idx.is_gt.any() and (~self.idx.is_gt).any():
                f0, t, f1 = self.gt_sched; frac = f0 if (i + self.offset) / self.n < t else f1
                pr = self.idx.p_gt if rng.random() < frac else self.idx.p_other
            else:
                pr = self.idx.p
            f, key, z, H, W, res, _, ds, src = self.idx.rows[rng.choice(len(self.idx.rows), p=pr)]
            s = float(rng.uniform(0.75, 1.33)) if math.isnan(res) else float(np.clip(rng.uniform(self.res_lo, self.res_hi) / res, 0.5, 2.0))
            c = max(32, min(int(round(P * s)), H, W)); y = int(rng.integers(0, H - c + 1)); x = int(rng.integers(0, W - c + 1))
            h = self._h5(self.root / f)
            img = self._read(h, key, z, "raw", y, x, c).astype(np.float32)
            if img.std() < 1.0:
                continue
            lab = msk = fgp = None
            if src == "gt":
                lab = self._read(h, key, z, "labels/cell", y, x, c).astype(np.int64); msk = self._read(h, key, z, "masks/cell", y, x, c) > 0
                if msk.mean() < 0.2:
                    continue
            elif src == "pl":
                hp = self._h5(self.proot / f)
                lab = self._read(hp, key, z, "pseudo", y, x, c).astype(np.int64); msk = np.ones_like(lab, bool)
                fgp = self._read(hp, key, z, "fg", y, x, c).astype(np.float32) / 255
            break
        else:
            raise RuntimeError("no usable crop")
        if c != P:
            img = _resize(img, P, "bilinear"); ix = np.minimum((np.arange(P) * c / P + c / (2 * P)).astype(int), c - 1)
            if lab is not None:
                lab = lab[np.ix_(ix, ix)]; msk = msk[np.ix_(ix, ix)]
            if fgp is not None:
                fgp = fgp[np.ix_(ix, ix)]
        k90 = int(rng.integers(4)); fx = rng.random() < 0.5; fy = rng.random() < 0.5
        def aug(a):
            a = np.rot90(a, k90)
            if fx: a = a[:, ::-1]
            if fy: a = a[::-1]
            return np.ascontiguousarray(a)
        img = aug((img - img.mean()) / (img.std() + 1e-6))
        prior = ridge_prior(img)
        if src == "sp":
            fine = superpixels(img, prior=prior, **self.sp_fine)
            coarse = _merge_weak(fine, prior, self.sp_merge)
            pos, neg = fine.astype(np.int64), coarse.astype(np.int64); valid = np.ones((P, P), bool)
            fg = np.zeros((P, P), np.float32); fgv = np.zeros((P, P), bool)
        else:
            lab, msk = aug(lab), aug(msk)
            pos = np.where(msk, lab, -1); neg = pos.copy(); valid = msk
            if src == "gt":
                fg = (lab > 0).astype(np.float32); fgv = msk
            else:
                fgp = aug(fgp); fg = (fgp > 0.5).astype(np.float32); fgv = (fgp < 0.3) | (fgp > 0.7)
        x_in = _photometric(img, rng, strong=True)
        t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dt)
        return {"img": t(x_in)[None], "pos": t(pos, torch.int64), "neg": t(neg, torch.int64), "fg": t(fg), "fg_valid": t(fgv, torch.bool),
                "valid": t(valid, torch.bool), "prior": t(prior), "src": torch.tensor({"gt": 0, "sp": 1, "pl": 2}[src]), "ds_id": torch.tensor(self.idx.ds_id[ds])}
