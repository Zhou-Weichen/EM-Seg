"""Visualise teacher pseudo-labels: for each dataset in <pseudo_root>/rows_*.csv pick --per-dataset random rows and
save <out>/<dataset>_<i>.png = [image | image + boundaries (red) | random-colour instances | teacher fg], 1024-px crops at
the native grid (whole image if smaller). Also writes <out>/summary.csv (n_instances, median instance area px, fg fraction,
boundary-pixel fraction) per shown row.
usage: python tools/pseudo_vis.py <pseudo_root> <out_dir> [--per-dataset 4 --crop 1024 --seed 0]
"""
import argparse, glob, os
from pathlib import Path
import h5py, numpy as np, pandas as pd
from PIL import Image

p = argparse.ArgumentParser(); p.add_argument("pseudo_root"); p.add_argument("out"); p.add_argument("--per-dataset", type=int, default=4)
p.add_argument("--crop", type=int, default=1024); p.add_argument("--seed", type=int, default=0); p.add_argument("--data-root", default=os.environ.get("EMDATA_ROOT", str(Path(__file__).resolve().parents[1] / "data")))
a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
rows = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{a.pseudo_root}/rows_*.csv"))]).fillna({"key": ""})
rows["dataset"] = rows.file.str.split("/").str[1]
summ = []
for ds, g in rows.groupby("dataset"):
    for i, r in enumerate(g.sample(min(len(g), a.per_dataset), random_state=a.seed).itertuples()):
        key = r.key if isinstance(r.key, str) and r.key else ""; z = int(r.z)
        with h5py.File(Path(a.data_root) / r.file, "r") as h, h5py.File(Path(a.pseudo_root) / r.file, "r") as hp:
            rd = lambda d: d[:] if z < 0 or key else d[z]
            img = rd(h[f"{key}/raw"] if key else h["raw"]); seg = rd(hp[f"{key}/pseudo"] if key else hp["pseudo"]); fg = rd(hp[f"{key}/fg"] if key else hp["fg"])
        H, W = img.shape; c = min(a.crop, H, W); y0 = int(rng.integers(0, H - c + 1)); x0 = int(rng.integers(0, W - c + 1))
        img, seg, fg = img[y0:y0 + c, x0:x0 + c], seg[y0:y0 + c, x0:x0 + c].astype(np.int64), fg[y0:y0 + c, x0:x0 + c]
        lo, hi = np.percentile(img, (1, 99)); g8 = np.clip((img - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        bnd = np.zeros_like(seg, bool); bnd[:, 1:] |= seg[:, 1:] != seg[:, :-1]; bnd[1:, :] |= seg[1:, :] != seg[:-1, :]
        rgb = np.stack([g8] * 3, -1); ov = rgb.copy(); ov[bnd] = (255, 0, 0)
        ids, inv = np.unique(seg, return_inverse=True); cols = rng.integers(40, 255, (len(ids), 3)).astype(np.uint8); cols[ids == 0] = 0
        col = cols[inv.reshape(seg.shape)]; col = (0.6 * col + 0.4 * rgb).astype(np.uint8)
        fgi = np.stack([fg] * 3, -1).astype(np.uint8)
        panel = np.concatenate([rgb, ov, col, fgi], 1)
        if c > 768:
            panel = np.asarray(Image.fromarray(panel).resize((panel.shape[1] * 768 // c, 768), Image.BILINEAR))
        Image.fromarray(panel).save(out / f"{ds}_{i}.png")
        areas = np.bincount(seg.ravel())[1:]; areas = areas[areas > 0]
        summ.append(dict(dataset=ds, file=r.file, key=key, z=z, crop=c, n_inst=len(areas), med_area=float(np.median(areas)) if len(areas) else 0,
                         fg_frac=float((fg > 127).mean()), bnd_frac=float(bnd.mean())))
pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
print(pd.DataFrame(summ).groupby("dataset")[["n_inst", "med_area", "fg_frac", "bnd_frac"]].median().round(3).to_string())
