"""Label-quality audit (edge-alignment check, after Depth Anything 3) for the cell-labelled datasets.
Per dataset, N random 512² crops at native resolution (valid fraction > 0.5): GT boundary B (instance transitions, 1 px)
vs image membrane ridge R (ridge prior >= 0.3).
  hit1 / hit3 : fraction of GT boundary pixels with a ridge pixel within 1 / 3 px (label → image agreement)
  best_shift  : integer (dy,dx) in [-4,4]² maximising hit1 (label misregistration if != 0); gain = hit1(best) − hit1(0)
  thick       : mean GT background (label 0) fraction inside valid (gap-style vs touching labels)
python tools/label_audit.py <out.json> [N]   (EMDATA_ROOT selects the corpus)
"""
import json, os, sys
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, h5py
from scipy import ndimage as ndi
from emseg.seg.prior import ridge_prior, boundary_prior

ROOT = Path(os.environ.get("EMDATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
P = 512


def one(args):
    f, key, z, H, W, ds, seed = args
    rng = np.random.default_rng(seed); c = min(P, H, W)
    h = h5py.File(ROOT / f, "r"); g = h[key] if key else h
    for _ in range(10):
        y = int(rng.integers(0, H - c + 1)); x = int(rng.integers(0, W - c + 1))
        sl = (slice(y, y + c), slice(x, x + c)) if key else (z, slice(y, y + c), slice(x, x + c))
        msk = g["masks/cell"][sl] > 0
        if msk.mean() > 0.5:
            break
    else:
        return None
    img = g["raw"][sl].astype(np.float32); lab = g["labels/cell"][sl].astype(np.int64)
    img = (img - img.mean()) / (img.std() + 1e-6)
    R = ridge_prior(img) >= 0.3
    B = boundary_prior(lab, dilate=0) > 0
    B &= ndi.binary_erosion(msk, iterations=4)
    if B.sum() < 50:
        return None
    d = ndi.distance_transform_edt(~R)
    out = {"ds": ds, "hit1": float((d[B] <= 1).mean()), "hit3": float((d[B] <= 3).mean()), "bg": float((lab[msk] == 0).mean())}
    best = (out["hit1"], 0, 0); By, Bx = np.nonzero(B)
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            yy, xx = By + dy, Bx + dx; ok = (yy >= 0) & (yy < c) & (xx >= 0) & (xx < c)
            v = float((d[yy[ok], xx[ok]] <= 1).mean())
            if v > best[0] + 1e-9:
                best = (v, dy, dx)
    out.update(best_hit1=best[0], dy=best[1], dx=best[2])
    return out


if __name__ == "__main__":
    out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    m = pd.read_csv(ROOT / "manifests/em_only.csv", low_memory=False)
    m = m[(m.split == "train") & m.classes.str.contains("cell", na=False)]
    jobs = []
    for ds, g in m.groupby("dataset"):
        for i, r in enumerate(g.sample(min(N, len(g)), random_state=0).itertuples()):
            jobs.append((r.file, r.key if isinstance(r.key, str) else "", int(r.z) if r.z == r.z else -1, int(r.H), int(r.W), ds, i))
    with mp.get_context("fork").Pool(min(48, os.cpu_count() or 8)) as pool:
        R = [r for r in pool.map(one, jobs) if r]
    df = pd.DataFrame(R)
    q = m.groupby("dataset").quality.first()
    s = df.groupby("ds").agg(n=("hit1", "size"), hit1=("hit1", "mean"), hit3=("hit3", "mean"), best_hit1=("best_hit1", "mean"),
                             shift_nonzero=("dy", lambda v: float(((v != 0) | (df.loc[v.index, "dx"] != 0)).mean())), bg=("bg", "mean"))
    s["gain"] = s.best_hit1 - s.hit1; s["quality"] = q.reindex(s.index)
    s = s.sort_values("hit1")
    json.dump({"per_dataset": s.reset_index().to_dict(orient="records"), "n_per_dataset": N}, open(out, "w"), indent=2)
    pd.set_option("display.width", 200); print(s.round(3).to_string())
