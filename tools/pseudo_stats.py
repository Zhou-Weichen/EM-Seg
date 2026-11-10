#!/usr/bin/env python
"""Per-row statistics of a teacher pseudo-label pass: n_inst, max_frac (largest instance / image), fg_frac, bnd_frac.
Usage: python tools/pseudo_stats.py <pseudo_root> [--workers 16]  → <pseudo_root>/stats.csv (file,key,z,n_inst,max_frac,fg_frac,bnd_frac).
Read by emseg.train.data.InstIndex to drop empty / noise slabs (one whole-image instance) from student training."""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import h5py, numpy as np, pandas as pd


def stats(args):
    root, file, key, z = args
    try:
        with h5py.File(Path(root) / file, "r") as h:
            g = h[key] if key else h
            lab = g["pseudo"][z]; fg = g["fg"][z]
    except Exception as e:
        return (file, key, z, -1, 1.0, 0.0, 0.0, f"{type(e).__name__}")
    ids, cnt = np.unique(lab, return_counts=True); nz = ids != 0
    n_inst = int(nz.sum()); max_frac = float(cnt[nz].max() / lab.size) if n_inst else 0.0
    bnd = (lab[:, 1:] != lab[:, :-1]).mean() + (lab[1:] != lab[:-1]).mean()
    return (file, key, z, n_inst, round(max_frac, 4), round(float((fg > 127).mean()), 4), round(float(bnd / 2), 5), "")


def main():
    p = argparse.ArgumentParser(); p.add_argument("root"); p.add_argument("--workers", type=int, default=16); a = p.parse_args()
    rows = pd.concat([pd.read_csv(f) for f in sorted(Path(a.root).glob("rows_*.csv"))]).fillna({"key": ""})
    jobs = [(a.root, r.file, str(r.key), int(r.z)) for r in rows.itertuples()]
    out = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, s in enumerate(ex.map(stats, jobs, chunksize=64)):
            out.append(s)
            if i % 5000 == 0: print(i, len(jobs), flush=True)
    df = pd.DataFrame(out, columns=["file", "key", "z", "n_inst", "max_frac", "fg_frac", "bnd_frac", "err"])
    df.to_csv(Path(a.root) / "stats.csv", index=False)
    print("rows", len(df), "errors", int((df.err != "").sum()), "n_inst<8", int((df.n_inst < 8).sum()), "max_frac>0.6", int((df.max_frac > 0.6).sum()))
    print("exit=0")


if __name__ == "__main__":
    main()
