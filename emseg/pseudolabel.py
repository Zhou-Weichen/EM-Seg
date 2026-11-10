"""Offline teacher pseudo-labels for student training (--exp I3).

For training rows without cell labels (excluding held-out datasets), at most --per-dataset rows per dataset: rescale the
slice to ~--target-res nm/px (factor clipped to [0.5, 2]; unknown resolution -> 1), predict fg + disaffinities with the
teacher's affinity head (tiled), mutex watershed (stride 2, fg 0.5, min size 25 — the default of emseg.seg.targets), resize back to
the native grid (nearest) and store under <out>/<file>: `pseudo` uint32 and `fg` uint8 (0–255) with the shape of `raw`
(volumes: only the selected z are written). <out>/rows.csv lists the labelled rows (file, key, z).

python -m emseg.pseudolabel --teacher <teacher ckpt> --out $EMDATA_ROOT/pseudo/<tag> [--shard i --n-shards n]
"""
from __future__ import annotations

import argparse, json, time
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from emseg.paths import emdata_root
from emseg.seg.mws import mws
from emseg.seg.targets import OFFSETS_2D, DEFAULTS
from emseg.finetune import build, infer


def _mws_job(args):
    prob, shape = args
    seg = mws(prob[1:].copy(), [list(o) for o in OFFSETS_2D], DEFAULTS["affinity_mws"], prob[0]).astype(np.uint32)
    fg = (prob[0] * 255).astype(np.uint8)
    if seg.shape != shape:
        iy = np.minimum((np.arange(shape[0]) * seg.shape[0] / shape[0]).astype(int), seg.shape[0] - 1)
        ix = np.minimum((np.arange(shape[1]) * seg.shape[1] / shape[1]).astype(int), seg.shape[1] - 1)
        seg = seg[np.ix_(iy, ix)]; fg = fg[np.ix_(iy, ix)]
    return seg, fg


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True); p.add_argument("--out", required=True)
    p.add_argument("--manifest", default=str(emdata_root() / "manifests/em_only.csv"), help="comma-separated csv list"); p.add_argument("--data-root", default=str(emdata_root()))
    p.add_argument("--exclude", default="snemi3d"); p.add_argument("--datasets", default="", help="comma-separated include list (default: all non-cell datasets)")
    p.add_argument("--per-dataset", type=int, default=400); p.add_argument("--target-res", type=float, default=8.0); p.add_argument("--min-std", type=float, default=5.0)
    p.add_argument("--shard", type=int, default=0); p.add_argument("--n-shards", type=int, default=1); p.add_argument("--workers", type=int, default=24); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m = pd.concat([pd.read_csv(f, low_memory=False) for f in a.manifest.split(",")], ignore_index=True)
    m = m[(m.split == "train") & ~m.dataset.isin(a.exclude.split(",")) & ~m.classes.str.contains("cell", na=False)]
    if a.datasets:
        m = m[m.dataset.isin(a.datasets.split(","))]
    m = m.groupby("dataset", group_keys=False).apply(lambda g: g.sample(min(len(g), a.per_dataset), random_state=a.seed))
    files = sorted(m.file.unique()); files = files[a.shard::a.n_shards]
    m = m[m.file.isin(files)]
    if a.shard == 0:
        json.dump({**vars(a), "n_rows_total_shard0": len(m), "start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(out / "config.json", "w"), indent=2)
    dev = torch.device("cuda"); model, it = build(a.teacher, zero_shot=True); model = model.to(dev).eval()
    rows = []; t0 = time.time(); n = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for f, g in m.groupby("file"):
            src = h5py.File(Path(a.data_root) / f, "r"); dst_p = out / f; dst_p.parent.mkdir(parents=True, exist_ok=True)
            try:
                dst = h5py.File(dst_p, "a")
            except OSError:                                      # truncated by a kill mid-write: redo the file
                dst_p.unlink(); dst = h5py.File(dst_p, "a")
            jobs = []
            def done(key, z):                                    # resume: row already written by an earlier (killed) pass
                try:
                    if key: return f"{key}/pseudo" in dst
                    return "pseudo" in dst and bool(dst["pseudo"][z].max() > 0)
                except Exception:
                    return False
            def write(job):
                nonlocal n
                (key, z, shape), j = job; seg, fg = j.get()
                if key:
                    for name, arr in (("pseudo", seg), ("fg", fg)):
                        if f"{key}/{name}" in dst: del dst[f"{key}/{name}"]
                        dst.create_dataset(f"{key}/{name}", data=arr, compression="gzip", chunks=True)
                else:
                    zshape = src["raw"].shape
                    for name, dt in (("pseudo", np.uint32), ("fg", np.uint8)):
                        if name not in dst:
                            dst.create_dataset(name, shape=zshape, dtype=dt, chunks=(1, min(256, zshape[1]), min(256, zshape[2])), compression="gzip")
                    dst["pseudo"][z] = seg; dst["fg"][z] = fg
                rows.append((f, key, z)); n += 1
            for r in g.itertuples():
                key = r.key if isinstance(r.key, str) else ""; z = int(r.z) if r.z == r.z else -1
                if done(key, z):
                    rows.append((f, key, z)); n += 1; continue
                raw = (src[f"{key}/raw"][:] if key else src["raw"][z]).astype(np.float32)
                if raw.std() < a.min_std:
                    continue
                img = (raw - raw.mean()) / (raw.std() + 1e-6)
                res = float(r.res_xy_nm) if r.res_xy_nm == r.res_xy_nm and r.res_xy_nm > 0 else float("nan")
                s = 1.0 if np.isnan(res) else float(np.clip(res / a.target_res, 0.5, 2.0))
                if min(img.shape) * s < 64:
                    continue
                if s != 1.0:                                     # predict + MWS on the rescaled grid (offsets in teacher pixels)
                    img = F.interpolate(torch.from_numpy(img)[None, None], scale_factor=s, mode="bilinear", align_corners=False, antialias=s < 1)[0, 0].numpy()
                prob = infer(model, img, dev)
                jobs.append(((key, z, raw.shape), pool.apply_async(_mws_job, ((prob, raw.shape),))))
                while len(jobs) > 2 * a.workers:
                    write(jobs.pop(0))
            while jobs:
                write(jobs.pop(0))
            dst.close(); src.close()
            print(f"{time.time() - t0:.0f}s {n} rows ({f})", flush=True)
    pd.DataFrame(rows, columns=["file", "key", "z"]).to_csv(out / f"rows_{a.shard}.csv", index=False)
    print("exit=0", flush=True)


if __name__ == "__main__":
    main()
